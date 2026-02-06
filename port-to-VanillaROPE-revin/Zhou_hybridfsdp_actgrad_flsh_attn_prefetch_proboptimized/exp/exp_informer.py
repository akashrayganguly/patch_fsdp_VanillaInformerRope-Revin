"""
exp_informer.py - MAXIMUM PERFORMANCE VERSION

OPTIMIZATIONS:
1. DataPrefetcher with CUDA streams - overlaps data loading with compute
2. no_sync() for gradient accumulation - reduces communication
3. Proper DeviceMesh for HYBRID_SHARD
4. Non-blocking data transfers
5. Optimized DataLoader settings
6. Timing instrumentation

The goal: Eliminate the 0% GPU utilization periods you're seeing
"""

import os
import time
import warnings
import numpy as np
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

# Activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.cuda.amp import GradScaler

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


# =============================================================================
# CUDA STREAM DATA PREFETCHER - KEY TO ELIMINATING IDLE TIME
# =============================================================================
class DataPrefetcher:
    """
    Prefetches data to GPU using a separate CUDA stream.
    
    This overlaps CPU→GPU data transfer with GPU computation:
    
    Without prefetcher:
        [Load batch 1] → [Compute batch 1] → [Load batch 2] → [Compute batch 2]
        GPU idle here ↑                      GPU idle here ↑
    
    With prefetcher:
        [Load batch 1] → [Compute batch 1] → [Compute batch 2] → [Compute batch 3]
                         [Load batch 2    ] → [Load batch 3    ] → [Load batch 4]
        No GPU idle! The next batch is ready when computation finishes.
    """
    
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.iterator = None
    
    def __iter__(self):
        self.iterator = iter(self.loader)
        self._preload()
        return self
    
    def _preload(self):
        """Preload the next batch asynchronously"""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            # Transfer all tensors to GPU asynchronously
            self.next_batch = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    self.next_batch.append(
                        item.to(self.device, non_blocking=True)
                    )
                else:
                    self.next_batch.append(item)
            self.next_batch = tuple(self.next_batch)
    
    def __next__(self):
        # Wait for the preloaded batch to be ready
        torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        # Record that these tensors are being used by current stream
        for item in batch:
            if isinstance(item, torch.Tensor):
                item.record_stream(torch.cuda.current_stream())
        
        # Start loading the next batch
        self._preload()
        
        return batch
    
    def __len__(self):
        return len(self.loader)


# =============================================================================
# AMP CONFIGURATION
# =============================================================================
def get_amp_config():
    """Auto-detect best mixed precision configuration"""
    if not torch.cuda.is_available():
        return {'supported': False, 'dtype': torch.float32, 'use_scaler': False, 'name': 'FP32'}
    
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return {'supported': True, 'dtype': torch.bfloat16, 'use_scaler': False, 'name': 'BF16'}
    
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        return {'supported': True, 'dtype': torch.float16, 'use_scaler': True, 'name': 'FP16'}
    
    return {'supported': False, 'dtype': torch.float32, 'use_scaler': False, 'name': 'FP32'}


# =============================================================================
# GPU DETECTION
# =============================================================================
def get_num_gpus_per_node():
    """Detect GPUs per node"""
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE')
    if local_world_size:
        return int(local_world_size)
    
    slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    if slurm_gpus:
        return int(slurm_gpus)
    
    return torch.cuda.device_count()


# =============================================================================
# HYBRID SHARDING SETUP
# =============================================================================
def setup_hybrid_sharding(args):
    """Setup DeviceMesh for HYBRID_SHARD"""
    if not dist.is_initialized():
        return None
    
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_gpus_per_node = get_num_gpus_per_node()
    num_nodes = world_size // num_gpus_per_node
    
    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"HYBRID SHARDING SETUP")
        print(f"  World: {world_size}, Nodes: {num_nodes}, GPUs/node: {num_gpus_per_node}")
        print(f"{'='*60}")
    
    if num_nodes <= 1:
        if global_rank == 0:
            print("  Single node - HYBRID_SHARD = FULL_SHARD")
        return None
    
    try:
        from torch.distributed.device_mesh import init_device_mesh
        
        mesh = init_device_mesh(
            "cuda",
            (num_nodes, num_gpus_per_node),
            mesh_dim_names=("replicate", "shard")
        )
        
        if global_rank == 0:
            print(f"  ✓ DeviceMesh: ({num_nodes}, {num_gpus_per_node})")
            print(f"  ✓ Shard within node, replicate across nodes")
        
        return mesh
    except Exception as e:
        if global_rank == 0:
            print(f"  ✗ DeviceMesh failed: {e}")
        return None


# =============================================================================
# TIMING UTILITIES
# =============================================================================
class PerformanceMonitor:
    """Track performance metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_time = 0.0
        self.count = 0
    
    def update(self, data_t, fwd_t, bwd_t, opt_t):
        self.data_time += data_t
        self.forward_time += fwd_t
        self.backward_time += bwd_t
        self.optimizer_time += opt_t
        self.count += 1
    
    def summary(self):
        if self.count == 0:
            return "No data"
        total = self.data_time + self.forward_time + self.backward_time + self.optimizer_time
        if total == 0:
            return "No time recorded"
        return (f"Data:{self.data_time/total*100:.0f}% "
                f"Fwd:{self.forward_time/total*100:.0f}% "
                f"Bwd:{self.backward_time/total*100:.0f}% "
                f"Opt:{self.optimizer_time/total*100:.0f}%")


# =============================================================================
# MAIN EXPERIMENT CLASS
# =============================================================================
class Exp_Informer(Exp_Basic):
    """Informer with MAXIMUM PERFORMANCE optimizations"""
    
    def __init__(self, args):
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.amp_config = get_amp_config()
        self.device_mesh = None
        self.perf_monitor = PerformanceMonitor()
        self.use_prefetcher = getattr(args, 'use_prefetcher', True)
        
        super(Exp_Informer, self).__init__(args)
        
        if self._should_print():
            print(f"\n{'='*60}")
            print(f"CONFIGURATION")
            print(f"  FSDP: {args.use_fsdp}, Strategy: {args.fsdp_sharding_strategy}")
            print(f"  AMP: {self.amp_config['name']}")
            print(f"  Grad Accum: {self.gradient_accumulation_steps}")
            print(f"  Data Prefetcher: {self.use_prefetcher}")
            print(f"{'='*60}\n")

    def _build_model(self):
        model_dict = {'informer': Informer, 'informerstack': InformerStack}
        
        e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
        
        model = model_dict[self.args.model](
            self.args.enc_in, self.args.dec_in, self.args.c_out,
            self.args.seq_len, self.args.label_len, self.args.pred_len,
            self.args.factor, self.args.d_model, self.args.n_heads,
            e_layers, self.args.d_layers, self.args.d_ff,
            self.args.dropout, self.args.attn, self.args.embed,
            self.args.freq, self.args.activation,
            self.args.output_attention, self.args.distil, self.args.mix,
            self.device
        ).float()

        if self._should_print():
            params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {params:,}")

        if self.args.use_fsdp and torch.cuda.is_available():
            model = self._wrap_model_with_fsdp(model)
        elif self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _apply_activation_checkpointing(self, model):
        """Apply activation checkpointing"""
        check_fn = lambda m: isinstance(m, (EncoderLayer, DecoderLayer))
        wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)
        
        if self._should_print():
            enc = sum(1 for m in model.modules() if isinstance(m, EncoderLayer))
            dec = sum(1 for m in model.modules() if isinstance(m, DecoderLayer))
            print(f"  Checkpointing: {enc} encoder, {dec} decoder layers")

    def _wrap_model_with_fsdp(self, model):
        """Wrap with FSDP"""
        if self._should_print():
            print(f"\nWrapping with FSDP...")

        # Activation checkpointing BEFORE FSDP
        if self.args.fsdp_activation_checkpointing:
            self._apply_activation_checkpointing(model)

        # Sharding strategy
        strategy_name = self.args.fsdp_sharding_strategy
        strategy = getattr(ShardingStrategy, strategy_name)
        is_hybrid = strategy_name in ['HYBRID_SHARD', '_HYBRID_SHARD_ZERO2']

        # DeviceMesh for hybrid
        if is_hybrid:
            self.device_mesh = setup_hybrid_sharding(self.args)

        # Mixed precision
        mp_policy = None
        if self.args.use_amp and self.amp_config['supported']:
            mp_policy = MixedPrecision(
                param_dtype=self.amp_config['dtype'],
                reduce_dtype=self.amp_config['dtype'],
                buffer_dtype=self.amp_config['dtype'],
            )

        # Auto-wrap policy
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={EncoderLayer, DecoderLayer},
        )

        # FSDP kwargs
        fsdp_kwargs = {
            'auto_wrap_policy': wrap_policy,
            'sharding_strategy': strategy,
            'mixed_precision': mp_policy,
            'device_id': torch.cuda.current_device(),
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'forward_prefetch': True,  # Overlap all-gather with compute
            'limit_all_gathers': True,
            'use_orig_params': True,
        }

        if self.args.fsdp_cpu_offload:
            fsdp_kwargs['cpu_offload'] = CPUOffload(offload_params=True)

        if is_hybrid and self.device_mesh is not None:
            fsdp_kwargs['device_mesh'] = self.device_mesh

        model = FSDP(model, **fsdp_kwargs)

        if self._should_print():
            print(f"  Strategy: {strategy_name}")
            print(f"  DeviceMesh: {'Yes' if self.device_mesh else 'No'}")
            print(f"  Forward Prefetch: Yes")

        return model

    def _get_data(self, flag):
        """Get data with optimized DataLoader settings"""
        args = self.args
        
        data_dict = {
            'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom, 'ECL': Dataset_Custom, 
            'Solar': Dataset_Custom, 'custom': Dataset_Custom,
        }
        Data = data_dict.get(args.data, Dataset_Custom)
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle, drop_last, batch_size = False, True, args.batch_size
        elif flag == 'pred':
            shuffle, drop_last, batch_size = False, False, 1
            Data = Dataset_Pred
        else:
            shuffle, drop_last, batch_size = True, True, args.batch_size

        data_set = Data(
            root_path=args.root_path, data_path=args.data_path, flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features, target=args.target, inverse=args.inverse,
            timeenc=timeenc, freq=args.freq if flag != 'pred' else args.detail_freq,
            cols=args.cols
        )

        if self._should_print():
            print(f'{flag} samples: {len(data_set)}')

        # OPTIMIZED DataLoader settings
        num_workers = getattr(args, 'num_workers', 8)
        prefetch = getattr(args, 'prefetch_factor', 4) if num_workers > 0 else None

        if args.use_fsdp:
            sampler = DistributedSampler(data_set, shuffle=shuffle, drop_last=drop_last)
            loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,  # Sampler handles this
                sampler=sampler,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=True,  # CRITICAL for fast transfer
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=prefetch,
            )
        else:
            loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, drop_last=drop_last, pin_memory=True
            )

        return data_set, loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """Validation"""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        with torch.no_grad():
            for batch in vali_loader:
                pred, true = self._process_one_batch(vali_data, *batch)
                total_loss += criterion(pred, true).detach()
                count += 1

        avg_loss = total_loss / count
        if self.args.use_fsdp and dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        self.model.train()
        return avg_loss.item()

    def train(self, setting):
        """
        OPTIMIZED Training Loop
        
        Key optimizations:
        1. DataPrefetcher for async data loading
        2. no_sync() for gradient accumulation
        3. Non-blocking tensor transfers
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if self._should_print() and not os.path.exists(path):
            os.makedirs(path)

        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True,
            use_fsdp=self.args.use_fsdp, global_rank=self.args.global_rank
        )

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        use_amp = self.args.use_amp and self.amp_config['supported']
        use_scaler = use_amp and self.amp_config['use_scaler']
        scaler = GradScaler() if use_scaler else None

        grad_accum = self.gradient_accumulation_steps

        if self._should_print():
            print(f"\n{'='*60}")
            print(f"TRAINING")
            print(f"  Steps/epoch: {train_steps}")
            print(f"  Grad accum: {grad_accum}")
            print(f"  Prefetcher: {self.use_prefetcher}")
            print(f"{'='*60}\n")

        for epoch in range(self.args.train_epochs):
            train_loss_sum = torch.tensor(0.0, device=self.device)
            train_loss_count = 0
            
            if self.args.use_fsdp and hasattr(train_loader, 'sampler'):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            self.perf_monitor.reset()
            epoch_start = time.time()
            iter_start = time.time()

            # Use prefetcher for async data loading
            if self.use_prefetcher:
                data_iter = DataPrefetcher(train_loader, self.device)
            else:
                data_iter = train_loader

            for i, batch in enumerate(data_iter):
                data_time = time.time() - iter_start
                
                # Determine sync
                is_accum = (i + 1) % grad_accum != 0
                is_last = (i + 1) == train_steps
                should_sync = not is_accum or is_last

                # no_sync context for gradient accumulation
                sync_ctx = self.model.no_sync() if (self.args.use_fsdp and not should_sync) else nullcontext()

                fwd_start = time.time()
                
                with sync_ctx:
                    if self.use_prefetcher:
                        # Data already on GPU from prefetcher
                        pred, true = self._process_one_batch_prefetched(train_data, *batch)
                    else:
                        pred, true = self._process_one_batch(train_data, *batch)
                    
                    loss = criterion(pred.float(), true.float())
                    loss_scaled = loss / grad_accum

                    fwd_time = time.time() - fwd_start

                    with torch.no_grad():
                        train_loss_sum += loss.detach()
                        train_loss_count += 1

                    bwd_start = time.time()
                    if use_scaler:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    bwd_time = time.time() - bwd_start

                # Optimizer step
                opt_time = 0
                if should_sync:
                    opt_start = time.time()
                    if use_scaler:
                        if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            self.model.clip_grad_norm_(self.args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                            self.model.clip_grad_norm_(self.args.max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad()
                    opt_time = time.time() - opt_start

                self.perf_monitor.update(data_time, fwd_time, bwd_time, opt_time)

                # Logging
                if (i + 1) % 100 == 0 and self._should_print():
                    speed = (time.time() - epoch_start) / (i + 1)
                    print(f"  Epoch {epoch+1} | Iter {i+1}/{train_steps} | "
                          f"Loss: {loss.item():.6f} | Speed: {speed:.2f}s/iter")
                    print(f"    Timing: {self.perf_monitor.summary()}")

                iter_start = time.time()

            # Epoch end
            epoch_time = time.time() - epoch_start
            train_loss = (train_loss_sum / train_loss_count).item()

            if self.args.use_fsdp and dist.is_initialized():
                loss_t = torch.tensor(train_loss, device=self.device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                train_loss = loss_t.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self._should_print():
                print(f"\n  Epoch {epoch+1} | {epoch_time:.0f}s | "
                      f"Train: {train_loss:.6f} | Vali: {vali_loss:.6f} | Test: {test_loss:.6f}")
                print(f"    Timing breakdown: {self.perf_monitor.summary()}\n")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if self._should_print():
                    print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        # Load best
        best_path = path + '/checkpoint.pth'
        self._load_checkpoint(best_path)
        return self.model

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """Process batch with data on CPU"""
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device, non_blocking=True)

        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=True)

        return outputs, batch_y

    def _process_one_batch_prefetched(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """Process batch with data already on GPU (from prefetcher)"""
        # Data is already on GPU, just ensure correct dtype
        batch_x = batch_x.float()
        batch_y_cpu = batch_y.float()  # Keep copy for labels
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], 
                                  device=self.device, dtype=torch.float32)
        else:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], 
                                 device=self.device, dtype=torch.float32)

        dec_inp = torch.cat([batch_y_cpu[:, :self.args.label_len, :], dec_inp], dim=1)

        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        true = batch_y_cpu[:, -self.args.pred_len:, f_dim:]

        return outputs, true

    def test(self, setting):
        """Test"""
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                pred, true = self._process_one_batch(test_data, *batch)
                preds.append(pred.cpu().numpy())
                trues.append(true.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if self._should_print():
            print(f'\nTest: MSE={mse:.6f}, MAE={mae:.6f}')
            folder = f'./results/{setting}/'
            os.makedirs(folder, exist_ok=True)
            np.save(f'{folder}metrics.npy', [mae, mse, rmse, mape, mspe])

    def predict(self, setting, load=False):
        """Predict"""
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            self._load_checkpoint(f'{self.args.checkpoints}/{setting}/checkpoint.pth')

        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for batch in pred_loader:
                pred, _ = self._process_one_batch(pred_data, *batch)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        
        if self._should_print():
            folder = f'./results/{setting}/'
            os.makedirs(folder, exist_ok=True)
            np.save(f'{folder}prediction.npy', preds)

    def _load_checkpoint(self, path):
        """Load checkpoint"""
        if not os.path.exists(path):
            return
        if self._should_print():
            print(f"Loading {path}")
        if self.args.use_fsdp:
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _should_print(self):
        if hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            return self.args.global_rank == 0
        return True

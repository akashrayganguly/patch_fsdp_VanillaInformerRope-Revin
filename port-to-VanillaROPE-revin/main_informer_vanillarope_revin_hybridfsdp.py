#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAXIMUM PERFORMANCE Main Script for VanillaInformerROPE-Revin-HybridFSDP

Target: 12 nodes × 2 A100 GPUs = 24 GPUs total

Changes from previous version:
1. PROJECT_DIR updated to match VanillaInformerROPE-Revin-HybridFSDP repo
2. use_prefetcher = True (CUDA stream data prefetching)
3. num_workers = 6 (safe for 8 CPUs per GPU on 2-GPU nodes)
4. prefetch_factor = 4 (more batches ready)
5. Debug timing enabled by default
6. use_revin, channel_period, max_len exposed in args
"""

import sys
import os
import torch
import torch.distributed as dist
from datetime import timedelta

# ============================================================================
# CRITICAL: Point to the correct repo directory
# ============================================================================
PROJECT_DIR = 'VanillaInformerROPE-Revin-hybridfsdp'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.tools import dotdict
from exp.exp_informer import Exp_Informer


def get_available_gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        num_gpus = int(os.environ.get('LOCAL_WORLD_SIZE', get_available_gpus()))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        num_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', get_available_gpus()))
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(num_gpus)
    else:
        rank, world_size, local_rank = 0, 1, 0
        num_gpus = max(1, get_available_gpus())
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['LOCAL_WORLD_SIZE'] = str(num_gpus)

    return rank, world_size, local_rank, num_gpus


def init_distributed_mode(args):
    if args.use_fsdp:
        rank, world_size, local_rank, num_gpus = setup_distributed()

        args.global_rank = rank
        args.world_size = world_size
        args.local_rank = local_rank
        args.num_gpus_per_node = num_gpus

        if not dist.is_initialized() and world_size > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://',
                                    world_size=world_size, rank=rank,
                                    timeout=timedelta(minutes=30))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = torch.device(f'cuda:{local_rank}')
            args.gpu = local_rank
        else:
            args.device = torch.device('cpu')
            args.gpu = None

        if rank == 0:
            num_nodes = world_size // num_gpus
            print(f"\n{'='*60}")
            print(f"DISTRIBUTED: {num_nodes} nodes x {num_gpus} GPUs = {world_size} total")
            print(f"{'='*60}\n")
    else:
        args.global_rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.num_gpus_per_node = get_available_gpus()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_args():
    args = dotdict()

    # Model
    args.model = 'informer'

    # Data
    args.data = 'custom'
    args.root_path = './ETDataset/ETT-small/'
    args.data_path = 'nc_by_meff_multiple_mtau_ECL.csv'
    args.features = 'M'
    args.target = 'data9'
    args.freq = 'h'
    args.checkpoints = './checkpoints'
    args.cols = None
    args.inverse = False

    # Sequences
    args.seq_len = 96 * 321
    args.label_len = 48 * 321
    args.pred_len = 96 * 321

    # Model params
    args.enc_in = 9
    args.dec_in = 9
    args.c_out = 9
    args.factor = 5
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.s_layers = [3, 2, 1]
    args.d_ff = 2048
    args.dropout = 0.05
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.distil = True
    args.output_attention = False
    args.mix = True
    args.padding = 0

    # RevIN and Channel Mixing (new params for ROPE-RevIN model)
    args.use_revin = True
    args.channel_period = 321
    args.channel_mix_size = args.channel_period       # Set to int (e.g. 321) to enable channel mixing
    args.max_len = 200000

    # Training
    args.batch_size = 16
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = True
    args.train_epochs = 10
    args.patience = 3

    # Gradient accumulation
    args.gradient_accumulation_steps = 4
    args.max_grad_norm = 1.0

    # ==========================================================================
    # DATA LOADING
    # With 2 A100 GPUs per node and --cpus-per-task=24, each GPU gets ~12 CPUs.
    # num_workers=6 leaves headroom for the main process and OS.
    # ==========================================================================
    args.num_workers = 6
    args.prefetch_factor = 4
    args.use_prefetcher = True         # CUDA stream prefetching

    # Experiment
    args.itr = 1
    args.des = 'hybrid_optimized'
    args.seed = 2021

    # GPU
    args.use_gpu = torch.cuda.is_available()
    args.use_fsdp = True
    args.use_multi_gpu = False
    args.gpu = 0
    args.devices = 'auto'
    args.device_ids = None

    # FSDP
    args.fsdp_sharding_strategy = 'HYBRID_SHARD'
    args.fsdp_auto_wrap_min_params = 1e6
    args.fsdp_backward_prefetch = 'BACKWARD_PRE'
    args.fsdp_cpu_offload = False
    args.fsdp_activation_checkpointing = False

    return args


def setup_data_parser(args):
    data_parser = {
        'custom': {'data': 'nc_by_meff_multiple_mtau_ECL.csv', 'T': 'data9',
                   'M': [9, 9, 9], 'S': [1, 1, 1], 'MS': [9, 9, 1]},
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    }
    if args.data in data_parser:
        info = data_parser[args.data]
        args.data_path = info['data']
        args.target = info['T']
        args.enc_in, args.dec_in, args.c_out = info[args.features]


def print_config(args):
    if args.use_fsdp and args.global_rank != 0:
        return

    num_nodes = args.world_size // args.num_gpus_per_node
    effective_batch = args.batch_size * args.gradient_accumulation_steps * args.world_size

    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Distributed: {num_nodes} nodes x {args.num_gpus_per_node} GPUs")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} accum x {args.world_size} = {effective_batch}")
    print(f"Strategy: {args.fsdp_sharding_strategy}")
    print(f"Data loading: {args.num_workers} workers, prefetch={args.prefetch_factor}, CUDA prefetch={args.use_prefetcher}")
    print(f"Sequences: {args.seq_len} -> {args.pred_len}")
    print(f"RevIN: {args.use_revin}")
    print(f"Channel Period: {args.channel_period}")
    print(f"Channel Mix Size: {args.channel_mix_size}")
    print(f"{'='*60}\n")


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    try:
        args = create_args()

        if args.use_fsdp:
            init_distributed_mode(args)
        else:
            args.global_rank = 0
            args.world_size = 1
            args.local_rank = 0
            args.num_gpus_per_node = get_available_gpus()

        setup_data_parser(args)
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        set_seed(args.seed + args.global_rank)
        print_config(args)

        for ii in range(args.itr):
            setting = f'{args.model}_{args.data}_{args.des}_{ii}'

            if args.global_rank == 0:
                print(f"\n>>> Starting: {setting}")

            exp = Exp_Informer(args)
            exp.train(setting)

            if args.global_rank == 0:
                print(f"\n>>> Testing: {setting}")
            exp.test(setting)

            torch.cuda.empty_cache()
            if args.use_fsdp and dist.is_initialized():
                dist.barrier()

        if args.global_rank == 0:
            print(f"\n{'='*60}")
            print("COMPLETED")
            print(f"{'='*60}\n")

        cleanup()

    except Exception as e:
        if 'args' in locals() and getattr(args, 'global_rank', 0) == 0:
            import traceback
            print(f"\nERROR: {e}")
            traceback.print_exc()
        cleanup()
        raise


if __name__ == "__main__":
    main()

"""
RevIN (Reversible Instance Normalization) - FSDP & BF16 Optimized

FIXES FROM ORIGINAL:
1. _get_statistics() now only stores on self (no return needed)
2. normalize() uses self.mean/self.stdev directly (no unpacking of None)
3. _normalize() signature matches usage (self, x) only
4. forward('norm') returns a TENSOR, not a tuple
5. forward('denorm') checks that statistics exist

This implementation is optimized for:
- FSDP (Fully Sharded Data Parallel) across multiple nodes/GPUs
- Mixed precision training (BF16/FP16)
- Gradient checkpointing compatibility

Usage:
    revin = RevIN(num_features=9)
    # In model forward:
    x_enc = revin(x_enc, 'norm')     # returns normalized tensor
    ...
    dec_out = revin(dec_out, 'denorm')  # returns denormalized tensor
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-3, affine=True, distributed=True):
        """
        Reversible Instance Normalization for Time Series

        FSDP-compatible implementation with distributed statistics synchronization.

        Args:
            num_features (int): Number of features/channels in your data
            eps (float): Epsilon for numerical stability
                        - Use 1e-3 for BF16/FP16 (default)
                        - Use 1e-5 for FP32 only
            affine (bool): If True, apply learnable affine transformation
            distributed (bool): If True, synchronize statistics across all GPUs
                               Should be True for FSDP/DDP training

        Input shape:
            [batch_size, sequence_length, num_features]
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.distributed = distributed

        # Initialize stored statistics to None
        # These are set during normalize() and read during denormalize()
        self.mean = None
        self.stdev = None

        if self.affine:
            self._init_params()

    def _init_params(self):
        """Initialize learnable affine parameters"""
        # Shape: [num_features]
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Compute mean and standard deviation with distributed synchronization.

        Computes instance-level statistics by reducing over the time dimension.
        In distributed training, statistics are averaged across all GPUs using
        all-reduce to ensure all ranks have identical normalization.

        Stores results on self.mean and self.stdev (detached).

        Args:
            x: [batch, seq_len, features]
        """
        # Reduce over time dimension (dim=1), keep batch and feature dims
        # For input shape [B, T, F], this gives [B, 1, F]
        dim2reduce = tuple(range(1, x.ndim - 1))

        # Compute local statistics on this GPU's data shard
        mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)

        # Synchronize statistics across all GPUs in distributed training
        if self.distributed and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(variance, op=dist.ReduceOp.AVG)

        # Compute standard deviation with numerical stability
        stdev = torch.sqrt(variance + self.eps)

        # Detach to prevent backpropagation through statistics
        self.mean = mean.detach()
        self.stdev = stdev.detach()

    def _normalize(self, x):
        """
        Apply normalization using stored self.mean and self.stdev.

        Args:
            x: [batch, seq_len, features]

        Returns:
            Normalized tensor of same shape
        """
        # Standardize: (x - mean) / std
        x = (x - self.mean) / self.stdev

        # Apply learnable affine transformation if enabled
        if self.affine:
            # Reshape affine params from [F] to [1, 1, F] for broadcasting
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            x = x * weight + bias

        return x

    def _denormalize(self, x):
        """
        Reverse normalization using stored self.mean and self.stdev.

        Args:
            x: [batch, seq_len, features]

        Returns:
            Denormalized tensor of same shape
        """
        # Reverse affine transformation if it was applied
        if self.affine:
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)

            # Invert: (x - bias) / weight
            x = (x - bias) / (weight + self.eps * self.eps)

        # Reverse standardization: x * std + mean
        x = x * self.stdev + self.mean

        return x

    def normalize(self, x):
        """
        Compute statistics and normalize x.

        Args:
            x: [batch, seq_len, features]

        Returns:
            Normalized tensor (same shape as x)
        """
        self._get_statistics(x)
        return self._normalize(x)

    def denormalize(self, x):
        """
        Denormalize x using previously stored statistics.

        Args:
            x: [batch, seq_len, features]

        Returns:
            Denormalized tensor (same shape as x)
        """
        if self.mean is None or self.stdev is None:
            raise RuntimeError(
                "RevIN.denormalize() called before normalize(). "
                "You must call normalize() first to compute statistics."
            )
        return self._denormalize(x)

    def forward(self, x, mode: str):
        """
        Forward pass with mode selection.

        Args:
            x: Input tensor [batch, seq_len, features]
            mode: 'norm' to normalize, 'denorm' to denormalize

        Returns:
            Tensor (same shape as x)
        """
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            return self.denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' not supported. Use 'norm' or 'denorm'")

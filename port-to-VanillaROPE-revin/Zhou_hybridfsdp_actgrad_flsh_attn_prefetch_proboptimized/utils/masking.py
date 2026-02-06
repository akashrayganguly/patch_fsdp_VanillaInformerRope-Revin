"""
masking.py - OPTIMIZED Mask Classes

OPTIMIZATIONS:
1. All masks created directly on GPU (avoid CPU-GPU transfers)
2. Reduced tensor operations
3. Better memory efficiency
4. torch.compile friendly
"""

import torch


class TriangularCausalMask:
    """
    Causal (lower-triangular) mask for autoregressive attention
    
    Creates mask where position i can only attend to positions <= i
    
    OPTIMIZATION: Created directly on target device
    """
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Create directly on device (avoid CPU->GPU transfer)
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), 
                diagonal=1
            )

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """
    OPTIMIZED Sparse mask for ProbAttention
    
    Creates causal masks for the selected top-u queries
    
    OPTIMIZATIONS:
    1. Direct device creation (no .to(device) calls)
    2. Reduced intermediate tensors
    3. Fused operations where possible
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Args:
            B: Batch size
            H: Number of heads
            L: Query sequence length
            index: [B, H, n_top] - Indices of selected queries
            scores: [B, H, n_top, L_K] - Attention scores to mask
            device: Target device
        """
        n_top = index.shape[-1]
        L_K = scores.shape[-1]
        
        # OPTIMIZATION: Build mask directly on device without intermediate tensors
        # For each selected query at position index[b,h,i], mask keys with position > index[b,h,i]
        
        # Key positions: [1, 1, 1, L_K]
        key_positions = torch.arange(L_K, device=device, dtype=torch.long).view(1, 1, 1, L_K)
        
        # Selected query positions: [B, H, n_top, 1]
        query_positions = index.unsqueeze(-1)
        
        # Causal mask: True where key_position > query_position (should be masked)
        self._mask = key_positions > query_positions  # [B, H, n_top, L_K]
    
    @property
    def mask(self):
        return self._mask


class ProbMaskLegacy:
    """
    Legacy ProbMask implementation (kept for reference/debugging)
    
    This is the original implementation - slower due to:
    1. Creating tensors on CPU then moving to GPU
    2. Multiple expand operations
    3. Advanced indexing overhead
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # Original implementation
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        
        # This advanced indexing is slow
        indicator = _mask_ex[
            torch.arange(B, device=device)[:, None, None],
            torch.arange(H, device=device)[None, :, None],
            index, 
            :
        ]
        self._mask = indicator.view(scores.shape)
    
    @property
    def mask(self):
        return self._mask


class LocalMask:
    """
    Local attention mask - each position attends only to nearby positions
    
    Useful for very long sequences where full attention is too expensive
    """
    def __init__(self, B, L, window_size, device="cpu"):
        """
        Args:
            B: Batch size
            L: Sequence length
            window_size: Size of local attention window
            device: Target device
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Create band matrix mask
            positions = torch.arange(L, device=device)
            # Mask where |i - j| > window_size
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]
            self._mask = (diff.abs() > window_size).unsqueeze(0).unsqueeze(0).expand(*mask_shape)
    
    @property
    def mask(self):
        return self._mask


class CombinedMask:
    """
    Combines causal mask with local mask for efficient long-range modeling
    """
    def __init__(self, B, L, window_size, device="cpu"):
        causal = TriangularCausalMask(B, L, device)
        local = LocalMask(B, L, window_size, device)
        # Combine: mask if either causal OR outside window
        self._mask = causal.mask | local.mask
    
    @property
    def mask(self):
        return self._mask

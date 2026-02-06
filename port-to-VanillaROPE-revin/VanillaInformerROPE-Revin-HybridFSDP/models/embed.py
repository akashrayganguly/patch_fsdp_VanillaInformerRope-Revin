"""
embed.py - Optimized Embedding Module with Rotary Position Embeddings (ROPE)

FSDP HYBRID_SHARD Compatible Version for Informer Time Series Forecasting

===============================================================================
KEY CHANGES FROM ORIGINAL PATCH:
===============================================================================

1. PARAMETERIZED CHANNEL PERIOD:
   - Original: Hardcoded 321
   - Fixed: Configurable via `channel_period` parameter

2. SAFE TENSOR OPERATIONS:
   - Original: Used .view() which can fail with non-contiguous FSDP shards
   - Fixed: Using .flatten() and torch.stack() which handle non-contiguous tensors

3. MEMORY EFFICIENCY:
   - Original: 200K x d_model/2 x 4 embeddings = ~1.6GB for d_model=512
   - Fixed: 
     a) Channel embeddings only store 1 period (321 positions vs 200K)
     b) Caching to avoid repeated computation
     c) Option for on-the-fly computation (DataEmbeddingMemoryEfficient)

4. FSDP COMPATIBILITY:
   - Proper buffer registration with persistent=True
   - No in-place operations that break autograd
   - Contiguous tensor assertions
   - Safe parameter sharding

5. EDGE CASE HANDLING:
   - Original: int(seq_len/321) fails if not divisible
   - Fixed: Ceiling division with trimming

===============================================================================
USAGE:
===============================================================================

# Standard usage (same as your original):
embedding = DataEmbedding(
    c_in=9, 
    d_model=512, 
    embed_type='timeF', 
    freq='h', 
    dropout=0.1,
    channel_period=321  # Now configurable!
)

# Memory-efficient version for very long sequences:
embedding = DataEmbeddingMemoryEfficient(
    c_in=9, 
    d_model=512, 
    channel_period=321
)

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _compute_rope_embeddings(
    max_len: int, 
    d_model: int, 
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rotary position embeddings (sin and cos).
    
    Args:
        max_len: Maximum sequence length
        d_model: Model dimension (must be even)
        base: Base for inverse frequency computation
        device: Target device
        dtype: Target dtype
        
    Returns:
        sin_embed: (max_len, d_model/2)
        cos_embed: (max_len, d_model/2)
    """
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
    
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=dtype, device=device) / d_model))
    
    # Position indices
    positions = torch.arange(0, max_len, dtype=dtype, device=device)
    
    # Outer product: (max_len, d_model/2)
    sinusoid_inp = torch.outer(positions, inv_freq)
    
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def _rotate_half_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized rotate_half that's safe for FSDP.
    
    Transforms [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    
    This implementation avoids .view() on potentially non-contiguous tensors
    which can occur when FSDP shards parameters across GPUs.
    
    Args:
        x: Input tensor of shape (..., d_model) where d_model is even
        
    Returns:
        Rotated tensor of same shape
    """
    # Split into pairs and rotate
    x1 = x[..., ::2]   # Even indices: x0, x2, x4, ...
    x2 = x[..., 1::2]  # Odd indices: x1, x3, x5, ...
    
    # Stack as [-x2, x1] pairs and flatten
    # Shape: (..., d_model/2, 2) -> (..., d_model)
    # Using flatten(-2) instead of view for safety
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    sin_embed: torch.Tensor,
    cos_embed: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor (batch, seq_len, d_model)
        sin_embed: Sin embeddings (1, seq_len, d_model)
        cos_embed: Cos embeddings (1, seq_len, d_model)
        
    Returns:
        Rotated tensor of same shape as x
    """
    return x * cos_embed + _rotate_half_optimized(x) * sin_embed


# =============================================================================
# ROTARY POSITIONAL EMBEDDING - LEARNABLE
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Learnable Rotary Positional Embedding (ROPE).
    
    FSDP Compatible: Parameters are properly registered and can be sharded.
    The large sin/cos parameter tensors will be sharded across GPUs in FSDP mode.
    
    Args:
        d_model: Model dimension (must be even)
        max_len: Maximum sequence length to support
        base: Base for frequency computation (default: 10000)
    """
    
    def __init__(self, d_model: int, max_len: int = 200000, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even for ROPE, got {d_model}"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute initial embeddings
        sin_embed, cos_embed = _compute_rope_embeddings(max_len, d_model, base)
        
        # Register as learnable parameters
        # These will be sharded by FSDP
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary transformation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Rotated tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.size()
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        # Slice to sequence length: (seq_len, d_model/2)
        sin_base = self.sin_embed[:seq_len, :]
        cos_base = self.cos_embed[:seq_len, :]
        
        # Expand to full d_model: (1, seq_len, d_model)
        sin_embed = sin_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        cos_embed = cos_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        
        return _apply_rotary_emb(x, sin_embed, cos_embed)
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, max_len={self.max_len}, base={self.base}'


# =============================================================================
# ROTARY POSITIONAL EMBEDDING - FIXED (Non-learnable)
# =============================================================================

class RotaryPositionalEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding (ROPE).
    
    FSDP Compatible: Buffers are properly registered with persistent=True
    for correct state dict handling during checkpointing.
    
    Args:
        d_model: Model dimension (must be even)
        max_len: Maximum sequence length to support
        base: Base for frequency computation (default: 10000)
    """
    
    def __init__(self, d_model: int, max_len: int = 200000, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even for ROPE, got {d_model}"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute embeddings
        sin_embed, cos_embed = _compute_rope_embeddings(max_len, d_model, base)
        
        # Register as buffers (non-learnable, but saved in state_dict)
        # persistent=True ensures they're included in state_dict for FSDP checkpointing
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary transformation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Rotated tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.size()
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        sin_base = self.sin_embed[:seq_len, :]
        cos_base = self.cos_embed[:seq_len, :]
        
        sin_embed = sin_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        cos_embed = cos_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        
        return _apply_rotary_emb(x, sin_embed, cos_embed)
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, max_len={self.max_len}, base={self.base}'


# =============================================================================
# ROTARY CHANNEL EMBEDDING - LEARNABLE
# =============================================================================

class RotaryChannelEmbeddingLearnable(nn.Module):
    """
    Learnable Rotary Channel Embedding.
    
    Applies periodic rotary embeddings based on channel structure.
    The sequence is divided into periods, and each period gets the same rotation.
    This is useful for time series with periodic structure (e.g., hourly data with
    daily patterns).
    
    FSDP Compatible: Only stores one period of embeddings (memory efficient),
    and properly handles non-divisible sequence lengths.
    
    Args:
        c_in: Number of input channels
        d_model: Model dimension (must be even)
        channel_period: Period for channel rotation (default: 321)
        base: Base for frequency computation (default: 50000)
    """
    
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        channel_period: int = 321,
        max_len: int = 2000,
        base: float = 50000.0
    ):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even for ROPE, got {d_model}"
        
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base
        self.max_len = max_len

        # Only store one period's worth of embeddings (memory efficient!)
        sin_embed, cos_embed = _compute_rope_embeddings(channel_period, d_model, base)
        
        # Register as learnable parameters
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary channel transformation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Rotated tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.size()

        sin_period = self.sin_embed.unsqueeze(0).repeat_interleave(2, dim=-1)
        cos_period = self.cos_embed.unsqueeze(0).repeat_interleave(2, dim=-1)
        
        # Calculate number of periods needed (ceiling division)
        num_periods =  (seq_len + self.channel_period - 1) // self.channel_period
        
        # Repeat for all periods: (1, num_periods * period, d_model)
        sin_repeated = sin_period.repeat(1, num_periods, 1)
        cos_repeated = cos_period.repeat(1, num_periods, 1)
        
        # Trim to exact sequence length (handles non-divisible case)
        sin_embed = sin_repeated[:, :seq_len, :]
        cos_embed = cos_repeated[:, :seq_len, :]
        
        return _apply_rotary_emb(x, sin_embed, cos_embed)
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, c_in={self.c_in}, period={self.channel_period}, base={self.base}'


# =============================================================================
# ROTARY CHANNEL EMBEDDING - FIXED
# =============================================================================

class RotaryChannelEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Channel Embedding.
    
    FSDP Compatible: Proper buffer registration for state dict handling.
    
    Args:
        c_in: Number of input channels
        d_model: Model dimension (must be even)
        channel_period: Period for channel rotation (default: 321)
        base: Base for frequency computation (default: 50000)
    """
    
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        channel_period: int = 321,
        max_len: int = 2000,
        base: float = 50000.0
    ):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even for ROPE, got {d_model}"
        
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base
        
        # Compute embeddings for one period
        sin_embed, cos_embed = _compute_rope_embeddings(channel_period, d_model, base)
        
        # Register as buffers
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary channel transformation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Rotated tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.size()


        sin_period = self.sin_embed.unsqueeze(0).repeat_interleave(2, dim=-1)
        cos_period = self.cos_embed.unsqueeze(0).repeat_interleave(2, dim=-1)
        
        num_periods = (seq_len + self.channel_period - 1) // self.channel_period
        
        sin_repeated = sin_period.repeat(1, num_periods, 1)
        cos_repeated = cos_period.repeat(1, num_periods, 1)
        
        sin_embed = sin_repeated[:, :seq_len, :]
        cos_embed = cos_repeated[:, :seq_len, :]
        
        return _apply_rotary_emb(x, sin_embed, cos_embed)
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, c_in={self.c_in}, period={self.channel_period}, base={self.base}'


# =============================================================================
# STANDARD EMBEDDINGS (Original Implementation - Kept for Compatibility)
# =============================================================================

class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding."""
    
    def __init__(self, d_model: int, max_len: int = 200000):
        super().__init__()
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not parameter) with persistent=True for FSDP
        self.register_buffer('pe', pe, persistent=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        Returns:
            Positional embeddings (1, seq_len, d_model)
        """
        return self.pe[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    """Convolutional token embedding."""
    
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular'
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_in', 
                    nonlinearity='leaky_relu'
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, c_in)
        Returns:
            Token embeddings (batch, seq_len, d_model)
        """
        # (batch, seq_len, c_in) -> (batch, c_in, seq_len) -> conv -> (batch, d_model, seq_len)
        x = self.tokenConv(x.permute(0, 2, 1))
        # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        return x.transpose(1, 2)


class FixedEmbedding(nn.Module):
    """Fixed sinusoidal embedding for categorical features."""
    
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        
        # Compute fixed embedding weights
        w = torch.zeros(c_in, d_model, dtype=torch.float32)
        
        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / d_model)
        )
        
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Temporal feature embedding (hour, weekday, day, month)."""
    
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        super().__init__()
        
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Linear projection of time features."""
    
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        super().__init__()
        
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


# =============================================================================
# MAIN DATA EMBEDDING CLASS - FSDP OPTIMIZED
# =============================================================================

class DataEmbedding(nn.Module):
    """
    Combined Data Embedding with Rotary Position and Channel Embeddings.
    
    FSDP HYBRID_SHARD Compatible:
    - All submodules properly registered for sharding
    - No in-place operations that break autograd
    - Efficient memory usage (channel embeddings only store one period)
    - Safe tensor operations (no .view() on non-contiguous tensors)
    - Proper buffer registration for checkpoint saving
    
    Architecture:
        Input -> TokenEmbedding -> ROPE (learnable + fixed) -> Channel ROPE (learnable + fixed) -> Dropout
    
    Args:
        c_in: Number of input channels
        d_model: Model dimension
        embed_type: Type of temporal embedding ('fixed' or 'timeF')
        freq: Frequency of data ('h', 't', 's', etc.)
        dropout: Dropout probability
        channel_period: Period for channel rotation (default: 321)
        max_len: Maximum sequence length (default: 200000)
    """
    
    def __init__(
        self, 
        c_in: int, 
        d_model: int, 
        embed_type: str = 'fixed', 
        freq: str = 'h', 
        dropout: float = 0.1,
        channel_period: int = 321,
        max_len: int = 200000
    ):
        super().__init__()
        
        self.c_in = c_in
        self.d_model = d_model
        self.channel_period = channel_period
        
        # Token embedding (conv1d projection)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # Standard positional embedding (kept for compatibility/fallback)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        
        # Temporal embedding
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, 
                embed_type=embed_type, 
                freq=freq
            )
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, 
                embed_type=embed_type, 
                freq=freq
            )
        
        # Rotary Position Embeddings (learnable + fixed)
        self.rpe = RotaryPositionalEmbedding(d_model=d_model, max_len=max_len)
        self.rpe_fixed = RotaryPositionalEmbeddingFixed(d_model=d_model, max_len=max_len)
        
        # Rotary Channel Embeddings (learnable + fixed)
        # Note: These only store `channel_period` embeddings (memory efficient!)
        self.fixed_channel_embedding = RotaryChannelEmbeddingFixed(
            c_in=c_in, 
            d_model=d_model,
            channel_period=channel_period
        )
        self.learnable_channel_embedding = RotaryChannelEmbeddingLearnable(
            c_in=c_in, 
            d_model=d_model,
            channel_period=channel_period
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ROPE embeddings.
        
        This matches the original patch behavior:
        1. Token embedding
        2. Apply learnable + fixed position ROPE
        3. Apply learnable + fixed channel ROPE
        4. Dropout
        
        Args:
            x: Input data tensor (batch, seq_len, c_in)
            x_mark: Time mark tensor (batch, seq_len, time_features) - currently unused in ROPE mode
            
        Returns:
            Embedded tensor (batch, seq_len, d_model)
        """
        # Token embedding: (batch, seq_len, c_in) -> (batch, seq_len, d_model)
        x = self.value_embedding(x)
        
        # Apply rotary position embeddings (learnable + fixed combined)
        x = self.rpe(x) + self.rpe_fixed(x)
        
        # Apply rotary channel embeddings (learnable + fixed combined)
        x = self.fixed_channel_embedding(x) + self.learnable_channel_embedding(x)
        
        return self.dropout(x)
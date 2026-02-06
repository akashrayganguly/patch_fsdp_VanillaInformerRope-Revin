import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class RotaryPositionalEmbedding(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding (ROPE).
    
    This implementation applies the rotary transformation using pointwise 
    multiplications. For an input x of shape (batch, seq_len, d_model) with even d_model,
    the rotation is applied as:
    
        x_rot = x * cos_embed_full + rotate_half(x) * sin_embed_full
    
    where cos_embed_full and sin_embed_full are constructed by concatenating the 
    fixed sin/cos embeddings for each half.
    """
    def __init__(self, d_model, max_len=200000):
        super(RotaryPositionalEmbedding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for ROPE."
        self.d_model = d_model

        # Compute the inverse frequency vector.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a positions vector (max_len positions).
        positions = torch.arange(0, max_len, dtype=torch.float)
        # Compute sinusoidal inputs: shape (max_len, d_model/2)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Pre-compute sin and cos embeddings.
        sin_embed = torch.sin(sinusoid_inp)  # (max_len, d_model/2)
        cos_embed = torch.cos(sinusoid_inp)  # (max_len, d_model/2)

        # Learnable ROPE
        # Register these as learnable parameters.
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

        # Fixed ROPE
        # Register buffers so they remain fixed.
        #self.register_buffer("sin_embed", sin_embed)
        #self.register_buffer("cos_embed", cos_embed)
    
    def rotate_half(self, x):
        """
        Helper function that rotates half of the dimensions of x.
        Splits x into two halves and returns [-x2, x1] concatenated along the last dim.
        """
        x_even = -x[..., 1::2]
        x_odd  = x[..., 0::2]

        # Stack b and a along a new last dimension.
        stacked = torch.stack([x_even, x_odd], dim=-1)  # Shape becomes (1, 2, 3, 2)

        # Reshape to flatten the last two dimensions into one.
        return stacked.view(x_even.shape[0], x_even.shape[1], -1)  # Shape: (1, 2, 6)
            
    def forward(self, x):
        """
        Applies the rotary transformation using pointwise multiplication.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch, seq_len, d_model) after applying ROPE.
        """
        batch, seq_len, d_model = x.size()
        # Get fixed sin and cos embeddings for the current sequence length.
        # Original shape: (seq_len, d_model/2) -> expand to (1, seq_len, d_model/2)
        sin_embed = self.sin_embed[:seq_len, :].unsqueeze(0)
        cos_embed = self.cos_embed[:seq_len, :].unsqueeze(0)
        # Instead of doing separate 2D operations, we simply expand them to full d_model.
        # This creates tensors of shape (1, seq_len, d_model) where the embedding for each
        # dimension pair is repeated.
        sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)
        cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)
        # Apply the rotary transformation with pointwise multiplication.
        return x * cos_embed + self.rotate_half(x) * sin_embed


class RotaryPositionalEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding (ROPE).
    
    This implementation applies the rotary transformation using pointwise 
    multiplications. For an input x of shape (batch, seq_len, d_model) with even d_model,
    the rotation is applied as:
    
        x_rot = x * cos_embed_full + rotate_half(x) * sin_embed_full
    
    where cos_embed_full and sin_embed_full are constructed by concatenating the 
    fixed sin/cos embeddings for each half.
    """
    def __init__(self, d_model, max_len=200000):
        super(RotaryPositionalEmbeddingFixed, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for ROPE."
        self.d_model = d_model

        # Compute the inverse frequency vector.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a positions vector (max_len positions).
        positions = torch.arange(0, max_len, dtype=torch.float)
        # Compute sinusoidal inputs: shape (max_len, d_model/2)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Pre-compute sin and cos embeddings.
        sin_embed = torch.sin(sinusoid_inp)  # (max_len, d_model/2)
        cos_embed = torch.cos(sinusoid_inp)  # (max_len, d_model/2)

        # Learnable ROPE
        # Register these as learnable parameters.
        #self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        #self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

        # Fixed ROPE
        # Register buffers so they remain fixed.
        self.register_buffer("sin_embed", sin_embed)
        self.register_buffer("cos_embed", cos_embed)
    
    def rotate_half(self, x):
        """
        Helper function that rotates half of the dimensions of x.
        Splits x into two halves and returns [-x2, x1] concatenated along the last dim.
        """
        x_even = -x[..., 1::2]
        x_odd  = x[..., 0::2]

        # Stack b and a along a new last dimension.
        stacked = torch.stack([x_even, x_odd], dim=-1)  # Shape becomes (1, 2, 3, 2)

        # Reshape to flatten the last two dimensions into one.
        return stacked.view(x_even.shape[0], x_even.shape[1], -1)  # Shape: (1, 2, 6)
            
    def forward(self, x):
        """
        Applies the rotary transformation using pointwise multiplication.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch, seq_len, d_model) after applying ROPE.
        """
        batch, seq_len, d_model = x.size()
        # Get fixed sin and cos embeddings for the current sequence length.
        # Original shape: (seq_len, d_model/2) -> expand to (1, seq_len, d_model/2)
        sin_embed = self.sin_embed[:seq_len, :].unsqueeze(0)
        cos_embed = self.cos_embed[:seq_len, :].unsqueeze(0)
        # Instead of doing separate 2D operations, we simply expand them to full d_model.
        # This creates tensors of shape (1, seq_len, d_model) where the embedding for each
        # dimension pair is repeated.
        sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)
        cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)
        # Apply the rotary transformation with pointwise multiplication.
        return x * cos_embed + self.rotate_half(x) * sin_embed





class RotaryChannelEmbeddingLearnable(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding (ROPE).
    
    This implementation applies the rotary transformation using pointwise 
    multiplications. For an input x of shape (batch, seq_len, d_model) with even d_model,
    the rotation is applied as:
    
        x_rot = x * cos_embed_full + rotate_half(x) * sin_embed_full
    
    where cos_embed_full and sin_embed_full are constructed by concatenating the 
    fixed sin/cos embeddings for each half.
    """
    def __init__(self, c_in, d_model, max_len=200000):
        super(RotaryChannelEmbeddingLearnable, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for ROPE."
        self.d_model = d_model
        self.c_in = c_in

        # Compute the inverse frequency vector.
        # Channel should be smaller
        inv_freq = 1.0 / (50000 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a positions vector (max_len positions).
        positions = torch.arange(0, max_len, dtype=torch.float)
        # Compute sinusoidal inputs: shape (max_len, d_model/2)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Pre-compute sin and cos embeddings.
        sin_embed = torch.sin(sinusoid_inp)  # (max_len, d_model/2)
        cos_embed = torch.cos(sinusoid_inp)  # (max_len, d_model/2)

        # Learnable ROPE
        # Register these as learnable parameters.
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

        # Fixed ROPE
        # Register buffers so they remain fixed.
        #self.register_buffer("sin_embed", sin_embed)
        #self.register_buffer("cos_embed", cos_embed)
    
    def rotate_half(self, x):
        """
        Helper function that rotates half of the dimensions of x.
        Splits x into two halves and returns [-x2, x1] concatenated along the last dim.
        """
        x_even = -x[..., 1::2]
        x_odd  = x[..., 0::2]

        # Stack b and a along a new last dimension.
        stacked = torch.stack([x_even, x_odd], dim=-1)  # Shape becomes (1, 2, 3, 2)

        # Reshape to flatten the last two dimensions into one.
        return stacked.view(x_even.shape[0], x_even.shape[1], -1)  # Shape: (1, 2, 6)
            
    def forward(self, x):
        """
        Applies the rotary transformation using pointwise multiplication.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch, seq_len, d_model) after applying ROPE.
        """
        batch, seq_len, d_model = x.size()
        # Get fixed sin and cos embeddings for the current sequence length.
        # Original shape: (seq_len, d_model/2) -> expand to (1, seq_len, d_model/2)
        sin_embed = self.sin_embed[:321, :].unsqueeze(0)
        cos_embed = self.cos_embed[:321, :].unsqueeze(0)
        
        # Instead of doing separate 2D operations, we simply expand them to full d_model.
        # This creates tensors of shape (1, seq_len, d_model) where the embedding for each
        # dimension pair is repeated.
        sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)
        cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)

        sin_embed = sin_embed.repeat(1, int(seq_len/321), 1)
        cos_embed = cos_embed.repeat(1, int(seq_len/321), 1)
        
        #sin_embed = F.pad(sin_embed, (0, 0, 1, 1))
        #cos_embed = F.pad(cos_embed, (0, 0, 1, 1))

        #print(f'{x[0,0,:]}    val: {int(seq_len/7)}', flush=True) 
        
        # Apply the rotary transformation with pointwise multiplication.
        return x * cos_embed + self.rotate_half(x) * sin_embed


class RotaryChannelEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding (ROPE).
    
    This implementation applies the rotary transformation using pointwise 
    multiplications. For an input x of shape (batch, seq_len, d_model) with even d_model,
    the rotation is applied as:
    
        x_rot = x * cos_embed_full + rotate_half(x) * sin_embed_full
    
    where cos_embed_full and sin_embed_full are constructed by concatenating the 
    fixed sin/cos embeddings for each half.
    """
    def __init__(self, c_in, d_model, max_len=200000):
        super(RotaryChannelEmbeddingFixed, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for ROPE."
        self.d_model = d_model
        self.c_in = c_in

        # Compute the inverse frequency vector.
        # Channel should be smaller
        inv_freq = 1.0 / (50000 ** (torch.arange(0, d_model, 2).float() / d_model))
        # Create a positions vector (max_len positions).
        positions = torch.arange(0, max_len, dtype=torch.float)
        # Compute sinusoidal inputs: shape (max_len, d_model/2)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        # Pre-compute sin and cos embeddings.
        sin_embed = torch.sin(sinusoid_inp)  # (max_len, d_model/2)
        cos_embed = torch.cos(sinusoid_inp)  # (max_len, d_model/2)

        # Learnable ROPE
        # Register these as learnable parameters.
        #self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        #self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

        # Fixed ROPE
        # Register buffers so they remain fixed.
        self.register_buffer("sin_embed", sin_embed)
        self.register_buffer("cos_embed", cos_embed)
    
    def rotate_half(self, x):
        """
        Helper function that rotates half of the dimensions of x.
        Splits x into two halves and returns [-x2, x1] concatenated along the last dim.
        """
        x_even = -x[..., 1::2]
        x_odd  = x[..., 0::2]

        # Stack b and a along a new last dimension.
        stacked = torch.stack([x_even, x_odd], dim=-1)  # Shape becomes (1, 2, 3, 2)

        # Reshape to flatten the last two dimensions into one.
        return stacked.view(x_even.shape[0], x_even.shape[1], -1)  # Shape: (1, 2, 6)
            
    def forward(self, x):
        """
        Applies the rotary transformation using pointwise multiplication.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch, seq_len, d_model) after applying ROPE.
        """
        batch, seq_len, d_model = x.size()
        # Get fixed sin and cos embeddings for the current sequence length.
        # Original shape: (seq_len, d_model/2) -> expand to (1, seq_len, d_model/2)
        sin_embed = self.sin_embed[:321, :].unsqueeze(0)
        cos_embed = self.cos_embed[:321, :].unsqueeze(0)
        
        # Instead of doing separate 2D operations, we simply expand them to full d_model.
        # This creates tensors of shape (1, seq_len, d_model) where the embedding for each
        # dimension pair is repeated.
        sin_embed = torch.repeat_interleave(sin_embed, repeats=2, dim=-1)
        cos_embed = torch.repeat_interleave(cos_embed, repeats=2, dim=-1)

        sin_embed = sin_embed.repeat(1, int(seq_len/321), 1)
        cos_embed = cos_embed.repeat(1, int(seq_len/321), 1)

        #sin_embed = F.pad(sin_embed, (0, 0, 1, 1))
        #cos_embed = F.pad(cos_embed, (0, 0, 1, 1))


        #print(f'{x[0,0,:]}    val: {int(seq_len/7)}', flush=True) 
        
        # Apply the rotary transformation with pointwise multiplication.
        return x * cos_embed + self.rotate_half(x) * sin_embed







# # === Example usage ===
# if __name__ == "__main__":
#     batch = 2
#     seq_len = 20
#     d_model = 512  # Must be even
#     dummy_input = torch.zeros(batch, seq_len, d_model)
    
#     rope = RotaryPositionalEmbedding(d_model, max_len=5000)
#     out = rope(dummy_input)
#     print("Output shape:", out.shape)  # Expected: (2, 20, 512)



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=200000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.rpe = RotaryPositionalEmbedding(d_model=d_model)
        self.rpe_fixed = RotaryPositionalEmbeddingFixed(d_model=d_model)

        self.fixed_channel_embedding = RotaryChannelEmbeddingFixed(c_in=c_in, d_model=d_model)
        self.learnable_channel_embedding = RotaryChannelEmbeddingLearnable(c_in=c_in, d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        #print(f'{x.size()}    Original', flush=True)
        x = self.value_embedding(x)
        #print(f'{x.size()}    Value Embedded', flush=True)
        x = self.rpe.forward(x) + self.rpe_fixed.forward(x) #+ self.temporal_embedding(x_mark) # self.position_embedding(x) #+ self.temporal_embedding(x_mark)
        #print(f'{x.size()}    RPE', flush=True)
        x = self.fixed_channel_embedding.forward(x) + self.learnable_channel_embedding.forward(x)
        
        #x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x) #+ self.temporal_embedding(x_mark)
        return self.dropout(x)

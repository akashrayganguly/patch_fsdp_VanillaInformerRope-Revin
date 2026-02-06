import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.norm4 = nn.LayerNorm(d_model)
        self.W = nn.Parameter(torch.empty(321, 321))
        nn.init.xavier_uniform_(self.W)
        self.c = 321

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        
        x = self.norm3(x + y)
        #return self.norm3(x+y)


        # Residual connection and normalization

        # Apply learnable matrix W after FFN
        batch_size, seq_len, d_model = x.shape
        n = seq_len // self.c  # Compute number of groups
        # Reshape to (batch_size, n, c, d_model) for group-wise transformation
        x_reshaped = x.view(batch_size, n, self.c, d_model)
        # Apply W to the c dimension: W (c, c) multiplies x_reshaped (..., c, d_model)
        x_transformed = torch.einsum('ij,bnjd->bnid', self.W, x_reshaped)
        # Reshape back to (batch_size, n*c, d_model)
        x = x_transformed.reshape(batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        return self.norm4(x)



class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

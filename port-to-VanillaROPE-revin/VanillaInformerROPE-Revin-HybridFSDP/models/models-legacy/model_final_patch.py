"""
model.py - Informer Model with ROPE Embeddings

Updated to support configurable channel_period for Rotary Channel Embeddings.
Compatible with FSDP HYBRID_SHARD distributed training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer Model for Long Sequence Time-Series Forecasting.
    
    Now with ROPE (Rotary Position Embeddings) support.
    
    Args:
        enc_in: Encoder input dimension (number of features)
        dec_in: Decoder input dimension
        c_out: Output dimension
        seq_len: Input sequence length
        label_len: Label length for decoder
        out_len: Prediction length
        factor: ProbSparse attention factor
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        attn: Attention type ('prob' or 'full')
        embed: Embedding type ('fixed', 'learned', 'timeF')
        freq: Data frequency ('h', 't', 's', etc.)
        activation: Activation function ('relu' or 'gelu')
        output_attention: Whether to output attention weights
        distil: Whether to use distillation (conv pooling)
        mix: Whether to use mix attention in decoder
        device: Target device
        channel_period: Period for rotary channel embedding (default: 321)
        max_len: Maximum sequence length for embeddings (default: 200000)
    """
    
    def __init__(
        self, 
        enc_in, 
        dec_in, 
        c_out, 
        seq_len, 
        label_len, 
        out_len,
        factor=5, 
        d_model=512, 
        n_heads=8, 
        e_layers=3, 
        d_layers=2, 
        d_ff=512,
        dropout=0.0, 
        attn='prob', 
        embed='fixed', 
        freq='h', 
        activation='gelu',
        output_attention=False, 
        distil=True, 
        mix=True,
        device=torch.device('cuda:0'),
        channel_period=321,
        max_len=200000
    ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding - Now with channel_period parameter
        self.enc_embedding = DataEmbedding(
            c_in=enc_in, 
            d_model=d_model, 
            embed_type=embed, 
            freq=freq, 
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )
        self.dec_embedding = DataEmbedding(
            c_in=dec_in, 
            d_model=d_model, 
            embed_type=embed, 
            freq=freq, 
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )
        
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.projection = nn.Linear(d_model, c_out, bias=True)
    
    def forward(
        self, 
        x_enc, 
        x_mark_enc, 
        x_dec, 
        x_mark_dec,
        enc_self_mask=None, 
        dec_self_mask=None, 
        dec_enc_mask=None
    ):
        # Encode
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decode
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    """
    InformerStack Model - Multiple encoder stacks with different input lengths.
    
    Now with ROPE (Rotary Position Embeddings) support.
    
    Args:
        enc_in: Encoder input dimension
        dec_in: Decoder input dimension
        c_out: Output dimension
        seq_len: Input sequence length
        label_len: Label length
        out_len: Prediction length
        factor: ProbSparse attention factor
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: List of encoder layer counts for each stack
        d_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        attn: Attention type
        embed: Embedding type
        freq: Data frequency
        activation: Activation function
        output_attention: Output attention weights
        distil: Use distillation
        mix: Use mix attention
        device: Target device
        channel_period: Period for rotary channel embedding
        max_len: Maximum sequence length
    """
    
    def __init__(
        self, 
        enc_in, 
        dec_in, 
        c_out, 
        seq_len, 
        label_len, 
        out_len,
        factor=5, 
        d_model=512, 
        n_heads=8, 
        e_layers=[3, 2, 1], 
        d_layers=2, 
        d_ff=512,
        dropout=0.0, 
        attn='prob', 
        embed='fixed', 
        freq='h', 
        activation='gelu',
        output_attention=False, 
        distil=True, 
        mix=True,
        device=torch.device('cuda:0'),
        channel_period=321,
        max_len=200000
    ):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding - Now with channel_period parameter
        self.enc_embedding = DataEmbedding(
            c_in=enc_in, 
            d_model=d_model, 
            embed_type=embed, 
            freq=freq, 
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )
        self.dec_embedding = DataEmbedding(
            c_in=dec_in, 
            d_model=d_model, 
            embed_type=embed, 
            freq=freq, 
            dropout=dropout,
            channel_period=channel_period,
            max_len=max_len
        )
        
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder stack
        inp_lens = list(range(len(e_layers)))  # [0, 1, 2, ...] customizable
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(d_model) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.projection = nn.Linear(d_model, c_out, bias=True)
    
    def forward(
        self, 
        x_enc, 
        x_mark_enc, 
        x_dec, 
        x_mark_dec,
        enc_self_mask=None, 
        dec_self_mask=None, 
        dec_enc_mask=None
    ):
        # Encode
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decode
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

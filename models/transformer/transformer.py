from typing import Optional
from functools import partial

import torch.nn as nn
from torch import Tensor

import models.layers as layers
from .mha import MultiHeadSelfAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        dim_feedforward: int,
        dropout: float = None,
        activation: str = "relu",
        norm_eps: float = 1.0e-5,
        pre_norm: bool = True
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            num_heads,
            embed_dim,
            dropout,
        )
        self.mlp = layers.MLP(embed_dim, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.dropout1 = layers.get_dropout(dropout)
        self.dropout2 = layers.get_dropout(dropout)
        self.identity1 = nn.Identity()
        self.identity2 = nn.Identity()
        self.pre_norm = pre_norm

    def pre_forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        x = self.norm1(seq)
        x = self.attention(
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        # for recording residual connection without dropout
        self.identity1(seq + x)
        seq = seq + self.dropout1(x)

        x = self.norm2(seq)
        x = self.mlp(x)
        # for recording residual connection without dropout
        self.identity2(seq + x)
        seq = seq + self.dropout2(x)
        return seq

    def post_forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        x = self.attention(
            seq,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x)
        # for recording residual connection without dropout
        self.identity1(seq + x)
        seq = seq + self.dropout1(x)

        x = self.mlp(seq)
        x = self.norm2(x)
        # for recording residual connection without dropout
        self.identity2(seq + x)
        seq = seq + self.dropout2(x)
        return seq

    def forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        if self.pre_norm:
            forward_fn = self.pre_forward
        else:
            forward_fn = self.post_forward
        return forward_fn(seq, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 12,
        num_heads: int = 8,
        embed_dim: int = 512,
        dim_feedforward: int = 2048,
        dropout: float = None,
        activation: str = "relu",
        final_norm: bool = True,
        norm_eps: float = 1.0e-5,
        pre_norm: bool = True
    ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm

        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps) if final_norm else None

        encoder_layer = partial(
            EncoderLayer,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_eps=norm_eps,
            pre_norm=pre_norm
        )
        self.layers = nn.ModuleList([encoder_layer() for _ in range(num_encoder_layers)])

    def pre_forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        for l_id, layer in enumerate(self.layers):
            seq = layer(
                seq,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            seq = self.norm(seq)
        return seq

    def post_forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        if self.norm is not None:
            seq = self.norm(seq)
        for layer in self.layers:
            seq = layer(
                seq,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return seq

    def forward(
        self,
        seq: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ):
        if self.pre_norm:
            forward_fn = self.pre_forward
        else:
            forward_fn = self.post_forward
        return forward_fn(seq, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

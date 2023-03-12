import torch

import torch.nn as nn
from torch import Tensor

from models.layers import get_dropout


class PosEncoding(nn.Module):
    """
    No position encoding
    """
    def __init__(self, dropout: float = None, *args, **kwargs):
        super().__init__()
        self.pos_drop = get_dropout(dropout)

    def forward(self, seq: Tensor):
        seq = self.pos_drop(seq)
        return seq


class PosEncoding_Learnable(PosEncoding):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        dropout: float = None,
        **kwargs
    ):
        super().__init__(dropout=dropout)
        self.pos_embed = nn.Parameter(torch.empty(num_tokens, 1, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, seq: Tensor):
        seq = self.pos_drop(seq + self.pos_embed)
        return seq


__REGISTERED_POS_ENCODING__ = {
    "identity": PosEncoding,
    "learnable": PosEncoding_Learnable
}


def get_pos_encoding(**pos_encoding_cfg) -> PosEncoding:
    name = pos_encoding_cfg["name"]
    pos_embed = __REGISTERED_POS_ENCODING__[name](**pos_encoding_cfg)
    return pos_embed

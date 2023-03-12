import torch.nn as nn
from torch import Tensor

import models.layers as layers
from models.utils import conv_1x1


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dim_feedforward: int,
        dropout: float = None,
        activation: str = "relu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout = layers.get_dropout(dropout)
        self.activation = layers.get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, 1e-6)
        nn.init.normal_(self.linear2.bias, 1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x


class MLP_2D(nn.Module):
    """
    2D MLP
    dimension: embed_dim -> [embed_dim * dim_expand] -> out_dim
    """
    def __init__(
        self,
        embed_dim: int,
        dim_expand: float,
        out_dim: int,
        dropout: float = None,
        activation: str = "relu"
    ):
        super().__init__()
        dim_feedforward = round(embed_dim * dim_expand)
        self.linear1 = conv_1x1(embed_dim, dim_feedforward, bias=True)
        self.linear2 = conv_1x1(dim_feedforward, out_dim, bias=True)
        self.dropout = layers.get_dropout(dropout)
        self.activation = layers.get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, 1e-6)
        nn.init.normal_(self.linear2.bias, 1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x

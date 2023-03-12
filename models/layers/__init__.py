from typing import Callable

import torch
import torch.nn as nn

from .norm import Norm_fn, LaryerNorm2D, ChannelNorm
from .nan_norm import NanBatchNorm1d, NanBatchNorm2d
from .drop_path import DropPath, get_drop_path
from .dropout import get_dropout
from .mlp import MLP, MLP_2D
from .patch_embed import PatchEmbed, get_patch_embedding
from .pos_encoding import PosEncoding, get_pos_encoding
from .interpolate import Interpolate


def get_activation_fn(activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    __SUPPORTED_ACTIVATION__ = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "glu": nn.GLU,
        "swish": nn.SiLU,
        "sigmoid": nn.Sigmoid,
        "hard_sigmoid": nn.Hardsigmoid,
        "none": nn.Identity
    }
    return __SUPPORTED_ACTIVATION__[activation_name]()


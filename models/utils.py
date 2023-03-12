import collections.abc
from itertools import repeat
from typing import Tuple
import math

import torch.nn as nn


def ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


single = ntuple(1)
pair = ntuple(2)
triple = ntuple(3)
quadruple = ntuple(4)


def conv_1x1(in_dim: int, out_dim: int, bias: bool = False):
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)


def conv_3x3(in_dim: int, out_dim: int, bias: bool = False):
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=bias)


def conv_out_shape(in_size: Tuple[int], conv: nn.Conv2d):
    def conv_size(size, padding: int, dilation: int, kernel_size: int, stride: int):
        out_size = (size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return math.floor(out_size)

    out_size = (
        conv_size(in_size[0], conv.padding[0], conv.dilation[0], conv.kernel_size[0], conv.stride[0]),
        conv_size(in_size[1], conv.padding[1], conv.dilation[1], conv.kernel_size[1], conv.stride[1])
    )
    return out_size

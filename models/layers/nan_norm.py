from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def _get_num(x: torch.Tensor, dim: Union[int, Sequence[int]] = (), keepdim: bool = False):
    return torch.sum(~x.isnan(), dim=dim, keepdim=keepdim)


def nan_var(
    input: torch.Tensor,
    dim: Union[int, Sequence[int]] = (),
    unbiased: bool = True,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Using traditional
    """
    n = _get_num(input, dim)
    # using Bessel's correction
    corr = 1
    if unbiased:
        corr = n / (n - 1)

    mean = torch.nanmean(input, dim=dim, keepdim=True)
    diff = torch.square(input - mean)
    return corr * torch.nanmean(diff, dim=dim, keepdim=keepdim)


def nan_batch_norm1d(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    B, _, N = input.shape
    if training:
        assert B * N > 1, f"Expected more than 1 value per channel when training, got input size {input.shape}"

    if training:
        dim = (0, 2)
        mean = input.nanmean(dim=dim)
        var = nan_var(input, dim=dim, unbiased=False)
        n = _get_num(input, dim=dim)
        with torch.no_grad():
            # update running mean and var
            running_mean.copy_(momentum * mean + (1 - momentum) * running_mean)
            # update running_var with unbiased var
            running_var.copy_(momentum * var * n / (n - 1) + (1 - momentum) * running_var)
    else:
        mean = running_mean
        var = running_var

    input = (input - mean[None, :, None]) / torch.sqrt(var[None, :, None] + eps)
    gamma = weight[None, :, None] if weight is not None else 1
    beta = bias[None, :, None] if bias is not None else 0
    return gamma * input + beta


@torch.no_grad()
def _clone_tensor(x: torch.Tensor, make_parameter: bool = False):
    if x is not None:
        x = x.clone()
        if make_parameter:
            x = nn.Parameter(x.clone())
        return x
    else:
        return None


class NanBatchNorm1d(nn.Module):
    def __init__(self, bn: _BatchNorm):
        super().__init__()
        self.weight: nn.Parameter
        self.bias: nn.Parameter
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        self.register_parameter("weight", _clone_tensor(bn.weight, make_parameter=True))
        self.register_parameter("bias", _clone_tensor(bn.bias, make_parameter=True))
        self.register_buffer("running_mean", _clone_tensor(bn.running_mean))
        self.register_buffer("running_var", _clone_tensor(bn.running_var))
        self.momentum = bn.momentum
        self.eps = bn.eps

    def forward(self, x: torch.Tensor):
        return nan_batch_norm1d(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )


class NanBatchNorm2d(NanBatchNorm1d):
    def __init__(self, bn: _BatchNorm):
        super().__init__(bn)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = super().forward(x)
        x = x.reshape(B, C, H, W)
        return x



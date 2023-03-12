from typing import Iterable

import torch
import torch.nn as nn


def normalize_sum_(x: torch.Tensor, dim: int = -1):
    """
    Normalize along given dimension, so that sum of this dimension is 1.
    """
    x /= x.sum(dim=dim, keepdim=True)
    x.nan_to_num_(0)
    return x


def normalize_max_(x: torch.Tensor, dim: int = -1):
    """
    Normalize along given dimension, so that sum of this dimension is 1.
    """
    x /= x.max(dim=dim, keepdim=True)[0]
    x.nan_to_num_(0)
    return x


def normalize_sum(x: torch.Tensor, dim: int = -1, detach_sum: bool = False):
    """
    Normalize along given dimension, so that sum of this dimension is 1.
    """
    sum_x = x.sum(dim=dim, keepdim=True)
    if detach_sum:
        sum_x.detach_()
    x = x / sum_x
    x = x.nan_to_num(0)
    return x


def normalize_max(x: torch.Tensor, dim: int = -1):
    """
    Normalize along given dimension, so that sum of this dimension is 1.
    """
    x = x / x.max(dim=dim, keepdim=True)[0]
    x = x.nan_to_num(0)
    return x


def normalize_sum_clamp(
    x: torch.Tensor,
    dim: int = -1,
    detach_sum: bool = False,
    min_val: float = 0
) -> torch.Tensor:
    return normalize_sum(x.clamp_min(min_val), dim, detach_sum=detach_sum)


def pair_wise_point_dist(h: int, w: int, pow: float = 2, device: torch.device = None) -> torch.Tensor:
    r"""
    Return point-wise distance [n, n] (n = h * w) on feature map with shape [h, w].
    D_{i,j} = ||p_i - p_j||_{pow}, where p_i = (x_i, y_i) is the 2d position of
    point i w.r.t. the 2d feature map.

    Note:
        The coordinates are permuted as [h, w]. Then, they are flattened [h*w] to
        calculate point wise distance.
    """
    i_ids = torch.arange(h, dtype=torch.float, device=device)
    j_ids = torch.arange(w, dtype=torch.float, device=device)
    p_i, p_j = torch.meshgrid(i_ids, j_ids, indexing="ij")
    p = torch.stack((p_i.flatten(), p_j.flatten()), dim=1)
    return torch.cdist(p, p, p=pow)


def pair_wise_point_sim(h: int, w: int, alpha: float = 1, pow: float = 2, device: torch.device = None) -> torch.Tensor:
    r"""
    Return point-wise distance [n, n] (n = h * w) on feature map with shape [h, w].
    Sim_{i,j} = 1 / (1 + ||p_i - p_j||_{pow} / alpha), where p_i = (x_i, y_i) is the 2d
    position of point i w.r.t. the 2d feature map.
    """
    assert alpha >= 0
    dist = pair_wise_point_dist(h, w, pow, device) / alpha
    dist = 1 / (1 + dist)
    return dist


class MyParameter(nn.Module):
    def __init__(
        self,
        shape: Iterable[int],
        dtype=torch.float,
        as_buffer: bool = False
    ) -> None:
        super().__init__()
        self.tensor: torch.Tensor
        val = torch.empty(shape, dtype=dtype)
        self.tensor = nn.Parameter(val, requires_grad=not as_buffer)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.tensor)

    def copy_(self, value: torch.Tensor):
        with torch.no_grad():
            self.tensor.copy_(value)

    def normalize_sum_(self, dim: int, min_val: float = 0):
        with torch.no_grad():
            normalize_sum_(self.tensor.clamp_min_(min_val), dim=dim)

from copy import deepcopy
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.normalization import LayerNorm


class ChannelNorm(nn.Module):
    def __init__(self, normalized_shape, dim: Tuple[int], elementwise_affine: bool = True):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = dim
        self.layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=self.dim)
        x = self.layer_norm(x - x_mean)
        return x


class LaryerNorm2D(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1.0e-6, elementwise_affine: bool = True):
        super().__init__()
        self.norm = LayerNorm(embed_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor):
        """
        Input: [bs, embed_dim, H, W]
        """
        # [bs, embed_dim, H, W] -> [bs, H, W, embed_dim]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class Norm_fn:
    def __init__(self, norm_cfg: Dict[str, Any]):
        self.norm_name = norm_cfg["name"]
        self.norm_cfg = norm_cfg

    def _bn_call(self, **runtime_kwargs) -> nn.Module:
        cfg = deepcopy(self.norm_cfg)
        cfg.update(runtime_kwargs)
        return BatchNorm2d(
            num_features=cfg["num_features"],
            eps=cfg.pop("eps", 1e-5),
            momentum=cfg.pop("momentum", 0.1),
            affine=cfg.pop("affine", True)
        )

    def _ln_call(self, **runtime_kwargs) -> nn.Module:
        cfg = deepcopy(self.norm_cfg)
        cfg.update(runtime_kwargs)
        return LayerNorm(
            normalized_shape=cfg["normalized_shape"],
            eps=cfg.pop("eps", 1e-5),
            elementwise_affine=cfg.pop("elementwise_affine", True),
        )

    def _cn_call(self, **runtime_kwargs) -> nn.Module:
        cfg = deepcopy(self.norm_cfg)
        cfg.update(runtime_kwargs)
        return ChannelNorm(
            normalized_shape=cfg["normalized_shape"],
            dim=cfg["dim"],
            elementwise_affine=cfg.pop("elementwise_affine", True),
        )

    def __call__(self, **runtime_kwargs) -> nn.Module:
        fn = {
            "bn": self._bn_call,
            "ln": self._ln_call,
            "cn": self._cn_call,
        }
        return fn[self.norm_name](**runtime_kwargs)

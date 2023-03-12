from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, out_size: Tuple[int, int], mode: str = "bilinear"):
        super().__init__()
        self.out_size = out_size
        self.mode = mode

    def forward(self, x: torch.Tensor):
        return F.interpolate(x, size=self.out_size, mode=self.mode, align_corners=True)

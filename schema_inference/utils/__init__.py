from typing import Dict, Tuple

import torch

from .dist_utils import LogArgs, DistLaunchArgs
from .model import load_pretrain_model
from .customs_param_group import customs_param_group
from .ingredient_model_wrapper import IngredientModelWrapper


def move_data_to_device(
    x: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = x.to(device)
    if targets is not None:
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(device)
    return x, targets

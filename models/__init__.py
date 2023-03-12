from typing import Any, Dict

import torch.nn as nn

from cv_lib.classification.models import get_model as get_official_models
from cv_lib.classification.models import register_models
from .vision_transformers import get_vit, get_deit


models = {}
register_models(models)


def get_defined_models(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    return get_official_models(model_cfg, num_classes)


__REGISTERED_MODELS__ = {
    "vit": get_vit,
    "deit": get_deit,
    "official_models": get_defined_models
}


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module, is_jit_model: bool = False):
        super().__init__()
        self.module = module
        self.is_jit_model = is_jit_model

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, dict):
            return output
        if self.is_jit_model and isinstance(output, tuple):
            output, _ = output
        ret = {
            "pred": output
        }
        return ret


def get_model(model_cfg: Dict[str, Any], num_classes: int, with_wrapper: bool = True) -> nn.Module:
    model = __REGISTERED_MODELS__[model_cfg["name"]](model_cfg, num_classes)
    if with_wrapper:
        model = ModelWrapper(model)
    return model

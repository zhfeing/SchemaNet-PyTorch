from typing import Any, Dict

from .base_loss import CELoss, Loss
from .schema_inference_loss import SchemaInferenceLoss


__REGISTERED_LOSS__ = {
    "ce_loss": CELoss,
    "schema_inference_loss": SchemaInferenceLoss
}


def get_loss_fn(loss_cfg: Dict[str, Any], **kwargs) -> Loss:
    name = loss_cfg["name"]
    cfg = loss_cfg.get("loss_cfg", dict())
    return __REGISTERED_LOSS__[name](**cfg, **kwargs)


from typing import Dict, Any

import torch.nn as nn

from .discretization import Discretization
from .visual_word_encoder import VisualWordEncoder, Adapter


def get_visual_word_encoder(discretization_cfg: Dict[str, Any], model: nn.Module) -> VisualWordEncoder:
    discretization = Discretization(**discretization_cfg["vocabulary"])
    encoder = VisualWordEncoder(model, discretization_cfg["encoder_layer"], discretization)
    return encoder

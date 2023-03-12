from typing import Dict

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .discretization import Discretization


class Adapter:
    def __init__(self):
        self.shape: torch.Size = None

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        self.cls_token = x[:1]
        return x[1:]

    def reconstruct(self, x: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.cls_token, x), dim=0)
        return x, match


class VisualWordEncoder:
    def __init__(
        self,
        model: nn.Module,
        encode_layer: str,
        discretization: Discretization
    ):
        """
        Args:
            encode_layer: layer name where the discretization will apply to its output
        """
        super().__init__()
        self.encode_layer = encode_layer
        self.discretization = discretization
        self.adapter = Adapter()

        self.hook = self.register_forward_hooks(model)
        self.mid_dict: Dict[str, torch.Tensor] = {
            "origin_seq": None,
            "encoded_seq": None,
            "match": None
        }

    def register_forward_hooks(self, model: nn.Module) -> RemovableHandle:
        raw_model = model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            raw_model = model.module
        for name, module in raw_model.named_modules():
            if name == self.encode_layer:
                # define hook
                def forward_hook(module, input, output):
                    self.mid_dict["origin_seq"] = output
                    output = self.adapter.adapt(output)
                    output, match = self.discretization(output)
                    output, match = self.adapter.reconstruct(output, match)
                    self.mid_dict["encoded_seq"] = output
                    self.mid_dict["match"] = match
                    return output

                handle = module.register_forward_hook(forward_hook)
                return handle

    def clear(self):
        self.hook.remove()
        for k in self.mid_dict:
            self.mid_dict[k] = None


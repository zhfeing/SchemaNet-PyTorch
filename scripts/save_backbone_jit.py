"""
save_decoupled_jit.py
This script save a discretization model to separate traced jit models as `encoder`, `discretization`, and `decoder`.
Warning:
    For saving memory, output jit model will be traced as a static function and, therefor,
    no longer support accessing to its parameters (as constants) nor submodules and will never be trainable.
    However, if input tensor `requires_grad == True`, the gradients can be correctly computed.
"""

import argparse
import os
import collections

import torch
import torch.nn as nn
import torch.jit
from torch.utils.hooks import RemovableHandle

from cv_lib.utils import get_cfg

from models import get_model, ModelWrapper
from schema_inference.utils import load_pretrain_model
from discretization import Discretization, Adapter


class Decoupling:
    def __init__(
        self,
        encode_layer: str,
        discretization: Discretization,
        extract_layer: str = None
    ):
        """
        Args:
            encode_layer: layer name where the discretization will apply to its output
            extract_layer: extract feature from given layer name
        """
        super().__init__()
        self.encode_layer = encode_layer
        self.extract_layer = extract_layer
        self.discretization = discretization
        self.adapter = Adapter()

        self.hook: RemovableHandle = None
        self.extract_layer_hook: RemovableHandle = None
        self.mid_feat: torch.Tensor = None
        self.discrete_feat: torch.Tensor = None
        self.extracted: torch.Tensor = None

    def register_backbone_hooks(self, model: nn.Module) -> RemovableHandle:
        self.clear()
        raw_model = model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            raw_model = model.module
        for name, module in raw_model.named_modules():
            if name == self.encode_layer:
                # define hook
                def forward_hook(module, input, output):
                    self.mid_feat = output

                self.hook = module.register_forward_hook(forward_hook)
            if name == self.extract_layer:
                def forward_hook(module, intput, output):
                    self.extracted = output
                self.extract_layer_hook = module.register_forward_hook(forward_hook)

    def register_cls_header_hooks(self, model: nn.Module) -> RemovableHandle:
        self.clear()
        raw_model = model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            raw_model = model.module
        for name, module in raw_model.named_modules():
            if name == self.encode_layer:
                # define hook
                def forward_hook(module, input, output):
                    return self.discrete_feat

                self.hook = module.register_forward_hook(forward_hook)
                return

    def clear(self):
        self.hook = None
        self.mid_feat = None
        self.discrete_feat = None
        self.match = None
        if self.hook is not None:
            self.hook.remove()
        if self.extract_layer_hook is not None:
            self.extract_layer_hook.remove()


class BackboneJitWrapper:
    def __init__(self, model: ModelWrapper, decoupling: Decoupling, model_input: torch.Tensor):
        self.model = model
        self.decoupling = decoupling
        self.model_input: torch.Tensor = model_input

    def backbone_forward(self, dummy_input: torch.Tensor):
        self.decoupling.register_backbone_hooks(self.model)
        self.model(dummy_input)
        ret = collections.OrderedDict()
        ret["mid_feat"] = self.decoupling.mid_feat
        extracted = self.decoupling.extracted
        if extracted is not None:
            ret["extracted"] = extracted
        return ret

    def cls_header_forward(self, dummy_input: torch.Tensor):
        self.decoupling.register_cls_header_hooks(self.model)
        self.decoupling.discrete_feat = dummy_input
        return self.model(self.model_input)

    def backbone_discretization_forward(self, dummy_input: torch.Tensor):
        backbone_out = self.backbone_forward(dummy_input)
        mid_feat = self.decoupling.adapter.adapt(backbone_out["mid_feat"])
        output, match = self.decoupling.discretization(mid_feat)
        output, match = self.decoupling.adapter.reconstruct(output, match)
        return output


class DiscretizationJitWrapper(nn.Module):
    def __init__(self, discretization: Discretization):
        super().__init__()
        self.discretization = discretization
        self.adapter = Adapter()

    def forward(self, dummy_input: torch.Tensor):
        seq = self.adapter.adapt(dummy_input)
        output, match = self.discretization(seq)
        output, match = self.adapter.reconstruct(output, match)
        return output, match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_fp", type=str)
    parser.add_argument("--ckpt_fp", type=str)
    parser.add_argument("--vocabulary_fp", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--extract_layer", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_cfg(args.cfg_fp)
    model_cfg = get_cfg(cfg["model"])
    discretization_cfg = cfg["discretization"]

    model: ModelWrapper = get_model(model_cfg["model"], args.num_classes, with_wrapper=True)
    discretization = Discretization(**discretization_cfg["vocabulary"])
    decoupling = Decoupling(
        discretization_cfg["encoder_layer"],
        discretization,
        extract_layer=args.extract_layer
    )

    # load state dict
    ckpt = torch.load(args.ckpt_fp, map_location="cpu")
    load_pretrain_model(ckpt, model)
    decoupling.discretization.initial_vocabulary(args.vocabulary_fp)

    model_input = torch.randn(1, args.img_channels, args.img_size, args.img_size).to(device)
    backbone_jit_wrapper = BackboneJitWrapper(model, decoupling, model_input)
    discretization_jit_wrapper = DiscretizationJitWrapper(discretization)

    model.eval().requires_grad_(False).to(device)
    discretization.eval().requires_grad_(False).to(device)
    discretization_jit_wrapper.eval().requires_grad_(False).to(device)

    # get mid seq
    mid_feat = backbone_jit_wrapper.backbone_forward(model_input)["mid_feat"]

    # tracing
    backbone_jit: torch.jit.ScriptModule = torch.jit.trace(
        backbone_jit_wrapper.backbone_forward,
        (model_input,),
        strict=False
    )
    discretization_jit: torch.jit.ScriptModule = torch.jit.trace(
        discretization_jit_wrapper,
        (mid_feat,),
        strict=False
    )
    cls_header_jit: torch.jit.ScriptModule = torch.jit.trace(
        backbone_jit_wrapper.cls_header_forward,
        (mid_feat,),
        strict=False
    )
    backbone_discretization_jit: torch.jit.ScriptModule = torch.jit.trace(
        backbone_jit_wrapper.backbone_discretization_forward,
        (model_input,),
        strict=False
    )

    torch.jit.save(backbone_jit, os.path.join(args.save_path, "backbone-jit.pth"))
    torch.jit.save(cls_header_jit, os.path.join(args.save_path, "cls_header-jit.pth"))
    torch.jit.save(discretization_jit, os.path.join(args.save_path, "discretization-jit.pth"))
    torch.jit.save(backbone_discretization_jit, os.path.join(args.save_path, "backbone_discretization-jit.pth"))

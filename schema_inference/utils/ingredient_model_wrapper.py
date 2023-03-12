import collections
from typing import Dict

import torch
import torch.nn as nn
from torch.jit import ScriptModule


class IngredientModelWrapper(nn.Module):
    """
    Always work in evaluation mode, although all sub-script modules are fixed
    Return:
        cls_token: [bs, 1, dim]
        feat: [bs, L, dim]
        feat_origin: [bs, L, dim]
        ingredients: [bs, L]
        attn: [bs, L, L]
        attn_cls: [bs, L]
    """
    def __init__(
        self,
        backbone_jit: ScriptModule,
        discretization_jit: ScriptModule = None
    ):
        super().__init__()
        self.backbone_jit = backbone_jit
        self.discretization_jit = discretization_jit
        self.discretization_tensor: torch.Tensor

        self.register_buffer("discretization_tensor", discretization_jit.discretization.vocabulary.weight)
        self.num_ingredients: int = self.discretization_tensor.shape[0]
        self.emb_dim: int = self.discretization_tensor.shape[1]

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(False)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = collections.OrderedDict()
        out_backbone = self.backbone_jit(x)
        mid_feat = out_backbone["mid_feat"]
        extracted_attn: torch.Tensor = out_backbone["extracted"]
        feat, ingredients = self.discretization_jit(mid_feat)
        # [1, bs, dim] -> [bs, 1, dim]
        ret["cls_token"] = torch.transpose(feat[:1], 0, 1)
        # [L, bs, dim] -> [bs, L, dim]
        ret["feat"] = torch.transpose(feat[1:], 0, 1)
        ret["feat_origin"] = torch.transpose(mid_feat[1:], 0, 1)
        # [L, bs] -> [bs, L]
        ret["ingredients"] = torch.transpose(ingredients, 0, 1)
        # get attention
        bs, L = ret["ingredients"].shape
        attn = torch.zeros(bs, L + 1, L + 1, device=x.device)
        # [bs * heads, L + 1, L + 1] -> [bs, heads, L + 1, L + 1]
        if extracted_attn is not None:
            attn_heads = extracted_attn.unflatten(0, (bs, -1))
            torch.mean(attn_heads, dim=1, out=attn)
        # [bs, L, L]
        ret["attn"] = attn[..., 1:, 1:]
        # [bs, L]
        ret["attn_cls"] = attn[..., 0, 1:]
        for k, v in ret.items():
            ret[k] = v.contiguous()
        return ret


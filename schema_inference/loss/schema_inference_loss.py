from typing import Dict
import collections

import torch
import torch.nn.functional as F

from .base_loss import Loss


class SchemaInferenceLoss(Loss):
    def __init__(
        self,
        re_a_vertex: float = 3,
        re_a_edge: float = 3,
        **kwargs
    ):
        super().__init__()
        self.re_a_vertex = re_a_vertex
        self.re_a_edge = re_a_edge

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred = output["pred"]
        if isinstance(pred, dict):
            pred = pred["pred"]
        gt = target["label"]

        ret = collections.OrderedDict()
        ret["cls"] = F.cross_entropy(pred, gt)
        ret.update(self.loss_sparsity(
            output["class_vertices"],
            output["class_edges"],
        ))
        return ret

    def loss_sparsity(self, vertex_weights: torch.Tensor, edge_weights: torch.Tensor):
        ret = collections.OrderedDict()
        entropy_vertex = entropy(vertex_weights).max(dim=0)[0]
        entropy_edge = entropy(edge_weights).max(dim=1)[0].mean()
        ret["entropy_vertex"] = entropy_vertex
        ret["entropy_edge"] = entropy_edge
        ret["re_entropy_vertex"] = rectify_linear(entropy_vertex, a=self.re_a_vertex)
        ret["re_entropy_edge"] = rectify_linear(entropy_edge, a=self.re_a_edge)
        return ret


def entropy(
    p: torch.Tensor,
    eps: float = 1.0e-7,
    dim: int = -1,
    keepdim: bool = False
):
    log_p = torch.log(p + eps)
    return -torch.sum(p * log_p, dim=dim, keepdim=keepdim)


def rectify_linear(
    x: torch.Tensor,
    a: float = 0
):
    if x > a:
        return x
    else:
        return a - 1 + 1.0 / (1 + a - x)

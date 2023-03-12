from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GNN


class Matcher(nn.Module):
    def __init__(self, similarity: str, num_codes: int, gnn_cfg: Dict[str, Any]):
        super().__init__()
        self.gnn = GNN(num_codes=num_codes, **gnn_cfg)
        SUPPORTED_SIM = {
            "cosine": self._cosine_sim,
            "euclidean": self._euclidean_sim,
            "inner_product": self._inner_product
        }
        self.similarity = SUPPORTED_SIM[similarity]

    def _cosine_sim(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        sim = torch.cosine_similarity(feat_1, feat_2, dim=-1)
        return (sim + 1) / 2

    def _euclidean_sim(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        dist = torch.linalg.vector_norm(feat_1 - feat_2, dim=-1)
        return 1 / (1 + dist)

    def _inner_product(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        dist = (feat_1 * feat_2).sum(-1)
        return dist

    def forward(
        self,
        instance_dict: Dict[str, List[torch.Tensor]],
        class_dict: Dict[str, torch.Tensor]
    ):
        ####################################################################
        ## compute instance features
        instance_ingredients = instance_dict["instance_ingredients"]    # [[n_1], ..., [n_bs]]
        instance_vertices = instance_dict["instance_vertices"]          # [[n_1], ..., [n_bs]]
        instance_edges = instance_dict["instance_edges"]                # [[n_1, n_1], ..., [n_bs, n_bs]]

        bs = len(instance_ingredients)
        sizes = [len(x) for x in instance_ingredients]
        max_size = max(sizes)
        # padding instance graphs
        feat_mask = torch.zeros(bs, max_size, dtype=torch.bool, device=instance_ingredients[0].device)
        for i, s in enumerate(sizes):
            feat_mask[i, s:].fill_(1)
            # fill dummy ingredients with `num_codes`, which is dummy in gnn's embedding layer
            instance_ingredients[i] = F.pad(instance_ingredients[i], (0, max_size - s), value=self.gnn.num_codes)
            instance_vertices[i] = F.pad(instance_vertices[i], (0, max_size - s))
            instance_edges[i] = F.pad(instance_edges[i], (0, max_size - s, 0, max_size - s))
        # [bs, dim]
        feat_instance: torch.Tensor = self.gnn(
            nodes=torch.stack(instance_vertices),
            edges=torch.stack(instance_edges),
            ingredients=torch.stack(instance_ingredients),
            feat_mask=feat_mask
        )

        ####################################################################
        ## compute ir-atlas features
        # [num_classes, dim]
        feat_kg: torch.Tensor = self.gnn(
            nodes=class_dict["class_vertices"],
            edges=class_dict["class_edges"],
            ingredients=class_dict["class_ingredients"]
        )

        feat_kg = feat_kg.expand(bs, -1, -1)
        feat_instance = feat_instance.unsqueeze(1).expand_as(feat_kg)

        sim = self.similarity(feat_instance, feat_kg)
        return sim

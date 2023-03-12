import collections
import string

import torch
import torch.nn as nn

from schema_inference.utils import IngredientModelWrapper
import cv_lib.utils as cv_utils


class BackboneFCPredictor(nn.Module):
    """
    Procedure:
        1. use backbone (vit) to obtain the features
        2. FC (codebook) -> max -> FC
    """
    def __init__(
        self,
        backbone: nn.Module(),
        codebook_size: int,
        num_classes: int,
        extract_name: string
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.extract_name = extract_name
        self.fc2 = nn.Linear(list(backbone.parameters())[9].shape[0], num_classes)
        nn.init.normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.extractor = cv_utils.MidExtractor(self.backbone, extract_names=[extract_name])
        pass

    def forward(self, x: torch.Tensor, requires_graph: bool = False, task: int = None):
        ret = collections.OrderedDict()
        with torch.no_grad():
            self.backbone(x)
        features = self.extractor.features[self.extract_name]  # omit the class token
        # feature size: h*w, bs, dim
        features = torch.mean(features.permute(1, 2, 0), dim=2)
        # feature size: bs, dim
        features = self.fc2(features)
        pred = features
        # vertices, edges = self.relation_graph(
        #     ingredients=output["ingredients"],
        #     attn=output["attn"],
        #     attn_cls=output["attn_cls"]
        # )
        # vertex_weights = self.relation_graph.get_vertex_weights()
        # edge_weights = self.relation_graph.get_edge_weights()
        # pred = self.matcher(
        #     instance_vertices=vertices,
        #     instance_edges=edges,
        #     kg_vertices=vertex_weights,
        #     kg_edges=edge_weights,
        #     task=task
        # )
        ret["pred"] = pred
        ret["vertex_weights"] = torch.tensor(0)
        ret["edge_weights"] = torch.tensor(0)
        # if requires_graph:
        #     ret["instance_vertices"] = vertices
        #     ret["instance_edges"] = edges
        return ret

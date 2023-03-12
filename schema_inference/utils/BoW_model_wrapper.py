import collections

import torch
import torch.nn as nn

from schema_inference.utils import IngredientModelWrapper


class BoWPredictor(nn.Module):
    """
    Procedure:
        1. use ingredient model to predict sequence of ingredient
        2. use dark KG to predict
    Prediction:
        "pred": dark kg prediction, shape: [bs, num_classes];
        "origin_pred": origin model prediction, shape: [bs, num_classes];
        "codes": codes predicted by origin model, shape: [bs, H, W];
        "attribution": attribution to codes w.r.t. each class, shape: [bs, num_classes, H, W]
    """
    def __init__(
        self,
        ingredient_wrapper: IngredientModelWrapper,
        num_classes: int
    ):
        super().__init__()
        self.ingredient_wrapper = ingredient_wrapper
        self.num_classes = num_classes
        self.num_ingredients = self.ingredient_wrapper.num_ingredients
        self.fc = nn.Linear(self.num_ingredients, self.num_classes)
        nn.init.normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, requires_graph: bool = False, task: int = None):
        batch_size = x.shape[0]
        ret = collections.OrderedDict()
        with torch.no_grad():
            output = self.ingredient_wrapper(x)

        features = torch.zeros((batch_size, self.num_ingredients)).to(x.device)
        for i in range(batch_size):
            features[i] = torch.bincount(output["ingredients"][i], minlength=self.num_ingredients)
        pred = self.fc(features)
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

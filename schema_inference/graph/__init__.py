import collections
from typing import Dict, List

import torch
import torch.nn as nn

from .schema_net import SchemaNet
from .match import Matcher
from .convert_graph import to_networkx

from schema_inference.utils import IngredientModelWrapper


class SchemaNetPredictor(nn.Module):
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
        schema_net: SchemaNet,
        matcher: Matcher
    ):
        super().__init__()
        self.ingredient_wrapper = ingredient_wrapper
        self.schema_net = schema_net
        self.matcher = matcher
        self.num_classes = schema_net.num_classes

    def forward(self, x: torch.Tensor, requires_graph: bool = False):
        ret = collections.OrderedDict()
        with torch.no_grad():
            output = self.ingredient_wrapper(x)
        instance_dict: Dict[str, List[torch.Tensor]] = self.schema_net(
            ingredients=output["ingredients"],
            attn=output["attn"],
            attn_cls=output["attn_cls"]
        )
        class_dict = self.schema_net.get_atlas()
        pred = self.matcher(
            instance_dict=instance_dict,
            class_dict=class_dict
        )
        ret["pred"] = pred
        ret.update(class_dict)
        if requires_graph:
            ret.update(instance_dict)
            ret["ingredients"] = output["ingredients"]
            ret["attn_cls"] = output["attn_cls"]
        return ret

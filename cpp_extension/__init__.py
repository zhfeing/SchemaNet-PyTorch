from typing import List, Dict

import torch

from .extension import (
    feat_to_v_attr,
    feat_to_instance_v,
    feat_to_e,
    feat_to_instance_e
)


__all__ = [
    "cpp_feat_to_v_attr",
    "cpp_feat_to_instance_v",
    "cpp_feat_to_e"
]


def cpp_feat_to_v_attr(
    ingredients: torch.LongTensor,
    attn_cls: torch.Tensor,
    n_vertices: int,
    mean: bool = False,
    ingredients_only: bool = False
) -> torch.Tensor:
    return feat_to_v_attr(ingredients, attn_cls, n_vertices, mean, ingredients_only)


def cpp_feat_to_instance_v(
    ingredients: torch.LongTensor,
    attn_cls: torch.Tensor,
    vertex_attribute_weights: torch.Tensor,
    mean: bool = False
) -> List[torch.Tensor]:
    return feat_to_instance_v(ingredients, attn_cls, vertex_attribute_weights, mean)


def cpp_feat_to_e(
    ingredients: torch.LongTensor,
    attn: torch.Tensor,
    geo_sim: torch.Tensor,
    class_ingredient_dict: List[Dict[int, int]],
    label: List[int],
    n_max: int,
    mean: bool = False
) -> torch.Tensor:
    return feat_to_e(
        ingredients,
        attn,
        geo_sim,
        class_ingredient_dict,
        label,
        n_max,
        mean
    )


def cpp_feat_to_instance_e(
    ingredients: torch.LongTensor,
    attn: torch.Tensor,
    geo_sim: torch.Tensor,
    batch_ingredient_dict: List[Dict[int, int]],
    edge_attribute_weights: torch.Tensor,
    mean: bool = False,
    remove_self_loop: bool = False
) -> List[torch.Tensor]:
    return feat_to_instance_e(
        ingredients,
        attn,
        geo_sim,
        batch_ingredient_dict,
        edge_attribute_weights,
        mean,
        remove_self_loop
    )


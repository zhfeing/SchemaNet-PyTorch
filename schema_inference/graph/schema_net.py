import logging
from typing import Tuple, List, Dict

import torch
import torch.nn as nn

import schema_inference.graph.utils as graph_utils


class SchemaNet(nn.Module):
    """
    IR-Atlas & instance IR-Graph generation
    Parameters:
        vertex_weights: [K, num_vertices], the weight of each vertex for K classes
        edge_weights: [K, num_vertices, num_vertices], K adjacent matrix for each classes
    Weights
        vertex_attribute_weights: [2, 1]
        edge_attribute_weights: [2, 1]
    Args:
        feat_h: original feature height
        feat_w: original feature width
        constant_vertex_attr: pos 0 for geometric attribution weight, 1 for attention attribution weight
        constant_edge_attr: pos 0 for geometric attribution weight, 1 for attention attribution weight
        clamp_vertex_attn: filter attentions to cls-token
        clamp_edge_attn: filter edge attentions
        remove_self_loop: remove self-loops in IR-graphs
        prune_node_threshold: edges connected to this node will be removed if its weight is too small
    """
    def __init__(
        self,
        num_vertices: int,
        num_classes: int = 10,
        dist_alpha: float = 1,
        dist_pow: float = 2,
        feat_h: int = 14,
        feat_w: int = 14,
        class_max_vertices: int = None,
        constant_vertex_attr: Tuple[float, float] = None,
        constant_edge_attr: Tuple[float, float] = None,
        clamp_vertex_attn: float = None,
        clamp_edge_attn: float = None,
        remove_self_loop: bool = False,
        prune_node_threshold: float = None,
        apply_normalize: bool = True,
        clamp_weights: bool = True
    ):
        super().__init__()
        self.logger = logging.getLogger("SchemaNet")
        # config
        self.num_vertices = num_vertices
        self.num_classes = num_classes
        self.dist_alpha = dist_alpha
        self.dist_pow = dist_pow
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.constant_vertex_attr = constant_vertex_attr
        self.constant_edge_attr = constant_edge_attr
        self.clamp_vertex_attn = clamp_vertex_attn
        self.clamp_edge_attn = clamp_edge_attn
        self.remove_self_loop = remove_self_loop
        self.prune_node_threshold = prune_node_threshold
        self.apply_normalize = apply_normalize
        self.clamp_weights = clamp_weights

        # tracking state
        self.n_tracked: torch.LongTensor
        self.register_buffer("n_tracked", torch.zeros(num_classes), persistent=False)

        if class_max_vertices is None:
            class_max_vertices = num_vertices
        else:
            assert class_max_vertices <= num_vertices

        # record the vertex id of each vertex for each class, shape: [num_classes, class_max_vertices]
        self.class_max_vertices = class_max_vertices
        self.class_ingredients = graph_utils.MyParameter(
            shape=(num_classes, class_max_vertices),
            dtype=torch.long,
            as_buffer=True
        )
        # real ingredient id (0 ~ num_vertices) -> class id (0 ~ class_max_vertices)
        self.class_ingredient_dict: List[Dict[int, int]] = list()

        # parameters
        self.vertex_weights = graph_utils.MyParameter(
            shape=(num_classes, class_max_vertices),
            as_buffer=False
        )
        self.edge_weights = graph_utils.MyParameter(
            shape=(num_classes, class_max_vertices, class_max_vertices),
            as_buffer=False
        )
        # weights
        self.vertex_attribute_weights = graph_utils.MyParameter(
            shape=(2, 1),
            as_buffer=constant_vertex_attr is not None
        )
        self.edge_attribute_weights = graph_utils.MyParameter(
            shape=(2, 1),
            as_buffer=constant_edge_attr is not None
        )
        self._reset_parameters()

    def _reset_parameters(self):
        # set weights
        nn.init.constant_(self.vertex_attribute_weights.tensor, 0.5)
        nn.init.constant_(self.edge_attribute_weights.tensor, 0.5)
        # set all values between 0 and 1, mean 0.5, 3*std = 0.5
        nn.init.trunc_normal_(self.vertex_weights.tensor, mean=0.5, std=1 / 6, a=0, b=1)
        nn.init.trunc_normal_(self.edge_weights.tensor, mean=0.5, std=1 / 6, a=0, b=1)
        self.vertex_weights.normalize_sum_(dim=-1)
        self.edge_weights.normalize_sum_(dim=-1)
        if self.constant_vertex_attr is not None:
            init = torch.tensor(self.constant_vertex_attr).reshape(2, 1)
            self.vertex_attribute_weights.copy_(init)
        if self.constant_edge_attr is not None:
            init = torch.tensor(self.constant_edge_attr).reshape(2, 1)
            self.edge_attribute_weights.copy_(init)
        self.normalize()

    def register_class_vertices(self, class_vertices: torch.LongTensor):
        self.class_ingredients.copy_(class_vertices)
        self.class_ingredient_dict.clear()
        for vertices in class_vertices:
            ingredient_dict = {k.item(): v for v, k in enumerate(vertices)}
            self.class_ingredient_dict.append(ingredient_dict)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        ret = super().load_state_dict(state_dict, strict)
        self.register_class_vertices(self.class_ingredients.tensor)
        return ret

    @torch.no_grad()
    def normalize(self):
        if self.clamp_weights:
            self.vertex_attribute_weights.tensor.clamp_(min=0.01, max=10)
            self.edge_attribute_weights.tensor.clamp_(min=0.01, max=10)
        if self.apply_normalize:
            self.vertex_weights.normalize_sum_(dim=-1)
            self.edge_weights.normalize_sum_(dim=-1)
            if self.remove_self_loop:
                self.edge_weights.tensor.diagonal(dim1=1, dim2=2).fill_(0)

    def get_class_vertices(self, detach: bool = False) -> torch.Tensor:
        vertex_weights = self.vertex_weights.tensor
        if detach:
            vertex_weights = vertex_weights.detach()
        # normalize
        vertex_weights = graph_utils.normalize_sum_clamp(vertex_weights, detach_sum=True, min_val=1.0e-5)
        return vertex_weights

    def get_class_edges(self, detach: bool = False) -> torch.Tensor:
        edge_weights = self.edge_weights.tensor
        if detach:
            edge_weights = edge_weights.detach()
        # apply mask
        if self.prune_node_threshold is not None:
            with torch.no_grad():
                # if a vertex has weight < threshold, then edge connected to this vertex is set to zero weight
                vertex_weights = self.get_class_vertices(detach=True)
                mask = (vertex_weights > self.prune_node_threshold).float()
                mask = mask.unsqueeze(-1)
                mask = torch.bmm(mask, mask.transpose(1, 2))
                edge_weights.masked_fill_(~mask.bool(), 0)
            # set the gradients of masked edges to zero as well
            edge_weights = edge_weights * mask
        # normalize
        edge_weights = graph_utils.normalize_sum_clamp(edge_weights, detach_sum=True)

        if self.remove_self_loop:
            with torch.no_grad():
                mask = torch.ones_like(edge_weights)
                mask.diagonal(dim1=1, dim2=2).fill_(0)
            edge_weights = edge_weights * mask
        return edge_weights

    def get_atlas(self, detach: bool = False) -> Dict[str, torch.Tensor]:
        class_vertices = self.get_class_vertices(detach)
        class_edges = self.get_class_edges(detach)
        return {
            "class_vertices": class_vertices,
            "class_edges": class_edges,
            "class_ingredients": self.class_ingredients.tensor
        }

    #########################################################################
    ## for initialization
    def feat_to_full_vertices(
        self,
        ingredients: torch.LongTensor,
        attn_cls: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert feature ingredients to vertex weights
        Args:
            ingredients: [bs, L]
            attn_cls: [bs, L]
        Return: node weights for each sample, shape [bs, num_vertices]
        """
        if self.clamp_vertex_attn is not None:
            attn_cls.masked_fill_(attn_cls < self.clamp_vertex_attn, float("-inf"))
        attn_cls = attn_cls.softmax(dim=-1)
        vertices_attr = self._feat_to_full_v(ingredients, attn_cls)
        # calculate weighted vertex weights
        graph_utils.normalize_max_(vertices_attr, dim=1)
        vertex_weights = vertices_attr @ self.vertex_attribute_weights.tensor
        return vertex_weights.squeeze_(-1)

    def _feat_to_full_v(
        self,
        ingredients: torch.LongTensor,
        attn_cls: torch.Tensor
    ) -> torch.Tensor:
        from cpp_extension import cpp_feat_to_v_attr
        return cpp_feat_to_v_attr(
            ingredients.cpu(),
            attn_cls.cpu(),
            n_vertices=self.num_vertices,
            mean=True
        ).to(ingredients.device)

    def feat_to_limited_edges(
        self,
        ingredients: torch.LongTensor,
        attn: torch.Tensor,
        label: torch.LongTensor
    ) -> torch.Tensor:
        """
        Convert feature ingredients to vertex weights
        Args:
            ingredients: [bs, L]
            attn: [bs, L, L]
        Return:
            batch edges, which are organized according to `self.class_ingredients`,
                shape: [bs, `self.class_max_vertices`, `self.class_max_vertices`]
        """
        if self.clamp_edge_attn is not None:
            attn.masked_fill_(attn < self.clamp_edge_attn, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        geo_sim = graph_utils.pair_wise_point_sim(
            h=self.feat_h,
            w=self.feat_w,
            alpha=self.dist_alpha,
            pow=self.dist_pow,
            device=ingredients.device
        )
        # [bs, num_vertices, num_vertices, 2]
        edges_attr = self._feat_to_e(ingredients, attn, geo_sim, label)
        graph_utils.normalize_sum_(edges_attr, dim=2)
        if self.remove_self_loop:
            edges_attr.diagonal(dim1=1, dim2=2).fill_(0)
        # calculate weighted vertex weights
        edges_attr = edges_attr @ self.edge_attribute_weights.tensor
        return edges_attr.squeeze_(-1)

    def _feat_to_e(
        self,
        ingredients: torch.LongTensor,
        attn: torch.Tensor,
        geo_sim: torch.Tensor,
        label: torch.LongTensor
    ) -> torch.Tensor:
        assert len(self.class_ingredient_dict) > 0, "run `register_class_vertices` before"
        from cpp_extension import cpp_feat_to_e
        edge = cpp_feat_to_e(
            ingredients.cpu(),
            attn.cpu(),
            geo_sim.cpu(),
            label=label.cpu().tolist(),
            class_ingredient_dict=self.class_ingredient_dict,
            n_max=self.class_max_vertices,
            mean=True
        ).to(ingredients.device)
        return edge

    #########################################################################
    ## for prediction
    def feat_to_instance_vertices(
        self,
        ingredients: torch.LongTensor,
        attn_cls: torch.Tensor
    ) -> Tuple[List[torch.LongTensor], List[torch.Tensor]]:
        """
        Convert feature ingredients to vertex weights for each instance
        Args:
            ingredients: [bs, L]
            attn_cls: [bs, L]
            drop_positions: [bs]
        Return:
            (
                instance ingredients, shape: [n_1], ..., [n_bs]
                instance vertices, shape: [n_1], ..., [n_bs]
            )
        """
        if self.clamp_vertex_attn is not None:
            attn_cls.masked_fill_(attn_cls < self.clamp_vertex_attn, float("-inf"))
        attn_cls = attn_cls.softmax(dim=-1).nan_to_num(0)
        instance_ingredients, instance_vertices, num_vertices = self._feat_to_instance_v_attr(
            ingredients,
            attn_cls
        )
        num_vertices = num_vertices.tolist()
        instance_ingredients = list(torch.split_with_sizes(instance_ingredients, num_vertices))
        instance_vertices = list(torch.split_with_sizes(instance_vertices, num_vertices))
        return instance_ingredients, instance_vertices

    def _feat_to_instance_v_attr(
        self,
        ingredients: torch.LongTensor,
        attn_cls: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        from cpp_extension import cpp_feat_to_instance_v
        return cpp_feat_to_instance_v(
            ingredients=ingredients.cpu(),
            attn_cls=attn_cls.cpu(),
            vertex_attribute_weights=self.vertex_attribute_weights.tensor,
            mean=True
        )

    def feat_to_instance_edges(
        self,
        ingredients: torch.LongTensor,
        attn: torch.Tensor,
        instance_ingredients: List[torch.LongTensor]
    ) -> List[torch.Tensor]:
        """
        Convert feature ingredients to vertex weights
        Args:
            ingredients: [bs, L]
            attn: [bs, L, L]
            instance_ingredients: lis of instance ingredients: [[n_1], ..., [n_bs]]
        Return:
        """
        if self.clamp_edge_attn is not None:
            attn.masked_fill_(attn < self.clamp_edge_attn, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        geo_sim = graph_utils.pair_wise_point_sim(
            h=self.feat_h,
            w=self.feat_w,
            alpha=self.dist_alpha,
            pow=self.dist_pow,
            device=ingredients.device
        )
        # convert instance_ingredients to map dict
        batch_ingredient_dict = list()
        for instance_i in instance_ingredients:
            d = {v: k for k, v in enumerate(instance_i.tolist())}
            batch_ingredient_dict.append(d)

        edges = self._feat_to_instance_e(
            ingredients=ingredients,
            attn=attn,
            geo_sim=geo_sim,
            batch_ingredient_dict=batch_ingredient_dict
        )
        return edges

    def _feat_to_instance_e(
        self,
        ingredients: torch.LongTensor,
        attn: torch.Tensor,
        geo_sim: torch.Tensor,
        batch_ingredient_dict: List[Dict[int, int]]
    ):
        from cpp_extension import cpp_feat_to_instance_e
        edges = cpp_feat_to_instance_e(
            ingredients.cpu(),
            attn.cpu(),
            geo_sim.cpu(),
            batch_ingredient_dict=batch_ingredient_dict,
            edge_attribute_weights=self.edge_attribute_weights.tensor,
            mean=True,
            remove_self_loop=self.remove_self_loop
        )
        return edges

    def forward(
        self,
        ingredients: torch.LongTensor,
        attn: torch.Tensor,
        attn_cls: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Convert ingredient sequence to instance IR-Graph
        Args:
            ingredients: [bs, L]
            attn: [bs, L, L]
            attn_cls: [bs, L]
        Return:
            nodes: [bs, num_vertices]
            edges: [bs, num_vertices, num_vertices]
        """
        instance_ingredients, instance_vertices = self.feat_to_instance_vertices(ingredients, attn_cls)
        instance_edges = self.feat_to_instance_edges(ingredients, attn, instance_ingredients)
        return {
            "instance_ingredients": instance_ingredients,
            "instance_vertices": instance_vertices,
            "instance_edges": instance_edges
        }

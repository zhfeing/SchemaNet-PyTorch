#pragma once
#include <torch/torch.h>
#include <vector>

namespace ext{
at::Tensor feat_to_v_attr(
    at::Tensor ingredients,
    at::Tensor attn_cls,
    int n_vertices,
    bool mean = false,
    bool ingredients_only = false
);

std::vector<at::Tensor> feat_to_instance_v(
    at::Tensor ingredients,
    at::Tensor attn_cls,
    at::Tensor vertex_attribute_weights,
    bool mean = false
);
}


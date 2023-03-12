#pragma once

#include <torch/torch.h>

#include <vector>
#include <unordered_map>

#include <utils.h>

namespace ext{

using HashDict = std::unordered_map<long, long>;
using HashDictList = std::vector<HashDict>;

at::Tensor feat_to_e(
    at::Tensor ingredients,
    at::Tensor attn,
    at::Tensor geo_sim,
    HashDictList class_ingredient_dict,
    LongContainer label,
    int n_max,
    bool mean = false
);

std::vector<at::Tensor> feat_to_instance_e(
    at::Tensor ingredients,
    at::Tensor attn,
    at::Tensor geo_sim,
    HashDictList batch_ingredient_dict,
    at::Tensor edge_attribute_weights,
    bool mean = false,
    bool remove_self_loop = false
);

}

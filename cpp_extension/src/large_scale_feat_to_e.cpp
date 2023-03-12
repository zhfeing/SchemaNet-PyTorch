#include <vector>
#include <map>
#include <exception>

#include <torch/torch.h>

#include <feat_to_e.h>
#include <utils.h>

#ifdef _DEBUG
#include <chrono>
#include <iostream>
using std::cout;
using std::endl;
#endif


using at::Tensor;

/*
(1) Get instance edge attributes:
    1. geometric similarity
    2. mean attention between each pair
(2) Computed weighted sum of above two terms

Args:
    ingredients: [bs, L], LongTensor
    attn: [bs, L, L]
    geo_sim: [L, L]
    batch_ingredient_dict: [bs] x {code_id: code_pos}
    label: [bs]
*/
std::vector<Tensor> ext::feat_to_instance_e(
    Tensor ingredients,
    Tensor attn,
    Tensor geo_sim,
    HashDictList batch_ingredient_dict,
    Tensor edge_attribute_weights,
    bool mean,
    bool remove_self_loop
)
{
#ifdef _DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif
    at::Device device = edge_attribute_weights.device();

    const size_t bs = ingredients.size(0);
    const int L = ingredients.size(1);

    std::vector<Tensor> edges(bs);

    if (batch_ingredient_dict.size() != bs)
    {
        throw std::runtime_error("Batch size is not compat with `batch_ingredient_dict`");
    }

    // accessors
    auto accessor_ingredients = ingredients.accessor<long, 2>();
    auto accessor_attn = attn.accessor<float, 3>();
    auto accessor_geo_sim = geo_sim.accessor<float, 2>();

    for (size_t b_id = 0; b_id < bs; ++b_id)
    {
        HashDict &ingredient_dict = batch_ingredient_dict[b_id];
        const int num_v = ingredient_dict.size();
        Tensor instance_edges = at::zeros({num_v, num_v, 2});

        // batch accessors
        auto batch_ingredients = accessor_ingredients[b_id];
        auto batch_attn = accessor_attn[b_id];
        auto accessor_instance_edges = instance_edges.accessor<float, 3>();

        // Code mapper: map a node to a list of positions appears in `code_ids`, also in the `ingredient_table`
        // E.g., code `16` will be mapped to `[0, 3, 7, 20]` if code `16` appears in these positions
        std::map<long, LongContainer> mapper;
        for (int idx = 0; idx < L; idx++)
        {
            const long c_id = batch_ingredients[idx];
            // code `c_id` belongs to this class
            if (mapper.find(c_id) == mapper.end())
            {
                // add a new code
                mapper[c_id] = LongContainer();
                mapper[c_id].reserve(NodeContainerInitSize);
            }
            // append a position of code `c_id`
            mapper[c_id].push_back(idx);
        }

        //! write edges
        // record relationships of code pairs (ci, cj)
        // the mean or sum of these values will be set as edge (ci, cj)
        FloatContainer attn_container;
        FloatContainer geo_sim_container;
        attn_container.reserve(EdgeContainerInitSize);
        geo_sim_container.reserve(EdgeContainerInitSize);
        // iter over code ci
        for (auto iter_ci = mapper.begin(); iter_ci != mapper.end(); iter_ci++)
        {
            // iter over code cj
            for (auto iter_cj = mapper.begin(); iter_cj != mapper.end(); iter_cj++)
            {
                // appearances of code ci & cj
                LongContainer &ci_pos = iter_ci->second;
                LongContainer &cj_pos = iter_cj->second;
                // append all pairs
                for (auto iter_pos_ci = ci_pos.begin(); iter_pos_ci != ci_pos.end(); iter_pos_ci++)
                {
                    for (auto iter_pos_cj = cj_pos.begin(); iter_pos_cj != cj_pos.end(); iter_pos_cj++)
                    {
                        attn_container.push_back(batch_attn[*iter_pos_ci][*iter_pos_cj]);
                        geo_sim_container.push_back(accessor_geo_sim[*iter_pos_ci][*iter_pos_cj]);
                    }
                }
                // accumulate to mean or sum
                long ci_idx = ingredient_dict[iter_ci->first];
                long cj_idx = ingredient_dict[iter_cj->first];
                auto edge = accessor_instance_edges[ci_idx][cj_idx];
                edge[0] = accumulate(geo_sim_container, mean);
                edge[1] = accumulate(attn_container, mean);
                attn_container.clear();
                geo_sim_container.clear();
            }
        }
        /*
        graph_utils.normalize_sum_(edges_attr, dim=2)
        if self.remove_self_loop:
            edges_attr.diagonal(dim1=1, dim2=2).fill_(0)
        # calculate weighted vertex weights
        edges_attr = edges_attr @ self.edge_attribute_weights.tensor
        return edges_attr.squeeze_(-1)
        */
        instance_edges = instance_edges.to(device);
        instance_edges.div_(instance_edges.sum(1, true)).nan_to_num_(0);
        if (remove_self_loop)
        {
            instance_edges.diagonal(0, 1).fill_(0);
        }
        instance_edges = instance_edges.matmul(edge_attribute_weights).squeeze(-1);
        edges[b_id] = instance_edges;
    }

#ifdef _DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "[DEBUG] Edge time: " << duration.count() << "us" << endl;
#endif
    return edges;
}

#include <map>
#include <vector>
#include <tuple>
#include <exception>

#include <torch/torch.h>

#include <feat_to_v.h>
#include <utils.h>

#ifdef _DEBUG
#include <chrono>
#include <iostream>
using std::cout;
using std::endl;
#endif


using at::Tensor;
using LoTensor = std::vector<Tensor>;


/*
(1) Get instance vertex attributes for each sample:
    1. count of each ingredient
    2. mean attention to cls token of each ingredient
(2) Computed weighted sum of above two terms

Args:
    ingredients: [bs, L]
    attn_cls: [bs, L]
    attr_weights: [2, 1]
    drop_positions: [bs, n_drop]

Return [
    1. concatenation of instance ingredients, each of which is LongTensor [n_i, 2],
    2. concatenation of instance vertices, each of which is Tensor [n_i]
    3. number of vertices of each instance, shape: [bs]
]
*/
LoTensor ext::feat_to_instance_v(
    Tensor ingredients,
    Tensor attn_cls,
    Tensor vertex_attribute_weights,
    bool mean
)
{
#ifdef _DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    at::Device device = vertex_attribute_weights.device();

    LoTensor instance_ingredients;
    LoTensor instance_vertices;

    // require both
    const size_t bs = ingredients.size(0);
    const int L = ingredients.size(1);

    std::vector<long> num_vertices(bs);

    // accessors
    auto accessor_ingredients = ingredients.accessor<long, 2>();
    auto accessor_attn_cls = attn_cls.accessor<float, 2>();

    for (size_t b_id = 0; b_id < bs; ++b_id)
    {
        // batch accessors
        auto batch_ingredients = accessor_ingredients[b_id];
        auto batch_attn_cls = accessor_attn_cls[b_id];

        // map a code to a list of its attrs as a code may appear many times
        std::map<long, int> count_aggregator;
        std::map<long, FloatContainer> attn_aggregator;

        // iter over all positions
        for (int i = 0; i < L; ++i)
        {
            // get code id (ci) and attention to cls token (ai)
            const long ci = batch_ingredients[i];
            const float ai = batch_attn_cls[i];
            // create slot for code ci
            if (count_aggregator.find(ci) == count_aggregator.end())
            {
                count_aggregator[ci] = 0;
                FloatContainer _a;
                _a.reserve(NodeContainerInitSize);
                attn_aggregator[ci] = _a;
            }
            // aggregation
            // count code appearance
            ++count_aggregator[ci];
            // collect attention
            attn_aggregator[ci].push_back(ai);
        }

        const long num_v = count_aggregator.size();
        Tensor attrs = at::empty({num_v, 2});
        Tensor v_ids = at::empty({num_v}, at::kLong);
        // accessors
        auto accessor_attrs = attrs.accessor<float, 2>();
        auto accessor_v_ids = v_ids.accessor<long, 1>();

        // fill to `attr` Tensor attribution `1`
        int i = 0;
        for (auto iter = count_aggregator.begin(); iter != count_aggregator.end(); ++iter, ++i)
        {
            long v = iter->first;
            accessor_v_ids[i] = v;
            accessor_attrs[i][0] = float(iter->second);
            accessor_attrs[i][1] = accumulate(attn_aggregator[v], mean);
        }

        /*
        # calculate weighted vertex weights
        graph_utils.normalize_max_(vertices_attr, dim=1)
        vertex_weights = vertices_attr @ self.vertex_attribute_weights.tensor
        vertex_weights.squeeze_(-1)
        */
        attrs = attrs.to(device);
        v_ids = v_ids.to(device);

        attrs.div_(std::get<0>(attrs.max(0, true))).nan_to_num_(0);
        attrs = attrs.matmul(vertex_attribute_weights).squeeze(-1);

        instance_vertices.push_back(attrs);
        instance_ingredients.push_back(v_ids);
        num_vertices[b_id] = num_v;
    }

#ifdef _DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "[DEBUG] feat2v time: " << duration.count() << "us" << endl;
#endif

    return {
        at::cat({instance_ingredients}),
        at::cat({instance_vertices}),
        at::from_blob(num_vertices.data(), {(long)num_vertices.size()}, at::kLong).clone()
    };
}


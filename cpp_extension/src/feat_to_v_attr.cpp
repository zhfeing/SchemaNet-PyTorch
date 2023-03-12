#include <map>

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


Tensor feat_to_v_attr_ingredient_only(
    Tensor ingredients,
    int n_vertices,
    bool mean
)
{
    const int bs = ingredients.size(0);
    const int L = ingredients.size(1);
    Tensor attr = at::zeros({bs, n_vertices, 2});
    
    // accessors
    auto accessor_ingredients = ingredients.accessor<long, 2>();
    auto accessor_attr = attr.accessor<float, 3>();

    for (int b_id = 0; b_id < bs; ++b_id)
    {
        // batch accessors
        auto batch_ingredients = accessor_ingredients[b_id];
        auto batch_attr = accessor_attr[b_id];

        // map a code to a list of its attrs as a code may appear many times
        std::map<long, int> count_aggregator;

        // iter over all positions
        for (int i = 0; i < L; ++i)
        {
            // get code id (ci)
            const long ci = batch_ingredients[i];
            // create slot for code ci
            if (count_aggregator.find(ci) == count_aggregator.end())
            {
                count_aggregator[ci] = 0;
            }
            // aggregation
            // count code appearance
            ++count_aggregator[ci];
        }
        // fill to `attr` Tensor attribution `1`
        for (auto iter = count_aggregator.begin(); iter != count_aggregator.end(); ++iter)
        {
            batch_attr[iter->first][0] = float(iter->second);
        }
    }
    return attr;
}


/*
Get instance vertex attributes:
    1. count of each ingredient
    2. mean attention to cls token of each ingredient
Args:
    ingredients: [bs, L]
    attn_cls: [bs, L]
*/
Tensor ext::feat_to_v_attr(
    Tensor ingredients,
    Tensor attn_cls,
    int n_vertices,
    bool mean,
    bool ingredients_only
)
{
#ifdef _DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif
    if (ingredients_only)
    {
        return feat_to_v_attr_ingredient_only(ingredients, n_vertices, mean);
    }
    // require both
    const int bs = ingredients.size(0);
    const int L = ingredients.size(1);
    Tensor attr = at::zeros({bs, n_vertices, 2});
    
    // accessors
    auto accessor_ingredients = ingredients.accessor<long, 2>();
    auto accessor_attn_cls = attn_cls.accessor<float, 2>();
    auto accessor_attr = attr.accessor<float, 3>();

    for (int b_id = 0; b_id < bs; ++b_id)
    {
        // batch accessors
        auto batch_ingredients = accessor_ingredients[b_id];
        auto batch_attn_cls = accessor_attn_cls[b_id];
        auto batch_attr = accessor_attr[b_id];

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
        // fill to `attr` Tensor attribution `1`
        for (auto iter = count_aggregator.begin(); iter != count_aggregator.end(); ++iter)
        {
            batch_attr[iter->first][0] = float(iter->second);
        }
        // fill to `attr` Tensor attribution `2`
        for (auto iter = attn_aggregator.begin(); iter != attn_aggregator.end(); ++iter)
        {
            batch_attr[iter->first][1] = accumulate(iter->second, mean);
        }
    }

#ifdef _DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "[DEBUG] Vertex attr time: " << duration.count() << "us" << endl;
#endif
    return attr;
}


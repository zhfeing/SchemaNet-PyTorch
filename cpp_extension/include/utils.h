#pragma once

#include <vector>

namespace ext{

template<typename _T>
using _T_Container = std::vector<_T>;
using FloatContainer = _T_Container<float>;
using LongContainer = _T_Container<long>;


constexpr int NodeContainerInitSize = 256;
constexpr int EdgeContainerInitSize = 1024;

float accumulate(FloatContainer &container, bool mean = false);

}

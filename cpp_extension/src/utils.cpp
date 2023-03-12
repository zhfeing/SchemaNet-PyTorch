#include <numeric>

#include <utils.h>


float ext::accumulate(FloatContainer &container, bool mean)
{
    // calculate sum / or mean
    float sum = std::accumulate(container.begin(), container.end(), .0f);
    if (mean)
    {
        sum = sum / container.size();
    }
    return sum;
}

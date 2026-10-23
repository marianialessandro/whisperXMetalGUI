#pragma once
#include <cmath>
#include <cstdint>

// Preserve the pinned Cython float32 Euclidean group order; see native_sklearn_notice.txt.
inline float wxkmeans_squared_distance(const float* a, const float* b, uint32_t features) {
    float total = 0.0f; uint32_t feature = 0;
    for (; feature + 4 <= features; feature += 4) {
        float d0 = a[feature] - b[feature], d1 = a[feature + 1] - b[feature + 1], d2 = a[feature + 2] - b[feature + 2], d3 = a[feature + 3] - b[feature + 3];
        volatile float second_square = d1 * d1;
        float group = std::fma(d0, d0, second_square);
        group = std::fma(d2, d2, group);
        group = std::fma(d3, d3, group);
        total += group;
    }
    for (; feature < features; ++feature) { float difference = a[feature] - b[feature]; total = std::fma(difference, difference, total); }
    return total;
}

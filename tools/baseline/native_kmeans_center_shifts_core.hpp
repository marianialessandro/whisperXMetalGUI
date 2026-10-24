#pragma once
#include "native_checks.hpp"
#include "native_kmeans_distance.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

// Preserve the pinned Cython float32 group order and scalar tail on existing owned buffers.
inline std::vector<float> wxkmeans_owned_center_shifts(const std::vector<float>& a, const std::vector<float>& b, uint32_t clusters, uint32_t features) {
    require(clusters > 0 && clusters <= 256 && features > 0 && features <= 1024 && a.size() == uint64_t(clusters) * features && b.size() == a.size(), "kmeans_lloyd_center_shifts_shape_invalid");
    std::vector<float> result(clusters);
    for (float value : a) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_lloyd_center_shifts_value_domain_invalid");
    for (float value : b) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_lloyd_center_shifts_value_domain_invalid");
    for (uint32_t cluster = 0; cluster < clusters; ++cluster) {
        uint64_t start = uint64_t(cluster) * features;
        result[cluster] = std::sqrt(wxkmeans_squared_distance(a.data() + start, b.data() + start, features));
        require(std::isfinite(result[cluster]) && result[cluster] >= 0, "kmeans_lloyd_center_shifts_result_invalid");
    }
    return result;
}

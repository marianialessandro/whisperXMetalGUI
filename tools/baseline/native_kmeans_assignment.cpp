#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include "native_checks.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
static_assert(sizeof(KmeansPotentialMetrics) == 16);
static_assert(sizeof(__LAPACK_int) == 4);

// Preserve sklearn 1.9.1's dense chunk SGEMM operand swap and first-minimum tie policy; see native_sklearn_notice.txt.
extern "C" int wxkmeans_lloyd_assignment(const float* values, const float* centers, const float* norms, uint32_t rows, uint32_t clusters, uint32_t features, int32_t* labels, float* distances, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values != nullptr && centers != nullptr && norms != nullptr && labels != nullptr && distances != nullptr && metrics != nullptr, "kmeans_lloyd_assignment_pointer_missing");
        require(rows > 0 && rows <= 256 && clusters > 0 && clusters <= 256 && features > 0 && features <= 1024, "kmeans_lloyd_assignment_shape_invalid");
        uintptr_t label_address = reinterpret_cast<uintptr_t>(labels), distance_address = reinterpret_cast<uintptr_t>(distances);
        require(label_address <= distance_address ? distance_address - label_address >= uint64_t(rows) * sizeof(int32_t) : label_address - distance_address >= uint64_t(rows) * clusters * sizeof(float), "kmeans_lloyd_assignment_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> x(uint64_t(rows) * features), c(uint64_t(clusters) * features), n(clusters), result(uint64_t(rows) * clusters);
        std::vector<int32_t> assigned(rows);
        std::memcpy(x.data(), values, x.size() * sizeof(float));
        std::memcpy(c.data(), centers, c.size() * sizeof(float));
        std::memcpy(n.data(), norms, n.size() * sizeof(float));
        auto valid = [](float value) { return std::isfinite(value) && std::abs(value) <= 0x1p32f && (value == 0 || std::abs(value) >= 0x1p-60f); };
        for (float value : x) require(valid(value), "kmeans_lloyd_assignment_value_domain_invalid");
        for (float value : c) require(valid(value), "kmeans_lloyd_assignment_value_domain_invalid");
        for (float value : n) require(std::isfinite(value) && value >= 0 && value <= 0x1p74f && (value == 0 || value >= 0x1p-120f), "kmeans_lloyd_assignment_norm_domain_invalid");
        for (uint32_t row = 0; row < rows; ++row) std::memcpy(result.data() + uint64_t(row) * clusters, n.data(), n.size() * sizeof(float));
        char transpose = 'T', normal = 'N';
        __LAPACK_int m = clusters, samples = rows, k = features, leading = features, ldc = clusters;
        float alpha = -2.0f, beta = 1.0f;
        sgemm_(&transpose, &normal, &m, &samples, &k, &alpha, c.data(), &leading, x.data(), &leading, &beta, result.data(), &ldc);
        for (float value : result) require(std::isfinite(value), "kmeans_lloyd_assignment_result_invalid");
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t label = 0; float minimum = result[uint64_t(row) * clusters];
            for (uint32_t cluster = 1; cluster < clusters; ++cluster) if (result[uint64_t(row) * clusters + cluster] < minimum) { label = cluster; minimum = result[uint64_t(row) * clusters + cluster]; }
            assigned[row] = int32_t(label);
        }
        std::memcpy(distances, result.data(), result.size() * sizeof(float));
        std::memcpy(labels, assigned.data(), assigned.size() * sizeof(int32_t));
        *metrics = {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; }
        return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_lloyd_assignment_failed", capacity - 1); error[capacity - 1] = 0; }
        return 1;
    }
}

// Match the single-chunk float32 FMA accumulation and double-literal reciprocal in sklearn's dense centroid update.
extern "C" int wxkmeans_lloyd_centroids(const float* values, const float* weights, const int32_t* labels, uint32_t rows, uint32_t clusters, uint32_t features, float* centers, float* cluster_weights, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values != nullptr && weights != nullptr && labels != nullptr && centers != nullptr && cluster_weights != nullptr && metrics != nullptr, "kmeans_lloyd_centroids_pointer_missing");
        require(rows > 0 && rows <= 256 && clusters > 0 && clusters <= 256 && features > 0 && features <= 1024, "kmeans_lloyd_centroids_shape_invalid");
        uintptr_t a = reinterpret_cast<uintptr_t>(centers), b = reinterpret_cast<uintptr_t>(cluster_weights);
        require(a <= b ? b - a >= uint64_t(clusters) * features * sizeof(float) : a - b >= uint64_t(clusters) * sizeof(float), "kmeans_lloyd_centroids_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> x(uint64_t(rows) * features), w(rows), result(uint64_t(clusters) * features, 0.0f), totals(clusters, 0.0f);
        std::vector<int32_t> assigned(rows);
        std::memcpy(x.data(), values, x.size() * sizeof(float));
        std::memcpy(w.data(), weights, w.size() * sizeof(float));
        std::memcpy(assigned.data(), labels, assigned.size() * sizeof(int32_t));
        for (float value : x) require(std::isfinite(value) && std::abs(value) <= 0x1p32f && (value == 0 || std::abs(value) >= 0x1p-60f), "kmeans_lloyd_centroids_value_domain_invalid");
        for (float value : w) require(std::isfinite(value) && value >= 0 && value <= 0x1p32f && (value == 0 || value >= 0x1p-32f), "kmeans_lloyd_centroids_weight_domain_invalid");
        for (int32_t label : assigned) require(label >= 0 && uint32_t(label) < clusters, "kmeans_lloyd_centroids_label_domain_invalid");
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t label = uint32_t(assigned[row]); totals[label] += w[row];
            for (uint32_t feature = 0; feature < features; ++feature) {
                uint64_t index = uint64_t(label) * features + feature;
                result[index] = std::fma(x[uint64_t(row) * features + feature], w[row], result[index]);
            }
        }
        for (uint32_t cluster = 0; cluster < clusters; ++cluster) {
            require(totals[cluster] > 0 && std::isfinite(totals[cluster]), "kmeans_lloyd_centroids_empty_cluster_not_supported");
            totals[cluster] = 0.0f + totals[cluster];
            float alpha = static_cast<float>(1.0 / static_cast<double>(totals[cluster]));
            for (uint32_t feature = 0; feature < features; ++feature) {
                uint64_t index = uint64_t(cluster) * features + feature;
                result[index] = (0.0f + result[index]) * alpha;
                require(std::isfinite(result[index]), "kmeans_lloyd_centroids_result_invalid");
            }
        }
        std::memcpy(centers, result.data(), result.size() * sizeof(float));
        std::memcpy(cluster_weights, totals.data(), totals.size() * sizeof(float));
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; }
        return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_lloyd_centroids_failed", capacity - 1); error[capacity - 1] = 0; }
        return 1;
    }
}

#include "native_checks.hpp"
#include "native_kmeans_pairwise.hpp"
#include "native_kmeans_selection.hpp"
#include "native_kmeans_relocation_core.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
static_assert(sizeof(KmeansPotentialMetrics) == 16);

// Preserve sklearn's empty-cluster transfer and ascending centroid averaging order; see native_sklearn_notice.txt.
extern "C" int wxkmeans_lloyd_relocated_centroids(const float* values, const float* weights, const int32_t* labels, const float* previous, uint32_t rows, uint32_t clusters, uint32_t features, float* centers, float* cluster_weights, float* distances, int32_t* selected, uint32_t* counts, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values && weights && labels && previous && centers && cluster_weights && distances && selected && counts && metrics, "kmeans_relocation_pointer_missing");
        require(rows > 0 && rows <= 256 && clusters > 0 && clusters <= rows && features > 0 && features <= 1024, "kmeans_relocation_shape_invalid");
        std::pair<uintptr_t, uint64_t> outputs[] = {{reinterpret_cast<uintptr_t>(centers), uint64_t(clusters) * features * 4}, {reinterpret_cast<uintptr_t>(cluster_weights), uint64_t(clusters) * 4}, {reinterpret_cast<uintptr_t>(distances), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(selected), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(counts), 8}};
        for (uint32_t i = 0; i < 5; ++i) for (uint32_t j = i + 1; j < 5; ++j) require(outputs[i].first <= outputs[j].first ? outputs[j].first - outputs[i].first >= outputs[i].second : outputs[i].first - outputs[j].first >= outputs[j].second, "kmeans_relocation_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> x(uint64_t(rows) * features), w(rows), old(uint64_t(clusters) * features), sums(old.size(), 0.0f), totals(clusters, 0.0f), row_distances(rows, 0.0f);
        std::vector<int32_t> assigned(rows), order(rows, -1), empty;
        std::memcpy(x.data(), reinterpret_cast<const unsigned char*>(values), x.size() * 4);
        std::memcpy(w.data(), reinterpret_cast<const unsigned char*>(weights), w.size() * 4);
        std::memcpy(assigned.data(), reinterpret_cast<const unsigned char*>(labels), assigned.size() * 4);
        std::memcpy(old.data(), reinterpret_cast<const unsigned char*>(previous), old.size() * 4);
        for (float value : x) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_relocation_value_domain_invalid");
        for (float value : old) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_relocation_value_domain_invalid");
        for (float value : w) require(std::isfinite(value) && value >= 0 && value <= 0x1p32f && (value == 0 || value >= 0x1p-32f), "kmeans_relocation_weight_domain_invalid");
        for (int32_t label : assigned) require(label >= 0 && uint32_t(label) < clusters, "kmeans_relocation_label_domain_invalid");
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t cluster = uint32_t(assigned[row]); totals[cluster] += w[row];
            for (uint32_t feature = 0; feature < features; ++feature) sums[uint64_t(cluster) * features + feature] = std::fma(x[uint64_t(row) * features + feature], w[row], sums[uint64_t(cluster) * features + feature]);
        }
        for (float& value : sums) value = 0.0f + value;
        for (float& value : totals) value = 0.0f + value;
        uint32_t moved = wxkmeans_finalize_centroids(x, w, assigned, old, rows, clusters, features, sums, totals, row_distances, order, empty);
        for (float value : sums) require(std::isfinite(value), "kmeans_relocation_result_invalid");
        for (float value : totals) require(std::isfinite(value), "kmeans_relocation_result_invalid");
        uint32_t result_counts[] = {uint32_t(empty.size()), moved};
        std::memcpy(reinterpret_cast<unsigned char*>(centers), sums.data(), sums.size() * 4);
        std::memcpy(reinterpret_cast<unsigned char*>(cluster_weights), totals.data(), totals.size() * 4);
        std::memcpy(reinterpret_cast<unsigned char*>(distances), row_distances.data(), row_distances.size() * 4);
        std::memcpy(reinterpret_cast<unsigned char*>(selected), order.data(), order.size() * 4);
        std::memcpy(reinterpret_cast<unsigned char*>(counts), result_counts, sizeof(result_counts));
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error && capacity) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1;
    } catch (...) {
        if (error && capacity) { std::strncpy(error, "kmeans_relocation_failed", capacity - 1); error[capacity - 1] = 0; } return 1;
    }
}

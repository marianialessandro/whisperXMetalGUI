#ifndef _OPENMP
#error "OpenMP is required for the optional two-chunk core"
#endif
#include "native_checks.hpp"
#include "native_kmeans_relocation_core.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int omp_get_num_threads(void);
extern "C" int omp_get_thread_num(void);
extern "C" int omp_get_dynamic(void);
struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
extern "C" int wxkmeans_lloyd_assignment(const float*, const float*, const float*, uint32_t, uint32_t, uint32_t, int32_t*, float*, KmeansPotentialMetrics*, char*, uint32_t);

static uint32_t wxkmeans_two_chunk_team(uint32_t rows, uint32_t threads) {
    require(rows > 0 && rows <= 512, "kmeans_two_chunks_shape_invalid");
    require(threads == 1 || threads == 2 || threads == 12, "kmeans_two_chunks_thread_count_not_supported");
    require(omp_get_dynamic() == 0 && std::getenv("KMP_FORCE_REDUCTION") == nullptr && std::getenv("KMP_DETERMINISTIC_REDUCTION") == nullptr, "kmeans_two_chunks_runtime_policy_not_supported");
    return std::min(threads, (rows + 255) / 256);
}

extern "C" int wxkmeans_lloyd_assignment_two_chunks(const float* values, const float* centers, const float* norms, uint32_t rows, uint32_t clusters, uint32_t features, uint32_t threads, int32_t* labels, float* distances, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values && centers && norms && labels && distances && metrics, "kmeans_two_chunks_pointer_missing");
        require(clusters > 0 && clusters <= 256 && features > 0 && features <= 1024, "kmeans_two_chunks_shape_invalid");
        uint32_t team = wxkmeans_two_chunk_team(rows, threads), chunks = (rows + 255) / 256;
        uintptr_t a = reinterpret_cast<uintptr_t>(labels), b = reinterpret_cast<uintptr_t>(distances);
        require(a <= b ? b - a >= uint64_t(rows) * 4 : a - b >= uint64_t(rows) * clusters * 4, "kmeans_two_chunks_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> x(uint64_t(rows) * features), c(uint64_t(clusters) * features), n(clusters), result(uint64_t(rows) * clusters);
        std::vector<int32_t> assigned(rows);
        std::memcpy(x.data(), reinterpret_cast<const unsigned char*>(values), x.size() * 4); std::memcpy(c.data(), reinterpret_cast<const unsigned char*>(centers), c.size() * 4); std::memcpy(n.data(), reinterpret_cast<const unsigned char*>(norms), n.size() * 4);
        int statuses[2] = {}; char messages[2][128] = {}; KmeansPotentialMetrics submetrics[2] = {}; int actual_threads = 0;
        #pragma omp parallel for num_threads(team) schedule(static)
        for (int chunk = 0; chunk < int(chunks); ++chunk) {
            if (chunk == 0) actual_threads = omp_get_num_threads();
            uint32_t start = uint32_t(chunk) * 256, length = std::min(256u, rows - start);
            statuses[chunk] = wxkmeans_lloyd_assignment(x.data() + uint64_t(start) * features, c.data(), n.data(), length, clusters, features, assigned.data() + start, result.data() + uint64_t(start) * clusters, &submetrics[chunk], messages[chunk], 128);
        }
        require(actual_threads == int(team), "kmeans_two_chunks_actual_team_not_supported");
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) require(statuses[chunk] == 0, messages[chunk]);
        std::memcpy(reinterpret_cast<unsigned char*>(labels), assigned.data(), assigned.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(distances), result.data(), result.size() * 4);
        *metrics = {1, chunks, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())}; return 0;
    } catch (const std::exception& failure) { if (error && capacity) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1; }
    catch (...) { if (error && capacity) { std::strncpy(error, "kmeans_two_chunks_failed", capacity - 1); error[capacity - 1] = 0; } return 1; }
}

extern "C" int wxkmeans_lloyd_centroids_two_chunks(const float* values, const float* weights, const int32_t* labels, const float* previous, uint32_t rows, uint32_t clusters, uint32_t features, uint32_t threads, float* centers, float* cluster_weights, float* distances, int32_t* selected, uint32_t* counts, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values && weights && labels && previous && centers && cluster_weights && distances && selected && counts && metrics, "kmeans_two_chunks_pointer_missing");
        require(clusters > 0 && clusters <= rows && clusters <= 256 && features > 0 && features <= 1024, "kmeans_two_chunks_shape_invalid");
        uint32_t team = wxkmeans_two_chunk_team(rows, threads), chunks = (rows + 255) / 256;
        std::pair<uintptr_t, uint64_t> outputs[] = {{reinterpret_cast<uintptr_t>(centers), uint64_t(clusters) * features * 4}, {reinterpret_cast<uintptr_t>(cluster_weights), uint64_t(clusters) * 4}, {reinterpret_cast<uintptr_t>(distances), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(selected), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(counts), 8}};
        for (uint32_t i = 0; i < 5; ++i) for (uint32_t j = i + 1; j < 5; ++j) require(outputs[i].first <= outputs[j].first ? outputs[j].first - outputs[i].first >= outputs[i].second : outputs[i].first - outputs[j].first >= outputs[j].second, "kmeans_two_chunks_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> x(uint64_t(rows) * features), w(rows), old(uint64_t(clusters) * features), sums(old.size(), 0.0f), totals(clusters, 0.0f), row_distances(rows, 0.0f), partials(uint64_t(team) * old.size(), 0.0f), partial_weights(uint64_t(team) * clusters, 0.0f);
        std::vector<int32_t> assigned(rows), order(rows, -1), empty;
        std::memcpy(x.data(), reinterpret_cast<const unsigned char*>(values), x.size() * 4); std::memcpy(w.data(), reinterpret_cast<const unsigned char*>(weights), w.size() * 4); std::memcpy(assigned.data(), reinterpret_cast<const unsigned char*>(labels), assigned.size() * 4); std::memcpy(old.data(), reinterpret_cast<const unsigned char*>(previous), old.size() * 4);
        for (float value : x) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_two_chunks_value_domain_invalid");
        for (float value : old) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_two_chunks_value_domain_invalid");
        for (float value : w) require(std::isfinite(value) && value >= 0 && value <= 0x1p32f && (value == 0 || value >= 0x1p-32f), "kmeans_two_chunks_weight_domain_invalid");
        for (int32_t label : assigned) require(label >= 0 && uint32_t(label) < clusters, "kmeans_two_chunks_label_domain_invalid");
        int actual_threads = 0;
        #pragma omp parallel num_threads(team)
        {
            int worker = omp_get_thread_num();
            if (worker == 0) actual_threads = omp_get_num_threads();
            #pragma omp for schedule(static)
            for (int chunk = 0; chunk < int(chunks); ++chunk) {
                uint32_t start = uint32_t(chunk) * 256, end = std::min(rows, start + 256);
                for (uint32_t row = start; row < end; ++row) {
                    uint32_t cluster = uint32_t(assigned[row]); partial_weights[uint64_t(worker) * clusters + cluster] += w[row];
                    for (uint32_t feature = 0; feature < features; ++feature) {
                        uint64_t index = uint64_t(worker) * old.size() + uint64_t(cluster) * features + feature;
                        partials[index] = std::fma(x[uint64_t(row) * features + feature], w[row], partials[index]);
                    }
                }
            }
        }
        require(actual_threads == int(team), "kmeans_two_chunks_actual_team_not_supported");
        for (uint32_t worker = 0; worker < team; ++worker) {
            for (uint32_t cluster = 0; cluster < clusters; ++cluster) totals[cluster] += partial_weights[uint64_t(worker) * clusters + cluster];
            for (uint64_t index = 0; index < sums.size(); ++index) sums[index] += partials[uint64_t(worker) * old.size() + index];
        }
        uint32_t moved = wxkmeans_finalize_centroids(x, w, assigned, old, rows, clusters, features, sums, totals, row_distances, order, empty);
        for (float value : sums) require(std::isfinite(value), "kmeans_two_chunks_result_invalid");
        for (float value : totals) require(std::isfinite(value), "kmeans_two_chunks_result_invalid");
        uint32_t result_counts[] = {uint32_t(empty.size()), moved};
        std::memcpy(reinterpret_cast<unsigned char*>(centers), sums.data(), sums.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(cluster_weights), totals.data(), totals.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(distances), row_distances.data(), row_distances.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(selected), order.data(), order.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(counts), result_counts, 8);
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())}; return 0;
    } catch (const std::exception& failure) { if (error && capacity) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1; }
    catch (...) { if (error && capacity) { std::strncpy(error, "kmeans_two_chunks_failed", capacity - 1); error[capacity - 1] = 0; } return 1; }
}

extern "C" int wxkmeans_lloyd_center_shifts(const float*, const float*, uint32_t, uint32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_lloyd_centroids_and_shifts(const float* values, const float* weights, const int32_t* labels, const float* previous, uint32_t rows, uint32_t clusters, uint32_t features, uint32_t threads, float* centers, float* cluster_weights, float* distances, int32_t* selected, uint32_t* counts, float* shifts, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values && weights && labels && previous && centers && cluster_weights && distances && selected && counts && shifts && metrics, "kmeans_fused_centroids_pointer_missing");
        require(rows > 0 && rows <= 512 && clusters > 0 && clusters <= rows && clusters <= 256 && features > 0 && features <= 1024, "kmeans_fused_centroids_shape_invalid");
        std::pair<uintptr_t, uint64_t> outputs[] = {{reinterpret_cast<uintptr_t>(centers), uint64_t(clusters) * features * 4}, {reinterpret_cast<uintptr_t>(cluster_weights), uint64_t(clusters) * 4}, {reinterpret_cast<uintptr_t>(distances), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(selected), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(counts), 8}, {reinterpret_cast<uintptr_t>(shifts), uint64_t(clusters) * 4}};
        for (uint32_t i = 0; i < 6; ++i) for (uint32_t j = i + 1; j < 6; ++j) require(outputs[i].first <= outputs[j].first ? outputs[j].first - outputs[i].first >= outputs[i].second : outputs[i].first - outputs[j].first >= outputs[j].second, "kmeans_fused_centroids_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> old(uint64_t(clusters) * features), updated(old.size()), totals(clusters), row_distances(rows), movement(clusters);
        std::vector<int32_t> order(rows); uint32_t relocated[2]; KmeansPotentialMetrics child;
        std::memcpy(old.data(), reinterpret_cast<const unsigned char*>(previous), old.size() * 4);
        int status = wxkmeans_lloyd_centroids_two_chunks(values, weights, labels, old.data(), rows, clusters, features, threads, updated.data(), totals.data(), row_distances.data(), order.data(), relocated, &child, error, capacity);
        if (status != 0) return status;
        status = wxkmeans_lloyd_center_shifts(updated.data(), old.data(), clusters, features, movement.data(), &child, error, capacity);
        if (status != 0) return status;
        std::memcpy(reinterpret_cast<unsigned char*>(centers), updated.data(), updated.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(cluster_weights), totals.data(), totals.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(distances), row_distances.data(), row_distances.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(selected), order.data(), order.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(counts), relocated, 8); std::memcpy(reinterpret_cast<unsigned char*>(shifts), movement.data(), movement.size() * 4);
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())}; return 0;
    } catch (const std::exception& failure) { if (error && capacity) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1; }
    catch (...) { if (error && capacity) { std::strncpy(error, "kmeans_fused_centroids_failed", capacity - 1); error[capacity - 1] = 0; } return 1; }
}

extern "C" int wxkmeans_lloyd_update(const float* values, const float* weights, const float* previous, const float* norms, uint32_t rows, uint32_t clusters, uint32_t features, uint32_t threads, int32_t* labels, float* scores, float* centers, float* cluster_weights, float* distances, int32_t* selected, uint32_t* counts, float* shifts, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values && weights && previous && norms && labels && scores && centers && cluster_weights && distances && selected && counts && shifts && metrics, "kmeans_update_pointer_missing");
        require(rows > 0 && rows <= 512 && clusters > 0 && clusters <= rows && clusters <= 256 && features > 0 && features <= 1024, "kmeans_update_shape_invalid");
        std::pair<uintptr_t, uint64_t> outputs[] = {{reinterpret_cast<uintptr_t>(labels), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(scores), uint64_t(rows) * clusters * 4}, {reinterpret_cast<uintptr_t>(centers), uint64_t(clusters) * features * 4}, {reinterpret_cast<uintptr_t>(cluster_weights), uint64_t(clusters) * 4}, {reinterpret_cast<uintptr_t>(distances), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(selected), uint64_t(rows) * 4}, {reinterpret_cast<uintptr_t>(counts), 8}, {reinterpret_cast<uintptr_t>(shifts), uint64_t(clusters) * 4}};
        for (uint32_t i = 0; i < 8; ++i) for (uint32_t j = i + 1; j < 8; ++j) require(outputs[i].first <= outputs[j].first ? outputs[j].first - outputs[i].first >= outputs[i].second : outputs[i].first - outputs[j].first >= outputs[j].second, "kmeans_update_outputs_overlap");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> old(uint64_t(clusters) * features), assigned_scores(uint64_t(rows) * clusters), updated(old.size()), totals(clusters), row_distances(rows), movement(clusters);
        std::vector<int32_t> assigned(rows), order(rows); uint32_t relocated[2]; KmeansPotentialMetrics assignment, child;
        std::memcpy(old.data(), reinterpret_cast<const unsigned char*>(previous), old.size() * 4);
        int status = wxkmeans_lloyd_assignment_two_chunks(values, old.data(), norms, rows, clusters, features, threads, assigned.data(), assigned_scores.data(), &assignment, error, capacity);
        if (status != 0) return status;
        status = wxkmeans_lloyd_centroids_and_shifts(values, weights, assigned.data(), old.data(), rows, clusters, features, threads, updated.data(), totals.data(), row_distances.data(), order.data(), relocated, movement.data(), &child, error, capacity);
        if (status != 0) return status;
        std::memcpy(reinterpret_cast<unsigned char*>(labels), assigned.data(), assigned.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(scores), assigned_scores.data(), assigned_scores.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(centers), updated.data(), updated.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(cluster_weights), totals.data(), totals.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(distances), row_distances.data(), row_distances.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(selected), order.data(), order.size() * 4); std::memcpy(reinterpret_cast<unsigned char*>(counts), relocated, 8); std::memcpy(reinterpret_cast<unsigned char*>(shifts), movement.data(), movement.size() * 4);
        *metrics = {1, assignment.blas_calls, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())}; return 0;
    } catch (const std::exception& failure) { if (error && capacity) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1; }
    catch (...) { if (error && capacity) { std::strncpy(error, "kmeans_update_failed", capacity - 1); error[capacity - 1] = 0; } return 1; }
}

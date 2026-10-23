#ifndef _OPENMP
#error "OpenMP is required for the optional inertia core"
#endif
#include "native_checks.hpp"
#include "native_kmeans_distance.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

extern "C" int omp_get_num_threads(void);
extern "C" int omp_get_dynamic(void);
struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
static_assert(sizeof(KmeansPotentialMetrics) == 16);

extern "C" const char* wxkmeans_inertia_runtime_path() {
    Dl_info info = {};
    return dladdr(reinterpret_cast<const void*>(&omp_get_num_threads), &info) && info.dli_fname != nullptr ? info.dli_fname : nullptr;
}

extern "C" int wxkmeans_inertia_dense(const float* values, const float* weights, const float* centers, const int32_t* labels, uint32_t rows, uint32_t clusters, uint32_t features, uint32_t threads, int32_t single_label, float* output, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(values != nullptr && weights != nullptr && centers != nullptr && labels != nullptr && output != nullptr && metrics != nullptr, "kmeans_inertia_pointer_missing");
        require(rows > 0 && rows <= 256 && clusters > 0 && clusters <= 256 && features > 0 && features <= 1024, "kmeans_inertia_shape_invalid");
        require(threads == 1 || threads == 2 || threads == 12, "kmeans_inertia_thread_count_not_supported");
        require(single_label >= -1 && single_label < int32_t(clusters), "kmeans_inertia_single_label_invalid");
        require(omp_get_dynamic() == 0 && std::getenv("KMP_FORCE_REDUCTION") == nullptr && std::getenv("KMP_DETERMINISTIC_REDUCTION") == nullptr, "kmeans_inertia_runtime_policy_not_supported");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> x(uint64_t(rows) * features), w(rows), c(uint64_t(clusters) * features);
        std::vector<int32_t> assigned(rows);
        std::memcpy(x.data(), reinterpret_cast<const unsigned char*>(values), x.size() * sizeof(float));
        std::memcpy(w.data(), reinterpret_cast<const unsigned char*>(weights), w.size() * sizeof(float));
        std::memcpy(c.data(), reinterpret_cast<const unsigned char*>(centers), c.size() * sizeof(float));
        std::memcpy(assigned.data(), reinterpret_cast<const unsigned char*>(labels), assigned.size() * sizeof(int32_t));
        for (float value : x) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_inertia_value_domain_invalid");
        for (float value : c) require(std::isfinite(value) && std::abs(value) <= 0x1p32f, "kmeans_inertia_value_domain_invalid");
        for (float value : w) require(std::isfinite(value) && value >= 0 && value <= 0x1p32f && (value == 0 || value >= 0x1p-32f), "kmeans_inertia_weight_domain_invalid");
        for (int32_t label : assigned) require(label >= 0 && uint32_t(label) < clusters, "kmeans_inertia_label_domain_invalid");
        float inertia = 0.0f; int actual_threads = 0;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:inertia)
        for (int row = 0; row < int(rows); ++row) {
            if (row == 0) actual_threads = omp_get_num_threads();
            int32_t label = assigned[row];
            if (single_label < 0 || single_label == label) {
                float distance = wxkmeans_squared_distance(x.data() + uint64_t(row) * features, c.data() + uint64_t(label) * features, features);
                inertia = std::fma(distance, w[row], inertia);
            }
        }
        require(actual_threads == int(threads), "kmeans_inertia_actual_team_not_supported");
        require(std::isfinite(inertia) && inertia >= 0, "kmeans_inertia_result_invalid");
        std::memcpy(reinterpret_cast<unsigned char*>(output), &inertia, sizeof(float));
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; }
        return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_inertia_failed", capacity - 1); error[capacity - 1] = 0; }
        return 1;
    }
}

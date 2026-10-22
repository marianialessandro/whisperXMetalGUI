#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include "native_checks.hpp"
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
static_assert(sizeof(KmeansPotentialMetrics) == 16);

extern "C" int wxkmeans_weighted_potentials(const float* distances, const float* weights, uint32_t rows, uint32_t columns, float* output, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(distances != nullptr && weights != nullptr && output != nullptr && metrics != nullptr, "kmeans_potential_pointer_missing");
        require(rows > 0 && rows <= 1024 && columns > 0 && columns <= 4096 && uint64_t(rows) * columns <= 1048576, "kmeans_potential_shape_invalid");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> left(uint64_t(rows) * columns), right(columns), result(rows);
        std::memcpy(left.data(), reinterpret_cast<const unsigned char*>(distances), left.size() * sizeof(float));
        std::memcpy(right.data(), reinterpret_cast<const unsigned char*>(weights), right.size() * sizeof(float));
        auto valid = [](float value, float minimum, float maximum) {
            uint32_t bits = std::bit_cast<uint32_t>(value), magnitude = bits & 0x7fffffffu;
            return magnitude == 0 || ((bits >> 31) == 0 && magnitude >= std::bit_cast<uint32_t>(minimum) && magnitude <= std::bit_cast<uint32_t>(maximum));
        };
        for (float value : left) require(valid(value, 0x1p-64f, 0x1p32f), "kmeans_potential_distance_domain_invalid");
        for (float value : right) require(valid(value, 0x1p-32f, 0x1p32f), "kmeans_potential_weight_domain_invalid");
        uint32_t blas_calls = 1;
        if (rows == 1) {
            // NumPy 2.2.6 matmul dispatches scalar outputs to one ILP64 sdot within this bounded domain.
            double total = 0.; total += cblas_sdot(columns, left.data(), 1, right.data(), 1); result[0] = static_cast<float>(total);
        } else if (columns == 1) {
            // NumPy uses the scalar-vector loop with a separately rounded float32 product and zero addition.
            for (uint32_t row = 0; row < rows; ++row) { volatile float product = left[row] * right[0]; result[row] = 0.0f + product; }
            blas_calls = 0;
        } else {
            cblas_sgemv(CblasColMajor, CblasTrans, columns, rows, 1.0f, left.data(), columns, right.data(), 1, 0.0f, result.data(), 1);
        }
        for (float value : result) require(valid(value, 0x1p-96f, 0x1p76f), "kmeans_potential_result_invalid");
        std::memcpy(reinterpret_cast<unsigned char*>(output), result.data(), result.size() * sizeof(float));
        *metrics = {1, blas_calls, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; }
        return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_potential_failed", capacity - 1); error[capacity - 1] = 0; }
        return 1;
    }
}

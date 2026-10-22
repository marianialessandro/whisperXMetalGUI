#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include "native_checks.hpp"
#include <bit>
#include <chrono>
#include <cmath>
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

extern "C" int wxkmeans_distance_products(const double* left, const double* right, uint32_t rows, uint32_t columns, uint32_t features, double* output, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(left != nullptr && right != nullptr && output != nullptr && metrics != nullptr, "kmeans_distance_product_pointer_missing");
        require(rows > 0 && rows <= 256 && columns > 0 && columns <= 256 && features > 0 && features <= 1024, "kmeans_distance_product_shape_invalid");
        auto started = std::chrono::steady_clock::now();
        std::vector<double> a(uint64_t(rows) * features), b(uint64_t(columns) * features), result(uint64_t(rows) * columns);
        std::memcpy(a.data(), reinterpret_cast<const unsigned char*>(left), a.size() * sizeof(double));
        std::memcpy(b.data(), reinterpret_cast<const unsigned char*>(right), b.size() * sizeof(double));
        auto valid = [](double value) {
            uint64_t magnitude = std::bit_cast<uint64_t>(value) & 0x7ffffffffffffffful;
            return magnitude == 0 || (magnitude >= std::bit_cast<uint64_t>(0x1p-200) && magnitude <= std::bit_cast<uint64_t>(0x1p200));
        };
        for (double value : a) require(valid(value), "kmeans_distance_product_value_domain_invalid");
        for (double value : b) require(valid(value), "kmeans_distance_product_value_domain_invalid");
        uint32_t blas_calls = 1;
        if (rows == 1 && columns == 1) {
            result[0] = 0.; result[0] += cblas_ddot(features, a.data(), 1, b.data(), 1);
        } else if (features == 1) {
            for (uint32_t row = 0; row < rows; ++row) for (uint32_t column = 0; column < columns; ++column) { volatile double product = a[row] * b[column]; result[uint64_t(row) * columns + column] = 0. + product; }
            blas_calls = 0;
        } else if (rows == 1) {
            cblas_dgemv(CblasColMajor, CblasTrans, features, columns, 1., b.data(), features, a.data(), 1, 0., result.data(), 1);
        } else if (columns == 1) {
            cblas_dgemv(CblasColMajor, CblasTrans, features, rows, 1., a.data(), features, b.data(), 1, 0., result.data(), 1);
        } else if (left == right && rows == columns) {
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, rows, features, 1., a.data(), features, 0., result.data(), columns);
            for (uint32_t row = 0; row < rows; ++row) for (uint32_t column = row + 1; column < columns; ++column) result[uint64_t(column) * columns + row] = result[uint64_t(row) * columns + column];
        } else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, columns, features, 1., a.data(), features, b.data(), features, 0., result.data(), columns);
        }
        for (double value : result) require(std::isfinite(value), "kmeans_distance_product_result_invalid");
        std::memcpy(reinterpret_cast<unsigned char*>(output), result.data(), result.size() * sizeof(double));
        *metrics = {1, blas_calls, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; }
        return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_distance_product_failed", capacity - 1); error[capacity - 1] = 0; }
        return 1;
    }
}

extern "C" int wxkmeans_upcast_chunk(const float* input, uint32_t rows, uint32_t features, double* output, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(input != nullptr && output != nullptr && metrics != nullptr, "kmeans_upcast_chunk_pointer_missing");
        require(rows > 0 && rows <= 256 && features > 0 && features <= 1024, "kmeans_upcast_chunk_shape_invalid");
        auto started = std::chrono::steady_clock::now();
        uint64_t count = uint64_t(rows) * features;
        std::vector<uint64_t> result(count);
        for (uint64_t index = 0; index < count; ++index) {
            uint32_t bits; std::memcpy(&bits, reinterpret_cast<const unsigned char*>(input) + index * sizeof(float), sizeof(bits));
            uint32_t exponent = (bits >> 23) & 255u, fraction = bits & 0x7fffffu;
            require(exponent != 255u, "kmeans_upcast_chunk_finite_input_required");
            uint64_t value = uint64_t(bits >> 31) << 63;
            if (exponent != 0) value |= (uint64_t(exponent + 896u) << 52) | (uint64_t(fraction) << 29);
            else if (fraction != 0) {
                uint32_t highest = 31u - uint32_t(std::countl_zero(fraction));
                value |= (uint64_t(highest + 874u) << 52) | (uint64_t(fraction - (1u << highest)) << (52u - highest));
            }
            result[index] = value;
        }
        std::memcpy(reinterpret_cast<unsigned char*>(output), result.data(), count * sizeof(double));
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())};
        return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; }
        return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_upcast_chunk_failed", capacity - 1); error[capacity - 1] = 0; }
        return 1;
    }
}

// NumPy pairwise reduction order is adapted under the notice in native_numpy_norm_notice.txt.
static float kmeans_pairwise_shift_sum(const float* values, uint32_t count) {
    if (count < 8) { float total = -0.0f; for (uint32_t index = 0; index < count; ++index) total += values[index]; return total; }
    if (count > 128) { uint32_t split = (count / 2) & ~7u; return kmeans_pairwise_shift_sum(values, split) + kmeans_pairwise_shift_sum(values + split, count - split); }
    float lanes[8]; for (uint32_t lane = 0; lane < 8; ++lane) lanes[lane] = values[lane];
    uint32_t index = 8;
    for (; index < count - count % 8; index += 8) for (uint32_t lane = 0; lane < 8; ++lane) lanes[lane] += values[index + lane];
    float total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; index < count; ++index) total += values[index];
    return total;
}

extern "C" int wxkmeans_lloyd_shift_total(const float* shifts, uint32_t count, float* output, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(shifts != nullptr && output != nullptr && metrics != nullptr, "kmeans_lloyd_shift_pointer_missing");
        require(count > 0 && count <= 1024, "kmeans_lloyd_shift_count_invalid");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> values(count), squares(count); std::memcpy(values.data(), reinterpret_cast<const unsigned char*>(shifts), count * sizeof(float));
        for (uint32_t index = 0; index < count; ++index) {
            uint32_t bits = std::bit_cast<uint32_t>(values[index]), magnitude = bits & 0x7fffffffu;
            require(magnitude == 0 || ((bits >> 31) == 0 && magnitude >= std::bit_cast<uint32_t>(0x1p-60f) && magnitude <= std::bit_cast<uint32_t>(0x1p32f)), "kmeans_lloyd_shift_value_domain_invalid");
            volatile float square = values[index] * values[index]; squares[index] = square;
        }
        float result = 0.0f + kmeans_pairwise_shift_sum(squares.data(), count);
        require(std::isfinite(result) && result >= 0, "kmeans_lloyd_shift_result_invalid");
        std::memcpy(reinterpret_cast<unsigned char*>(output), &result, sizeof(result));
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())}; return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_lloyd_shift_failed", capacity - 1); error[capacity - 1] = 0; } return 1;
    }
}

extern "C" int wxkmeans_lloyd_labels_equal(const int32_t* labels, const int32_t* previous, uint32_t count, uint32_t* output, KmeansPotentialMetrics* metrics, char* error, uint32_t capacity) {
    try {
        require(labels != nullptr && previous != nullptr && output != nullptr && metrics != nullptr, "kmeans_lloyd_labels_pointer_missing");
        require(count > 0 && count <= 4096, "kmeans_lloyd_labels_count_invalid");
        auto started = std::chrono::steady_clock::now();
        std::vector<int32_t> a(count), b(count); std::memcpy(a.data(), reinterpret_cast<const unsigned char*>(labels), count * sizeof(int32_t)); std::memcpy(b.data(), reinterpret_cast<const unsigned char*>(previous), count * sizeof(int32_t));
        uint32_t result = 1;
        for (uint32_t index = 0; index < count; ++index) {
            require(a[index] >= -1 && a[index] <= 4095 && b[index] >= -1 && b[index] <= 4095, "kmeans_lloyd_labels_value_domain_invalid");
            if (a[index] != b[index]) result = 0;
        }
        std::memcpy(reinterpret_cast<unsigned char*>(output), &result, sizeof(result));
        *metrics = {1, 0, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count())}; return 0;
    } catch (const std::exception& failure) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, failure.what(), capacity - 1); error[capacity - 1] = 0; } return 1;
    } catch (...) {
        if (error != nullptr && capacity > 0) { std::strncpy(error, "kmeans_lloyd_labels_failed", capacity - 1); error[capacity - 1] = 0; } return 1;
    }
}

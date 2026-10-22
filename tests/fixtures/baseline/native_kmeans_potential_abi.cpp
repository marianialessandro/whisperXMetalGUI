#include <cassert>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
extern "C" int wxkmeans_weighted_potentials(const float*, const float*, uint32_t, uint32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_distance_products(const double*, const double*, uint32_t, uint32_t, uint32_t, double*, KmeansPotentialMetrics*, char*, uint32_t);

int main() {
    float left[] = {1, 2, 3, 4}, weights[] = {2, 3}, output[] = {91, 92};
    KmeansPotentialMetrics metrics = {}; char error[128] = {};
    assert(wxkmeans_weighted_potentials(left, weights, 2, 2, output, &metrics, error, sizeof(error)) == 0 && output[0] == 8 && output[1] == 18 && metrics.version == 1 && metrics.blas_calls == 1);
    assert(wxkmeans_weighted_potentials(left, weights, 1, 2, output, &metrics, error, sizeof(error)) == 0 && output[0] == 8);
    assert(wxkmeans_weighted_potentials(left, weights, 2, 1, output, &metrics, error, sizeof(error)) == 0 && output[0] == 2 && output[1] == 4 && metrics.blas_calls == 0);
    output[0] = 91; output[1] = 92;
    assert(wxkmeans_weighted_potentials(nullptr, weights, 2, 2, output, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_weighted_potentials(left, nullptr, 2, 2, output, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_weighted_potentials(left, weights, 2, 2, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_weighted_potentials(left, weights, 2, 2, output, nullptr, error, sizeof(error)) != 0);
    for (uint32_t rows : {0u, 1025u}) assert(wxkmeans_weighted_potentials(left, weights, rows, 2, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91);
    for (uint32_t columns : {0u, 4097u}) assert(wxkmeans_weighted_potentials(left, weights, 2, columns, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91);
    assert(wxkmeans_weighted_potentials(left, weights, 1024, 4096, output, &metrics, error, sizeof(error)) != 0);
    for (float value : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), -1.0f, 0x1p-65f, 0x1p33f}) {
        left[3] = value;
        assert(wxkmeans_weighted_potentials(left, weights, 2, 2, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91 && output[1] == 92);
    }
    left[3] = 4;
    for (float value : {-1.0f, 0x1p-33f, 0x1p33f, std::numeric_limits<float>::infinity()}) {
        weights[1] = value;
        assert(wxkmeans_weighted_potentials(left, weights, 2, 2, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91 && output[1] == 92);
    }
    weights[1] = 3;
    unsigned char unaligned_left[sizeof(left) + 1], unaligned_weights[sizeof(weights) + 1], unaligned_output[sizeof(output) + 1];
    std::memcpy(unaligned_left + 1, left, sizeof(left)); std::memcpy(unaligned_weights + 1, weights, sizeof(weights));
    assert(wxkmeans_weighted_potentials(reinterpret_cast<float*>(unaligned_left + 1), reinterpret_cast<float*>(unaligned_weights + 1), 2, 2, reinterpret_cast<float*>(unaligned_output + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(output, unaligned_output + 1, sizeof(output)); assert(output[0] == 8 && output[1] == 18);
    assert(wxkmeans_weighted_potentials(left, weights, 2, 2, left, &metrics, error, sizeof(error)) == 0 && left[0] == 8 && left[1] == 18);
    float zero[] = {-0.0f};
    assert(wxkmeans_weighted_potentials(zero, zero, 1, 1, output, &metrics, nullptr, 0) == 0 && std::bit_cast<uint32_t>(output[0]) == 0);
    assert(wxkmeans_weighted_potentials(nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr, 0) != 0);
    char tiny[] = {'x'}; assert(wxkmeans_weighted_potentials(nullptr, nullptr, 0, 0, nullptr, nullptr, tiny, 1) != 0 && tiny[0] == 0);
    double a[] = {1, 2, 3, 4}, b[] = {2, 3, 4, 5}, products[] = {91, 92, 93, 94};
    assert(wxkmeans_distance_products(a, b, 2, 2, 2, products, &metrics, error, sizeof(error)) == 0 && products[0] == 8 && products[1] == 14 && products[2] == 18 && products[3] == 32);
    assert(wxkmeans_distance_products(a, a, 2, 2, 2, products, &metrics, error, sizeof(error)) == 0 && products[0] == 5 && products[1] == 11 && products[2] == 11 && products[3] == 25);
    assert(wxkmeans_distance_products(a, b, 1, 1, 2, products, &metrics, error, sizeof(error)) == 0 && products[0] == 8);
    products[0] = 91;
    assert(wxkmeans_distance_products(nullptr, b, 2, 2, 2, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
    assert(wxkmeans_distance_products(a, nullptr, 2, 2, 2, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
    assert(wxkmeans_distance_products(a, b, 2, 2, 2, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_distance_products(a, b, 2, 2, 2, products, nullptr, error, sizeof(error)) != 0 && products[0] == 91);
    for (uint32_t count : {0u, 257u}) {
        assert(wxkmeans_distance_products(a, b, count, 2, 2, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
        assert(wxkmeans_distance_products(a, b, 2, count, 2, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
    }
    for (uint32_t features : {0u, 1025u}) assert(wxkmeans_distance_products(a, b, 2, 2, features, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
    for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 0x1p-201, 0x1p201}) {
        a[3] = value;
        assert(wxkmeans_distance_products(a, b, 2, 2, 2, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
        a[3] = 4; b[3] = value;
        assert(wxkmeans_distance_products(a, b, 2, 2, 2, products, &metrics, error, sizeof(error)) != 0 && products[0] == 91);
        b[3] = 5;
    }
    unsigned char unaligned_a[sizeof(a) + 1], unaligned_b[sizeof(b) + 1], unaligned_products[sizeof(products) + 1];
    std::memcpy(unaligned_a + 1, a, sizeof(a)); std::memcpy(unaligned_b + 1, b, sizeof(b));
    assert(wxkmeans_distance_products(reinterpret_cast<double*>(unaligned_a + 1), reinterpret_cast<double*>(unaligned_b + 1), 2, 2, 2, reinterpret_cast<double*>(unaligned_products + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(products, unaligned_products + 1, sizeof(products)); assert(products[0] == 8 && products[3] == 32);
    assert(wxkmeans_distance_products(a, b, 2, 2, 2, a, &metrics, error, sizeof(error)) == 0 && a[0] == 8 && a[3] == 32);
    assert(wxkmeans_distance_products(nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr, 0) != 0);
    tiny[0] = 'x'; assert(wxkmeans_distance_products(nullptr, nullptr, 0, 0, 0, nullptr, nullptr, tiny, 1) != 0 && tiny[0] == 0);

}

#include <cassert>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
extern "C" int wxkmeans_weighted_potentials(const float*, const float*, uint32_t, uint32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_distance_products(const double*, const double*, uint32_t, uint32_t, uint32_t, double*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_upcast_chunk(const float*, uint32_t, uint32_t, double*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_lloyd_shift_total(const float*, uint32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_lloyd_labels_equal(const int32_t*, const int32_t*, uint32_t, uint32_t*, KmeansPotentialMetrics*, char*, uint32_t);

extern "C" int wxkmeans_same_clustering(const int32_t*, const int32_t*, uint32_t, uint32_t, uint32_t*, KmeansPotentialMetrics*, char*, uint32_t);

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

    float chunk[] = {-0.0f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()};
    double converted[] = {91, 92, 93};
    assert(wxkmeans_upcast_chunk(chunk, 1, 3, converted, &metrics, error, sizeof(error)) == 0 && metrics.version == 1 && metrics.blas_calls == 0);
    assert(std::bit_cast<uint64_t>(converted[0]) == 0x8000000000000000ul && converted[1] == 0x1p-149 && converted[2] == static_cast<double>(std::numeric_limits<float>::max()));
    converted[0] = 91; converted[1] = 92; converted[2] = 93;
    assert(wxkmeans_upcast_chunk(nullptr, 1, 3, converted, &metrics, error, sizeof(error)) != 0 && converted[0] == 91);
    assert(wxkmeans_upcast_chunk(chunk, 1, 3, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_upcast_chunk(chunk, 1, 3, converted, nullptr, error, sizeof(error)) != 0 && converted[0] == 91);
    for (uint32_t rows : {0u, 257u}) assert(wxkmeans_upcast_chunk(chunk, rows, 3, converted, &metrics, error, sizeof(error)) != 0 && converted[0] == 91);
    for (uint32_t features : {0u, 1025u}) assert(wxkmeans_upcast_chunk(chunk, 1, features, converted, &metrics, error, sizeof(error)) != 0 && converted[0] == 91);
    for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc00001u}) {
        chunk[2] = std::bit_cast<float>(bits);
        assert(wxkmeans_upcast_chunk(chunk, 1, 3, converted, &metrics, error, sizeof(error)) != 0 && converted[0] == 91 && converted[1] == 92 && converted[2] == 93);
    }
    chunk[2] = 3;
    unsigned char unaligned_chunk[sizeof(chunk) + 1], unaligned_converted[sizeof(converted) + 1];
    std::memcpy(unaligned_chunk + 1, chunk, sizeof(chunk));
    assert(wxkmeans_upcast_chunk(reinterpret_cast<float*>(unaligned_chunk + 1), 1, 3, reinterpret_cast<double*>(unaligned_converted + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(converted, unaligned_converted + 1, sizeof(converted)); assert(converted[1] == 0x1p-149 && converted[2] == 3);
    unsigned char overlapping[sizeof(converted)]; std::memcpy(overlapping, chunk, sizeof(chunk));
    assert(wxkmeans_upcast_chunk(reinterpret_cast<float*>(overlapping), 1, 3, reinterpret_cast<double*>(overlapping), &metrics, error, sizeof(error)) == 0);
    std::memcpy(converted, overlapping, sizeof(converted)); assert(converted[1] == 0x1p-149 && converted[2] == 3);
    assert(wxkmeans_upcast_chunk(nullptr, 0, 0, nullptr, nullptr, nullptr, 0) != 0);
    tiny[0] = 'x'; assert(wxkmeans_upcast_chunk(nullptr, 0, 0, nullptr, nullptr, tiny, 1) != 0 && tiny[0] == 0);

    float shifts[] = {3, 4}, total = 91;
    assert(wxkmeans_lloyd_shift_total(shifts, 2, &total, &metrics, error, sizeof(error)) == 0 && total == 25 && metrics.blas_calls == 0);
    total = 91;
    assert(wxkmeans_lloyd_shift_total(nullptr, 2, &total, &metrics, error, sizeof(error)) != 0 && total == 91);
    assert(wxkmeans_lloyd_shift_total(shifts, 2, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_shift_total(shifts, 2, &total, nullptr, error, sizeof(error)) != 0 && total == 91);
    for (uint32_t count : {0u, 1025u}) assert(wxkmeans_lloyd_shift_total(shifts, count, &total, &metrics, error, sizeof(error)) != 0 && total == 91);
    for (float value : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 0x1p-61f, 0x1p33f}) {
        shifts[1] = value;
        assert(wxkmeans_lloyd_shift_total(shifts, 2, &total, &metrics, error, sizeof(error)) != 0 && total == 91);
    }
    shifts[1] = 4;
    unsigned char unaligned_shifts[sizeof(shifts) + 1], unaligned_total[sizeof(total) + 1]; std::memcpy(unaligned_shifts + 1, shifts, sizeof(shifts));
    assert(wxkmeans_lloyd_shift_total(reinterpret_cast<float*>(unaligned_shifts + 1), 2, reinterpret_cast<float*>(unaligned_total + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(&total, unaligned_total + 1, sizeof(total)); assert(total == 25);
    assert(wxkmeans_lloyd_shift_total(shifts, 2, shifts, &metrics, error, sizeof(error)) == 0 && shifts[0] == 25);
    assert(wxkmeans_lloyd_shift_total(nullptr, 0, nullptr, nullptr, nullptr, 0) != 0);

    int32_t labels[] = {0, 1, 4095}, previous[] = {0, 1, 4095}; uint32_t equal = 91;
    assert(wxkmeans_lloyd_labels_equal(labels, previous, 3, &equal, &metrics, error, sizeof(error)) == 0 && equal == 1 && metrics.blas_calls == 0);
    previous[0] = -1;
    assert(wxkmeans_lloyd_labels_equal(labels, previous, 3, &equal, &metrics, error, sizeof(error)) == 0 && equal == 0);
    equal = 91;
    assert(wxkmeans_lloyd_labels_equal(nullptr, previous, 3, &equal, &metrics, error, sizeof(error)) != 0 && equal == 91);
    assert(wxkmeans_lloyd_labels_equal(labels, nullptr, 3, &equal, &metrics, error, sizeof(error)) != 0 && equal == 91);
    assert(wxkmeans_lloyd_labels_equal(labels, previous, 3, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_labels_equal(labels, previous, 3, &equal, nullptr, error, sizeof(error)) != 0 && equal == 91);
    for (uint32_t count : {0u, 4097u}) assert(wxkmeans_lloyd_labels_equal(labels, previous, count, &equal, &metrics, error, sizeof(error)) != 0 && equal == 91);
    for (int32_t value : {-2, 4096, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()}) {
        previous[2] = value;
        assert(wxkmeans_lloyd_labels_equal(labels, previous, 3, &equal, &metrics, error, sizeof(error)) != 0 && equal == 91);
    }
    previous[2] = 4095;
    unsigned char unaligned_labels[sizeof(labels) + 1], unaligned_equal[sizeof(equal) + 1]; std::memcpy(unaligned_labels + 1, labels, sizeof(labels));
    assert(wxkmeans_lloyd_labels_equal(reinterpret_cast<int32_t*>(unaligned_labels + 1), labels, 3, reinterpret_cast<uint32_t*>(unaligned_equal + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(&equal, unaligned_equal + 1, sizeof(equal)); assert(equal == 1);
    assert(wxkmeans_lloyd_labels_equal(labels, labels, 3, reinterpret_cast<uint32_t*>(labels), &metrics, error, sizeof(error)) == 0 && labels[0] == 1);
    assert(wxkmeans_lloyd_labels_equal(nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0) != 0);

    int32_t map_a[] = {0, 1, 0}, map_b[] = {1, 0, 1}; uint32_t equivalent = 91;
    assert(wxkmeans_same_clustering(map_a, map_b, 3, 2, &equivalent, &metrics, error, sizeof(error)) == 0 && equivalent == 1);
    map_b[2] = 0;
    assert(wxkmeans_same_clustering(map_a, map_b, 3, 2, &equivalent, &metrics, error, sizeof(error)) == 0 && equivalent == 0);
    equivalent = 91;
    assert(wxkmeans_same_clustering(nullptr, map_b, 3, 2, &equivalent, &metrics, error, sizeof(error)) != 0 && equivalent == 91);
    assert(wxkmeans_same_clustering(map_a, nullptr, 3, 2, &equivalent, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_same_clustering(map_a, map_b, 3, 2, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_same_clustering(map_a, map_b, 3, 2, &equivalent, nullptr, error, sizeof(error)) != 0);
    for (uint32_t count : {0u, 4097u}) assert(wxkmeans_same_clustering(map_a, map_b, count, 2, &equivalent, &metrics, error, sizeof(error)) != 0 && equivalent == 91);
    for (uint32_t clusters : {0u, 4097u}) assert(wxkmeans_same_clustering(map_a, map_b, 3, clusters, &equivalent, &metrics, error, sizeof(error)) != 0 && equivalent == 91);
    for (int32_t value : {-1, 2, std::numeric_limits<int32_t>::max()}) {
        map_b[2] = value;
        assert(wxkmeans_same_clustering(map_a, map_b, 3, 2, &equivalent, &metrics, error, sizeof(error)) != 0 && equivalent == 91);
    }
    map_b[2] = 1;
    unsigned char unaligned_map[sizeof(map_a) + 1], unaligned_equivalent[sizeof(equivalent) + 1]; std::memcpy(unaligned_map + 1, map_a, sizeof(map_a));
    assert(wxkmeans_same_clustering(reinterpret_cast<int32_t*>(unaligned_map + 1), map_b, 3, 2, reinterpret_cast<uint32_t*>(unaligned_equivalent + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(&equivalent, unaligned_equivalent + 1, sizeof(equivalent)); assert(equivalent == 1);
    assert(wxkmeans_same_clustering(map_a, map_b, 3, 2, reinterpret_cast<uint32_t*>(map_a), &metrics, error, sizeof(error)) == 0 && map_a[0] == 1);
    assert(wxkmeans_same_clustering(nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr, 0) != 0);

}

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
extern "C" int wxkmeans_lloyd_relocated_centroids(const float*, const float*, const int32_t*, const float*, uint32_t, uint32_t, uint32_t, float*, float*, float*, int32_t*, uint32_t*, KmeansPotentialMetrics*, char*, uint32_t);
int main() {
    float x[] = {0, 2, -2, 1, -1, 0}, weights[] = {1, 1, 1, 1, 1, 1}, old[] = {0, 0, 0}, centers[3] = {}, totals[3] = {}, distances[6] = {};
    int32_t labels[6] = {}, selected[6] = {}; uint32_t counts[2] = {}; KmeansPotentialMetrics metrics = {}; char error[128] = {};
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) == 0);
    assert(centers[0] == 0 && centers[1] == -2 && centers[2] == 2 && totals[0] == 4 && selected[0] == 2 && selected[1] == 1 && counts[0] == 2 && counts[1] == 2 && metrics.blas_calls == 0);
    centers[0] = 91; counts[0] = 92;
    assert(wxkmeans_lloyd_relocated_centroids(nullptr, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91 && counts[0] == 92);
    assert(wxkmeans_lloyd_relocated_centroids(x, nullptr, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, nullptr, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, nullptr, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, nullptr, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, nullptr, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, nullptr, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, nullptr, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, nullptr, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, centers, distances, selected, counts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 257, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 7, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    x[5] = std::numeric_limits<float>::quiet_NaN();
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91 && counts[0] == 92);
    x[5] = 0; labels[5] = 3;
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91);
    labels[5] = 0; weights[5] = -1;
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, centers, totals, distances, selected, counts, &metrics, error, sizeof(error)) != 0);
    weights[5] = 1;
    unsigned char unaligned_input[sizeof(x) + 1], unaligned_output[sizeof(centers) + 1]; std::memcpy(unaligned_input + 1, x, sizeof(x));
    assert(wxkmeans_lloyd_relocated_centroids(reinterpret_cast<float*>(unaligned_input + 1), weights, labels, old, 6, 3, 1, reinterpret_cast<float*>(unaligned_output + 1), totals, distances, selected, counts, &metrics, error, sizeof(error)) == 0);
    std::memcpy(centers, unaligned_output + 1, sizeof(centers)); assert(centers[1] == -2);
    assert(wxkmeans_lloyd_relocated_centroids(x, weights, labels, old, 6, 3, 1, old, totals, distances, selected, counts, &metrics, error, sizeof(error)) == 0 && old[1] == -2);
    std::vector<float> large_x(256), large_weights(256, 1), large_old(128, 0), large_centers(128), large_totals(128), large_distances(256); std::vector<int32_t> large_labels(256, 0), large_order(256);
    for (int i = 0; i < 256; ++i) large_x[i] = float((i + 128) % 256);
    assert(wxkmeans_lloyd_relocated_centroids(large_x.data(), large_weights.data(), large_labels.data(), large_old.data(), 256, 128, 1, large_centers.data(), large_totals.data(), large_distances.data(), large_order.data(), counts, &metrics, error, sizeof(error)) == 0 && counts[0] == 127);
}

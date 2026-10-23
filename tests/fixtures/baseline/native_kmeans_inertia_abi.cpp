#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>

struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
extern "C" int wxkmeans_inertia_dense(const float*, const float*, const float*, const int32_t*, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);

int main(int argc, char**) {
    float x[] = {1, 2, 3, 4}, weights[] = {1, 3}, centers[] = {0, 0, 5, 5}, output = 91;
    int32_t labels[] = {0, 0}; KmeansPotentialMetrics metrics = {}; char error[128] = {};
    if (argc > 1) {
        assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91 && std::strstr(error, "actual_team_not_supported") != nullptr);
        return 0;
    }
    for (uint32_t threads : {1u, 2u, 12u}) {
        assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, threads, -1, &output, &metrics, error, sizeof(error)) == 0 && output == 80 && metrics.version == 1 && metrics.blas_calls == 0);
        assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, threads, 1, &output, &metrics, error, sizeof(error)) == 0 && output == 0);
    }
    output = 91;
    assert(wxkmeans_inertia_dense(nullptr, weights, centers, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91);
    assert(wxkmeans_inertia_dense(x, nullptr, centers, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_inertia_dense(x, weights, nullptr, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_inertia_dense(x, weights, centers, nullptr, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, &output, nullptr, error, sizeof(error)) != 0);
    for (uint32_t size : {0u, 257u}) {
        assert(wxkmeans_inertia_dense(x, weights, centers, labels, size, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91);
        assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, size, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0);
    }
    for (uint32_t size : {0u, 1025u}) assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, size, 12, -1, &output, &metrics, error, sizeof(error)) != 0);
    for (uint32_t threads : {0u, 3u, 4u, 7u, 64u}) assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, threads, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91);
    for (int32_t single : {-2, 2}) assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, single, &output, &metrics, error, sizeof(error)) != 0);
    x[3] = std::numeric_limits<float>::quiet_NaN();
    assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91);
    x[3] = 4; labels[1] = 2;
    assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91);
    labels[1] = 0; weights[1] = -1;
    assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, &output, &metrics, error, sizeof(error)) != 0 && output == 91);
    weights[1] = 3;
    unsigned char unaligned[sizeof(x) + 1], result[sizeof(float) + 1]; std::memcpy(unaligned + 1, x, sizeof(x));
    assert(wxkmeans_inertia_dense(reinterpret_cast<float*>(unaligned + 1), weights, centers, labels, 2, 2, 2, 12, -1, reinterpret_cast<float*>(result + 1), &metrics, error, sizeof(error)) == 0);
    std::memcpy(&output, result + 1, sizeof(float)); assert(output == 80);
    assert(wxkmeans_inertia_dense(x, weights, centers, labels, 2, 2, 2, 12, -1, x, &metrics, error, sizeof(error)) == 0 && x[0] == 80);
}

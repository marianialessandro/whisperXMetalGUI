#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
struct KmeansPotentialMetrics { uint32_t version, blas_calls; uint64_t wall_ns; };
extern "C" int wxkmeans_lloyd_assignment_two_chunks(const float*, const float*, const float*, uint32_t, uint32_t, uint32_t, uint32_t, int32_t*, float*, KmeansPotentialMetrics*, char*, uint32_t);
extern "C" int wxkmeans_lloyd_centroids_two_chunks(const float*, const float*, const int32_t*, const float*, uint32_t, uint32_t, uint32_t, uint32_t, float*, float*, float*, int32_t*, uint32_t*, KmeansPotentialMetrics*, char*, uint32_t);
extern "C" int wxkmeans_lloyd_centroids_and_shifts(const float*, const float*, const int32_t*, const float*, uint32_t, uint32_t, uint32_t, uint32_t, float*, float*, float*, int32_t*, uint32_t*, float*, KmeansPotentialMetrics*, char*, uint32_t);
extern "C" int wxkmeans_inertia_dense_two_chunks(const float*, const float*, const float*, const int32_t*, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);
extern "C" int wxkmeans_inertia_dense(const float*, const float*, const float*, const int32_t*, uint32_t, uint32_t, uint32_t, uint32_t, int32_t, float*, KmeansPotentialMetrics*, char*, uint32_t);
int main(int argc, char**) {
    std::vector<float> x(1024), weights(512, 1), scores(1024, 92), distances(512);
    for (int row = 0; row < 512; ++row) { x[row * 2] = row < 256 ? 1 : 3; x[row * 2 + 1] = row < 256 ? 2 : 4; }
    float previous[] = {1, 2, 3, 4}, norms[] = {5, 25}, centers[] = {91, 91, 91, 91}, totals[] = {93, 93}, inertia = 94;
    std::vector<int32_t> labels(512, 91), selected(512); uint32_t counts[] = {95, 96}; KmeansPotentialMetrics metrics = {}; char error[128] = {};
    float fused_shifts[] = {97, 98};
    if (argc > 1) {
        assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 12, labels.data(), scores.data(), &metrics, error, sizeof(error)) != 0 && labels[0] == 91 && scores[0] == 92);
        std::fill(labels.begin(), labels.end(), 0);
        assert(wxkmeans_lloyd_centroids_and_shifts(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, fused_shifts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91 && fused_shifts[0] == 97 && std::strcmp(error, "kmeans_two_chunks_actual_team_not_supported") == 0);
        return 0;
    }
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 12, labels.data(), scores.data(), &metrics, error, sizeof(error)) == 0 && labels[0] == 0 && labels[511] == 1 && metrics.blas_calls == 2);
    assert(wxkmeans_lloyd_centroids_two_chunks(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, &metrics, error, sizeof(error)) == 0 && centers[0] == 1 && centers[3] == 4 && totals[0] == 256 && totals[1] == 256 && counts[0] == 0);
    assert(wxkmeans_lloyd_centroids_and_shifts(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, fused_shifts, &metrics, error, sizeof(error)) == 0 && centers[0] == 1 && centers[3] == 4 && fused_shifts[0] == 0 && fused_shifts[1] == 0);
    centers[0] = 91; totals[0] = 93; distances[0] = 94; selected[0] = 95; counts[0] = 96; fused_shifts[0] = 97;
    weights.back() = -1;
    assert(wxkmeans_lloyd_centroids_and_shifts(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, fused_shifts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91 && totals[0] == 93 && distances[0] == 94 && selected[0] == 95 && counts[0] == 96 && fused_shifts[0] == 97);
    weights.back() = 1;
    assert(wxkmeans_lloyd_centroids_and_shifts(x.data(), weights.data(), labels.data(), previous, 513, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, fused_shifts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_centroids_and_shifts(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, centers, &metrics, error, sizeof(error)) != 0 && centers[0] == 91 && fused_shifts[0] == 97);
    assert(wxkmeans_lloyd_centroids_and_shifts(nullptr, weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, fused_shifts, &metrics, error, sizeof(error)) != 0);
    std::vector<unsigned char> fused_alias(17); std::memcpy(fused_alias.data() + 1, previous, 16);
    assert(wxkmeans_lloyd_centroids_and_shifts(x.data(), weights.data(), labels.data(), reinterpret_cast<float*>(fused_alias.data() + 1), 512, 2, 2, 12, reinterpret_cast<float*>(fused_alias.data() + 1), totals, distances.data(), selected.data(), counts, fused_shifts, &metrics, error, sizeof(error)) == 0 && fused_shifts[0] == 0);
    assert(wxkmeans_inertia_dense_two_chunks(x.data(), weights.data(), previous, labels.data(), 512, 2, 2, 12, -1, &inertia, &metrics, error, sizeof(error)) == 0 && inertia == 0);
    inertia = 94;
    assert(wxkmeans_inertia_dense(x.data(), weights.data(), previous, labels.data(), 512, 2, 2, 12, -1, &inertia, &metrics, error, sizeof(error)) != 0 && inertia == 94);
    assert(wxkmeans_lloyd_assignment_two_chunks(nullptr, previous, norms, 512, 2, 2, 12, labels.data(), scores.data(), &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 513, 2, 2, 12, labels.data(), scores.data(), &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 3, labels.data(), scores.data(), &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 12, labels.data(), reinterpret_cast<float*>(labels.data()), &metrics, error, sizeof(error)) != 0);
    labels.assign(512, 91); scores.assign(1024, 92); x.back() = std::numeric_limits<float>::quiet_NaN();
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 12, labels.data(), scores.data(), &metrics, error, sizeof(error)) != 0 && labels[0] == 91 && labels.back() == 91 && scores[0] == 92);
    x.back() = 4;
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 12, labels.data(), scores.data(), &metrics, error, sizeof(error)) == 0);
    weights.back() = -1; centers[0] = 91;
    assert(wxkmeans_lloyd_centroids_two_chunks(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, &metrics, error, sizeof(error)) != 0 && centers[0] == 91);
    weights.back() = 1;
    assert(wxkmeans_lloyd_centroids_two_chunks(x.data(), weights.data(), labels.data(), previous, 513, 2, 2, 12, centers, totals, distances.data(), selected.data(), counts, &metrics, error, sizeof(error)) != 0);
    assert(wxkmeans_lloyd_centroids_two_chunks(x.data(), weights.data(), labels.data(), previous, 512, 2, 2, 12, centers, centers, distances.data(), selected.data(), counts, &metrics, error, sizeof(error)) != 0);
    std::vector<unsigned char> unaligned(x.size() * 4 + 1), unaligned_labels(labels.size() * 4 + 1); std::memcpy(unaligned.data() + 1, x.data(), x.size() * 4);
    assert(wxkmeans_lloyd_assignment_two_chunks(reinterpret_cast<float*>(unaligned.data() + 1), previous, norms, 512, 2, 2, 12, reinterpret_cast<int32_t*>(unaligned_labels.data() + 1), scores.data(), &metrics, error, sizeof(error)) == 0);
    assert(wxkmeans_lloyd_centroids_two_chunks(reinterpret_cast<float*>(unaligned.data() + 1), weights.data(), labels.data(), previous, 512, 2, 2, 12, previous, totals, distances.data(), selected.data(), counts, &metrics, error, sizeof(error)) == 0 && previous[0] == 1 && previous[3] == 4);
    assert(wxkmeans_lloyd_assignment_two_chunks(x.data(), previous, norms, 512, 2, 2, 12, labels.data(), x.data(), &metrics, error, sizeof(error)) == 0 && x[0] == -5 && x.back() == -25);
}

#pragma once
#include "native_kmeans_pairwise.hpp"
#include "native_kmeans_selection.hpp"
#include <cmath>
#include <vector>

// Finalize already merged Lloyd sums using the pinned sklearn/NumPy relocation and averaging policy.
inline uint32_t wxkmeans_finalize_centroids(const std::vector<float>& x, const std::vector<float>& w, const std::vector<int32_t>& assigned, const std::vector<float>& old, uint32_t rows, uint32_t clusters, uint32_t features, std::vector<float>& sums, std::vector<float>& totals, std::vector<float>& row_distances, std::vector<int32_t>& order, std::vector<int32_t>& empty) {
    std::vector<float> squares(features);
        for (uint32_t cluster = 0; cluster < clusters; ++cluster) if (totals[cluster] == 0) empty.push_back(int32_t(cluster));
        uint32_t moved = 0;
        if (!empty.empty()) {
            float maximum = 0.0f;
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t feature = 0; feature < features; ++feature) {
                    volatile float difference = x[uint64_t(row) * features + feature] - old[uint64_t(assigned[row]) * features + feature];
                    volatile float square = difference * difference; squares[feature] = square;
                }
                row_distances[row] = 0.0f + kmeans_pairwise_shift_sum(squares.data(), features);
                maximum = std::max(maximum, row_distances[row]);
            }
            std::vector<int32_t> partition(rows);
            for (uint32_t row = 0; row < rows; ++row) partition[row] = int32_t(row);
            wxselection::select(row_distances.data(), partition.data(), int(rows), int(rows - empty.size()));
            for (uint32_t i = 0; i < empty.size(); ++i) order[i] = partition[rows - 1 - i];
            if (maximum != 0) {
                moved = uint32_t(empty.size());
                for (uint32_t i = 0; i < empty.size(); ++i) {
                    uint32_t row = uint32_t(order[i]), donor = uint32_t(assigned[row]), target = uint32_t(empty[i]);
                    for (uint32_t feature = 0; feature < features; ++feature) {
                        float value = x[uint64_t(row) * features + feature];
                        sums[uint64_t(donor) * features + feature] = std::fma(-value, w[row], sums[uint64_t(donor) * features + feature]);
                        volatile float product = value * w[row]; sums[uint64_t(target) * features + feature] = product;
                    }
                    totals[target] = w[row]; totals[donor] -= w[row];
                }
            }
        }
        uint32_t largest = 0;
        for (uint32_t cluster = 1; cluster < clusters; ++cluster) if (totals[cluster] > totals[largest]) largest = cluster;
        for (uint32_t cluster = 0; cluster < clusters; ++cluster) {
            if (totals[cluster] > 0) {
                float alpha = static_cast<float>(1.0 / static_cast<double>(totals[cluster]));
                for (uint32_t feature = 0; feature < features; ++feature) sums[uint64_t(cluster) * features + feature] *= alpha;
            } else {
                for (uint32_t feature = 0; feature < features; ++feature) sums[uint64_t(cluster) * features + feature] = sums[uint64_t(largest) * features + feature];
            }
        }
    return moved;
}

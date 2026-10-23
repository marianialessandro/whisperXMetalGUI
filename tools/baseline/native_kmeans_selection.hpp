#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>

// Specialize NumPy 2.2.6 introselect for one kth and finite float32 keys; see native_numpy_norm_notice.txt.
namespace wxselection {
inline void select(const float* keys, int32_t* indices, int count, int kth);
inline int median_five(const float* keys, int32_t* indices) {
    auto less = [&](int a, int b) { return keys[indices[a]] < keys[indices[b]]; };
    if (less(1, 0)) std::swap(indices[1], indices[0]);
    if (less(4, 3)) std::swap(indices[4], indices[3]);
    if (less(3, 0)) std::swap(indices[3], indices[0]);
    if (less(4, 1)) std::swap(indices[4], indices[1]);
    if (less(2, 1)) std::swap(indices[2], indices[1]);
    return less(3, 2) ? (less(3, 1) ? 1 : 3) : 2;
}
inline int median_groups(const float* keys, int32_t* indices, int count) {
    int groups = count / 5;
    for (int group = 0; group < groups; ++group) {
        int middle = median_five(keys, indices + group * 5);
        std::swap(indices[group * 5 + middle], indices[group]);
    }
    if (groups > 2) select(keys, indices, groups, groups / 2);
    return groups / 2;
}
inline void select(const float* keys, int32_t* indices, int count, int kth) {
    int low = 0, high = count - 1;
    if (kth < 3) {
        for (int position = 0; position <= kth; ++position) {
            int minimum = position;
            for (int item = position + 1; item < count; ++item) if (keys[indices[item]] < keys[indices[minimum]]) minimum = item;
            std::swap(indices[position], indices[minimum]);
        }
        return;
    }
    if (kth == count - 1) {
        int maximum = low;
        for (int item = low + 1; item < count; ++item) if (!(keys[indices[item]] < keys[indices[maximum]])) maximum = item;
        std::swap(indices[kth], indices[maximum]);
        return;
    }
    int depth = 2 * (int(std::bit_width(uint32_t(count))) - 1);
    while (low + 1 < high) {
        int left = low + 1, right = high;
        if (depth > 0 || right - left < 5) {
            int middle = low + (high - low) / 2;
            if (keys[indices[high]] < keys[indices[middle]]) std::swap(indices[high], indices[middle]);
            if (keys[indices[high]] < keys[indices[low]]) std::swap(indices[high], indices[low]);
            if (keys[indices[low]] < keys[indices[middle]]) std::swap(indices[low], indices[middle]);
            std::swap(indices[middle], indices[low + 1]);
        } else {
            int middle = left + median_groups(keys, indices + left, right - left);
            std::swap(indices[middle], indices[low]);
            --left; ++right;
        }
        --depth;
        float pivot = keys[indices[low]];
        for (;;) {
            do { ++left; } while (keys[indices[left]] < pivot);
            do { --right; } while (pivot < keys[indices[right]]);
            if (right < left) break;
            std::swap(indices[left], indices[right]);
        }
        std::swap(indices[low], indices[right]);
        if (right >= kth) high = right - 1;
        if (right <= kth) low = left;
    }
    if (high == low + 1 && keys[indices[high]] < keys[indices[low]]) std::swap(indices[high], indices[low]);
}
}

#pragma once
#include <cstdint>

// Preserve the NumPy 2.2.6 pairwise float32 reduction; see native_numpy_norm_notice.txt.
inline float kmeans_pairwise_shift_sum(const float* values, uint32_t count) {
    if (count < 8) { float total = -0.0f; for (uint32_t index = 0; index < count; ++index) total += values[index]; return total; }
    if (count > 128) { uint32_t split = (count / 2) & ~7u; return kmeans_pairwise_shift_sum(values, split) + kmeans_pairwise_shift_sum(values + split, count - split); }
    float lanes[8]; for (uint32_t lane = 0; lane < 8; ++lane) lanes[lane] = values[lane];
    uint32_t index = 8;
    for (; index < count - count % 8; index += 8) for (uint32_t lane = 0; lane < 8; ++lane) lanes[lane] += values[index + lane];
    float total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; index < count; ++index) total += values[index];
    return total;
}


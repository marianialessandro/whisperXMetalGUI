#include <metal_stdlib>
using namespace metal;
struct EmbeddingDimensions { uint rows, columns; };

float embedding_square(device const float* values, uint index) { return values[index] * values[index]; }

// Match NumPy's contiguous float32 pairwise summation order before square root and division.
float embedding_leaf(device const float* values, uint start, uint count) {
    float result = -0.0f;
    if (count < 8) {
        for (uint i = 0; i < count; ++i) result += embedding_square(values, start + i);
        return result;
    }
    float sums[8];
    for (uint lane = 0; lane < 8; ++lane) sums[lane] = embedding_square(values, start + lane);
    uint i = 8;
    for (; i < count - count % 8; i += 8) for (uint lane = 0; lane < 8; ++lane) sums[lane] += embedding_square(values, start + i + lane);
    result = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    for (; i < count; ++i) result += embedding_square(values, start + i);
    return result;
}

float embedding_sum(device const float* values, uint count) {
    uint starts[16], lengths[16], states[16], depth = 0;
    float left[16], result = 0;
    starts[0] = 0; lengths[0] = count; states[0] = 0;
    while (true) {
        uint length = lengths[depth], start = starts[depth];
        if (length <= 128) result = embedding_leaf(values, start, length);
        else {
            uint split = (length / 2) & ~7u;
            if (states[depth] == 0) {
                states[depth] = 1; ++depth; starts[depth] = start; lengths[depth] = split; states[depth] = 0;
                continue;
            }
            if (states[depth] == 1) {
                left[depth] = result; states[depth] = 2; ++depth; starts[depth] = start + split; lengths[depth] = length - split; states[depth] = 0;
                continue;
            }
            result = left[depth] + result;
        }
        if (depth == 0) return result;
        --depth;
    }
}

kernel void embedding_normalize(device const float* input [[buffer(0)]], constant EmbeddingDimensions& d [[buffer(1)]], device float* output [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    uint start = row * d.columns;
    float norm = precise::sqrt(embedding_sum(input + start, d.columns));
    for (uint column = 0; column < d.columns; ++column) output[start + column] = input[start + column] / norm;
}

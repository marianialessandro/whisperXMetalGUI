#include <metal_stdlib>
using namespace metal;

struct Dimensions { uint frames; uint vocabulary; uint count; int blank; };

kernel void ctc_row_max(device const float* emission [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device float* values [[buffer(2)]], device uint* indices [[buffer(3)]], uint frame [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float scores[256];
    threadgroup uint columns[256];
    float best = -INFINITY;
    uint column = UINT_MAX;
    for (uint j = lane; j < dimensions.vocabulary; j += 256) {
        float value = emission[frame * dimensions.vocabulary + j];
        if (value > best || (value == best && j < column)) { best = value; column = j; }
    }
    scores[lane] = best; columns[lane] = column;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            float other = scores[lane + stride];
            uint index = columns[lane + stride];
            if (other > scores[lane] || (other == scores[lane] && index < columns[lane])) { scores[lane] = other; columns[lane] = index; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) { values[frame] = scores[0]; indices[frame] = columns[0]; }
}

kernel void ctc_validate(device const float* emission [[buffer(0)]], device const int* tokens [[buffer(1)]], constant Dimensions& dimensions [[buffer(2)]], device atomic_uint* invalid [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index == 0 && (dimensions.blank < 0 || dimensions.blank >= int(dimensions.vocabulary))) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
    if (index < dimensions.frames * dimensions.vocabulary && (!(emission[index] <= 0) || isnan(emission[index]))) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
    if (index < dimensions.count && (tokens[index] < 0 || tokens[index] >= int(dimensions.vocabulary))) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
}

kernel void ctc_prefix(device const float* emission [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device float* prefix [[buffer(2)]]) {
    float high = 0, low = 0;
    prefix[0] = 0;
    for (uint t = 0; t < dimensions.frames; ++t) {
        float value = emission[t * dimensions.vocabulary + dimensions.blank];
        if (isinf(high) || isinf(value)) { high += value; low = 0; }
        else {
            float sum = high + value;
            if (isinf(sum)) { high = sum; low = 0; }
            else {
                float virtual_value = sum - high;
                float error = (high - (sum - virtual_value)) + (value - virtual_value);
                float correction = low + error;
                float next = sum + correction;
                low = correction - (next - sum); high = next;
            }
        }
        prefix[t + 1] = high;
    }
}

kernel void ctc_trellis(device const float* emission [[buffer(0)]], device const int* tokens [[buffer(1)]], constant Dimensions& dimensions [[buffer(2)]], device const float* prefix [[buffer(3)]], device float* trellis [[buffer(4)]], uint lane [[thread_position_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    uint stride = dimensions.count + 1;
    for (uint t = lane; t <= dimensions.frames; t += width) trellis[t * stride] = t >= dimensions.frames + 1 - dimensions.count ? INFINITY : prefix[t];
    for (uint j = lane + 1; j <= dimensions.count; j += width) trellis[j] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint t = 0; t < dimensions.frames; ++t) {
        for (uint j = lane + 1; j <= dimensions.count; j += width) {
            float stayed = trellis[t * stride + j] + emission[t * dimensions.vocabulary + dimensions.blank];
            float changed = trellis[t * stride + j - 1] + emission[t * dimensions.vocabulary + tokens[j - 1]];
            trellis[(t + 1) * stride + j] = isnan(stayed) || isnan(changed) ? NAN : max(stayed, changed);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void ctc_backtrack(device const float* emission [[buffer(0)]], device const int* tokens [[buffer(1)]], constant Dimensions& dimensions [[buffer(2)]], device const float* trellis [[buffer(3)]], device float* path [[buffer(4)]]) {
    for (uint i = 0; i < dimensions.frames; ++i) { path[1 + i * 3] = -1; path[2 + i * 3] = -1; path[3 + i * 3] = 0; }
    uint stride = dimensions.count + 1;
    uint start = 0;
    float best = trellis[dimensions.count];
    for (uint t = 1; t <= dimensions.frames; ++t) {
        float score = trellis[t * stride + dimensions.count];
        if (!isnan(best) && (isnan(score) || score > best)) { best = score; start = t; }
    }
    uint j = dimensions.count, length = 0;
    for (uint t = start; t > 0; --t) {
        float stayed = trellis[(t - 1) * stride + j] + emission[(t - 1) * dimensions.vocabulary + dimensions.blank];
        float changed = trellis[(t - 1) * stride + j - 1] + emission[(t - 1) * dimensions.vocabulary + tokens[j - 1]];
        bool advance = changed > stayed;
        path[1 + length * 3] = j - 1; path[2 + length * 3] = t - 1;
        path[3 + length * 3] = exp(emission[(t - 1) * dimensions.vocabulary + (advance ? tokens[j - 1] : dimensions.blank)]);
        ++length;
        if (advance) { --j; if (j == 0) break; }
    }
    path[0] = j == 0 ? float(length) : -1;
}

kernel void ctc_trellis_compact(device const float* emission [[buffer(0)]], device const int* tokens [[buffer(1)]], constant Dimensions& dimensions [[buffer(2)]], device const float* prefix [[buffer(3)]], device float* rows [[buffer(4)]], device uchar* decisions [[buffer(5)]], device uint* endpoint [[buffer(6)]], uint lane [[thread_position_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    uint stride = dimensions.count + 1;
    for (uint j = lane; j <= dimensions.count; j += width) rows[j] = j == 0 ? prefix[0] : -INFINITY;
    float best = -INFINITY;
    uint start = 0;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint t = 0; t < dimensions.frames; ++t) {
        uint previous = (t % 2) * stride, next = ((t + 1) % 2) * stride;
        if (lane == 0) rows[next] = t + 1 >= dimensions.frames + 1 - dimensions.count ? INFINITY : prefix[t + 1];
        for (uint j = lane + 1; j <= dimensions.count; j += width) {
            float stayed = rows[previous + j] + emission[t * dimensions.vocabulary + dimensions.blank];
            float changed = rows[previous + j - 1] + emission[t * dimensions.vocabulary + tokens[j - 1]];
            rows[next + j] = isnan(stayed) || isnan(changed) ? NAN : max(stayed, changed);
            decisions[t * dimensions.count + j - 1] = changed > stayed;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0) {
            float score = rows[next + dimensions.count];
            if (!isnan(best) && (isnan(score) || score > best)) { best = score; start = t + 1; }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (lane == 0) endpoint[0] = start;
}

kernel void ctc_backtrack_compact(device const float* emission [[buffer(0)]], device const int* tokens [[buffer(1)]], constant Dimensions& dimensions [[buffer(2)]], device const float* unused_rows [[buffer(3)]], device float* path [[buffer(4)]], device const uchar* decisions [[buffer(5)]], device const uint* endpoint [[buffer(6)]]) {
    for (uint i = 0; i < dimensions.frames; ++i) { path[1 + i * 3] = -1; path[2 + i * 3] = -1; path[3 + i * 3] = 0; }
    uint j = dimensions.count, length = 0;
    for (uint t = endpoint[0]; t > 0; --t) {
        bool advance = decisions[(t - 1) * dimensions.count + j - 1] != 0;
        path[1 + length * 3] = j - 1; path[2 + length * 3] = t - 1;
        path[3 + length * 3] = exp(emission[(t - 1) * dimensions.vocabulary + (advance ? tokens[j - 1] : dimensions.blank)]);
        ++length;
        if (advance) { --j; if (j == 0) break; }
    }
    path[0] = j == 0 ? float(length) : -1;
}

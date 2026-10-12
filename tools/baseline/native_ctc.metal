#include <metal_stdlib>
using namespace metal;

struct Dimensions { uint frames; uint vocabulary; uint count; int blank; };

ulong shift_jam(ulong value, uint shift) {
    if (shift == 0) return value;
    if (shift >= 64) return value != 0;
    return (value >> shift) | ((value & ((1ul << shift) - 1)) != 0);
}

ulong positive_float_to_double(uint bits) {
    uint exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 0) {
        if (fraction == 0) return 0;
        uint highest = 31 - clz(fraction);
        return (ulong(highest + 874) << 52) | ((ulong(fraction) << (52 - highest)) & ((1ul << 52) - 1));
    }
    return (ulong(exponent + 896) << 52) | (ulong(fraction) << 29);
}

// Positive finite binary64 addition with guard/round/sticky bits and nearest-even rounding.
ulong add_positive_double(ulong a, ulong b) {
    if (a == 0) return b;
    if (b == 0) return a;
    if (a < b) { ulong swap = a; a = b; b = swap; }
    uint exponent = uint(a >> 52), difference = exponent - uint(b >> 52);
    ulong mask = (1ul << 52) - 1;
    ulong sum = (((a & mask) | (1ul << 52)) << 3) + shift_jam(((b & mask) | (1ul << 52)) << 3, difference);
    if (sum >= (1ul << 56)) { sum = shift_jam(sum, 1); ++exponent; }
    ulong rounded = sum >> 3, remainder = sum & 7;
    if (remainder > 4 || (remainder == 4 && (rounded & 1))) ++rounded;
    if (rounded == (1ul << 53)) { rounded >>= 1; ++exponent; }
    return (ulong(exponent) << 52) | (rounded & mask);
}

// Signed finite binary64 addition for bounded probability compensation; cancellation stays in the normal binary64 range.
ulong add_signed_double(ulong a, ulong b) {
    ulong sign = 1ul << 63, magnitude = sign - 1, mask = (1ul << 52) - 1;
    if ((a & magnitude) == 0) return b;
    if ((b & magnitude) == 0) return a;
    if ((a & sign) == (b & sign)) return add_positive_double(a & magnitude, b & magnitude) | (a & sign);
    if ((a & magnitude) < (b & magnitude)) { ulong swap = a; a = b; b = swap; }
    uint exponent = uint((a & magnitude) >> 52), difference = exponent - uint((b & magnitude) >> 52);
    ulong delta = (((a & mask) | (1ul << 52)) << 3) - shift_jam(((b & mask) | (1ul << 52)) << 3, difference);
    if (delta == 0) return 0;
    while (delta < (1ul << 55)) { delta <<= 1; --exponent; }
    ulong rounded = delta >> 3, remainder = delta & 7;
    if (remainder > 4 || (remainder == 4 && (rounded & 1))) ++rounded;
    if (rounded == (1ul << 53)) { rounded >>= 1; ++exponent; }
    return (a & sign) | (ulong(exponent) << 52) | (rounded & mask);
}

ulong divide_positive_double(ulong bits, uint count) {
    if (bits == 0) return 0;
    uint shifts = 31 - clz(count), exponent = uint(bits >> 52) - shifts;
    ulong mask = (1ul << 52) - 1, significand = (bits & mask) | (1ul << 52);
    ulong quotient = significand / count, remainder = significand % count;
    for (uint i = 0; i < shifts; ++i) {
        remainder <<= 1; quotient <<= 1;
        if (remainder >= count) { ++quotient; remainder -= count; }
    }
    if (quotient < (1ul << 52)) {
        --exponent; remainder <<= 1; quotient <<= 1;
        if (remainder >= count) { ++quotient; remainder -= count; }
    }
    if (remainder * 2 > count || (remainder * 2 == count && (quotient & 1))) ++quotient;
    if (quotient == (1ul << 53)) { quotient >>= 1; ++exponent; }
    return (ulong(exponent) << 52) | (quotient & mask);
}

kernel void ctc_merge_repeats(device const uint* packed [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device int2* bounds [[buffer(2)]], device ulong* means [[buffer(3)]], uint token [[thread_position_in_grid]]) {
    if (token >= dimensions.count) return;
    int length = int(as_type<float>(packed[0]));
    int start = -1, end = -1;
    ulong sum = 0, correction = 0;
    uint count = 0;
    if (length >= 0 && length <= int(dimensions.frames)) {
        for (int i = length - 1; i >= 0; --i) {
            if (uint(as_type<float>(packed[1 + i * 3])) == token) {
                int frame = int(as_type<float>(packed[2 + i * 3]));
                if (start < 0) start = frame;
                end = frame + 1;
                ulong value = positive_float_to_double(packed[3 + i * 3]);
                ulong next = add_positive_double(sum, value);
                ulong error = sum >= value ? add_signed_double(add_signed_double(sum, next ^ (1ul << 63)), value) : add_signed_double(add_signed_double(value, next ^ (1ul << 63)), sum);
                correction = add_signed_double(correction, error);
                sum = next;
                ++count;
            }
        }
    }
    bounds[token] = int2(start, end);
    means[token] = count ? divide_positive_double(add_signed_double(sum, correction), count) : 0x7ff8000000000000ul;
}

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

kernel void ctc_validate_wildcard(device const float* emission [[buffer(0)]], device const int* tokens [[buffer(1)]], constant Dimensions& dimensions [[buffer(2)]], device atomic_uint* invalid [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index == 0 && (dimensions.blank < 0 || dimensions.blank >= int(dimensions.vocabulary))) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
    if (index < dimensions.frames * dimensions.vocabulary && (!(emission[index] <= 0) || isnan(emission[index]))) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
    if (index < dimensions.count && (tokens[index] < 0 || tokens[index] > int(dimensions.vocabulary))) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed);
}

kernel void ctc_expand_wildcard(device const float* emission [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device float* expanded [[buffer(2)]], uint frame [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float scores[256];
    threadgroup uint columns[256];
    float best = -INFINITY;
    uint column = UINT_MAX;
    for (uint j = lane; j < dimensions.vocabulary; j += 256) {
        float value = emission[frame * dimensions.vocabulary + j];
        expanded[frame * (dimensions.vocabulary + 1) + j] = value;
        if (j != uint(dimensions.blank) && (value > best || (value == best && j < column))) { best = value; column = j; }
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
    if (lane == 0) expanded[frame * (dimensions.vocabulary + 1) + dimensions.vocabulary] = scores[0];
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

// Binary64 timestamp comparisons use integer keys; no floating-point narrowing or arithmetic is performed.
kernel void ctc_timestamp_bounds(device const ulong* values [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device ulong* output [[buffer(2)]], uint lane [[thread_position_in_threadgroup]]) {
    threadgroup ulong minima[256], maxima[256];
    threadgroup uint counts[256], invalid[256];
    ulong low = ~0ul, high = 0ul;
    uint count = 0, bad = 0;
    for (uint i = lane; i < dimensions.count; i += 256) {
        ulong bits = values[i], magnitude = bits & 0x7ffffffffffffffful;
        if (magnitude > 0x7ff0000000000000ul) continue;
        if (bits == 0x8000000000000000ul) { bad = 1; continue; }
        ulong key = (bits >> 63) ? ~bits : bits ^ 0x8000000000000000ul;
        low = min(low, key); high = max(high, key); ++count;
    }
    minima[lane] = low; maxima[lane] = high; counts[lane] = count; invalid[lane] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            minima[lane] = min(minima[lane], minima[lane + stride]); maxima[lane] = max(maxima[lane], maxima[lane + stride]);
            counts[lane] += counts[lane + stride]; invalid[lane] |= invalid[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        ulong low_key = minima[0], high_key = maxima[0];
        output[0] = counts[0] == 0 ? 0x7ff8000000000000ul : ((low_key >> 63) ? low_key ^ 0x8000000000000000ul : ~low_key);
        output[1] = counts[0] == 0 ? 0x7ff8000000000000ul : ((high_key >> 63) ? high_key ^ 0x8000000000000000ul : ~high_key);
        output[2] = invalid[0];
    }
}

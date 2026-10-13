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

// One group per CSR query shares a single immutable binary64 timestamp input and one command-buffer completion.
kernel void ctc_timestamp_bounds_batch(device const ulong* values [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device const uint* indices [[buffer(2)]], device const uint* offsets [[buffer(3)]], device ulong* output [[buffer(4)]], device atomic_uint* invalid [[buffer(5)]], uint lane [[thread_position_in_threadgroup]], uint query [[threadgroup_position_in_grid]]) {
    threadgroup ulong minima[256], maxima[256];
    threadgroup uint counts[256];
    uint begin = offsets[query], end = offsets[query + 1];
    bool valid = begin <= end && end <= dimensions.count && (query != 0 || begin == 0) && (query + 1 != dimensions.frames || end == dimensions.count);
    if (!valid) {
        if (lane == 0) { atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed); output[query * 2] = output[query * 2 + 1] = 0x7ff8000000000000ul; }
        return;
    }
    ulong low = ~0ul, high = 0ul;
    uint count = 0;
    for (uint i = begin + lane; i < end; i += 256) {
        uint index = indices[i];
        if (index >= dimensions.vocabulary) { atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed); continue; }
        ulong bits = values[index], magnitude = bits & 0x7ffffffffffffffful;
        if (magnitude > 0x7ff0000000000000ul) continue;
        if (bits == 0x8000000000000000ul) { atomic_fetch_or_explicit(invalid, 2u, memory_order_relaxed); continue; }
        ulong key = (bits >> 63) ? ~bits : bits ^ 0x8000000000000000ul;
        low = min(low, key); high = max(high, key); ++count;
    }
    minima[lane] = low; maxima[lane] = high; counts[lane] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) { minima[lane] = min(minima[lane], minima[lane + stride]); maxima[lane] = max(maxima[lane], maxima[lane + stride]); counts[lane] += counts[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        ulong low_key = minima[0], high_key = maxima[0];
        output[query * 2] = counts[0] == 0 ? 0x7ff8000000000000ul : ((low_key >> 63) ? low_key ^ 0x8000000000000000ul : ~low_key);
        output[query * 2 + 1] = counts[0] == 0 ? 0x7ff8000000000000ul : ((high_key >> 63) ? high_key ^ 0x8000000000000000ul : ~high_key);
    }
}

ulong word_score(device const ulong* values, device const uint* indices, uint index, uint stride) {
    ulong bits = values[indices[index] * stride];
    return (bits & 0x7ffffffffffffffful) > 0x7ff0000000000000ul ? 0ul : bits;
}

// Reproduce the pinned NumPy pairwise leaf order, retaining NaN positions as positive zero.
ulong word_sum_leaf(device const ulong* values, device const uint* indices, uint begin, uint count, uint stride) {
    ulong result = 0;
    if (count < 8) {
        for (uint i = 0; i < count; ++i) result = add_positive_double(result, word_score(values, indices, begin + i, stride));
        return result;
    }
    ulong sums[8];
    for (uint lane = 0; lane < 8; ++lane) sums[lane] = word_score(values, indices, begin + lane, stride);
    uint i = 8;
    for (; i < count - count % 8; i += 8) {
        for (uint lane = 0; lane < 8; ++lane) sums[lane] = add_positive_double(sums[lane], word_score(values, indices, begin + i + lane, stride));
    }
    result = add_positive_double(add_positive_double(add_positive_double(sums[0], sums[1]), add_positive_double(sums[2], sums[3])), add_positive_double(add_positive_double(sums[4], sums[5]), add_positive_double(sums[6], sums[7])));
    for (; i < count; ++i) result = add_positive_double(result, word_score(values, indices, begin + i, stride));
    return result;
}

// Bounded explicit stack replaces recursion; splits preserve multiples of the eight-accumulator unroll factor.
ulong word_pairwise_sum(device const ulong* values, device const uint* indices, uint begin, uint count, uint stride) {
    uint starts[16], lengths[16], states[16], depth = 0;
    ulong left[16], result = 0;
    starts[0] = begin; lengths[0] = count; states[0] = 0;
    while (true) {
        uint length = lengths[depth], start = starts[depth];
        if (length <= 128) result = word_sum_leaf(values, indices, start, length, stride);
        else {
            uint split = (length / 2) & ~7u;
            if (states[depth] == 0) {
                states[depth] = 1; ++depth;
                starts[depth] = start; lengths[depth] = split; states[depth] = 0;
                continue;
            }
            if (states[depth] == 1) {
                left[depth] = result; states[depth] = 2; ++depth;
                starts[depth] = start + split; lengths[depth] = length - split; states[depth] = 0;
                continue;
            }
            result = add_positive_double(left[depth], result);
        }
        if (depth == 0) return result;
        --depth;
    }
}

kernel void ctc_word_means_batch(device const ulong* values [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device const uint* indices [[buffer(2)]], device const uint* offsets [[buffer(3)]], device ulong* output [[buffer(4)]], device atomic_uint* invalid [[buffer(5)]], uint query [[thread_position_in_grid]]) {
    if (query >= dimensions.frames) return;
    uint begin = offsets[query], end = offsets[query + 1], count = 0;
    if (begin > end || end > dimensions.count || end - begin > 32768 || (query == 0 && begin != 0) || (query + 1 == dimensions.frames && end != dimensions.count)) {
        atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed); output[query] = 0x7ff8000000000000ul; return;
    }
    uint bad = 0;
    for (uint i = begin; i < end; ++i) {
        uint index = indices[i];
        if (index >= dimensions.vocabulary) { bad |= 1u; continue; }
        ulong bits = values[index], magnitude = bits & 0x7ffffffffffffffful;
        if (magnitude > 0x7ff0000000000000ul) continue;
        if (bits != 0 && (bits < 0x3f50624dd2f1a9fcul || bits > 0x3ff0000000000000ul)) { bad |= 2u; continue; }
        ++count;
    }
    if (bad) { atomic_fetch_or_explicit(invalid, bad, memory_order_relaxed); output[query] = 0x7ff8000000000000ul; return; }
    ulong sum = 0;
    for (uint start = begin; start < end; start += 8192) sum = add_positive_double(sum, word_pairwise_sum(values, indices, start, min(8192u, end - start), 1));
    output[query] = count ? divide_positive_double(sum, count) : 0x7ff8000000000000ul;
}

// Shared row queries select start minimum, end maximum and/or word mean from interleaved character triples.
kernel void ctc_alignment_stats_batch(device const ulong* values [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device const uint* indices [[buffer(2)]], device const uint* offsets [[buffer(3)]], device const uint* modes [[buffer(4)]], device ulong* output [[buffer(5)]], device atomic_uint* invalid [[buffer(6)]], uint lane [[thread_position_in_threadgroup]], uint query [[threadgroup_position_in_grid]]) {
    threadgroup ulong minima[256], maxima[256];
    threadgroup uint counts[256], errors[256];
    uint begin = offsets[query], end = offsets[query + 1], mode = modes[query];
    if (mode == 0 || mode > 7 || begin > end || end > dimensions.count || end - begin > 32768 || (query == 0 && begin != 0) || (query + 1 == dimensions.frames && end != dimensions.count)) {
        if (lane == 0) { atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed); for (uint j = 0; j < 3; ++j) output[query * 3 + j] = 0x7ff8000000000000ul; }
        return;
    }
    ulong low = ~0ul, high = 0ul;
    uint count = 0, bad = 0;
    for (uint i = begin + lane; i < end; i += 256) {
        uint row = indices[i];
        if (row >= dimensions.vocabulary) { bad |= 1u; continue; }
        if (mode & 1) {
            ulong bits = values[row * 3], magnitude = bits & 0x7ffffffffffffffful;
            if (magnitude <= 0x7ff0000000000000ul) {
                if (bits == 0x8000000000000000ul) bad |= 2u;
                else low = min(low, (bits >> 63) ? ~bits : bits ^ 0x8000000000000000ul);
            }
        }
        if (mode & 2) {
            ulong bits = values[row * 3 + 1], magnitude = bits & 0x7ffffffffffffffful;
            if (magnitude <= 0x7ff0000000000000ul) {
                if (bits == 0x8000000000000000ul) bad |= 2u;
                else high = max(high, (bits >> 63) ? ~bits : bits ^ 0x8000000000000000ul);
            }
        }
        if (mode & 4) {
            ulong bits = values[row * 3 + 2], magnitude = bits & 0x7ffffffffffffffful;
            if (magnitude > 0x7ff0000000000000ul) continue;
            if (bits != 0 && (bits < 0x3f50624dd2f1a9fcul || bits > 0x3ff0000000000000ul)) bad |= 4u;
            else ++count;
        }
    }
    minima[lane] = low; maxima[lane] = high; counts[lane] = count; errors[lane] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) { minima[lane] = min(minima[lane], minima[lane + stride]); maxima[lane] = max(maxima[lane], maxima[lane + stride]); counts[lane] += counts[lane + stride]; errors[lane] |= errors[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint j = 0; j < 3; ++j) output[query * 3 + j] = 0x7ff8000000000000ul;
        if (errors[0]) { atomic_fetch_or_explicit(invalid, errors[0], memory_order_relaxed); return; }
        ulong low_key = minima[0], high_key = maxima[0];
        if ((mode & 1) && low_key != ~0ul) output[query * 3] = (low_key >> 63) ? low_key ^ 0x8000000000000000ul : ~low_key;
        if ((mode & 2) && high_key != 0ul) output[query * 3 + 1] = (high_key >> 63) ? high_key ^ 0x8000000000000000ul : ~high_key;
        if ((mode & 4) && counts[0]) {
            ulong sum = 0;
            for (uint start = begin; start < end; start += 8192) sum = add_positive_double(sum, word_pairwise_sum(values + 2, indices, start, min(8192u, end - start), 3));
            output[query * 3 + 2] = divide_positive_double(sum, counts[0]);
        }
    }
}

// Fixed decimal round-to-three: scale the exact binary64 significand in integers before applying ties-to-even.
ulong python_round3(ulong bits) {
    ulong sign = bits & 0x8000000000000000ul, magnitude = bits & 0x7ffffffffffffffful;
    uint exponent = uint(magnitude >> 52);
    if (exponent >= 1066) return bits;
    if (exponent == 0) return sign;
    uint shift = 1075 - exponent;
    if (shift >= 64) return sign;
    ulong scaled = ((magnitude & 0xffffffffffffful) | (1ul << 52)) * 1000ul;
    ulong rounded = scaled >> shift, remainder = scaled & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    if (rounded == 0) return sign;
    uint highest = 63 - uint(clz(rounded));
    ulong integer_bits = (ulong(highest + 1023) << 52) | ((rounded << (52 - highest)) & 0xffffffffffffful);
    return sign | divide_positive_double(integer_bits, 1000);
}

kernel void ctc_round3(device const ulong* values [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < dimensions.count) output[index] = python_round3(values[index]);
}

// Multiplication of a normal positive binary64 value by a bounded frame integer, rounded before addition.
ulong multiply_frame_double(ulong bits, uint frame) {
    if (frame == 0) return 0;
    ulong mask = (1ul << 52) - 1, significand = (bits & mask) | (1ul << 52);
    ulong low = significand * ulong(frame), high = mulhi(significand, ulong(frame));
    uint highest = high ? 127 - uint(clz(high)) : 63 - uint(clz(low)), shift = highest - 52;
    ulong rounded = shift ? (low >> shift) | (high << (64 - shift)) : low;
    if (shift) {
        ulong remainder = low & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    }
    uint exponent = uint(bits >> 52) + highest - 52;
    if (rounded == (1ul << 53)) { rounded >>= 1; ++exponent; }
    return (ulong(exponent) << 52) | (rounded & mask);
}

kernel void ctc_scale_round_segments(device const int2* bounds [[buffer(0)]], constant Dimensions& dimensions [[buffer(1)]], device const ulong* means [[buffer(2)]], device const ulong* scale [[buffer(3)]], device ulong* output [[buffer(4)]], device atomic_uint* invalid [[buffer(5)]], uint token [[thread_position_in_grid]]) {
    if (token >= dimensions.count) return;
    if (token == 0) for (uint field = 0; field < 3; ++field) output[dimensions.count * 3 + field] = 0x7ff8000000000000ul;
    int2 frames = bounds[token];
    ulong score = means[token];
    if (frames.x < 0 || frames.y <= frames.x || frames.y > int(dimensions.frames) || score > 0x3ff0000000000000ul) { atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed); return; }
    output[token * 3] = python_round3(add_positive_double(multiply_frame_double(scale[0], uint(frames.x)), scale[1]));
    output[token * 3 + 1] = python_round3(add_positive_double(multiply_frame_double(scale[0], uint(frames.y)), scale[1]));
    output[token * 3 + 2] = python_round3(score);
}

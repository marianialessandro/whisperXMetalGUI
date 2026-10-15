struct DoubleDimensions { uint rows, columns, sequential; };

// Normal bounded binary64 multiplication with an exact 106-bit significand product and nearest-even rounding.
ulong multiply_positive_binary64(ulong a, ulong b) {
    if (a == 0 || b == 0) return 0;
    ulong mask = (1ul << 52) - 1, x = (a & mask) | (1ul << 52), y = (b & mask) | (1ul << 52);
    ulong x0 = uint(x), x1 = x >> 32, y0 = uint(y), y1 = y >> 32;
    ulong p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
    ulong middle = (p00 >> 32) + uint(p01) + uint(p10);
    ulong low = (middle << 32) | uint(p00), high = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
    uint shift = (high & (1ul << 41)) ? 53 : 52;
    int exponent = int(a >> 52) + int(b >> 52) - 1023 + int(shift == 53);
    ulong rounded = (high << (64 - shift)) | (low >> shift), remainder = low & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    if (rounded == (1ul << 53)) { rounded >>= 1; ++exponent; }
    return (ulong(exponent) << 52) | (rounded & mask);
}

// Long division emits all 53 significand bits, followed by exact nearest-even rounding from the remainder.
ulong divide_positive_binary64(ulong a, ulong b) {
    if (a == 0) return 0;
    ulong mask = (1ul << 52) - 1, x = (a & mask) | (1ul << 52), y = (b & mask) | (1ul << 52);
    int exponent = int(a >> 52) - int(b >> 52) + 1023;
    if (x < y) { x <<= 1; --exponent; }
    ulong quotient = 1, remainder = x - y;
    for (uint i = 0; i < 52; ++i) {
        quotient <<= 1; remainder <<= 1;
        if (remainder >= y) { ++quotient; remainder -= y; }
    }
    if (remainder * 2 > y || (remainder * 2 == y && (quotient & 1))) ++quotient;
    if (quotient == (1ul << 53)) { quotient >>= 1; ++exponent; }
    return (ulong(exponent) << 52) | (quotient & mask);
}

// Digit-by-digit integer square root of the 105/106-bit radicand preserves binary64 rounding without float arithmetic.
ulong sqrt_positive_binary64(ulong a) {
    if (a == 0) return 0;
    ulong mask = (1ul << 52) - 1, significand = (a & mask) | (1ul << 52);
    int exponent = int(a >> 52) - 1023, odd = exponent & 1;
    uint shift = 52 + uint(odd);
    ulong low = significand << shift, high = significand >> (64 - shift), root = 0, remainder = 0;
    for (int pair = 52; pair >= 0; --pair) {
        uint position = uint(pair) * 2;
        ulong digit = position >= 64 ? (high >> (position - 64)) & 3ul : (low >> position) & 3ul;
        remainder = (remainder << 2) | digit;
        ulong trial = (root << 2) | 1ul;
        root <<= 1;
        if (remainder >= trial) { remainder -= trial; ++root; }
    }
    if (remainder > root) ++root;
    exponent = (exponent - odd) / 2 + 1023;
    if (root == (1ul << 53)) { root >>= 1; ++exponent; }
    return (ulong(exponent) << 52) | (root & mask);
}

ulong double_square(device const ulong* values, uint index, bool squared = true) {
    ulong value = values[index] & ((1ul << 63) - 1);
    return squared ? multiply_positive_binary64(value, value) : value;
}

ulong double_leaf(device const ulong* values, uint start, uint count, bool squared = true) {
    ulong result = 0;
    if (count < 8) {
        for (uint i = 0; i < count; ++i) result = add_positive_double(result, double_square(values, start + i, squared));
        return result;
    }
    ulong sums[8];
    for (uint lane = 0; lane < 8; ++lane) sums[lane] = double_square(values, start + lane, squared);
    uint i = 8;
    for (; i < count - count % 8; i += 8) for (uint lane = 0; lane < 8; ++lane) sums[lane] = add_positive_double(sums[lane], double_square(values, start + i + lane, squared));
    result = add_positive_double(add_positive_double(add_positive_double(sums[0], sums[1]), add_positive_double(sums[2], sums[3])), add_positive_double(add_positive_double(sums[4], sums[5]), add_positive_double(sums[6], sums[7])));
    for (; i < count; ++i) result = add_positive_double(result, double_square(values, start + i, squared));
    return result;
}

ulong double_sum(device const ulong* values, uint count, uint sequential, bool squared = true) {
    if (sequential) {
        ulong result = 0;
        for (uint i = 0; i < count; ++i) result = add_positive_double(result, double_square(values, i, squared));
        return result;
    }
    uint starts[16], lengths[16], states[16], depth = 0;
    ulong left[16], result = 0;
    starts[0] = 0; lengths[0] = count; states[0] = 0;
    while (true) {
        uint length = lengths[depth], start = starts[depth];
        if (length <= 128) result = double_leaf(values, start, length, squared);
        else {
            uint split = (length / 2) & ~7u;
            if (states[depth] == 0) { states[depth] = 1; ++depth; starts[depth] = start; lengths[depth] = split; states[depth] = 0; continue; }
            if (states[depth] == 1) { left[depth] = result; states[depth] = 2; ++depth; starts[depth] = start + split; lengths[depth] = length - split; states[depth] = 0; continue; }
            result = add_positive_double(left[depth], result);
        }
        if (depth == 0) return result;
        --depth;
    }
}

kernel void double_normalize(device const ulong* input [[buffer(0)]], constant DoubleDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    uint start = row * d.columns;
    ulong norm = sqrt_positive_binary64(double_sum(input + start, d.columns, d.sequential)), sign = 1ul << 63;
    for (uint column = 0; column < d.columns; ++column) {
        ulong value = input[start + column];
        output[start + column] = divide_positive_binary64(value & (sign - 1), norm) | (value & sign);
    }
}

struct MatrixDimensions { uint rows, inner, columns; };

// Zero-sign handling here follows IEEE addition, including the positive-zero result of opposite signed zeros.
ulong matrix_add(ulong a, ulong b) {
    ulong sign = 1ul << 63;
    if (((a | b) & (sign - 1)) == 0) return a & b & sign;
    return add_signed_double(a, b);
}

// Each output has an explicit increasing-inner-index sum with separately rounded products and additions.
kernel void double_matmul(device const ulong* left [[buffer(0)]], constant MatrixDimensions& d [[buffer(1)]], device const ulong* right [[buffer(2)]], device ulong* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    uint row = index / d.columns, column = index % d.columns;
    ulong sign = 1ul << 63, sum = 0;
    for (uint inner = 0; inner < d.inner; ++inner) {
        ulong a = left[row * d.inner + inner], b = right[inner * d.columns + column];
        ulong product = multiply_positive_binary64(a & (sign - 1), b & (sign - 1)) | ((a ^ b) & sign);
        sum = matrix_add(sum, product);
    }
    output[index] = sum;
}

struct ElementDimensions { uint rows, columns, operation, scale_dimension; };

ulong dimension_binary64(uint value) {
    uint highest = 31u - clz(value);
    return (ulong(highest + 1023) << 52) | ((ulong(value) << (52u - highest)) & ((1ul << 52) - 1));
}

// Centering and scaling keep separate binary64 rounding steps; scalar square root is computed from integer dimension metadata on the GPU.
kernel void double_center_scale(device const ulong* input [[buffer(0)]], constant ElementDimensions& d [[buffer(1)]], device const ulong* mean [[buffer(2)]], device ulong* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    ulong value = input[index], sign = 1ul << 63;
    if (d.operation == 0) output[index] = matrix_add(value, mean[index % d.columns] ^ sign);
    else output[index] = multiply_positive_binary64(value & (sign - 1), sqrt_positive_binary64(dimension_binary64(d.scale_dimension))) | (value & sign);
}

// The scalar log(2*pi) is pinned to the observed NumPy 2.2.6 binary64 value, not recomputed with lower precision.
kernel void double_vbx_terms(device const ulong* input [[buffer(0)]], constant DoubleDimensions& d [[buffer(1)]], device const ulong* phi [[buffer(2)]], device ulong* terms [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index < d.rows) {
        ulong log_term = multiply_positive_binary64(dimension_binary64(d.columns), 0x3ffd67f1c864beb4ul);
        ulong sum = double_sum(input + index * d.columns, d.columns, d.sequential);
        terms[index] = multiply_positive_binary64(add_positive_double(sum, log_term), 0x3fe0000000000000ul) | (1ul << 63);
    }
    if (index < d.columns) {
        ulong value = phi[index], sign = 1ul << 63;
        terms[d.rows + index] = sqrt_positive_binary64(value & (sign - 1)) | (value & sign);
    }
}

kernel void double_vbx_rho(device const ulong* input [[buffer(0)]], constant DoubleDimensions& d [[buffer(1)]], device const ulong* terms [[buffer(2)]], device ulong* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    ulong a = input[index], b = terms[d.rows + index % d.columns], sign = 1ul << 63;
    output[index] = multiply_positive_binary64(a & (sign - 1), b & (sign - 1)) | ((a ^ b) & sign);
}

struct StatisticsDimensions { uint rows, columns, features, operation, sequential, reserved; ulong fa, fb; };

// Responsibilities are staged as contiguous speaker rows; their original axis-zero reduction order is explicit.
kernel void double_vbx_gamma_sum(device const ulong* input [[buffer(0)]], constant StatisticsDimensions& d [[buffer(1)]], device ulong* sums [[buffer(2)]], uint speaker [[thread_position_in_grid]]) {
    if (speaker < d.rows) sums[speaker] = double_sum(input + speaker * d.columns, d.columns, d.sequential, false);
}

// Division and each multiplication are separately rounded in the original Python expression order.
kernel void double_vbx_statistics(device const ulong* input [[buffer(0)]], constant StatisticsDimensions& d [[buffer(1)]], device const ulong* right [[buffer(2)]], device const ulong* sums [[buffer(3)]], device ulong* output [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.features) return;
    ulong ratio = divide_positive_binary64(d.fa, d.fb), sign = 1ul << 63;
    if (d.operation == 0) {
        ulong weighted = multiply_positive_binary64(ratio, sums[index / d.features]);
        ulong precision = add_positive_double(0x3ff0000000000000ul, multiply_positive_binary64(weighted, right[index % d.features] & (sign - 1)));
        output[index] = divide_positive_binary64(0x3ff0000000000000ul, precision);
    } else {
        ulong value = right[index];
        output[index] = multiply_positive_binary64(multiply_positive_binary64(ratio, input[index]), value & (sign - 1)) | (value & sign);
    }
}

struct ScoreDimensions { uint rows, columns, operation, reserved; ulong fa; };

// The original square, sum, penalty scaling, subtraction, addition and Fa scaling each round separately.
kernel void double_vbx_scores(device const ulong* input [[buffer(0)]], constant ScoreDimensions& d [[buffer(1)]], device const ulong* right [[buffer(2)]], device ulong* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    ulong sign = 1ul << 63;
    if (d.operation == 0) {
        ulong alpha = right[index] & (sign - 1);
        output[index] = matrix_add(input[index], multiply_positive_binary64(alpha, alpha));
    } else {
        ulong penalty = right[index % d.columns];
        ulong scaled = multiply_positive_binary64(penalty & (sign - 1), 0x3fe0000000000000ul) | (penalty & sign);
        ulong value = matrix_add(matrix_add(input[index], scaled ^ sign), right[d.columns + index / d.columns]);
        output[index] = multiply_positive_binary64(value & (sign - 1), d.fa) | (value & sign);
    }
}

// VBx prior normalization keeps the speaker sum pairwise and divides each mass only after the total is complete.
kernel void double_vbx_prior(device const ulong* masses [[buffer(0)]], constant StatisticsDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    ulong total = double_sum(masses, d.rows, 0, false);
    for (uint speaker = 0; speaker < d.rows; ++speaker) output[speaker] = divide_positive_binary64(masses[speaker], total);
}

// Log-prior addition and frame normalization retain the original two separately rounded binary64 operations.
kernel void double_vbx_posterior(device const ulong* input [[buffer(0)]], constant ScoreDimensions& d [[buffer(1)]], device const ulong* right [[buffer(2)]], device ulong* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    ulong value = right[d.operation == 0 ? index % d.columns : index / d.columns];
    output[index] = matrix_add(input[index], d.operation == 0 ? value : value ^ (1ul << 63));
}

// Signed reductions reproduce NumPy pairwise leaf order and the outer positive-zero accumulator.
ulong signed_leaf(device const ulong* values, uint start, uint count) {
    ulong result = 1ul << 63;
    if (count < 8) {
        for (uint i = 0; i < count; ++i) result = matrix_add(result, values[start + i]);
        return result;
    }
    ulong sums[8];
    for (uint lane = 0; lane < 8; ++lane) sums[lane] = values[start + lane];
    uint i = 8;
    for (; i < count - count % 8; i += 8) for (uint lane = 0; lane < 8; ++lane) sums[lane] = matrix_add(sums[lane], values[start + i + lane]);
    result = matrix_add(matrix_add(matrix_add(sums[0], sums[1]), matrix_add(sums[2], sums[3])), matrix_add(matrix_add(sums[4], sums[5]), matrix_add(sums[6], sums[7])));
    for (; i < count; ++i) result = matrix_add(result, values[start + i]);
    return result;
}

ulong signed_sum(device const ulong* values, uint count, uint sequential) {
    if (sequential) {
        ulong result = 0;
        for (uint i = 0; i < count; ++i) result = matrix_add(result, values[i]);
        return result;
    }
    uint starts[16], lengths[16], states[16], depth = 0;
    ulong left[16], result = 0;
    starts[0] = 0; lengths[0] = count; states[0] = 0;
    while (true) {
        uint length = lengths[depth], start = starts[depth];
        if (length <= 128) result = signed_leaf(values, start, length);
        else {
            uint split = (length / 2) & ~7u;
            if (states[depth] == 0) { states[depth] = 1; ++depth; starts[depth] = start; lengths[depth] = split; states[depth] = 0; continue; }
            if (states[depth] == 1) { left[depth] = result; states[depth] = 2; ++depth; starts[depth] = start + split; lengths[depth] = length - split; states[depth] = 0; continue; }
            result = matrix_add(left[depth], result);
        }
        if (depth == 0) return matrix_add(0, result);
        --depth;
    }
}


// NumPy 2.2.6 reduces long contiguous vectors in buffered chunks of 8192 elements before the outer signed accumulation.
ulong numpy_signed_sum(device const ulong* values, uint count) {
    ulong total = 0;
    for (uint start = 0; start < count; start += 8192) total = matrix_add(total, signed_sum(values + start, min(8192u, count - start), 0));
    return total;
}

struct ElboDimensions { uint rows, columns, operation, column_order; ulong fb, likelihood; };

kernel void double_vbx_likelihood(device const ulong* input [[buffer(0)]], constant ElboDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index == 0) output[0] = numpy_signed_sum(input, d.columns);
}

// Each ELBO subtraction, square and final addition is separately rounded before reduction in physical C/F order.
kernel void double_vbx_elbo_terms(device const ulong* input [[buffer(0)]], constant ElboDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    uint count = d.rows * d.columns;
    if (index >= count) return;
    ulong sign = 1ul << 63, alpha = input[2 * count + index] & (sign - 1);
    ulong term = matrix_add(input[index], input[count + index] ^ sign);
    term = matrix_add(term, multiply_positive_binary64(alpha, alpha) ^ sign);
    uint destination = d.column_order ? (index % d.columns) * d.rows + index / d.columns : index;
    output[destination] = matrix_add(term, 0x3ff0000000000000ul);
}

kernel void double_vbx_elbo_final(device const ulong* input [[buffer(0)]], constant ElboDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    ulong sign = 1ul << 63, sum = numpy_signed_sum(input, d.rows * d.columns);
    ulong scale = multiply_positive_binary64(d.fb, 0x3fe0000000000000ul);
    ulong penalty = multiply_positive_binary64(scale, sum & (sign - 1)) | (sum & sign);
    output[0] = matrix_add(d.likelihood, penalty);
}

// Algorithms adapted from OpenLibm e_log.c and e_exp.c for finite normal inputs using separately rounded integer-emulated binary64 operations.
// Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice is preserved.
ulong df_sub(ulong a, ulong b) { return matrix_add(a, b ^ (1ul << 63)); }
ulong df_mul(ulong a, ulong b) { ulong sign = 1ul << 63; return multiply_positive_binary64(a & (sign - 1), b & (sign - 1)) | ((a ^ b) & sign); }
ulong df_div(ulong a, ulong b) { ulong sign = 1ul << 63; return divide_positive_binary64(a & (sign - 1), b & (sign - 1)) | ((a ^ b) & sign); }
ulong df_integer(int value) { return value == 0 ? 0 : dimension_binary64(uint(value < 0 ? -value : value)) | (value < 0 ? 1ul << 63 : 0); }
int df_truncate(ulong value) {
    ulong sign = 1ul << 63, magnitude = value & (sign - 1);
    int exponent = int(magnitude >> 52) - 1023;
    if (exponent < 0) return 0;
    ulong significand = (magnitude & ((1ul << 52) - 1)) | (1ul << 52);
    int integer = int(significand >> uint(52 - exponent));
    return value & sign ? -integer : integer;
}

ulong bounded_binary64_log(ulong value) {
    ulong one = 0x3ff0000000000000ul, ln_hi = 0x3fe62e42fee00000ul, ln_lo = 0x3dea39ef35793c76ul;
    uint hx = uint(value >> 32), low = uint(value);
    int k = int(hx >> 20) - 1023;
    hx &= 0x000fffffu;
    uint adjust = (hx + 0x95f64u) & 0x100000u;
    ulong normalized = (ulong(hx | (adjust ^ 0x3ff00000u)) << 32) | low;
    k += int(adjust >> 20);
    ulong f = df_sub(normalized, one), dk = df_integer(k);
    if ((0x000fffffu & (2u + hx)) < 3u) {
        if ((f & ((1ul << 63) - 1)) == 0) return k == 0 ? 0 : matrix_add(df_mul(dk, ln_hi), df_mul(dk, ln_lo));
        ulong correction = df_mul(df_mul(f, f), df_sub(0x3fe0000000000000ul, df_mul(0x3fd5555555555555ul, f)));
        return k == 0 ? df_sub(f, correction) : df_sub(df_mul(dk, ln_hi), df_sub(df_sub(correction, df_mul(dk, ln_lo)), f));
    }
    ulong ratio = df_div(f, matrix_add(0x4000000000000000ul, f)), z = df_mul(ratio, ratio), w = df_mul(z, z);
    ulong even = df_mul(w, matrix_add(0x3fd999999997fa04ul, df_mul(w, matrix_add(0x3fcc71c51d8e78aful, df_mul(w, 0x3fc39a09d078c69ful)))));
    ulong odd = df_mul(z, matrix_add(0x3fe5555555555593ul, df_mul(w, matrix_add(0x3fd2492494229359ul, df_mul(w, matrix_add(0x3fc7466496cb03deul, df_mul(w, 0x3fc2f112df3e5244ul)))))));
    ulong correction = matrix_add(odd, even);
    int upper = int(hx) - 0x6147a, lower = 0x6b851 - int(hx);
    if ((upper | lower) > 0) {
        ulong square_half = df_mul(df_mul(0x3fe0000000000000ul, f), f);
        ulong reduced = df_mul(ratio, matrix_add(square_half, correction));
        return k == 0 ? df_sub(f, df_sub(square_half, reduced)) : df_sub(df_mul(dk, ln_hi), df_sub(df_sub(square_half, matrix_add(reduced, df_mul(dk, ln_lo))), f));
    }
    ulong reduced = df_mul(ratio, df_sub(f, correction));
    return k == 0 ? df_sub(f, reduced) : df_sub(df_mul(dk, ln_hi), df_sub(df_sub(reduced, df_mul(dk, ln_lo)), f));
}

ulong bounded_binary64_exp(ulong value) {
    ulong one = 0x3ff0000000000000ul, two = 0x4000000000000000ul, sign = 1ul << 63, ln_hi = 0x3fe62e42fee00000ul, ln_lo = 0x3dea39ef35793c76ul;
    uint high = uint((value & (sign - 1)) >> 32);
    bool negative = (value & sign) != 0;
    int k = 0;
    ulong hi = 0, lo = 0;
    if (high > 0x3fd62e42u) {
        if (high < 0x3ff0a2b2u) {
            hi = df_sub(value, ln_hi | (negative ? sign : 0)); lo = ln_lo | (negative ? sign : 0); k = negative ? -1 : 1;
        } else {
            k = df_truncate(matrix_add(df_mul(0x3ff71547652b82feul, value), 0x3fe0000000000000ul | (negative ? sign : 0)));
            ulong integer = df_integer(k);
            hi = df_sub(value, df_mul(integer, ln_hi)); lo = df_mul(integer, ln_lo);
        }
        value = df_sub(hi, lo);
    } else if (high < 0x3e300000u) return matrix_add(one, value);
    ulong square = df_mul(value, value);
    ulong polynomial = matrix_add(0xbebbbd41c5d26bf1ul, df_mul(square, 0x3e66376972bea4d0ul));
    polynomial = matrix_add(0x3f11566aaf25de2cul, df_mul(square, polynomial));
    polynomial = matrix_add(0xbf66c16c16bebd93ul, df_mul(square, polynomial));
    polynomial = matrix_add(0x3fc555555555553eul, df_mul(square, polynomial));
    ulong correction = df_sub(value, df_mul(square, polynomial));
    if (k == 0) return df_sub(one, df_sub(df_div(df_mul(value, correction), df_sub(correction, two)), value));
    ulong result = df_sub(one, df_sub(df_sub(lo, df_div(df_mul(value, correction), df_sub(two, correction))), hi));
    return (ulong(int(result >> 52) + k) << 52) | (result & ((1ul << 52) - 1));
}

struct TranscendentalDimensions { uint count, operation; };
kernel void double_vbx_transcendental(device const ulong* input [[buffer(0)]], constant TranscendentalDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < d.count) output[index] = d.operation == 0 ? bounded_binary64_log(input[index]) : bounded_binary64_exp(input[index]);
}

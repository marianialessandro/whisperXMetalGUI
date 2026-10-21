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

// The finite nonnegative log1p specialization follows OpenLibm s_log1p.c, developed at SunPro under the Sun notice above.
ulong bounded_binary64_log1p(ulong value) {
    ulong one = 0x3ff0000000000000ul, half_value = 0x3fe0000000000000ul, ln_hi = 0x3fe62e42fee00000ul, ln_lo = 0x3dea39ef35793c76ul;
    uint hx = uint((value & ((1ul << 63) - 1)) >> 32), hu = 0;
    int k = 1;
    ulong f = 0, correction = 0;
    if (hx < 0x3fda827au) {
        if (hx < 0x3e200000u) return hx < 0x3c900000u ? value : df_sub(value, df_mul(df_mul(value, value), half_value));
        k = 0; f = value; hu = 1;
    }
    if (k != 0) {
        ulong u = matrix_add(one, value);
        hu = uint(u >> 32); k = int(hu >> 20) - 1023;
        correction = df_div(k > 0 ? df_sub(one, df_sub(u, value)) : df_sub(value, df_sub(u, one)), u);
        hu &= 0x000fffffu;
        if (hu < 0x6a09eu) u = (ulong(hu | 0x3ff00000u) << 32) | uint(u);
        else { ++k; u = (ulong(hu | 0x3fe00000u) << 32) | uint(u); hu = (0x00100000u - hu) >> 2; }
        f = df_sub(u, one);
    }
    ulong square_half = df_mul(df_mul(half_value, f), f), dk = df_integer(k);
    if (hu == 0) {
        if ((f & ((1ul << 63) - 1)) == 0) return k == 0 ? 0 : matrix_add(df_mul(dk, ln_hi), matrix_add(correction, df_mul(dk, ln_lo)));
        ulong reduced = df_mul(square_half, df_sub(one, df_mul(0x3fe5555555555555ul, f)));
        return k == 0 ? df_sub(f, reduced) : df_sub(df_mul(dk, ln_hi), df_sub(df_sub(reduced, matrix_add(df_mul(dk, ln_lo), correction)), f));
    }
    ulong ratio = df_div(f, matrix_add(0x4000000000000000ul, f)), z = df_mul(ratio, ratio);
    ulong coefficients[7] = {0x3fe5555555555593ul, 0x3fd999999997fa04ul, 0x3fd2492494229359ul, 0x3fcc71c51d8e78aful, 0x3fc7466496cb03deul, 0x3fc39a09d078c69ful, 0x3fc2f112df3e5244ul};
    ulong polynomial = coefficients[6];
    for (int i = 5; i >= 0; --i) polynomial = matrix_add(coefficients[i], df_mul(z, polynomial));
    ulong reduced = df_mul(ratio, matrix_add(square_half, df_mul(z, polynomial)));
    return k == 0 ? df_sub(f, df_sub(square_half, reduced)) : df_sub(df_mul(dk, ln_hi), df_sub(df_sub(square_half, matrix_add(reduced, matrix_add(df_mul(dk, ln_lo), correction))), f));
}

kernel void double_vbx_log1p(device const ulong* input [[buffer(0)]], constant TranscendentalDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < d.count) output[index] = bounded_binary64_log1p(input[index]);
}

struct LogSumExpDimensions { uint rows, columns, sequential; };

bool binary64_equal(ulong a, ulong b) { return a == b || (((a | b) & ((1ul << 63) - 1)) == 0); }
bool binary64_greater_equal(ulong a, ulong b) {
    ulong sign = 1ul << 63;
    if (binary64_equal(a, b)) return true;
    if ((a ^ b) & sign) return (a & sign) == 0;
    return (a & sign) ? a <= b : a >= b;
}

// All repeated maxima are separated before exponentiation, as in the pinned finite unweighted SciPy path.
kernel void double_vbx_lse_terms(device const ulong* input [[buffer(0)]], constant LogSumExpDimensions& d [[buffer(1)]], device ulong* exponentials [[buffer(2)]], device ulong* metadata [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    uint start = row * d.columns, count = 0;
    ulong maximum = input[start], invalid = 0;
    for (uint column = 1; column < d.columns; ++column) if (binary64_greater_equal(input[start + column], maximum)) maximum = input[start + column];
    for (uint column = 0; column < d.columns; ++column) {
        ulong value = input[start + column];
        if (binary64_equal(value, maximum)) { ++count; exponentials[start + column] = 0; }
        else {
            ulong delta = df_sub(value, maximum);
            if ((delta & ((1ul << 63) - 1)) > 0x404e000000000000ul) { invalid = 1; exponentials[start + column] = 0; }
            else exponentials[start + column] = bounded_binary64_exp(delta);
        }
    }
    metadata[row * 3] = maximum; metadata[row * 3 + 1] = dimension_binary64(count); metadata[row * 3 + 2] = invalid;
}

kernel void double_vbx_lse_final(device const ulong* exponentials [[buffer(0)]], constant LogSumExpDimensions& d [[buffer(1)]], device const ulong* metadata [[buffer(2)]], device ulong* output [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    if (metadata[row * 3 + 2]) { output[row] = 0x7ff8000000000000ul; return; }
    ulong sum = double_sum(exponentials + row * d.columns, d.columns, d.sequential, false), mass = metadata[row * 3 + 1];
    ulong scaled = divide_positive_binary64(sum, mass);
    output[row] = matrix_add(matrix_add(bounded_binary64_log1p(scaled), bounded_binary64_log(mass)), metadata[row * 3]);
}

struct InitializationDimensions { uint rows, columns, operation, sequential; };

// Uniform-prior division and initial responsibility normalization retain binary64 rounding without changing the random sample sequence.
kernel void double_vbx_initialize(device const ulong* input [[buffer(0)]], constant InitializationDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (d.operation == 0) {
        if (index < d.columns) output[index] = divide_positive_binary64(0x3ff0000000000000ul, dimension_binary64(d.columns));
        return;
    }
    if (index >= d.rows) return;
    uint start = index * d.columns;
    ulong mass = double_sum(input + start, d.columns, d.sequential, false), sign = 1ul << 63;
    for (uint column = 0; column < d.columns; ++column) {
        ulong value = input[start + column];
        output[start + column] = divide_positive_binary64(value & (sign - 1), mass) | (value & sign);
    }
}

struct QinitDimensions { uint rows, columns; ulong smoothing; };

// Source-pinned cluster initialization is one-hot, so the shifted maximum is exactly the nonnegative smoothing value.
kernel void double_vbx_qinit_exp(device const ulong* input [[buffer(0)]], constant QinitDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    ulong scaled = df_mul(input[index], d.smoothing);
    ulong shifted = matrix_add(scaled, d.smoothing ^ (1ul << 63));
    output[index] = bounded_binary64_exp(shifted);
}

struct CentroidDimensions { uint speakers, frames, features, sequential; };

// Weighted centroid products accumulate in increasing frame order before separately rounded division by the NumPy-order speaker mass.
kernel void double_vbx_centroids(device const ulong* weights [[buffer(0)]], constant CentroidDimensions& d [[buffer(1)]], device const ulong* features [[buffer(2)]], device ulong* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.speakers * d.features) return;
    uint speaker = index / d.features, feature = index % d.features;
    ulong sum = 0, sign = 1ul << 63;
    for (uint frame = 0; frame < d.frames; ++frame) sum = matrix_add(sum, df_mul(weights[speaker * d.frames + frame], features[frame * d.features + feature]));
    ulong mass = double_sum(weights + speaker * d.frames, d.frames, d.sequential, false);
    output[index] = divide_positive_binary64(sum & (sign - 1), mass) | (sum & sign);
}

struct OneHotDimensions { uint rows, columns; };

// Integer cluster labels generate an exact binary64 one-hot matrix without a host matrix fill or scatter.
kernel void double_vbx_one_hot(device const long* labels [[buffer(0)]], constant OneHotDimensions& d [[buffer(1)]], device ulong* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < d.rows * d.columns) output[index] = labels[index / d.columns] == long(index % d.columns) ? 0x3ff0000000000000ul : 0;
}

struct CosineDimensions { uint rows, centroids, features; };

// Cosine uses sequential squared norms and dot products with separately rounded binary64 arithmetic.
kernel void double_vbx_cosine_norms(device const ulong* embeddings [[buffer(0)]], constant CosineDimensions& d [[buffer(1)]], device const ulong* centroids [[buffer(2)]], device ulong* norms [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows + d.centroids) return;
    device const ulong* values = index < d.rows ? embeddings + index * d.features : centroids + (index - d.rows) * d.features;
    ulong sum = 0, mask = (1ul << 63) - 1;
    for (uint feature = 0; feature < d.features; ++feature) {
        ulong value = values[feature] & mask;
        if (value > 0x7ff0000000000000ul) { norms[index] = 0x7ff8000000000000ul; return; }
        sum = add_positive_double(sum, multiply_positive_binary64(value, value));
    }
    norms[index] = sqrt_positive_binary64(sum);
}

// Undefined zero-norm or NaN pairs remain NaN; finite cosine values are clamped to [-1,1] before the final subtraction.
ulong cosine_distance_value(device const ulong* embeddings, constant CosineDimensions& d, device const ulong* centroids, device const ulong* norms, uint index) {
    uint row = index / d.centroids, centroid = index % d.centroids;
    ulong a = norms[row], b = norms[d.rows + centroid], one = 0x3ff0000000000000ul, sign = 1ul << 63;
    if (a == 0 || b == 0 || a > 0x7ff0000000000000ul || b > 0x7ff0000000000000ul) { return 0x7ff8000000000000ul; }
    ulong dot = 0;
    for (uint feature = 0; feature < d.features; ++feature) dot = matrix_add(dot, df_mul(embeddings[row * d.features + feature], centroids[centroid * d.features + feature]));
    ulong cosine = df_div(dot, multiply_positive_binary64(a, b));
    if ((cosine & (sign - 1)) > one) cosine = one | (cosine & sign);
    return df_sub(one, cosine);
}


kernel void double_vbx_cosine_distances(device const ulong* embeddings [[buffer(0)]], constant CosineDimensions& d [[buffer(1)]], device const ulong* centroids [[buffer(2)]], device const ulong* norms [[buffer(3)]], device ulong* output [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index < d.rows * d.centroids) output[index] = cosine_distance_value(embeddings, d, centroids, norms, index);
}

// Retain both separately rounded subtractions in 2 - (1 - cosine), with undefined pairs propagated before binary64 arithmetic.
kernel void double_vbx_cosine_scores(device const ulong* embeddings [[buffer(0)]], constant CosineDimensions& d [[buffer(1)]], device const ulong* centroids [[buffer(2)]], device const ulong* norms [[buffer(3)]], device ulong* output [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.centroids) return;
    ulong distance = cosine_distance_value(embeddings, d, centroids, norms, index);
    output[index] = distance > 0x7ff0000000000000ul ? distance : df_sub(0x4000000000000000ul, distance);
}

struct ArgmaxDimensions { uint rows, columns; };

bool binary64_argmax_greater(ulong a, ulong b) {
    ulong sign = 1ul << 63, mask = sign - 1;
    if (((a | b) & mask) == 0) return false;
    if ((a & sign) != (b & sign)) return (a & sign) == 0;
    return (a & sign) ? (a & mask) < (b & mask) : a > b;
}

// NumPy argmax prefers the first NaN, otherwise the first maximum, treating opposite signed zeros as equal.
kernel void double_vbx_argmax(device const ulong* scores [[buffer(0)]], constant ArgmaxDimensions& d [[buffer(1)]], device long* output [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    uint winner = 0;
    ulong best = scores[row * d.columns];
    for (uint column = 0; column < d.columns; ++column) {
        ulong value = scores[row * d.columns + column];
        if ((value & ((1ul << 63) - 1)) > 0x7ff0000000000000ul) { winner = column; break; }
        if (binary64_argmax_greater(value, best)) { winner = column; best = value; }
    }
    output[row] = long(winner);
}

struct InactiveDimensions { uint rows, columns, frames, blocks; };

// Scores are bounded nonnegative values or NaN, so integer magnitudes preserve finite minimum comparisons.
kernel void double_vbx_score_min_partials(device const ulong* scores [[buffer(0)]], constant InactiveDimensions& d [[buffer(1)]], device ulong* partials [[buffer(2)]], uint block [[thread_position_in_grid]]) {
    if (block >= d.blocks) return;
    uint end = min((block + 1) * 1024, d.rows * d.columns);
    ulong value = 0x4000000000000000ul;
    for (uint index = block * 1024; index < end; ++index) {
        ulong candidate = scores[index] & ((1ul << 63) - 1);
        if (candidate > 0x7ff0000000000000ul) { value = 0x7ff8000000000000ul; break; }
        value = min(value, candidate);
    }
    partials[block] = value;
}

// Binary segmentation has zero sum exactly when every frame is zero, including negative zero.
kernel void double_vbx_inactive_mask(device const ulong* scores [[buffer(0)]], constant InactiveDimensions& d [[buffer(1)]], device const ulong* segments [[buffer(2)]], device const ulong* partials [[buffer(3)]], device ulong* output [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    bool active = false;
    for (uint frame = 0; frame < d.frames; ++frame) if ((segments[row * d.frames + frame] & ((1ul << 63) - 1)) != 0) { active = true; break; }
    ulong replacement = 0;
    if (!active) {
        ulong minimum = 0x4000000000000000ul;
        for (uint block = 0; block < d.blocks; ++block) {
            ulong value = partials[block];
            if (value > 0x7ff0000000000000ul) { minimum = 0x7ff8000000000000ul; break; }
            minimum = min(minimum, value);
        }
        // Subtracting one rounds to negative one at or below 2^-54, including subnormal minima.
        replacement = minimum > 0x7ff0000000000000ul ? minimum : minimum <= 0x3c90000000000000ul ? 0xbff0000000000000ul : df_sub(minimum, 0x3ff0000000000000ul);
    }
    for (uint column = 0; column < d.columns; ++column) output[row * d.columns + column] = active ? scores[row * d.columns + column] : replacement;
}

struct CleanupDimensions { uint rows, columns, blocks; };

// NaNs are ignored for the minimum; zero ties are canonicalized only for replacement values.
kernel void double_vbx_nan_min_partials(device const ulong* scores [[buffer(0)]], constant CleanupDimensions& d [[buffer(1)]], device ulong* partials [[buffer(2)]], uint block [[thread_position_in_grid]]) {
    if (block >= d.blocks) return;
    ulong minimum = 0x7ff0000000000000ul, mask = (1ul << 63) - 1;
    uint end = min((block + 1) * 1024, d.rows * d.columns);
    for (uint index = block * 1024; index < end; ++index) {
        ulong value = scores[index];
        if ((value & mask) <= 0x7ff0000000000000ul && binary64_argmax_greater(minimum, value)) minimum = value;
    }
    partials[block] = (minimum & mask) == 0 ? 0 : minimum;
}

kernel void double_vbx_nan_cleanup(device const ulong* scores [[buffer(0)]], constant CleanupDimensions& d [[buffer(1)]], device const ulong* partials [[buffer(2)]], device ulong* output [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    ulong minimum = 0x7ff0000000000000ul, mask = (1ul << 63) - 1;
    for (uint block = 0; block < d.blocks; ++block) if (binary64_argmax_greater(minimum, partials[block])) minimum = partials[block];
    ulong replacement = minimum == 0x7ff0000000000000ul ? 0x7ff8000000000000ul : minimum;
    for (uint column = 0; column < d.columns; ++column) {
        ulong value = scores[row * d.columns + column];
        output[row * d.columns + column] = (value & mask) > 0x7ff0000000000000ul ? replacement : value;
    }
}

struct KmeansCentroidDimensions { uint rows, features, clusters; };

kernel void float_vbx_kmeans_centroids(device const float* features [[buffer(0)]], constant KmeansCentroidDimensions& d [[buffer(1)]], device const int* labels [[buffer(2)]], device float* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.clusters * d.features) return;
    uint cluster = index / d.features, feature = index % d.features, count = 0;
    float sum = 0.0f;
    for (uint row = 0; row < d.rows; ++row) {
        if (labels[row] == int(cluster)) {
            sum += features[row * d.features + feature];
            ++count;
        }
    }
    output[index] = sum / float(count);
}

struct FloatShiftDimensions { uint rows, columns, add; };

kernel void float_vbx_shift(device const float* values [[buffer(0)]], constant FloatShiftDimensions& d [[buffer(1)]], device const float* mean [[buffer(2)]], device float* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    float offset = mean[index % d.columns];
    output[index] = d.add ? values[index] + offset : values[index] - offset;
}

struct KmeansToleranceDimensions { uint rows, columns, output_offset; float tolerance; };

float kmeans_float_leaf(device const float* values, uint start, uint count) {
    float result = -0.0f;
    if (count < 8) {
        for (uint i = 0; i < count; ++i) result += values[start + i];
        return result;
    }
    float sums[8];
    for (uint lane = 0; lane < 8; ++lane) sums[lane] = values[start + lane];
    uint i = 8;
    for (; i < count - count % 8; i += 8) for (uint lane = 0; lane < 8; ++lane) sums[lane] += values[start + i + lane];
    result = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    for (; i < count; ++i) result += values[start + i];
    return result;
}

float kmeans_float_sum(device const float* values, uint count) {
    uint starts[16], lengths[16], states[16], depth = 0;
    float left[16], result = 0;
    starts[0] = 0; lengths[0] = count; states[0] = 0;
    while (true) {
        uint length = lengths[depth], start = starts[depth];
        if (length <= 128) result = kmeans_float_leaf(values, start, length);
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

kernel void float_kmeans_tolerance_column_mean(device const float* values [[buffer(0)]], constant KmeansToleranceDimensions& d [[buffer(1)]], device float* output [[buffer(2)]], uint column [[thread_position_in_grid]]) {
    if (column >= d.columns) return;
    float sum = 0.0f;
    for (uint row = 0; row < d.rows; ++row) sum += values[row * d.columns + column];
    float result = sum / float(d.rows);
    if ((d.output_offset != 0 && sum > 0 && sum < float(d.rows) * 0x1p-126f) || (sum != 0 && result == 0) || (result != 0 && (as_type<uint>(result) & 0x7f800000u) == 0)) result = as_type<float>(0x7fc00000u);
    output[d.output_offset + column] = result;
}

kernel void float_kmeans_tolerance_square(device const float* values [[buffer(0)]], constant KmeansToleranceDimensions& d [[buffer(1)]], device const float* means [[buffer(2)]], device float* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    float delta = values[index] - means[index % d.columns];
    float square = delta * delta;
    if ((delta != 0 && abs(delta) < 0x1p-63f) || (delta != 0 && square == 0) || (square != 0 && (as_type<uint>(square) & 0x7f800000u) == 0)) square = as_type<float>(0x7fc00000u);
    output[index] = square;
}

kernel void float_kmeans_tolerance_final(device const float* statistics [[buffer(0)]], constant KmeansToleranceDimensions& d [[buffer(1)]], device float* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    float sum = kmeans_float_sum(statistics + d.columns, d.columns);
    float mean = sum / float(d.columns), result = mean * d.tolerance;
    if ((sum > 0 && sum < float(d.columns) * 0x1p-126f) || (sum > 0 && mean == 0) || (mean != 0 && (as_type<uint>(mean) & 0x7f800000u) == 0) || (mean > 0 && result == 0) || (result != 0 && (as_type<uint>(result) & 0x7f800000u) == 0)) result = as_type<float>(0x7fc00000u);
    output[0] = result;
}

struct KmeansNormDimensions { uint rows, columns; };

// Match the pinned NumPy ARM NEON four-lane fused accumulation, with reversed four-vector blocks and a pairwise horizontal sum.
kernel void float_kmeans_squared_norms(device const float* values [[buffer(0)]], constant KmeansNormDimensions& d [[buffer(1)]], device float* output [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    if (row >= d.rows) return;
    float4 accum = float4(0.0f);
    uint column = 0;
    for (; column + 16 <= d.columns; column += 16) {
        for (int block = 3; block >= 0; --block) {
            uint start = row * d.columns + column + uint(block) * 4;
            float4 x = float4(values[start], values[start + 1], values[start + 2], values[start + 3]);
            accum = fma(x, x, accum);
        }
    }
    for (; column < d.columns; column += 4) {
        float4 x = float4(0.0f);
        for (uint lane = 0; lane < 4 && column + lane < d.columns; ++lane) x[lane] = values[row * d.columns + column + lane];
        accum = fma(x, x, accum);
    }
    output[row] = 0.0f + ((accum.x + accum.y) + (accum.z + accum.w));
}

struct KmeansPrefixDimensions { uint count; };

// Store products before accumulation to preserve NumPy's separate float32 multiplication and cumulative addition.
kernel void float_kmeans_weighted_products(device const float* weights [[buffer(0)]], constant KmeansPrefixDimensions& d [[buffer(1)]], device const float* distances [[buffer(2)]], device float* products [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index < d.count) products[index] = weights[index] * distances[index];
}

kernel void float_kmeans_weighted_prefix(device const float* products [[buffer(0)]], constant KmeansPrefixDimensions& d [[buffer(1)]], device float* output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    float accum = products[0];
    output[0] = accum;
    for (uint i = 1; i < d.count; ++i) {
        accum += products[i];
        output[i] = accum;
    }
}

struct KmeansMinimumDimensions { uint rows, columns; };

kernel void float_kmeans_candidate_minimum(device const float* candidates [[buffer(0)]], constant KmeansMinimumDimensions& d [[buffer(1)]], device const float* closest [[buffer(2)]], device float* output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.rows * d.columns) return;
    float a = closest[index % d.columns], b = candidates[index];
    output[index] = a == 0.0f && b == 0.0f ? as_type<float>(as_type<uint>(a) | as_type<uint>(b)) : (a < b ? a : b);
}

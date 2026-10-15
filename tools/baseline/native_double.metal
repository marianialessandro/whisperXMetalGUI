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

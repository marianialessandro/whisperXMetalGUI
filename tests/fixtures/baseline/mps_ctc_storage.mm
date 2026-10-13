#include "native_ctc_core.hpp"
#include <fstream>
#include <iostream>
#include <iterator>

static void same_path(const NativeResult& left, const NativeResult& right) {
    require(left.found == right.found && left.path.size() == right.path.size(), "borrowed_path_shape_mismatch");
    for (size_t index = 0; index < left.path.size(); ++index) require(left.path[index].token == right.path[index].token && left.path[index].frame == right.path[index].frame && std::memcmp(&left.path[index].score, &right.path[index].score, 4) == 0, "borrowed_path_value_mismatch");
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 2, "shader_argument_required");
            std::ifstream stream(argv[1]);
            std::string shader((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
            NativeCTCContext context(shader.data(), shader.size(), true, true);
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            const Dimensions dimensions{19, 6, 4, 0};
            std::vector<float> emission(19 * 6, -2.0f), prefixed(3 + 19 * 6, 1.0f);
            for (uint32_t row = 0; row < 19; ++row) emission[row * 6 + 1 + row % 4] = -0.1f;
            std::memcpy(prefixed.data() + 3, emission.data(), emission.size() * 4);
            id<MTLBuffer> borrowed = [device newBufferWithBytes:prefixed.data() length:prefixed.size() * 4 options:MTLResourceStorageModeShared];
            const int32_t tokens[]{1, 2, 2, 4};
            auto expected = context.compute(emission.data(), emission.size(), tokens, 4, dimensions);
            for (int run = 0; run < 8; ++run) same_path(expected, context.compute(nullptr, emission.size(), tokens, 4, dimensions, false, false, false, borrowed, 12));
            const int32_t wildcard_tokens[]{1, 2, 6, 4};
            auto wildcard = context.compute(emission.data(), emission.size(), wildcard_tokens, 4, dimensions, true);
            same_path(wildcard, context.compute(nullptr, emission.size(), wildcard_tokens, 4, dimensions, true, false, false, borrowed, 12));
            auto resident = context.compute(nullptr, emission.size(), tokens, 4, dimensions, false, true, true, borrowed, 12);
            require(resident.generation != 0, "resident_generation_missing");
            try { context.compute(nullptr, emission.size(), tokens, 4, dimensions, false, false, false, borrowed, 13); throw InventoryError("unaligned_borrow_accepted"); }
            catch (const InventoryError& error) { require(std::string(error.what()) == "borrowed_emission_range_or_device_invalid", "wrong_borrowed_error"); }
            uint64_t output[12]{};
            try { context.scale_round_segments(resident.generation, 4, 0, 0, output, 12); throw InventoryError("stale_resident_accepted"); }
            catch (const InventoryError& error) { require(std::string(error.what()) == "resident_segment_generation_invalid", "wrong_resident_error"); }
            for (uint64_t offset : {uint64_t(16), uint64_t(-4)}) {
                try { context.compute(nullptr, emission.size(), tokens, 4, dimensions, false, false, false, borrowed, offset); throw InventoryError("invalid_range_accepted"); }
                catch (const InventoryError& error) { require(std::string(error.what()) == "borrowed_emission_range_or_device_invalid", "wrong_range_error"); }
            }
            same_path(expected, context.compute(emission.data(), emission.size(), tokens, 4, dimensions));
            same_path(expected, context.compute(nullptr, emission.size(), tokens, 4, dimensions, false, false, false, borrowed, 12));
            auto retained = context.compute(nullptr, emission.size(), tokens, 4, dimensions, false, true, true, borrowed, 12);
            const double ratio = .02, offset = 0.; uint64_t ratio_bits, offset_bits;
            std::memcpy(&ratio_bits, &ratio, 8); std::memcpy(&offset_bits, &offset, 8);
            uint64_t characters[12]; context.scale_round_segments(retained.generation, 4, ratio_bits, offset_bits, characters, 12);
            const uint32_t indices[]{0, 1}, offsets[]{0, 1, 2}, modes[]{7, 7}; uint64_t statistics[8];
            auto stats = context.alignment_stats_batch(nullptr, 15, 5, indices, 2, offsets, 3, modes, 2, statistics, 8, retained.generation, ratio_bits, offset_bits, true, true);
            const uint32_t mapping[]{0, 3, 1, 4}; uint64_t actual[4], copy[4]{statistics[0], statistics[3], statistics[1], statistics[4]}, reference[4];
            context.interpolate_retained_statistics(stats.generation, mapping, 4, actual, 4);
            context.interpolate_nearest(copy, 4, reference, 4, 2);
            require(std::memcmp(actual, reference, sizeof(actual)) == 0, "retained_nearest_mismatch");
            const uint32_t invalid_mapping[]{2, UINT32_MAX, 1, 4}; std::memset(actual, 0xda, sizeof(actual));
            try { context.interpolate_retained_statistics(stats.generation, invalid_mapping, 4, actual, 4); throw InventoryError("invalid_retained_mapping_accepted"); }
            catch (const InventoryError& error) { require(std::string(error.what()) == "resident_nearest_indices_invalid", "wrong_retained_mapping_error"); }
            const uint8_t* untouched = reinterpret_cast<const uint8_t*>(actual);
            for (size_t i = 0; i < sizeof(actual); ++i) require(untouched[i] == 0xda, "retained_error_modified_output");
            context.invalidate_resident();
            try { context.interpolate_retained_statistics(stats.generation, mapping, 4, actual, 4); throw InventoryError("stale_statistics_accepted"); }
            catch (const InventoryError& error) { require(std::string(error.what()) == "resident_nearest_generation_invalid", "wrong_statistics_generation_error"); }
            std::cout << "borrowed_storage_sanitizer_verified\n";
            return 0;
        } catch (const std::exception&) { std::cerr << "borrowed_storage_harness_failed\n"; return 2; }
    }
}

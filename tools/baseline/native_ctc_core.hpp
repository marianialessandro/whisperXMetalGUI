#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

struct Dimensions { uint32_t frames; uint32_t vocabulary; uint32_t count; int32_t blank; };
static_assert(sizeof(Dimensions) == 16);
struct Timing { uint64_t wall_ns; double gpu_seconds; };
struct NativePoint { int32_t token; int32_t frame; float score; };
struct NativeSegment { int32_t start; int32_t end; double score; };
struct NativeResult { std::vector<NativePoint> path; bool found; std::vector<Timing> timings; uint64_t dp_bytes; uint64_t allocated_bytes; std::vector<NativeSegment> segments; int32_t path_length; uint64_t generation = 0; };
struct NativeNearestResult { Timing timing; uint64_t allocated_bytes; uint32_t counts[4]; };
struct NativeDirectStatsResult { Timing timing; uint64_t allocated_bytes; };
struct NativeBatchBoundsResult { std::vector<uint64_t> bounds; Timing timing; uint64_t allocated_bytes; };
struct NativeBoundsResult { uint64_t minimum, maximum; Timing timing; uint64_t allocated_bytes; };
struct NativeMaxResult { std::vector<float> values; std::vector<uint32_t> indices; std::vector<Timing> timings; uint64_t allocated_bytes; };

static void encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline, NSArray<id<MTLBuffer>>* buffers, const Dimensions& dimensions, NSUInteger dimension_index, NSUInteger grid, NSUInteger group, NSString* label, NSUInteger first_buffer_offset = 0) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "metal_encoder_creation_failed");
    encoder.label = label; [encoder setComputePipelineState:pipeline];
    NSUInteger index = 0, ordinal = 0;
    for (id<MTLBuffer> buffer in buffers) { if (index == dimension_index) ++index; [encoder setBuffer:buffer offset:ordinal++ == 0 ? first_buffer_offset : 0 atIndex:index++]; }
    [encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:dimension_index];
    [encoder dispatchThreads:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [encoder endEncoding];
}

static Timing complete(id<MTLCommandBuffer> command, std::chrono::steady_clock::time_point started) {
    [command commit]; [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "metal_command_failed");
    auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    require(command.GPUEndTime >= command.GPUStartTime && command.GPUStartTime > 0, "metal_gpu_timestamps_unavailable");
    return {static_cast<uint64_t>(wall), command.GPUEndTime - command.GPUStartTime};
}

static Timing execute(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, NSArray<id<MTLBuffer>>* buffers, const Dimensions& dimensions, NSUInteger dimension_index, NSUInteger grid, NSUInteger group, NSString* label, NSUInteger first_buffer_offset = 0) {
    auto started = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [queue commandBuffer]; require(command != nil, "metal_command_creation_failed");
    command.label = label;
    encode(command, pipeline, buffers, dimensions, dimension_index, grid, group, label, first_buffer_offset);
    return complete(command, started);
}

// Synchronous development context; callers must finish all calls before destroying it.
class NativeCTCContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> max_pipeline_;
    id<MTLComputePipelineState> wildcard_validate_, wildcard_expand_;
    id<MTLComputePipelineState> segment_pipeline_, bounds_pipeline_, batch_bounds_pipeline_, word_means_pipeline_, alignment_stats_pipeline_, round3_pipeline_, scale_round_pipeline_, word_round_pipeline_, nearest_neighbors_pipeline_, nearest_fill_pipeline_;
    NSMutableArray<id<MTLComputePipelineState>>* pipelines_;
    NSArray<NSString*>* names_;
    id<MTLBuffer> e_, t_, invalid_, prefix_, rows_, path_, decisions_, endpoint_;
    id<MTLBuffer> max_values_, max_indices_;
    id<MTLBuffer> expanded_;
    id<MTLBuffer> segment_bounds_, segment_means_, bounds_input_, bounds_output_, bounds_indices_, bounds_offsets_, stats_modes_, scale_parameters_, rounded_segments_, rounded_words_, nearest_neighbors_, nearest_counts_;
    uint64_t generation_counter_ = 0, resident_generation_ = 0, rounded_generation_ = 0, rounded_ratio_ = 0, rounded_offset_ = 0;
    uint32_t resident_count_ = 0, resident_frames_ = 0;
    bool compact_, batched_;
    std::mutex mutex_;

    void reserve(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer == nil || buffer.length < bytes) buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil, "metal_allocation_failed");
    }

    uint64_t allocated_bytes() const { return e_.length + t_.length + invalid_.length + prefix_.length + rows_.length + path_.length + decisions_.length + endpoint_.length + max_values_.length + max_indices_.length + expanded_.length + segment_bounds_.length + segment_means_.length + bounds_input_.length + bounds_output_.length + bounds_indices_.length + bounds_offsets_.length + stats_modes_.length + scale_parameters_.length + rounded_segments_.length + rounded_words_.length + nearest_neighbors_.length + nearest_counts_.length; }

    void pipeline(id<MTLComputePipelineState> __strong& state, NSString* name) {
        if (state != nil) return;
        id<MTLFunction> function = [library_ newFunctionWithName:name]; require(function != nil, "optional_shader_function_missing");
        NSError* error = nil;
        state = [device_ newComputePipelineStateWithFunction:function error:&error];
        require(state != nil && state.maxTotalThreadsPerThreadgroup >= 256, "optional_pipeline_creation_failed");
    }

public:
    NativeCTCContext(const char* shader, uint64_t shader_bytes, bool compact, bool batched) : compact_(compact), batched_(batched) {
        require(shader != nullptr && shader_bytes > 0 && shader_bytes <= 1024 * 1024, "shader_budget_invalid");
        device_ = MTLCreateSystemDefaultDevice(); require(device_ != nil && [device_ supportsFamily:MTLGPUFamilyApple7], "apple_metal_required");
        NSString* source = [[NSString alloc] initWithBytes:shader length:shader_bytes encoding:NSUTF8StringEncoding]; require(source != nil, "shader_encoding_invalid");
        MTLCompileOptions* compile = [[MTLCompileOptions alloc] init]; compile.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:source options:compile error:&error]; require(library != nil, "shader_compilation_failed");
        library_ = library;
        pipelines_ = [[NSMutableArray alloc] init];
        names_ = compact ? @[@"ctc_validate", @"ctc_prefix", @"ctc_trellis_compact", @"ctc_backtrack_compact"] : @[@"ctc_validate", @"ctc_prefix", @"ctc_trellis", @"ctc_backtrack"];
        for (NSString* name in names_) {
            id<MTLFunction> function = [library newFunctionWithName:name]; require(function != nil, "shader_function_missing");
            id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error]; require(pipeline != nil && pipeline.maxTotalThreadsPerThreadgroup >= 256, "pipeline_creation_failed");
            [pipelines_ addObject:pipeline];
        }
        queue_ = [device_ newCommandQueue]; require(queue_ != nil, "metal_queue_creation_failed");
    }

    NativeResult compute(const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, const Dimensions& source, bool wildcard = false, bool segments = false, bool resident = false, id<MTLBuffer> borrowed_emission = nil, uint64_t borrowed_offset = 0) {
        std::lock_guard<std::mutex> guard(mutex_);
        resident_generation_ = 0; rounded_generation_ = 0;
        require(!resident || segments, "resident_requires_segments");
        Dimensions d = source;
        require((emission != nullptr || borrowed_emission != nil) && tokens != nullptr && d.frames > 0 && d.frames <= 32768 && d.vocabulary > 0 && d.vocabulary <= uint64_t(1) << 24 && d.count > 0 && d.count <= d.frames, "input_shape_invalid");
        uint64_t elements = uint64_t(d.frames) * d.vocabulary, cells = uint64_t(d.frames + 1) * (d.count + 1);
        require(emission_elements == elements && token_elements == d.count, "input_lengths_invalid");
        require(elements * 4 <= 64 * 1024 * 1024 && cells <= 16 * 1024 * 1024, "input_budget_invalid");
        require(borrowed_emission != nil || borrowed_offset == 0, "borrowed_offset_without_buffer");
        if (borrowed_emission != nil) require(borrowed_emission.device == device_ && borrowed_offset % 4 == 0 && borrowed_offset <= borrowed_emission.length && elements * 4 <= borrowed_emission.length - borrowed_offset, "borrowed_emission_range_or_device_invalid");
        if (wildcard) {
            require(batched_ && d.vocabulary >= 2 && uint64_t(d.frames) * (d.vocabulary + 1) * 4 <= 64 * 1024 * 1024, "wildcard_shape_or_budget_invalid");
            pipeline(wildcard_validate_, @"ctc_validate_wildcard"); pipeline(wildcard_expand_, @"ctc_expand_wildcard");
            reserve(expanded_, uint64_t(d.frames) * (d.vocabulary + 1) * 4);
        }
        if (segments) {
            require(batched_, "segment_mode_requires_batched");
            pipeline(segment_pipeline_, @"ctc_merge_repeats");
            reserve(segment_bounds_, uint64_t(d.count) * 8); reserve(segment_means_, uint64_t(d.count) * 8);
        }
        uint64_t dp_bytes = compact_ ? 2 * uint64_t(d.count + 1) * 4 : cells * 4;
        if (borrowed_emission == nil) reserve(e_, elements * 4);
        reserve(t_, uint64_t(d.count) * 4); reserve(invalid_, 4);
        reserve(prefix_, uint64_t(d.frames + 1) * 4); reserve(rows_, dp_bytes); reserve(path_, (1 + uint64_t(d.frames) * 3) * 4);
        if (compact_) { reserve(decisions_, uint64_t(d.frames) * d.count); reserve(endpoint_, 4); }
        if (borrowed_emission == nil) std::memcpy(e_.contents, emission, elements * 4);
        id<MTLBuffer> source_emission = borrowed_emission != nil ? borrowed_emission : e_;
        std::memcpy(t_.contents, tokens, uint64_t(d.count) * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        std::vector<Timing> timings;
        timings.push_back(execute(queue_, wildcard ? wildcard_validate_ : pipelines_[0], @[source_emission, t_, invalid_], d, 2, elements, 256, wildcard ? @"ctc_validate_wildcard" : names_[0], borrowed_offset));
        require(*static_cast<uint32_t*>(invalid_.contents) == 0, "input_values_invalid");
        id<MTLBuffer> input = wildcard ? expanded_ : source_emission;
        const uint64_t input_offset = wildcard ? 0 : borrowed_offset;
        if (wildcard) ++d.vocabulary;
        NSArray<id<MTLBuffer>>* trellis = compact_ ? @[input, t_, prefix_, rows_, decisions_, endpoint_] : @[input, t_, prefix_, rows_];
        NSArray<id<MTLBuffer>>* backtrack = compact_ ? @[input, t_, rows_, path_, decisions_, endpoint_] : @[input, t_, rows_, path_];
        if (batched_) {
            auto started = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "metal_command_creation_failed");
            command.label = wildcard ? @"ctc_wildcard_prefix_trellis_backtrack" : @"ctc_prefix_trellis_backtrack";
            // Default tracked buffers preserve dependencies between sequential encoders.
            if (wildcard) encode(command, wildcard_expand_, @[source_emission, expanded_], source, 1, uint64_t(d.frames) * 256, 256, @"ctc_expand_wildcard", borrowed_offset);
            encode(command, pipelines_[1], @[input, prefix_], d, 1, 1, 1, names_[1], input_offset);
            encode(command, pipelines_[2], trellis, d, 2, 256, 256, names_[2], input_offset);
            encode(command, pipelines_[3], backtrack, d, 2, 1, 1, names_[3], input_offset);
            if (segments) encode(command, segment_pipeline_, @[path_, segment_bounds_, segment_means_], d, 1, d.count, 256, @"ctc_merge_repeats");
            timings.push_back(complete(command, started));
        } else {
            timings.push_back(execute(queue_, pipelines_[1], @[input, prefix_], d, 1, 1, 1, names_[1], input_offset));
            timings.push_back(execute(queue_, pipelines_[2], trellis, d, 2, 256, 256, names_[2], input_offset));
            timings.push_back(execute(queue_, pipelines_[3], backtrack, d, 2, 1, 1, names_[3], input_offset));
        }
        const float* packed = static_cast<const float*>(path_.contents);
        require(std::isfinite(packed[0]) && packed[0] >= -1 && packed[0] <= d.frames, "gpu_path_invalid");
        int32_t length = static_cast<int32_t>(packed[0]);
        NativeResult result{{}, length >= 0, timings, dp_bytes + (compact_ ? uint64_t(d.frames) * d.count + 4 : 0), allocated_bytes(), {}, length};
        if (length >= 0 && !segments) {
            result.path.reserve(length);
            for (int32_t i = length - 1; i >= 0; --i) result.path.push_back({static_cast<int32_t>(packed[1 + i * 3]), static_cast<int32_t>(packed[2 + i * 3]), packed[3 + i * 3]});
        }
        if (length >= 0 && segments && !resident) {
            const int32_t* bounds = static_cast<const int32_t*>(segment_bounds_.contents);
            const uint64_t* means = static_cast<const uint64_t*>(segment_means_.contents);
            for (uint32_t token = 0; token < d.count; ++token) {
                double score; std::memcpy(&score, means + token, sizeof(score));
                require(bounds[token * 2] >= 0 && bounds[token * 2 + 1] > bounds[token * 2] && std::isfinite(score), "gpu_character_segment_invalid");
                result.segments.push_back({bounds[token * 2], bounds[token * 2 + 1], score});
            }
        }
        if (length >= 0 && resident) {
            require(generation_counter_ != UINT64_MAX, "resident_generation_exhausted");
            resident_generation_ = ++generation_counter_; resident_count_ = d.count; resident_frames_ = d.frames;
            result.generation = resident_generation_;
        }
        return result;
    }

    NativeDirectStatsResult scale_round_segments(uint64_t generation, uint32_t count, uint64_t ratio, uint64_t offset, uint64_t* output, uint32_t capacity) {
        std::lock_guard<std::mutex> guard(mutex_);
        rounded_generation_ = 0;
        require(generation != 0 && generation == resident_generation_ && count == resident_count_, "resident_segment_generation_invalid");
        require(output != nullptr && capacity >= uint64_t(count) * 3, "resident_segment_output_invalid");
        uint64_t ratio_exponent = ratio >> 52, offset_exponent = offset >> 52;
        require(ratio_exponent >= 1000 && ratio_exponent <= 1033 && (offset == 0 || (offset_exponent >= 1000 && offset_exponent <= 1063)), "resident_segment_scale_domain_invalid");
        pipeline(scale_round_pipeline_, @"ctc_scale_round_segments");
        reserve(scale_parameters_, 16); reserve(rounded_segments_, uint64_t(count + 1) * 24); reserve(invalid_, 4);
        uint64_t scale[]{ratio, offset}; std::memcpy(scale_parameters_.contents, scale, sizeof(scale));
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        auto timing = execute(queue_, scale_round_pipeline_, @[segment_bounds_, segment_means_, scale_parameters_, rounded_segments_, invalid_], {resident_frames_, 0, count, 0}, 1, count, 256, @"ctc_scale_round_segments");
        require(*static_cast<uint32_t*>(invalid_.contents) == 0, "resident_character_segment_invalid");
        std::memcpy(output, rounded_segments_.contents, uint64_t(count) * 24);
        rounded_generation_ = generation; rounded_ratio_ = ratio; rounded_offset_ = offset;
        return {timing, allocated_bytes()};
    }

    NativeBatchBoundsResult timestamp_bounds_batch(const uint64_t* values, uint64_t count, const uint32_t* indices, uint64_t index_count, const uint32_t* offsets, uint64_t offset_count) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(values != nullptr && indices != nullptr && offsets != nullptr && count <= 65536 && index_count <= 4 * 1024 * 1024 && offset_count >= 2 && offset_count <= 32769, "timestamp_batch_shape_invalid");
        uint32_t queries = static_cast<uint32_t>(offset_count - 1);
        pipeline(batch_bounds_pipeline_, @"ctc_timestamp_bounds_batch");
        reserve(bounds_input_, std::max(uint64_t(8), count * 8)); reserve(bounds_output_, uint64_t(queries) * 16);
        reserve(bounds_indices_, std::max(uint64_t(4), index_count * 4)); reserve(bounds_offsets_, offset_count * 4); reserve(invalid_, 4);
        if (count) std::memcpy(bounds_input_.contents, values, count * 8);
        if (index_count) std::memcpy(bounds_indices_.contents, indices, index_count * 4);
        std::memcpy(bounds_offsets_.contents, offsets, offset_count * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        auto timing = execute(queue_, batch_bounds_pipeline_, @[bounds_input_, bounds_indices_, bounds_offsets_, bounds_output_, invalid_], {queries, static_cast<uint32_t>(count), static_cast<uint32_t>(index_count), 0}, 1, uint64_t(queries) * 256, 256, @"ctc_timestamp_bounds_batch");
        uint32_t invalid = *static_cast<uint32_t*>(invalid_.contents);
        require(!(invalid & 1), "timestamp_batch_indices_invalid"); require(!(invalid & 2), "timestamp_negative_zero_not_supported");
        const uint64_t* output = static_cast<const uint64_t*>(bounds_output_.contents);
        return {{output, output + uint64_t(queries) * 2}, timing, allocated_bytes()};
    }

    NativeBatchBoundsResult word_means_batch(const uint64_t* values, uint64_t count, const uint32_t* indices, uint64_t index_count, const uint32_t* offsets, uint64_t offset_count) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(values != nullptr && indices != nullptr && offsets != nullptr && count <= 32768 && index_count <= 4 * 1024 * 1024 && offset_count >= 2 && offset_count <= 32769, "word_means_shape_invalid");
        uint32_t queries = static_cast<uint32_t>(offset_count - 1);
        pipeline(word_means_pipeline_, @"ctc_word_means_batch");
        reserve(bounds_input_, std::max(uint64_t(8), count * 8)); reserve(bounds_output_, uint64_t(queries) * 8);
        reserve(bounds_indices_, std::max(uint64_t(4), index_count * 4)); reserve(bounds_offsets_, offset_count * 4); reserve(invalid_, 4);
        if (count) std::memcpy(bounds_input_.contents, values, count * 8);
        if (index_count) std::memcpy(bounds_indices_.contents, indices, index_count * 4);
        std::memcpy(bounds_offsets_.contents, offsets, offset_count * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        auto timing = execute(queue_, word_means_pipeline_, @[bounds_input_, bounds_indices_, bounds_offsets_, bounds_output_, invalid_], {queries, static_cast<uint32_t>(count), static_cast<uint32_t>(index_count), 0}, 1, queries, 256, @"ctc_word_means_batch");
        uint32_t invalid = *static_cast<uint32_t*>(invalid_.contents);
        require(!(invalid & 1), "word_means_indices_invalid"); require(!(invalid & 2), "word_means_values_invalid");
        const uint64_t* output = static_cast<const uint64_t*>(bounds_output_.contents);
        return {{output, output + queries}, timing, allocated_bytes()};
    }

    NativeDirectStatsResult alignment_stats_batch(const uint64_t* values, uint64_t elements, uint32_t rows, const uint32_t* indices, uint64_t index_count, const uint32_t* offsets, uint64_t offset_count, const uint32_t* modes, uint64_t mode_count, uint64_t* caller_output, uint64_t output_capacity, uint64_t generation = 0, uint64_t ratio = 0, uint64_t offset = 0, bool word_rounding = false) {
        std::lock_guard<std::mutex> guard(mutex_);
        bool resident = generation != 0;
        if (resident) require(generation == resident_generation_ && generation == rounded_generation_ && ratio == rounded_ratio_ && offset == rounded_offset_ && rows == resident_count_ + 1, "resident_statistics_generation_or_scale_invalid");
        require((resident || values != nullptr) && indices != nullptr && offsets != nullptr && modes != nullptr && rows <= (resident ? 32769u : 32768u) && elements == uint64_t(rows) * 3 && index_count <= 4 * 1024 * 1024 && offset_count >= 2 && offset_count <= 32769 && mode_count == offset_count - 1, "alignment_stats_shape_invalid");
        uint32_t queries = static_cast<uint32_t>(mode_count);
        require(caller_output != nullptr && output_capacity >= uint64_t(queries) * (word_rounding ? 4 : 3), "alignment_stats_output_invalid");
        pipeline(alignment_stats_pipeline_, @"ctc_alignment_stats_batch");
        if (!resident) reserve(bounds_input_, std::max(uint64_t(8), elements * 8));
        reserve(bounds_output_, uint64_t(queries) * 24);
        reserve(bounds_indices_, std::max(uint64_t(4), index_count * 4)); reserve(bounds_offsets_, offset_count * 4); reserve(stats_modes_, mode_count * 4); reserve(invalid_, 4);
        if (!resident && elements) std::memcpy(bounds_input_.contents, values, elements * 8);
        if (index_count) std::memcpy(bounds_indices_.contents, indices, index_count * 4);
        std::memcpy(bounds_offsets_.contents, offsets, offset_count * 4); std::memcpy(stats_modes_.contents, modes, mode_count * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        Timing timing;
        if (word_rounding) {
            require(resident, "word_rounding_requires_resident_statistics");
            pipeline(word_round_pipeline_, @"ctc_round_word_means"); reserve(rounded_words_, uint64_t(queries) * 8);
            auto started = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "metal_command_creation_failed");
            command.label = @"ctc_resident_stats_and_word_rounding";
            encode(command, alignment_stats_pipeline_, @[rounded_segments_, bounds_indices_, bounds_offsets_, stats_modes_, bounds_output_, invalid_], {queries, rows, static_cast<uint32_t>(index_count), 0}, 1, uint64_t(queries) * 256, 256, @"ctc_alignment_stats_batch");
            encode(command, word_round_pipeline_, @[bounds_output_, rounded_words_], {0, 0, queries, 0}, 1, queries, 256, @"ctc_round_word_means");
            timing = complete(command, started);
        } else timing = execute(queue_, alignment_stats_pipeline_, @[resident ? rounded_segments_ : bounds_input_, bounds_indices_, bounds_offsets_, stats_modes_, bounds_output_, invalid_], {queries, rows, static_cast<uint32_t>(index_count), 0}, 1, uint64_t(queries) * 256, 256, @"ctc_alignment_stats_batch");
        uint32_t invalid = *static_cast<uint32_t*>(invalid_.contents);
        require(!(invalid & 1), "alignment_stats_metadata_invalid"); require(!(invalid & 2), "timestamp_negative_zero_not_supported"); require(!(invalid & 4), "word_means_values_invalid");
        const uint64_t* output = static_cast<const uint64_t*>(bounds_output_.contents);
        std::memcpy(caller_output, output, uint64_t(queries) * 24);
        if (word_rounding) std::memcpy(caller_output + uint64_t(queries) * 3, rounded_words_.contents, uint64_t(queries) * 8);
        return {timing, allocated_bytes()};
    }

    NativeNearestResult interpolate_nearest(const uint64_t* values, uint64_t count, uint64_t* output, uint64_t capacity, uint32_t columns = 1) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(values != nullptr && output != nullptr && columns >= 1 && columns <= 2 && count > 0 && count % columns == 0 && count / columns <= 32768 && capacity >= count, "nearest_shape_or_output_invalid");
        pipeline(nearest_neighbors_pipeline_, @"ctc_nearest_neighbors"); pipeline(nearest_fill_pipeline_, @"ctc_nearest_fill");
        reserve(bounds_input_, count * 8); reserve(bounds_output_, count * 8); reserve(nearest_neighbors_, count * 8); reserve(nearest_counts_, columns * 8);
        std::memcpy(bounds_input_.contents, values, count * 8);
        auto started = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "metal_command_creation_failed");
        command.label = @"ctc_nearest_interpolation";
        Dimensions dimensions{static_cast<uint32_t>(count / columns), columns, static_cast<uint32_t>(count), 0};
        encode(command, nearest_neighbors_pipeline_, @[bounds_input_, nearest_neighbors_, nearest_counts_], dimensions, 1, columns, 1, @"ctc_nearest_neighbors");
        encode(command, nearest_fill_pipeline_, @[bounds_input_, nearest_neighbors_, bounds_output_], dimensions, 1, count, 256, @"ctc_nearest_fill");
        auto timing = complete(command, started);
        std::memcpy(output, bounds_output_.contents, count * 8);
        NativeNearestResult result{timing, allocated_bytes(), {}};
        std::memcpy(result.counts, nearest_counts_.contents, columns * 8);
        return result;
    }

    NativeDirectStatsResult round3(const uint64_t* values, uint64_t count, uint64_t* output, uint64_t capacity) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(values != nullptr && output != nullptr && count > 0 && count <= 98304 && capacity >= count, "round3_shape_or_output_invalid");
        pipeline(round3_pipeline_, @"ctc_round3");
        reserve(bounds_input_, count * 8); reserve(bounds_output_, count * 8);
        std::memcpy(bounds_input_.contents, values, count * 8);
        auto timing = execute(queue_, round3_pipeline_, @[bounds_input_, bounds_output_], {0, 0, static_cast<uint32_t>(count), 0}, 1, count, 256, @"ctc_round3");
        std::memcpy(output, bounds_output_.contents, count * 8);
        return {timing, allocated_bytes()};
    }

    NativeBoundsResult timestamp_bounds(const uint64_t* values, uint64_t count) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(values != nullptr && count <= 32768, "timestamp_shape_invalid");
        pipeline(bounds_pipeline_, @"ctc_timestamp_bounds");
        reserve(bounds_input_, std::max(uint64_t(8), count * 8)); reserve(bounds_output_, 24);
        if (count) std::memcpy(bounds_input_.contents, values, count * 8);
        auto timing = execute(queue_, bounds_pipeline_, @[bounds_input_, bounds_output_], {0, 0, static_cast<uint32_t>(count), 0}, 1, 256, 256, @"ctc_timestamp_bounds");
        const uint64_t* output = static_cast<const uint64_t*>(bounds_output_.contents);
        require(output[2] == 0, "timestamp_negative_zero_not_supported");
        return {output[0], output[1], timing, allocated_bytes()};
    }

    NativeMaxResult row_max(const float* emission, uint64_t emission_elements, uint32_t frames, uint32_t vocabulary) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(emission != nullptr && frames > 0 && frames <= 32768 && vocabulary > 0 && vocabulary <= uint64_t(1) << 24, "input_shape_invalid");
        uint64_t elements = uint64_t(frames) * vocabulary;
        require(emission_elements == elements && elements * 4 <= 64 * 1024 * 1024, "input_lengths_or_budget_invalid");
        if (max_pipeline_ == nil) {
            id<MTLFunction> function = [library_ newFunctionWithName:@"ctc_row_max"]; require(function != nil, "row_max_shader_function_missing");
            NSError* error = nil;
            max_pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
            require(max_pipeline_ != nil && max_pipeline_.maxTotalThreadsPerThreadgroup >= 256, "row_max_pipeline_creation_failed");
        }
        reserve(e_, elements * 4); reserve(t_, 4); reserve(invalid_, 4);
        reserve(max_values_, uint64_t(frames) * 4); reserve(max_indices_, uint64_t(frames) * 4);
        std::memcpy(e_.contents, emission, elements * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        Dimensions d{frames, vocabulary, 0, 0};
        std::vector<Timing> timings;
        timings.push_back(execute(queue_, pipelines_[0], @[e_, t_, invalid_], d, 2, elements, 256, names_[0]));
        require(*static_cast<uint32_t*>(invalid_.contents) == 0, "input_values_invalid");
        timings.push_back(execute(queue_, max_pipeline_, @[e_, max_values_, max_indices_], d, 1, uint64_t(frames) * 256, 256, @"ctc_row_max"));
        const float* values = static_cast<const float*>(max_values_.contents);
        const uint32_t* indices = static_cast<const uint32_t*>(max_indices_.contents);
        return {{values, values + frames}, {indices, indices + frames}, timings, allocated_bytes()};
    }
};

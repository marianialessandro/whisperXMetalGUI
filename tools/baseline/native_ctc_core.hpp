#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
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
struct NativeResult { std::vector<NativePoint> path; bool found; std::vector<Timing> timings; uint64_t dp_bytes; uint64_t allocated_bytes; std::vector<NativeSegment> segments; int32_t path_length; };
struct NativeMaxResult { std::vector<float> values; std::vector<uint32_t> indices; std::vector<Timing> timings; uint64_t allocated_bytes; };

static void encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline, NSArray<id<MTLBuffer>>* buffers, const Dimensions& dimensions, NSUInteger dimension_index, NSUInteger grid, NSUInteger group, NSString* label) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "metal_encoder_creation_failed");
    encoder.label = label; [encoder setComputePipelineState:pipeline];
    NSUInteger index = 0;
    for (id<MTLBuffer> buffer in buffers) { if (index == dimension_index) ++index; [encoder setBuffer:buffer offset:0 atIndex:index++]; }
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

static Timing execute(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, NSArray<id<MTLBuffer>>* buffers, const Dimensions& dimensions, NSUInteger dimension_index, NSUInteger grid, NSUInteger group, NSString* label) {
    auto started = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [queue commandBuffer]; require(command != nil, "metal_command_creation_failed");
    command.label = label;
    encode(command, pipeline, buffers, dimensions, dimension_index, grid, group, label);
    return complete(command, started);
}

// Synchronous development context; callers must finish all calls before destroying it.
class NativeCTCContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> max_pipeline_;
    id<MTLComputePipelineState> wildcard_validate_, wildcard_expand_;
    id<MTLComputePipelineState> segment_pipeline_;
    NSMutableArray<id<MTLComputePipelineState>>* pipelines_;
    NSArray<NSString*>* names_;
    id<MTLBuffer> e_, t_, invalid_, prefix_, rows_, path_, decisions_, endpoint_;
    id<MTLBuffer> max_values_, max_indices_;
    id<MTLBuffer> expanded_;
    id<MTLBuffer> segment_bounds_, segment_means_;
    bool compact_, batched_;
    std::mutex mutex_;

    void reserve(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer == nil || buffer.length < bytes) buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil, "metal_allocation_failed");
    }

    uint64_t allocated_bytes() const { return e_.length + t_.length + invalid_.length + prefix_.length + rows_.length + path_.length + decisions_.length + endpoint_.length + max_values_.length + max_indices_.length + expanded_.length + segment_bounds_.length + segment_means_.length; }

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

    NativeResult compute(const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, const Dimensions& source, bool wildcard = false, bool segments = false) {
        std::lock_guard<std::mutex> guard(mutex_);
        Dimensions d = source;
        require(emission != nullptr && tokens != nullptr && d.frames > 0 && d.frames <= 32768 && d.vocabulary > 0 && d.vocabulary <= uint64_t(1) << 24 && d.count > 0 && d.count <= d.frames, "input_shape_invalid");
        uint64_t elements = uint64_t(d.frames) * d.vocabulary, cells = uint64_t(d.frames + 1) * (d.count + 1);
        require(emission_elements == elements && token_elements == d.count, "input_lengths_invalid");
        require(elements * 4 <= 64 * 1024 * 1024 && cells <= 16 * 1024 * 1024, "input_budget_invalid");
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
        reserve(e_, elements * 4); reserve(t_, uint64_t(d.count) * 4); reserve(invalid_, 4);
        reserve(prefix_, uint64_t(d.frames + 1) * 4); reserve(rows_, dp_bytes); reserve(path_, (1 + uint64_t(d.frames) * 3) * 4);
        if (compact_) { reserve(decisions_, uint64_t(d.frames) * d.count); reserve(endpoint_, 4); }
        std::memcpy(e_.contents, emission, elements * 4); std::memcpy(t_.contents, tokens, uint64_t(d.count) * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        std::vector<Timing> timings;
        timings.push_back(execute(queue_, wildcard ? wildcard_validate_ : pipelines_[0], @[e_, t_, invalid_], d, 2, elements, 256, wildcard ? @"ctc_validate_wildcard" : names_[0]));
        require(*static_cast<uint32_t*>(invalid_.contents) == 0, "input_values_invalid");
        id<MTLBuffer> input = wildcard ? expanded_ : e_;
        if (wildcard) ++d.vocabulary;
        NSArray<id<MTLBuffer>>* trellis = compact_ ? @[input, t_, prefix_, rows_, decisions_, endpoint_] : @[input, t_, prefix_, rows_];
        NSArray<id<MTLBuffer>>* backtrack = compact_ ? @[input, t_, rows_, path_, decisions_, endpoint_] : @[input, t_, rows_, path_];
        if (batched_) {
            auto started = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "metal_command_creation_failed");
            command.label = wildcard ? @"ctc_wildcard_prefix_trellis_backtrack" : @"ctc_prefix_trellis_backtrack";
            // Default tracked buffers preserve dependencies between sequential encoders.
            if (wildcard) encode(command, wildcard_expand_, @[e_, expanded_], source, 1, uint64_t(d.frames) * 256, 256, @"ctc_expand_wildcard");
            encode(command, pipelines_[1], @[input, prefix_], d, 1, 1, 1, names_[1]);
            encode(command, pipelines_[2], trellis, d, 2, 256, 256, names_[2]);
            encode(command, pipelines_[3], backtrack, d, 2, 1, 1, names_[3]);
            if (segments) encode(command, segment_pipeline_, @[path_, segment_bounds_, segment_means_], d, 1, d.count, 256, @"ctc_merge_repeats");
            timings.push_back(complete(command, started));
        } else {
            timings.push_back(execute(queue_, pipelines_[1], @[e_, prefix_], d, 1, 1, 1, names_[1]));
            timings.push_back(execute(queue_, pipelines_[2], trellis, d, 2, 256, 256, names_[2]));
            timings.push_back(execute(queue_, pipelines_[3], backtrack, d, 2, 1, 1, names_[3]));
        }
        const float* packed = static_cast<const float*>(path_.contents);
        require(std::isfinite(packed[0]) && packed[0] >= -1 && packed[0] <= d.frames, "gpu_path_invalid");
        int32_t length = static_cast<int32_t>(packed[0]);
        NativeResult result{{}, length >= 0, timings, dp_bytes + (compact_ ? uint64_t(d.frames) * d.count + 4 : 0), allocated_bytes(), {}, length};
        if (length >= 0 && !segments) {
            result.path.reserve(length);
            for (int32_t i = length - 1; i >= 0; --i) result.path.push_back({static_cast<int32_t>(packed[1 + i * 3]), static_cast<int32_t>(packed[2 + i * 3]), packed[3 + i * 3]});
        }
        if (length >= 0 && segments) {
            const int32_t* bounds = static_cast<const int32_t*>(segment_bounds_.contents);
            const uint64_t* means = static_cast<const uint64_t*>(segment_means_.contents);
            for (uint32_t token = 0; token < d.count; ++token) {
                double score; std::memcpy(&score, means + token, sizeof(score));
                require(bounds[token * 2] >= 0 && bounds[token * 2 + 1] > bounds[token * 2] && std::isfinite(score), "gpu_character_segment_invalid");
                result.segments.push_back({bounds[token * 2], bounds[token * 2 + 1], score});
            }
        }
        return result;
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

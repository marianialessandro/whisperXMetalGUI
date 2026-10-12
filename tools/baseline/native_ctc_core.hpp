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
struct NativeResult { std::vector<NativePoint> path; bool found; std::vector<Timing> timings; uint64_t dp_bytes; uint64_t allocated_bytes; };

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
    NSMutableArray<id<MTLComputePipelineState>>* pipelines_;
    NSArray<NSString*>* names_;
    id<MTLBuffer> e_, t_, invalid_, prefix_, rows_, path_, decisions_, endpoint_;
    bool compact_, batched_;
    std::mutex mutex_;

    void reserve(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer == nil || buffer.length < bytes) buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil, "metal_allocation_failed");
    }

public:
    NativeCTCContext(const char* shader, uint64_t shader_bytes, bool compact, bool batched) : compact_(compact), batched_(batched) {
        require(shader != nullptr && shader_bytes > 0 && shader_bytes <= 1024 * 1024, "shader_budget_invalid");
        device_ = MTLCreateSystemDefaultDevice(); require(device_ != nil && [device_ supportsFamily:MTLGPUFamilyApple7], "apple_metal_required");
        NSString* source = [[NSString alloc] initWithBytes:shader length:shader_bytes encoding:NSUTF8StringEncoding]; require(source != nil, "shader_encoding_invalid");
        MTLCompileOptions* compile = [[MTLCompileOptions alloc] init]; compile.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:source options:compile error:&error]; require(library != nil, "shader_compilation_failed");
        pipelines_ = [[NSMutableArray alloc] init];
        names_ = compact ? @[@"ctc_validate", @"ctc_prefix", @"ctc_trellis_compact", @"ctc_backtrack_compact"] : @[@"ctc_validate", @"ctc_prefix", @"ctc_trellis", @"ctc_backtrack"];
        for (NSString* name in names_) {
            id<MTLFunction> function = [library newFunctionWithName:name]; require(function != nil, "shader_function_missing");
            id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error]; require(pipeline != nil && pipeline.maxTotalThreadsPerThreadgroup >= 256, "pipeline_creation_failed");
            [pipelines_ addObject:pipeline];
        }
        queue_ = [device_ newCommandQueue]; require(queue_ != nil, "metal_queue_creation_failed");
    }

    NativeResult compute(const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, const Dimensions& d) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(emission != nullptr && tokens != nullptr && d.frames > 0 && d.frames <= 32768 && d.vocabulary > 0 && d.vocabulary <= uint64_t(1) << 24 && d.count > 0 && d.count <= d.frames, "input_shape_invalid");
        uint64_t elements = uint64_t(d.frames) * d.vocabulary, cells = uint64_t(d.frames + 1) * (d.count + 1);
        require(emission_elements == elements && token_elements == d.count, "input_lengths_invalid");
        require(elements * 4 <= 64 * 1024 * 1024 && cells <= 16 * 1024 * 1024, "input_budget_invalid");
        uint64_t dp_bytes = compact_ ? 2 * uint64_t(d.count + 1) * 4 : cells * 4;
        reserve(e_, elements * 4); reserve(t_, uint64_t(d.count) * 4); reserve(invalid_, 4);
        reserve(prefix_, uint64_t(d.frames + 1) * 4); reserve(rows_, dp_bytes); reserve(path_, (1 + uint64_t(d.frames) * 3) * 4);
        if (compact_) { reserve(decisions_, uint64_t(d.frames) * d.count); reserve(endpoint_, 4); }
        std::memcpy(e_.contents, emission, elements * 4); std::memcpy(t_.contents, tokens, uint64_t(d.count) * 4);
        *static_cast<uint32_t*>(invalid_.contents) = 0;
        std::vector<Timing> timings;
        timings.push_back(execute(queue_, pipelines_[0], @[e_, t_, invalid_], d, 2, elements, 256, names_[0]));
        require(*static_cast<uint32_t*>(invalid_.contents) == 0, "input_values_invalid");
        NSArray<id<MTLBuffer>>* trellis = compact_ ? @[e_, t_, prefix_, rows_, decisions_, endpoint_] : @[e_, t_, prefix_, rows_];
        NSArray<id<MTLBuffer>>* backtrack = compact_ ? @[e_, t_, rows_, path_, decisions_, endpoint_] : @[e_, t_, rows_, path_];
        if (batched_) {
            auto started = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "metal_command_creation_failed");
            command.label = @"ctc_prefix_trellis_backtrack";
            // Default tracked buffers preserve dependencies between sequential encoders.
            encode(command, pipelines_[1], @[e_, prefix_], d, 1, 1, 1, names_[1]);
            encode(command, pipelines_[2], trellis, d, 2, 256, 256, names_[2]);
            encode(command, pipelines_[3], backtrack, d, 2, 1, 1, names_[3]);
            timings.push_back(complete(command, started));
        } else {
            timings.push_back(execute(queue_, pipelines_[1], @[e_, prefix_], d, 1, 1, 1, names_[1]));
            timings.push_back(execute(queue_, pipelines_[2], trellis, d, 2, 256, 256, names_[2]));
            timings.push_back(execute(queue_, pipelines_[3], backtrack, d, 2, 1, 1, names_[3]));
        }
        const float* packed = static_cast<const float*>(path_.contents);
        require(std::isfinite(packed[0]) && packed[0] >= -1 && packed[0] <= d.frames, "gpu_path_invalid");
        int32_t length = static_cast<int32_t>(packed[0]);
        NativeResult result{{}, length >= 0, timings, dp_bytes + (compact_ ? uint64_t(d.frames) * d.count + 4 : 0), e_.length + t_.length + invalid_.length + prefix_.length + rows_.length + path_.length + decisions_.length + endpoint_.length};
        if (length >= 0) {
            result.path.reserve(length);
            for (int32_t i = length - 1; i >= 0; --i) result.path.push_back({static_cast<int32_t>(packed[1 + i * 3]), static_cast<int32_t>(packed[2 + i * 3]), packed[3 + i * 3]});
        }
        return result;
    }
};

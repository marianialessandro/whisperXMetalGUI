#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

struct SpeakerDimensions { uint32_t turns, queries, speakers, nearest; };
struct SpeakerMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
static_assert(sizeof(SpeakerMetrics) == 32);

class SpeakerContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> order_pipeline_, assign_pipeline_;
    id<MTLBuffer> turns_, labels_, queries_, order_, sums_, output_;
    std::mutex mutex_;

    void reserve(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer == nil || buffer.length < bytes) buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil, "speaker_allocation_failed");
    }

    static void validate_times(const double* times, uint32_t count) {
        require(times != nullptr, "speaker_times_missing");
        for (uint32_t i = 0; i < count * 2; ++i) require(std::isfinite(times[i]) && !std::signbit(times[i]) && times[i] <= 0x1p32 && (times[i] == 0 || times[i] >= 0x1p-900), "speaker_timestamp_domain_invalid");
        for (uint32_t i = 0; i < count; ++i) require(times[i * 2] <= times[i * 2 + 1], "speaker_interval_reversed");
    }

public:
    SpeakerContext(const char* source, uint64_t size) {
        require(source != nullptr && size > 0 && size <= 1024 * 1024, "speaker_shader_budget_invalid");
        device_ = MTLCreateSystemDefaultDevice(); require(device_ != nil && [device_ supportsFamily:MTLGPUFamilyApple7], "speaker_apple_metal_required");
        NSString* shader = [[NSString alloc] initWithBytes:source length:size encoding:NSUTF8StringEncoding]; require(shader != nil, "speaker_shader_encoding_invalid");
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init]; options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:shader options:options error:&error]; require(library != nil, "speaker_shader_compilation_failed");
        order_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"speaker_stable_order"] error:&error];
        assign_pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"speaker_assign"] error:&error];
        require(order_pipeline_ != nil && assign_pipeline_ != nil, "speaker_pipeline_creation_failed");
        queue_ = [device_ newCommandQueue]; require(queue_ != nil, "speaker_queue_creation_failed");
    }

    SpeakerMetrics assign(const double* turns, const uint32_t* labels, const double* queries, SpeakerDimensions d, int32_t* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(d.turns > 0 && d.turns <= 4096 && d.queries > 0 && d.queries <= 4096 && d.speakers > 0 && d.speakers <= 256 && d.nearest <= 1, "speaker_shape_budget_invalid");
        require(uint64_t(d.turns) * d.queries <= 16 * 1024 * 1024 && labels != nullptr && output != nullptr, "speaker_work_budget_or_output_invalid");
        validate_times(turns, d.turns); validate_times(queries, d.queries);
        for (uint32_t i = 0; i < d.turns; ++i) require(labels[i] < d.speakers, "speaker_label_out_of_range");
        auto started = std::chrono::steady_clock::now();
        reserve(turns_, uint64_t(d.turns) * 16); reserve(labels_, uint64_t(d.turns) * 4); reserve(order_, uint64_t(d.turns) * 4);
        reserve(queries_, uint64_t(d.queries) * 16); reserve(sums_, uint64_t(d.queries) * d.speakers * 8); reserve(output_, uint64_t(d.queries) * 4);
        std::memcpy(turns_.contents, turns, uint64_t(d.turns) * 16); std::memcpy(labels_.contents, labels, uint64_t(d.turns) * 4); std::memcpy(queries_.contents, queries, uint64_t(d.queries) * 16);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "speaker_command_creation_failed");
        command.label = @"WhisperX native speaker intersections";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "speaker_encoder_creation_failed");
        [encoder setComputePipelineState:order_pipeline_]; [encoder setBuffer:turns_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:order_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.turns, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, order_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "speaker_encoder_creation_failed");
        [encoder setComputePipelineState:assign_pipeline_]; [encoder setBuffer:turns_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:order_ offset:0 atIndex:2]; [encoder setBuffer:labels_ offset:0 atIndex:3]; [encoder setBuffer:queries_ offset:0 atIndex:4]; [encoder setBuffer:sums_ offset:0 atIndex:5]; [encoder setBuffer:output_ offset:0 atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(d.queries, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, assign_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "speaker_command_failed");
        std::memcpy(output, output_.contents, uint64_t(d.queries) * 4);
        uint64_t wall = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        return {1, 2, wall, command.GPUEndTime - command.GPUStartTime, turns_.length + labels_.length + order_.length + queries_.length + sums_.length + output_.length};
    }
};

static void speaker_error(char* output, uint32_t capacity, const char* code) { if (output != nullptr && capacity) { std::strncpy(output, code, capacity - 1); output[capacity - 1] = 0; } }
extern "C" int wxspeaker_create(const char* source, uint64_t size, void** output, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(output != nullptr, "speaker_context_output_missing"); *output = nullptr; *output = new SpeakerContext(source, size); return 0; }
        catch (const InventoryError& failure) { speaker_error(error, capacity, failure.what()); return 1; }
        catch (...) { speaker_error(error, capacity, "speaker_creation_failed"); return 1; }
    }
}
extern "C" int wxspeaker_assign(void* context, const double* turns, const uint32_t* labels, const double* queries, uint32_t turn_count, uint32_t query_count, uint32_t speakers, uint32_t nearest, int32_t* output, SpeakerMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "speaker_context_or_metrics_missing"); *metrics = static_cast<SpeakerContext*>(context)->assign(turns, labels, queries, {turn_count, query_count, speakers, nearest}, output); return 0; }
        catch (const InventoryError& failure) { speaker_error(error, capacity, failure.what()); return 1; }
        catch (...) { speaker_error(error, capacity, "speaker_execution_failed"); return 1; }
    }
}
extern "C" void wxspeaker_destroy(void* context) { @autoreleasepool { delete static_cast<SpeakerContext*>(context); } }

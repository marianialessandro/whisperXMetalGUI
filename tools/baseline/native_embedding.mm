#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

struct EmbeddingDimensions { uint32_t rows, columns; };
struct EmbeddingMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
static_assert(sizeof(EmbeddingMetrics) == 32);

class EmbeddingContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    id<MTLBuffer> input_, output_;
    std::mutex mutex_;
    void reserve(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer == nil || buffer.length < bytes) buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil, "embedding_allocation_failed");
    }
public:
    EmbeddingContext(const char* source, uint64_t size) {
        require(source != nullptr && size > 0 && size <= 1024 * 1024, "embedding_shader_budget_invalid");
        device_ = MTLCreateSystemDefaultDevice(); require(device_ != nil && [device_ supportsFamily:MTLGPUFamilyApple7], "embedding_apple_metal_required");
        NSString* shader = [[NSString alloc] initWithBytes:source length:size encoding:NSUTF8StringEncoding]; require(shader != nil, "embedding_shader_encoding_invalid");
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init]; options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:shader options:options error:&error]; require(library != nil, "embedding_shader_compilation_failed");
        pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"embedding_normalize"] error:&error]; require(pipeline_ != nil, "embedding_pipeline_creation_failed");
        queue_ = [device_ newCommandQueue]; require(queue_ != nil, "embedding_queue_creation_failed");
    }
    EmbeddingMetrics normalize(const float* input, EmbeddingDimensions d, float* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && input != nullptr && output != nullptr, "embedding_shape_or_pointer_invalid");
        for (uint32_t row = 0; row < d.rows; ++row) {
            bool nonzero = false;
            for (uint32_t column = 0; column < d.columns; ++column) {
                float value = input[uint64_t(row) * d.columns + column];
                require(std::isfinite(value) && std::abs(value) <= 0x1p32f && (value == 0 || std::abs(value) >= 0x1p-32f), "embedding_value_domain_invalid");
                nonzero |= value != 0;
            }
            require(nonzero, "embedding_zero_row_not_supported");
        }
        auto started = std::chrono::steady_clock::now();
        uint64_t bytes = uint64_t(d.rows) * d.columns * sizeof(float);
        reserve(input_, bytes); reserve(output_, bytes); std::memcpy(input_.contents, input, bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "embedding_command_creation_failed");
        command.label = @"WhisperX experimental embedding normalization";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "embedding_encoder_creation_failed");
        [encoder setComputePipelineState:pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "embedding_command_failed");
        std::memcpy(output, output_.contents, bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length};
    }
};
static void embedding_error(char* output, uint32_t capacity, const char* code) { if (output != nullptr && capacity) { std::strncpy(output, code, capacity - 1); output[capacity - 1] = 0; } }
extern "C" int wxembedding_create(const char* source, uint64_t size, void** output, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(output != nullptr, "embedding_context_output_missing"); *output = nullptr; *output = new EmbeddingContext(source, size); return 0; }
        catch (const InventoryError& failure) { embedding_error(error, capacity, failure.what()); return 1; }
        catch (...) { embedding_error(error, capacity, "embedding_creation_failed"); return 1; }
    }
}
extern "C" int wxembedding_normalize(void* context, const float* input, uint32_t rows, uint32_t columns, float* output, EmbeddingMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "embedding_context_or_metrics_missing"); *metrics = static_cast<EmbeddingContext*>(context)->normalize(input, {rows, columns}, output); return 0; }
        catch (const InventoryError& failure) { embedding_error(error, capacity, failure.what()); return 1; }
        catch (...) { embedding_error(error, capacity, "embedding_execution_failed"); return 1; }
    }
}
extern "C" void wxembedding_destroy(void* context) { @autoreleasepool { delete static_cast<EmbeddingContext*>(context); } }

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

struct DoubleDimensions { uint32_t rows, columns, sequential; };
struct DoubleMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
static_assert(sizeof(DoubleMetrics) == 32);

class DoubleContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    id<MTLBuffer> input_, output_;
    std::mutex mutex_;
    void reserve(id<MTLBuffer> __strong& buffer, uint64_t bytes) {
        if (buffer == nil || buffer.length < bytes) buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil, "double_allocation_failed");
    }
public:
    DoubleContext(const char* source, uint64_t size) {
        require(source != nullptr && size > 0 && size <= 1024 * 1024, "double_shader_budget_invalid");
        device_ = MTLCreateSystemDefaultDevice(); require(device_ != nil && [device_ supportsFamily:MTLGPUFamilyApple7], "double_apple_metal_required");
        NSString* shader = [[NSString alloc] initWithBytes:source length:size encoding:NSUTF8StringEncoding]; require(shader != nil, "double_shader_encoding_invalid");
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init]; options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:shader options:options error:&error]; require(library != nil, "double_shader_compilation_failed");
        pipeline_ = [device_ newComputePipelineStateWithFunction:[library newFunctionWithName:@"double_normalize"] error:&error]; require(pipeline_ != nil, "double_pipeline_creation_failed");
        queue_ = [device_ newCommandQueue]; require(queue_ != nil, "double_queue_creation_failed");
    }
    DoubleMetrics normalize(const double* input, DoubleDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.sequential <= 1 && input != nullptr && output != nullptr, "double_shape_or_pointer_invalid");
        for (uint32_t row = 0; row < d.rows; ++row) {
            bool nonzero = false;
            for (uint32_t column = 0; column < d.columns; ++column) {
                double value = input[uint64_t(row) * d.columns + column];
                require(std::isfinite(value) && std::abs(value) <= 0x1p200 && (value == 0 || std::abs(value) >= 0x1p-200), "double_value_domain_invalid");
                nonzero |= value != 0;
            }
            require(nonzero, "double_zero_row_not_supported");
        }
        auto started = std::chrono::steady_clock::now();
        uint64_t bytes = uint64_t(d.rows) * d.columns * sizeof(double);
        reserve(input_, bytes); reserve(output_, bytes); std::memcpy(input_.contents, input, bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental binary64 normalization";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length};
    }
};
static void double_error(char* output, uint32_t capacity, const char* code) { if (output != nullptr && capacity) { std::strncpy(output, code, capacity - 1); output[capacity - 1] = 0; } }
extern "C" int wxdouble_create(const char* source, uint64_t size, void** output, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(output != nullptr, "double_context_output_missing"); *output = nullptr; *output = new DoubleContext(source, size); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_creation_failed"); return 1; }
    }
}
extern "C" int wxdouble_normalize(void* context, const double* input, uint32_t rows, uint32_t columns, uint32_t sequential, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->normalize(input, {rows, columns, sequential}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_execution_failed"); return 1; }
    }
}
extern "C" void wxdouble_destroy(void* context) { @autoreleasepool { delete static_cast<DoubleContext*>(context); } }

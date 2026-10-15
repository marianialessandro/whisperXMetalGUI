#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

struct DoubleDimensions { uint32_t rows, columns, sequential; };
struct MatrixDimensions { uint32_t rows, inner, columns; };
struct ElementDimensions { uint32_t rows, columns, operation, scale_dimension; };
struct DoubleMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
static_assert(sizeof(DoubleMetrics) == 32);

class DoubleContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_, matmul_pipeline_, element_pipeline_;
    id<MTLBuffer> input_, output_, right_;
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
        id<MTLFunction> matmul_function = [library newFunctionWithName:@"double_matmul"];
        if (matmul_function != nil) matmul_pipeline_ = [device_ newComputePipelineStateWithFunction:matmul_function error:&error];
        id<MTLFunction> element_function = [library newFunctionWithName:@"double_center_scale"];
        if (element_function != nil) element_pipeline_ = [device_ newComputePipelineStateWithFunction:element_function error:&error];
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
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length};
    }
    DoubleMetrics matmul(const double* left, const double* right, MatrixDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(matmul_pipeline_ != nil, "double_matrix_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.inner > 0 && d.inner <= 1024 && d.columns > 0 && d.columns <= 1024, "double_matrix_shape_invalid");
        require(uint64_t(d.rows) * d.inner * d.columns <= 32 * 1024 * 1024 && left != nullptr && right != nullptr && output != nullptr, "double_matrix_work_or_pointer_invalid");
        for (uint64_t i = 0; i < uint64_t(d.rows) * d.inner; ++i) require(std::isfinite(left[i]) && std::abs(left[i]) <= 0x1p100 && (left[i] == 0 || std::abs(left[i]) >= 0x1p-100), "double_matrix_value_domain_invalid");
        for (uint64_t i = 0; i < uint64_t(d.inner) * d.columns; ++i) require(std::isfinite(right[i]) && std::abs(right[i]) <= 0x1p100 && (right[i] == 0 || std::abs(right[i]) >= 0x1p-100), "double_matrix_value_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        uint64_t left_bytes = uint64_t(d.rows) * d.inner * sizeof(double), right_bytes = uint64_t(d.inner) * d.columns * sizeof(double), output_bytes = uint64_t(d.rows) * d.columns * sizeof(double);
        reserve(input_, left_bytes); reserve(right_, right_bytes); reserve(output_, output_bytes);
        std::memcpy(input_.contents, left, left_bytes); std::memcpy(right_.contents, right, right_bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental binary64 matrix product";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:matmul_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(uint64_t(d.rows) * d.columns, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, matmul_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, output_bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length};
    }

    DoubleMetrics elementwise(const double* input, const double* mean, ElementDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(element_pipeline_ != nil, "double_element_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.operation <= 1 && input != nullptr && output != nullptr, "double_element_shape_or_pointer_invalid");
        require(d.operation == 0 ? mean != nullptr && d.scale_dimension == 0 : mean == nullptr && d.scale_dimension > 0 && d.scale_dimension <= 1024, "double_element_operation_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns, bytes = count * sizeof(double), mean_bytes = d.operation == 0 ? uint64_t(d.columns) * sizeof(double) : 0;
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]) && std::abs(input[i]) <= 0x1p450 && (input[i] == 0 || std::abs(input[i]) >= 0x1p-450), "double_element_value_domain_invalid");
        for (uint32_t i = 0; i < (d.operation == 0 ? d.columns : 0); ++i) require(std::isfinite(mean[i]) && std::abs(mean[i]) <= 0x1p450 && (mean[i] == 0 || std::abs(mean[i]) >= 0x1p-450), "double_element_value_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, bytes); reserve(output_, bytes); reserve(right_, mean_bytes ? mean_bytes : sizeof(double));
        std::memcpy(input_.contents, input, bytes); if (mean_bytes) std::memcpy(right_.contents, mean, mean_bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental binary64 centering and scaling";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:element_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, element_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length};
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
extern "C" int wxdouble_matmul(void* context, const double* left, const double* right, uint32_t rows, uint32_t inner, uint32_t columns, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->matmul(left, right, {rows, inner, columns}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_matrix_execution_failed"); return 1; }
    }
}
extern "C" int wxdouble_elementwise(void* context, const double* input, const double* mean, uint32_t rows, uint32_t columns, uint32_t operation, uint32_t scale_dimension, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->elementwise(input, mean, {rows, columns, operation, scale_dimension}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_element_execution_failed"); return 1; }
    }
}
extern "C" void wxdouble_destroy(void* context) { @autoreleasepool { delete static_cast<DoubleContext*>(context); } }

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
struct StatisticsDimensions { uint32_t rows, columns, features, operation, sequential, reserved; uint64_t fa, fb; };
static_assert(sizeof(StatisticsDimensions) == 40);
struct ScoreDimensions { uint32_t rows, columns, operation, reserved; uint64_t fa; };
static_assert(sizeof(ScoreDimensions) == 24);
struct ElboDimensions { uint32_t rows, columns, operation, column_order; uint64_t fb, likelihood; };
static_assert(sizeof(ElboDimensions) == 32);
struct TranscendentalDimensions { uint32_t count, operation; };
struct LogSumExpDimensions { uint32_t rows, columns, sequential; };
struct QinitDimensions { uint32_t rows, columns; uint64_t smoothing; };
struct InitializationDimensions { uint32_t rows, columns, operation, sequential; };
struct DoubleMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
static_assert(sizeof(DoubleMetrics) == 32);

class DoubleContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_, matmul_pipeline_, element_pipeline_, terms_pipeline_, rho_pipeline_, gamma_pipeline_, statistics_pipeline_, score_pipeline_, prior_pipeline_, posterior_pipeline_, likelihood_pipeline_, elbo_terms_pipeline_, elbo_final_pipeline_, transcendental_pipeline_, log1p_pipeline_, lse_terms_pipeline_, lse_final_pipeline_, initialization_pipeline_, qinit_pipeline_;
    id<MTLBuffer> input_, output_, right_, terms_;
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
        id<MTLFunction> terms_function = [library newFunctionWithName:@"double_vbx_terms"], rho_function = [library newFunctionWithName:@"double_vbx_rho"];
        if (terms_function != nil) terms_pipeline_ = [device_ newComputePipelineStateWithFunction:terms_function error:&error];
        if (rho_function != nil) rho_pipeline_ = [device_ newComputePipelineStateWithFunction:rho_function error:&error];
        id<MTLFunction> gamma_function = [library newFunctionWithName:@"double_vbx_gamma_sum"], statistics_function = [library newFunctionWithName:@"double_vbx_statistics"];
        if (gamma_function != nil) gamma_pipeline_ = [device_ newComputePipelineStateWithFunction:gamma_function error:&error];
        if (statistics_function != nil) statistics_pipeline_ = [device_ newComputePipelineStateWithFunction:statistics_function error:&error];
        id<MTLFunction> score_function = [library newFunctionWithName:@"double_vbx_scores"];
        if (score_function != nil) score_pipeline_ = [device_ newComputePipelineStateWithFunction:score_function error:&error];
        id<MTLFunction> prior_function = [library newFunctionWithName:@"double_vbx_prior"];
        if (prior_function != nil) prior_pipeline_ = [device_ newComputePipelineStateWithFunction:prior_function error:&error];
        id<MTLFunction> posterior_function = [library newFunctionWithName:@"double_vbx_posterior"];
        if (posterior_function != nil) posterior_pipeline_ = [device_ newComputePipelineStateWithFunction:posterior_function error:&error];
        id<MTLFunction> likelihood_function = [library newFunctionWithName:@"double_vbx_likelihood"], elbo_terms_function = [library newFunctionWithName:@"double_vbx_elbo_terms"], elbo_final_function = [library newFunctionWithName:@"double_vbx_elbo_final"];
        if (likelihood_function != nil) likelihood_pipeline_ = [device_ newComputePipelineStateWithFunction:likelihood_function error:&error];
        if (elbo_terms_function != nil) elbo_terms_pipeline_ = [device_ newComputePipelineStateWithFunction:elbo_terms_function error:&error];
        if (elbo_final_function != nil) elbo_final_pipeline_ = [device_ newComputePipelineStateWithFunction:elbo_final_function error:&error];
        id<MTLFunction> transcendental_function = [library newFunctionWithName:@"double_vbx_transcendental"];
        if (transcendental_function != nil) transcendental_pipeline_ = [device_ newComputePipelineStateWithFunction:transcendental_function error:&error];
        id<MTLFunction> log1p_function = [library newFunctionWithName:@"double_vbx_log1p"];
        if (log1p_function != nil) log1p_pipeline_ = [device_ newComputePipelineStateWithFunction:log1p_function error:&error];
        id<MTLFunction> lse_terms_function = [library newFunctionWithName:@"double_vbx_lse_terms"], lse_final_function = [library newFunctionWithName:@"double_vbx_lse_final"];
        if (lse_terms_function != nil) lse_terms_pipeline_ = [device_ newComputePipelineStateWithFunction:lse_terms_function error:&error];
        if (lse_final_function != nil) lse_final_pipeline_ = [device_ newComputePipelineStateWithFunction:lse_final_function error:&error];
        id<MTLFunction> qinit_function = [library newFunctionWithName:@"double_vbx_qinit_exp"];
        if (qinit_function != nil) qinit_pipeline_ = [device_ newComputePipelineStateWithFunction:qinit_function error:&error];
        id<MTLFunction> initialization_function = [library newFunctionWithName:@"double_vbx_initialize"];
        if (initialization_function != nil) initialization_pipeline_ = [device_ newComputePipelineStateWithFunction:initialization_function error:&error];
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
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
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
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
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
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics precompute_vbx(const double* input, const double* phi, DoubleDimensions d, double* g, double* v, double* rho) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(terms_pipeline_ != nil && rho_pipeline_ != nil, "double_vbx_kernels_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.sequential <= 1 && input != nullptr && phi != nullptr && g != nullptr && v != nullptr && rho != nullptr, "double_vbx_shape_or_pointer_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns, bytes = count * sizeof(double);
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]) && std::abs(input[i]) <= 0x1p200 && (input[i] == 0 || std::abs(input[i]) >= 0x1p-200), "double_vbx_value_domain_invalid");
        for (uint32_t i = 0; i < d.columns; ++i) require(std::isfinite(phi[i]) && phi[i] >= 0 && phi[i] <= 0x1p200 && (phi[i] == 0 || phi[i] >= 0x1p-200), "double_vbx_phi_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, bytes); reserve(output_, bytes); reserve(right_, uint64_t(d.columns) * sizeof(double)); reserve(terms_, uint64_t(d.rows + d.columns) * sizeof(double));
        std::memcpy(input_.contents, input, bytes); std::memcpy(right_.contents, phi, uint64_t(d.columns) * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx precomputed terms";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:terms_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:terms_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(MAX(d.rows, d.columns), 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, terms_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:rho_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, rho_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(g, terms_.contents, uint64_t(d.rows) * sizeof(double)); std::memcpy(v, static_cast<const double*>(terms_.contents) + d.rows, uint64_t(d.columns) * sizeof(double)); std::memcpy(rho, output_.contents, bytes);
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics statistics_vbx(const double* input, const double* right, StatisticsDimensions d, double fa, double fb, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(statistics_pipeline_ != nil && (d.operation != 0 || gamma_pipeline_ != nil), "double_vbx_statistics_kernels_unavailable");
        require(d.rows > 0 && d.rows <= 1024 && d.columns > 0 && d.columns <= 4096 && d.features > 0 && d.features <= 1024 && d.operation <= 1 && d.sequential <= 1 && input != nullptr && right != nullptr && output != nullptr, "double_vbx_statistics_shape_or_pointer_invalid");
        require(d.operation == 0 || d.columns == d.features, "double_vbx_statistics_shape_invalid");
        require(std::isfinite(fa) && std::isfinite(fb) && fa >= 0x1p-20 && fa <= 0x1p20 && fb >= 0x1p-20 && fb <= 0x1p20, "double_vbx_statistics_scalar_domain_invalid");
        std::memcpy(&d.fa, &fa, sizeof(fa)); std::memcpy(&d.fb, &fb, sizeof(fb));
        uint64_t count = uint64_t(d.rows) * d.columns, out_count = uint64_t(d.rows) * d.features, right_count = d.operation == 0 ? d.features : out_count;
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]) && (d.operation == 0 ? input[i] >= 0 : input[i] > 0) && input[i] <= 1 && (input[i] == 0 || input[i] >= 0x1p-160), "double_vbx_statistics_input_domain_invalid");
        for (uint64_t i = 0; i < right_count; ++i) require(std::isfinite(right[i]) && (d.operation != 0 || right[i] >= 0) && std::abs(right[i]) <= 0x1p100 && (right[i] == 0 || std::abs(right[i]) >= 0x1p-100), "double_vbx_statistics_right_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(right_, right_count * sizeof(double)); reserve(output_, out_count * sizeof(double)); reserve(terms_, uint64_t(d.rows) * sizeof(double));
        std::memcpy(input_.contents, input, count * sizeof(double)); std::memcpy(right_.contents, right, right_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx statistics";
        if (d.operation == 0) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
            [encoder setComputePipelineState:gamma_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, gamma_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:statistics_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:terms_ offset:0 atIndex:3]; [encoder setBuffer:output_ offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(out_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, statistics_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, out_count * sizeof(double));
        return {1, d.operation == 0 ? 2u : 1u, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics scores_vbx(const double* input, const double* right, ScoreDimensions d, double fa, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(score_pipeline_ != nil, "double_vbx_scores_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.operation <= 1 && input != nullptr && right != nullptr && output != nullptr, "double_vbx_scores_shape_or_pointer_invalid");
        require(d.operation != 0 || d.rows <= 1024, "double_vbx_scores_speaker_budget_invalid");
        require(std::isfinite(fa) && fa >= 0x1p-20 && fa <= 0x1p20 && (d.operation != 0 || fa == 1.), "double_vbx_scores_scalar_domain_invalid");
        std::memcpy(&d.fa, &fa, sizeof(fa));
        uint64_t count = uint64_t(d.rows) * d.columns, right_count = d.operation == 0 ? count : uint64_t(d.rows) + d.columns;
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]) && (d.operation != 0 || input[i] >= 0) && std::abs(input[i]) <= 0x1p200 && (input[i] == 0 || std::abs(input[i]) >= 0x1p-200), "double_vbx_scores_input_domain_invalid");
        for (uint64_t i = 0; i < right_count; ++i) require(std::isfinite(right[i]) && std::abs(right[i]) <= 0x1p200 && (right[i] == 0 || std::abs(right[i]) >= 0x1p-200), "double_vbx_scores_right_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(right_, right_count * sizeof(double)); reserve(output_, count * sizeof(double));
        std::memcpy(input_.contents, input, count * sizeof(double)); std::memcpy(right_.contents, right, right_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx score arithmetic";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:score_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, score_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, count * sizeof(double));
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics prior_vbx(const double* input, StatisticsDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(gamma_pipeline_ != nil && prior_pipeline_ != nil, "double_vbx_prior_kernels_unavailable");
        require(d.rows > 0 && d.rows <= 1024 && d.columns > 0 && d.columns <= 4096 && d.sequential <= 1 && input != nullptr && output != nullptr, "double_vbx_prior_shape_or_pointer_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns;
        bool nonzero = false;
        for (uint64_t i = 0; i < count; ++i) {
            require(std::isfinite(input[i]) && input[i] >= 0 && input[i] <= 1 && (input[i] == 0 || input[i] >= 0x1p-200), "double_vbx_prior_input_domain_invalid");
            nonzero |= input[i] != 0;
        }
        require(nonzero, "double_vbx_prior_zero_mass_not_supported");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(terms_, uint64_t(d.rows) * sizeof(double)); reserve(output_, uint64_t(d.rows) * sizeof(double));
        std::memcpy(input_.contents, input, count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx prior update";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:gamma_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, gamma_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:prior_pipeline_]; [encoder setBuffer:terms_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, uint64_t(d.rows) * sizeof(double));
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics posterior_vbx(const double* input, const double* right, ScoreDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(posterior_pipeline_ != nil, "double_vbx_posterior_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.operation <= 1 && input != nullptr && right != nullptr && output != nullptr, "double_vbx_posterior_shape_or_pointer_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns, right_count = d.operation == 0 ? d.columns : d.rows;
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]) && std::abs(input[i]) <= 0x1p200 && (input[i] == 0 || std::abs(input[i]) >= 0x1p-200), "double_vbx_posterior_input_domain_invalid");
        for (uint64_t i = 0; i < right_count; ++i) require(std::isfinite(right[i]) && std::abs(right[i]) <= 0x1p200 && (right[i] == 0 || std::abs(right[i]) >= 0x1p-200), "double_vbx_posterior_right_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(right_, right_count * sizeof(double)); reserve(output_, count * sizeof(double));
        std::memcpy(input_.contents, input, count * sizeof(double)); std::memcpy(right_.contents, right, right_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx posterior arithmetic";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:posterior_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, posterior_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, count * sizeof(double));
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics elbo_vbx(const double* input, ElboDimensions d, double fb, double likelihood, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(d.operation <= 1, "double_vbx_elbo_operation_invalid");
        require(d.operation == 0 ? likelihood_pipeline_ != nil : elbo_terms_pipeline_ != nil && elbo_final_pipeline_ != nil, "double_vbx_elbo_kernels_unavailable");
        require(d.operation == 0 ? d.rows == 1 && d.columns > 0 && d.columns <= 4096 && d.column_order == 0 : d.rows > 0 && d.rows <= 1024 && d.columns > 0 && d.columns <= 1024 && d.column_order <= 1, "double_vbx_elbo_shape_invalid");
        require(input != nullptr && output != nullptr, "double_vbx_elbo_pointer_invalid");
        require(std::isfinite(fb) && fb >= 0x1p-20 && fb <= 0x1p20 && std::isfinite(likelihood) && std::abs(likelihood) <= 0x1p450 && (likelihood == 0 || std::abs(likelihood) >= 0x1p-450), "double_vbx_elbo_scalar_domain_invalid");
        std::memcpy(&d.fb, &fb, sizeof(fb)); std::memcpy(&d.likelihood, &likelihood, sizeof(likelihood));
        uint64_t count = uint64_t(d.rows) * d.columns, input_count = count * (d.operation == 0 ? 1 : 3);
        for (uint64_t i = 0; i < input_count; ++i) require(std::isfinite(input[i]) && std::abs(input[i]) <= 0x1p200 && (input[i] == 0 || std::abs(input[i]) >= 0x1p-200) && (d.operation == 0 || i < count || i >= 2 * count || input[i] > 0), "double_vbx_elbo_input_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, input_count * sizeof(double)); reserve(output_, count * sizeof(double)); reserve(terms_, sizeof(double));
        std::memcpy(input_.contents, input, input_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx ELBO arithmetic";
        id<MTLComputePipelineState> pipeline = d.operation == 0 ? likelihood_pipeline_ : elbo_terms_pipeline_;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:pipeline]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:d.operation == 0 ? terms_ : output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.operation == 0 ? 1 : count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        if (d.operation == 1) {
            encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
            [encoder setComputePipelineState:elbo_final_pipeline_]; [encoder setBuffer:output_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [encoder endEncoding];
        }
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, terms_.contents, sizeof(double));
        return {1, d.operation == 0 ? 1u : 2u, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics transcendental_vbx(const double* input, TranscendentalDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        id<MTLComputePipelineState> pipeline = d.operation == 2 ? log1p_pipeline_ : transcendental_pipeline_;
        require(pipeline != nil, "double_vbx_transcendental_kernel_unavailable");
        require(d.count > 0 && d.count <= 65536 && d.operation <= 2 && input != nullptr && output != nullptr, "double_vbx_transcendental_shape_or_pointer_invalid");
        for (uint32_t i = 0; i < d.count; ++i) require(std::isfinite(input[i]) && (d.operation == 0 ? input[i] >= 0x1p-200 && input[i] <= 0x1p200 : d.operation == 1 ? input[i] >= -60. && input[i] <= 0. && (input[i] == 0 || std::abs(input[i]) >= 0x1p-200) : input[i] >= 0. && input[i] <= 1024. && (input[i] == 0 || input[i] >= 0x1p-200)), "double_vbx_transcendental_input_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        uint64_t bytes = uint64_t(d.count) * sizeof(double);
        reserve(input_, bytes); reserve(output_, bytes); std::memcpy(input_.contents, input, bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental binary64 log/exp";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:pipeline]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics logsumexp_vbx(const double* input, LogSumExpDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(lse_terms_pipeline_ != nil && lse_final_pipeline_ != nil, "double_vbx_logsumexp_kernels_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && uint64_t(d.rows) * d.columns <= 65536 && d.sequential <= 1 && input != nullptr && output != nullptr, "double_vbx_logsumexp_shape_or_pointer_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns;
        for (uint64_t i = 0; i < count; ++i) require(std::isfinite(input[i]) && std::abs(input[i]) <= 0x1p200 && (input[i] == 0 || std::abs(input[i]) >= 0x1p-200), "double_vbx_logsumexp_input_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(output_, count * sizeof(double)); reserve(right_, uint64_t(d.rows) * 3 * sizeof(double)); reserve(terms_, uint64_t(d.rows) * sizeof(double));
        std::memcpy(input_.contents, input, count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental finite VBx logsumexp";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:lse_terms_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2]; [encoder setBuffer:right_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, lse_terms_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:lse_final_pipeline_]; [encoder setBuffer:output_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:terms_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, lse_final_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        for (uint32_t row = 0; row < d.rows; ++row) require(std::isfinite(static_cast<const double*>(terms_.contents)[row]), "double_vbx_logsumexp_shift_domain_invalid");
        std::memcpy(output, terms_.contents, uint64_t(d.rows) * sizeof(double));
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics qinit_vbx(const double* input, uint32_t rows, uint32_t columns, double smoothing, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(qinit_pipeline_ != nil && initialization_pipeline_ != nil, "double_vbx_qinit_kernel_unavailable");
        uint64_t count = uint64_t(rows) * columns, bytes = count * sizeof(double);
        require(rows > 0 && rows <= 4096 && columns > 0 && columns <= 1024 && count <= 65536 && input != nullptr && output != nullptr, "double_vbx_qinit_shape_or_pointer_invalid");
        require(std::isfinite(smoothing) && smoothing >= 0 && smoothing <= 60 && (smoothing == 0 || smoothing >= 0x1p-200), "double_vbx_qinit_smoothing_domain_invalid");
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t ones = 0;
            for (uint32_t column = 0; column < columns; ++column) {
                double value = input[uint64_t(row) * columns + column];
                require(value == 0. || value == 1., "double_vbx_qinit_one_hot_required");
                ones += value == 1.;
            }
            require(ones == 1, "double_vbx_qinit_one_hot_required");
        }
        auto started = std::chrono::steady_clock::now();
        reserve(input_, bytes); reserve(output_, bytes); std::memcpy(input_.contents, input, bytes);
        QinitDimensions d = {rows, columns, 0}; std::memcpy(&d.smoothing, &smoothing, sizeof(double));
        InitializationDimensions normalization = {rows, columns, 1, 0};
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx initial softmax";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:qinit_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, qinit_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:initialization_pipeline_]; [encoder setBuffer:output_ offset:0 atIndex:0]; [encoder setBytes:&normalization length:sizeof(normalization) atIndex:1]; [encoder setBuffer:input_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, initialization_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, input_.contents, bytes);
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics initialize_vbx(const double* input, InitializationDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(initialization_pipeline_ != nil, "double_vbx_initialization_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.operation <= 1 && d.sequential <= 1 && output != nullptr, "double_vbx_initialization_shape_or_pointer_invalid");
        require(d.operation == 0 ? input == nullptr && d.rows == 1 && d.sequential == 0 : input != nullptr, "double_vbx_initialization_operation_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns, bytes = count * sizeof(double);
        if (d.operation == 1) {
            for (uint32_t row = 0; row < d.rows; ++row) {
                bool nonzero = false;
                for (uint32_t column = 0; column < d.columns; ++column) {
                    double value = input[uint64_t(row) * d.columns + column];
                    require(std::isfinite(value) && value >= 0 && value <= 0x1p200 && (value == 0 || value >= 0x1p-200), "double_vbx_initialization_input_domain_invalid");
                    nonzero |= value != 0;
                }
                require(nonzero, "double_vbx_initialization_zero_row_not_supported");
            }
        }
        auto started = std::chrono::steady_clock::now();
        reserve(input_, d.operation == 0 ? sizeof(double) : bytes); reserve(output_, bytes);
        if (d.operation == 1) std::memcpy(input_.contents, input, bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx initialization";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:initialization_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.operation == 0 ? d.columns : d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, initialization_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
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
extern "C" int wxdouble_vbx_precompute(void* context, const double* input, const double* phi, uint32_t rows, uint32_t columns, uint32_t sequential, double* g, double* v, double* rho, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->precompute_vbx(input, phi, {rows, columns, sequential}, g, v, rho); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_precompute_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_statistics(void* context, const double* input, const double* right, uint32_t rows, uint32_t columns, uint32_t features, uint32_t operation, uint32_t sequential, double fa, double fb, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->statistics_vbx(input, right, {rows, columns, features, operation, sequential, 0, 0, 0}, fa, fb, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_statistics_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_scores(void* context, const double* input, const double* right, uint32_t rows, uint32_t columns, uint32_t operation, double fa, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->scores_vbx(input, right, {rows, columns, operation, 0, 0}, fa, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_scores_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_prior(void* context, const double* input, uint32_t speakers, uint32_t frames, uint32_t sequential, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->prior_vbx(input, {speakers, frames, 0, 0, sequential, 0, 0, 0}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_prior_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_posterior(void* context, const double* input, const double* right, uint32_t rows, uint32_t columns, uint32_t operation, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->posterior_vbx(input, right, {rows, columns, operation, 0, 0}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_posterior_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_elbo(void* context, const double* input, uint32_t rows, uint32_t columns, uint32_t operation, uint32_t column_order, double fb, double likelihood, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->elbo_vbx(input, {rows, columns, operation, column_order, 0, 0}, fb, likelihood, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_elbo_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_transcendental(void* context, const double* input, uint32_t count, uint32_t operation, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->transcendental_vbx(input, {count, operation}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_transcendental_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_logsumexp(void* context, const double* input, uint32_t rows, uint32_t columns, uint32_t sequential, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->logsumexp_vbx(input, {rows, columns, sequential}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_logsumexp_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_initialize(void* context, const double* input, uint32_t rows, uint32_t columns, uint32_t operation, uint32_t sequential, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->initialize_vbx(input, {rows, columns, operation, sequential}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_initialization_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_qinit(void* context, const double* input, uint32_t rows, uint32_t columns, double smoothing, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->qinit_vbx(input, rows, columns, smoothing, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_qinit_failed"); return 1; }
    }
}
extern "C" void wxdouble_destroy(void* context) { @autoreleasepool { delete static_cast<DoubleContext*>(context); } }

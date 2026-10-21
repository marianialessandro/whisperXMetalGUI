#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_checks.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

struct DoubleDimensions { uint32_t rows, columns, sequential; };
struct KmeansCentroidDimensions { uint32_t rows, features, clusters; };
struct FloatShiftDimensions { uint32_t rows, columns, add; };
struct KmeansToleranceDimensions { uint32_t rows, columns, output_offset; float tolerance; };
struct KmeansNormDimensions { uint32_t rows, columns; };
struct KmeansPrefixDimensions { uint32_t count; };
static_assert(sizeof(KmeansToleranceDimensions) == 16);
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
struct CleanupDimensions { uint32_t rows, columns, blocks; };
struct InactiveDimensions { uint32_t rows, columns, frames, blocks; };
struct ArgmaxDimensions { uint32_t rows, columns; };
struct CosineDimensions { uint32_t rows, centroids, features; };
struct OneHotDimensions { uint32_t rows, columns; };
struct CentroidDimensions { uint32_t speakers, frames, features, sequential; };
struct QinitDimensions { uint32_t rows, columns; uint64_t smoothing; };
struct InitializationDimensions { uint32_t rows, columns, operation, sequential; };
struct DoubleMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
static_assert(sizeof(DoubleMetrics) == 32);

class DoubleContext {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> kmeans_centroid_pipeline_;
    id<MTLComputePipelineState> float_shift_pipeline_;
    id<MTLComputePipelineState> tolerance_mean_pipeline_, tolerance_square_pipeline_, tolerance_final_pipeline_;
    id<MTLComputePipelineState> kmeans_norm_pipeline_, kmeans_weighted_products_pipeline_, kmeans_weighted_prefix_pipeline_;
    id<MTLComputePipelineState> pipeline_, matmul_pipeline_, element_pipeline_, terms_pipeline_, rho_pipeline_, gamma_pipeline_, statistics_pipeline_, score_pipeline_, prior_pipeline_, posterior_pipeline_, likelihood_pipeline_, elbo_terms_pipeline_, elbo_final_pipeline_, transcendental_pipeline_, log1p_pipeline_, lse_terms_pipeline_, lse_final_pipeline_, initialization_pipeline_, qinit_pipeline_, centroid_pipeline_, one_hot_pipeline_, cosine_norm_pipeline_, cosine_distance_pipeline_, cosine_score_pipeline_, argmax_pipeline_, minimum_pipeline_, inactive_pipeline_, nan_minimum_pipeline_, nan_cleanup_pipeline_;
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
        id<MTLFunction> nan_minimum_function = [library newFunctionWithName:@"double_vbx_nan_min_partials"], nan_cleanup_function = [library newFunctionWithName:@"double_vbx_nan_cleanup"];
        if (nan_minimum_function != nil) nan_minimum_pipeline_ = [device_ newComputePipelineStateWithFunction:nan_minimum_function error:&error];
        if (nan_cleanup_function != nil) nan_cleanup_pipeline_ = [device_ newComputePipelineStateWithFunction:nan_cleanup_function error:&error];
        id<MTLFunction> minimum_function = [library newFunctionWithName:@"double_vbx_score_min_partials"], inactive_function = [library newFunctionWithName:@"double_vbx_inactive_mask"];
        if (minimum_function != nil) minimum_pipeline_ = [device_ newComputePipelineStateWithFunction:minimum_function error:&error];
        if (inactive_function != nil) inactive_pipeline_ = [device_ newComputePipelineStateWithFunction:inactive_function error:&error];
        id<MTLFunction> argmax_function = [library newFunctionWithName:@"double_vbx_argmax"];
        if (argmax_function != nil) argmax_pipeline_ = [device_ newComputePipelineStateWithFunction:argmax_function error:&error];
        id<MTLFunction> cosine_score_function = [library newFunctionWithName:@"double_vbx_cosine_scores"];
        if (cosine_score_function != nil) cosine_score_pipeline_ = [device_ newComputePipelineStateWithFunction:cosine_score_function error:&error];
        id<MTLFunction> cosine_norm_function = [library newFunctionWithName:@"double_vbx_cosine_norms"], cosine_distance_function = [library newFunctionWithName:@"double_vbx_cosine_distances"];
        if (cosine_norm_function != nil) cosine_norm_pipeline_ = [device_ newComputePipelineStateWithFunction:cosine_norm_function error:&error];
        if (cosine_distance_function != nil) cosine_distance_pipeline_ = [device_ newComputePipelineStateWithFunction:cosine_distance_function error:&error];
        id<MTLFunction> one_hot_function = [library newFunctionWithName:@"double_vbx_one_hot"];
        if (one_hot_function != nil) one_hot_pipeline_ = [device_ newComputePipelineStateWithFunction:one_hot_function error:&error];
        id<MTLFunction> kmeans_centroid_function = [library newFunctionWithName:@"float_vbx_kmeans_centroids"];
        if (kmeans_centroid_function != nil) kmeans_centroid_pipeline_ = [device_ newComputePipelineStateWithFunction:kmeans_centroid_function error:&error];
        id<MTLFunction> float_shift_function = [library newFunctionWithName:@"float_vbx_shift"];
        if (float_shift_function != nil) float_shift_pipeline_ = [device_ newComputePipelineStateWithFunction:float_shift_function error:&error];
        id<MTLFunction> tolerance_mean_function = [library newFunctionWithName:@"float_kmeans_tolerance_column_mean"], tolerance_square_function = [library newFunctionWithName:@"float_kmeans_tolerance_square"], tolerance_final_function = [library newFunctionWithName:@"float_kmeans_tolerance_final"];
        if (tolerance_mean_function != nil) tolerance_mean_pipeline_ = [device_ newComputePipelineStateWithFunction:tolerance_mean_function error:&error];
        if (tolerance_square_function != nil) tolerance_square_pipeline_ = [device_ newComputePipelineStateWithFunction:tolerance_square_function error:&error];
        if (tolerance_final_function != nil) tolerance_final_pipeline_ = [device_ newComputePipelineStateWithFunction:tolerance_final_function error:&error];
        id<MTLFunction> kmeans_norm_function = [library newFunctionWithName:@"float_kmeans_squared_norms"];
        if (kmeans_norm_function != nil) kmeans_norm_pipeline_ = [device_ newComputePipelineStateWithFunction:kmeans_norm_function error:&error];
        id<MTLFunction> weighted_products_function = [library newFunctionWithName:@"float_kmeans_weighted_products"], weighted_prefix_function = [library newFunctionWithName:@"float_kmeans_weighted_prefix"];
        if (weighted_products_function != nil) kmeans_weighted_products_pipeline_ = [device_ newComputePipelineStateWithFunction:weighted_products_function error:&error];
        if (weighted_prefix_function != nil) kmeans_weighted_prefix_pipeline_ = [device_ newComputePipelineStateWithFunction:weighted_prefix_function error:&error];
        id<MTLFunction> centroid_function = [library newFunctionWithName:@"double_vbx_centroids"];
        if (centroid_function != nil) centroid_pipeline_ = [device_ newComputePipelineStateWithFunction:centroid_function error:&error];
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

    DoubleMetrics kmeans_weighted_prefix(const float* weights, const float* distances, KmeansPrefixDimensions d, float* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(kmeans_weighted_products_pipeline_ != nil && kmeans_weighted_prefix_pipeline_ != nil, "double_kmeans_weighted_prefix_kernel_unavailable");
        require(weights != nullptr && distances != nullptr && output != nullptr && d.count > 0 && d.count <= 4096, "double_kmeans_weighted_prefix_shape_or_pointer_invalid");
        for (uint32_t index = 0; index < d.count; ++index) {
            require(std::isfinite(weights[index]) && weights[index] >= 0 && weights[index] <= 0x1p32f && (weights[index] == 0 || weights[index] >= 0x1p-32f), "double_kmeans_weighted_prefix_weight_domain_invalid");
            require(std::isfinite(distances[index]) && distances[index] >= 0 && distances[index] <= 0x1p32f && (distances[index] == 0 || distances[index] >= 0x1p-64f), "double_kmeans_weighted_prefix_distance_domain_invalid");
        }
        auto started = std::chrono::steady_clock::now();
        uint64_t bytes = uint64_t(d.count) * sizeof(float);
        reserve(input_, bytes); reserve(right_, bytes); reserve(output_, bytes); reserve(terms_, bytes);
        std::memcpy(input_.contents, weights, bytes); std::memcpy(right_.contents, distances, bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental KMeans++ weighted prefix";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:kmeans_weighted_products_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(d.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, kmeans_weighted_products_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:kmeans_weighted_prefix_pipeline_]; [encoder setBuffer:output_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const float* result = static_cast<const float*>(terms_.contents);
        for (uint32_t index = 0; index < d.count; ++index) require(std::isfinite(result[index]) && result[index] >= 0 && result[index] <= 0x1p76f, "double_kmeans_weighted_prefix_result_invalid");
        std::memcpy(output, result, bytes);
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics kmeans_squared_norms(const float* values, KmeansNormDimensions d, float* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(kmeans_norm_pipeline_ != nil, "double_kmeans_squared_norms_kernel_unavailable");
        require(values != nullptr && output != nullptr && d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024, "double_kmeans_squared_norms_shape_or_pointer_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns;
        for (uint64_t index = 0; index < count; ++index) require(std::isfinite(values[index]) && std::abs(values[index]) <= 0x1p32f && (values[index] == 0 || std::abs(values[index]) >= 0x1p-32f), "double_kmeans_squared_norms_value_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(float)); reserve(output_, uint64_t(d.rows) * sizeof(float));
        std::memcpy(input_.contents, values, count * sizeof(float));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental KMeans float32 squared norms";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:kmeans_norm_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, kmeans_norm_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const float* result = static_cast<const float*>(output_.contents);
        for (uint32_t row = 0; row < d.rows; ++row) require(std::isfinite(result[row]) && result[row] >= 0 && result[row] <= 0x1p74f, "double_kmeans_squared_norms_result_invalid");
        std::memcpy(output, result, uint64_t(d.rows) * sizeof(float));
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics kmeans_tolerance(const float* values, KmeansToleranceDimensions d, float* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(tolerance_mean_pipeline_ != nil && tolerance_square_pipeline_ != nil && tolerance_final_pipeline_ != nil, "double_kmeans_tolerance_kernel_unavailable");
        require(values != nullptr && output != nullptr && d.rows > 0 && d.rows <= 4096 && d.columns >= 2 && d.columns <= 1024 && std::isfinite(d.tolerance) && d.tolerance >= 0x1p-32f && d.tolerance <= 0x1p16f, "double_kmeans_tolerance_shape_pointer_or_scalar_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns;
        for (uint64_t index = 0; index < count; ++index) require(std::isfinite(values[index]) && std::abs(values[index]) <= 0x1p32f && (values[index] == 0 || std::abs(values[index]) >= 0x1p-32f), "double_kmeans_tolerance_value_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(float)); reserve(right_, uint64_t(2) * d.columns * sizeof(float)); reserve(output_, count * sizeof(float)); reserve(terms_, sizeof(float));
        std::memcpy(input_.contents, values, count * sizeof(float));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental KMeans float32 tolerance";
        auto encode_mean = [&](id<MTLBuffer> source, const KmeansToleranceDimensions& dimensions) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
            [encoder setComputePipelineState:tolerance_mean_pipeline_]; [encoder setBuffer:source offset:0 atIndex:0]; [encoder setBytes:&dimensions length:sizeof(dimensions) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(d.columns, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, tolerance_mean_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        };
        encode_mean(input_, d);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:tolerance_square_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, tolerance_square_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        KmeansToleranceDimensions variance_dimensions = d; variance_dimensions.output_offset = d.columns;
        encode_mean(output_, variance_dimensions);
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:tolerance_final_pipeline_]; [encoder setBuffer:right_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        float result = *static_cast<const float*>(terms_.contents);
        require(std::isfinite(result) && result >= 0 && result <= 0x1p82f && std::fpclassify(result) != FP_SUBNORMAL, "double_kmeans_tolerance_intermediate_not_supported");
        std::memcpy(output, &result, sizeof(float));
        return {1, 4, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics float_shift_vbx(const float* values, const float* mean, FloatShiftDimensions d, float* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(float_shift_pipeline_ != nil, "double_float_shift_kernel_unavailable");
        require(values != nullptr && mean != nullptr && output != nullptr && d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && d.add <= 1, "double_float_shift_shape_or_pointer_invalid");
        uint64_t count = uint64_t(d.rows) * d.columns;
        for (uint64_t index = 0; index < count; ++index) require(std::isfinite(values[index]) && std::abs(values[index]) <= 0x1p50f && (values[index] == 0 || std::abs(values[index]) >= 0x1p-100f), "double_float_shift_value_domain_invalid");
        for (uint32_t column = 0; column < d.columns; ++column) require(std::isfinite(mean[column]) && std::abs(mean[column]) <= 0x1p50f && (mean[column] == 0 || std::abs(mean[column]) >= 0x1p-100f), "double_float_shift_mean_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(float)); reserve(right_, uint64_t(d.columns) * sizeof(float)); reserve(output_, count * sizeof(float));
        std::memcpy(input_.contents, values, count * sizeof(float)); std::memcpy(right_.contents, mean, uint64_t(d.columns) * sizeof(float));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental KMeans float32 shift";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:float_shift_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, float_shift_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const float* result = static_cast<const float*>(output_.contents);
        for (uint64_t index = 0; index < count; ++index) require(std::isfinite(result[index]) && std::abs(result[index]) <= 0x1p51f && std::fpclassify(result[index]) != FP_SUBNORMAL, "double_float_shift_result_invalid");
        std::memcpy(output, result, count * sizeof(float));
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics kmeans_centroids_vbx(const float* features, const int32_t* labels, KmeansCentroidDimensions d, float* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(kmeans_centroid_pipeline_ != nil, "double_vbx_kmeans_centroid_kernel_unavailable");
        require(features != nullptr && labels != nullptr && output != nullptr && d.rows > 0 && d.rows <= 4096 && d.features >= 2 && d.features <= 1024 && d.clusters > 0 && d.clusters <= 1024 && d.clusters <= d.rows && uint64_t(d.rows) * d.features * d.clusters <= 32 * 1024 * 1024, "double_vbx_kmeans_centroid_shape_or_pointer_invalid");
        std::vector<bool> present(d.clusters, false);
        for (uint32_t row = 0; row < d.rows; ++row) {
            require(labels[row] >= 0 && uint32_t(labels[row]) < d.clusters, "double_vbx_kmeans_label_invalid");
            present[labels[row]] = true;
        }
        for (bool value : present) require(value, "double_vbx_kmeans_empty_cluster_not_supported");
        uint64_t count = uint64_t(d.rows) * d.features, result_count = uint64_t(d.clusters) * d.features;
        for (uint64_t index = 0; index < count; ++index) require(std::isfinite(features[index]) && std::abs(features[index]) <= 0x1p32f && (features[index] == 0 || std::abs(features[index]) >= 0x1p-32f), "double_vbx_kmeans_feature_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(float)); reserve(right_, uint64_t(d.rows) * sizeof(int32_t)); reserve(output_, result_count * sizeof(float));
        std::memcpy(input_.contents, features, count * sizeof(float)); std::memcpy(right_.contents, labels, uint64_t(d.rows) * sizeof(int32_t));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental KMeans float32 centroids";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:kmeans_centroid_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(result_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, kmeans_centroid_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const float* result = static_cast<const float*>(output_.contents);
        for (uint64_t index = 0; index < result_count; ++index) require(std::isfinite(result[index]) && std::abs(result[index]) <= 0x1p32f, "double_vbx_kmeans_result_invalid");
        std::memcpy(output, result, result_count * sizeof(float));
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics nan_cleanup_vbx(const double* scores, uint32_t rows, uint32_t columns, double* output, uint32_t* all_nan) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(nan_minimum_pipeline_ != nil && nan_cleanup_pipeline_ != nil, "double_vbx_nan_cleanup_kernel_unavailable");
        require(rows > 0 && rows <= 4096 && columns > 0 && columns <= 1024 && scores != nullptr && output != nullptr && all_nan != nullptr, "double_vbx_nan_cleanup_shape_or_pointer_invalid");
        uint64_t count = uint64_t(rows) * columns;
        uintptr_t a = reinterpret_cast<uintptr_t>(output), b = reinterpret_cast<uintptr_t>(all_nan);
        require(a <= b ? b - a >= count * sizeof(double) : a - b >= sizeof(uint32_t), "double_vbx_nan_cleanup_outputs_overlap");
        bool any_finite = false;
        for (uint64_t index = 0; index < count; ++index) {
            require(std::isnan(scores[index]) || (std::isfinite(scores[index]) && scores[index] >= -1 && scores[index] <= 2), "double_vbx_nan_cleanup_value_domain_invalid");
            any_finite |= std::isfinite(scores[index]);
        }
        CleanupDimensions d = {rows, columns, uint32_t((count + 1023) / 1024)};
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(output_, count * sizeof(double)); reserve(terms_, uint64_t(d.blocks) * sizeof(double)); std::memcpy(input_.contents, scores, count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental constrained NaN cleanup";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:nan_minimum_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, nan_minimum_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:nan_cleanup_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, nan_cleanup_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const double* result = static_cast<const double*>(output_.contents);
        for (uint64_t index = 0; index < count; ++index) require(any_finite ? std::isfinite(result[index]) && result[index] >= -1 && result[index] <= 2 : std::isnan(result[index]), "double_vbx_nan_cleanup_result_invalid");
        std::memcpy(output, result, count * sizeof(double)); *all_nan = !any_finite;
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics inactive_mask_vbx(const double* scores, const double* segments, uint32_t rows, uint32_t columns, uint32_t frames, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(minimum_pipeline_ != nil && inactive_pipeline_ != nil, "double_vbx_inactive_kernel_unavailable");
        require(rows > 0 && rows <= 4096 && columns > 0 && columns <= 1024 && frames > 0 && frames <= 4096 && uint64_t(rows) * frames <= 1024 * 1024 && scores != nullptr && segments != nullptr && output != nullptr, "double_vbx_inactive_shape_or_pointer_invalid");
        uint64_t count = uint64_t(rows) * columns, segment_count = uint64_t(rows) * frames;
        for (uint64_t index = 0; index < count; ++index) require(std::isnan(scores[index]) || (std::isfinite(scores[index]) && scores[index] >= 0 && scores[index] <= 2), "double_vbx_inactive_score_domain_invalid");
        for (uint64_t index = 0; index < segment_count; ++index) require(segments[index] == 0 || segments[index] == 1, "double_vbx_inactive_binary_segments_required");
        InactiveDimensions d = {rows, columns, frames, uint32_t((count + 1023) / 1024)};
        auto started = std::chrono::steady_clock::now();
        reserve(input_, count * sizeof(double)); reserve(right_, segment_count * sizeof(double)); reserve(output_, count * sizeof(double)); reserve(terms_, uint64_t(d.blocks) * sizeof(double));
        std::memcpy(input_.contents, scores, count * sizeof(double)); std::memcpy(right_.contents, segments, segment_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental inactive speaker mask";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:minimum_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, minimum_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:inactive_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:terms_ offset:0 atIndex:3]; [encoder setBuffer:output_ offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, inactive_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const double* result = static_cast<const double*>(output_.contents);
        for (uint64_t index = 0; index < count; ++index) require(std::isnan(result[index]) || (std::isfinite(result[index]) && result[index] >= -1 && result[index] <= 2), "double_vbx_inactive_result_invalid");
        std::memcpy(output, result, count * sizeof(double));
        return {1, 2, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics qinit_from_labels_vbx(const int64_t* labels, uint32_t rows, uint32_t columns, double smoothing, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(one_hot_pipeline_ != nil && qinit_pipeline_ != nil && initialization_pipeline_ != nil, "double_vbx_resident_initialization_kernel_unavailable");
        uint64_t count = uint64_t(rows) * columns, bytes = count * sizeof(double);
        require(rows > 0 && rows <= 4096 && columns > 0 && columns <= 1024 && count <= 65536 && labels != nullptr && output != nullptr, "double_vbx_resident_initialization_shape_or_pointer_invalid");
        require(std::isfinite(smoothing) && smoothing >= 0 && smoothing <= 60 && (smoothing == 0 || smoothing >= 0x1p-200), "double_vbx_qinit_smoothing_domain_invalid");
        for (uint32_t row = 0; row < rows; ++row) require(labels[row] >= 0 && labels[row] < columns, "double_vbx_one_hot_label_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, bytes); reserve(output_, bytes); std::memcpy(input_.contents, labels, uint64_t(rows) * sizeof(int64_t));
        OneHotDimensions one_hot = {rows, columns}; QinitDimensions qinit = {rows, columns, 0}; InitializationDimensions normalization = {rows, columns, 1, 0};
        std::memcpy(&qinit.smoothing, &smoothing, sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental resident cluster initialization";
        auto encode = [&](id<MTLComputePipelineState> pipeline, id<MTLBuffer> source, const void* dimensions, uint64_t size, id<MTLBuffer> destination, uint64_t threads) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
            [encoder setComputePipelineState:pipeline]; [encoder setBuffer:source offset:0 atIndex:0]; [encoder setBytes:dimensions length:size atIndex:1]; [encoder setBuffer:destination offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        };
        encode(one_hot_pipeline_, input_, &one_hot, sizeof(one_hot), output_, count);
        encode(qinit_pipeline_, output_, &qinit, sizeof(qinit), input_, count);
        encode(initialization_pipeline_, input_, &normalization, sizeof(normalization), output_, rows);
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const double* result = static_cast<const double*>(output_.contents);
        for (uint64_t index = 0; index < count; ++index) require(std::isfinite(result[index]) && result[index] >= 0 && result[index] <= 1, "double_vbx_resident_initialization_result_invalid");
        std::memcpy(output, result, bytes);
        return {1, 3, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics argmax_vbx(const double* scores, ArgmaxDimensions d, int64_t* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(argmax_pipeline_ != nil, "double_vbx_argmax_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && scores != nullptr && output != nullptr, "double_vbx_argmax_shape_or_pointer_invalid");
        auto started = std::chrono::steady_clock::now();
        uint64_t bytes = uint64_t(d.rows) * d.columns * sizeof(double), result_bytes = uint64_t(d.rows) * sizeof(int64_t);
        reserve(input_, bytes); reserve(output_, result_bytes); std::memcpy(input_.contents, scores, bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx unconstrained argmax";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:argmax_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, argmax_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const int64_t* result = static_cast<const int64_t*>(output_.contents);
        for (uint32_t row = 0; row < d.rows; ++row) require(result[row] >= 0 && result[row] < d.columns, "double_vbx_argmax_result_invalid");
        std::memcpy(output, result, result_bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics cosine_vbx(const double* embeddings, const double* centroids, CosineDimensions d, double* output, bool soft_scores = false, int64_t* labels = nullptr) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(labels == nullptr || (soft_scores && argmax_pipeline_ != nil), "double_vbx_cosine_argmax_kernel_unavailable");
        id<MTLComputePipelineState> final_pipeline = soft_scores ? cosine_score_pipeline_ : cosine_distance_pipeline_;
        require(cosine_norm_pipeline_ != nil && final_pipeline != nil, "double_vbx_cosine_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.centroids > 0 && d.centroids <= 1024 && d.features > 0 && d.features <= 1024, "double_vbx_cosine_shape_invalid");
        uint64_t count = uint64_t(d.rows) * d.centroids, input_count = uint64_t(d.rows) * d.features, centroid_count = uint64_t(d.centroids) * d.features;
        require(count * d.features <= 32 * 1024 * 1024 && embeddings != nullptr && centroids != nullptr && output != nullptr, "double_vbx_cosine_work_or_pointer_invalid");
        std::vector<uint8_t> undefined(d.rows + d.centroids, 0);
        for (uint32_t index = 0; index < d.rows + d.centroids; ++index) {
            const double* values = index < d.rows ? embeddings + uint64_t(index) * d.features : centroids + uint64_t(index - d.rows) * d.features;
            bool nonzero = false, nan = false;
            for (uint32_t feature = 0; feature < d.features; ++feature) {
                double value = values[feature];
                require(std::isnan(value) || (std::isfinite(value) && std::abs(value) <= 0x1p100 && (value == 0 || std::abs(value) >= 0x1p-100)), "double_vbx_cosine_value_domain_invalid");
                nan |= std::isnan(value); nonzero |= value != 0;
            }
            undefined[index] = nan || !nonzero;
        }
        auto started = std::chrono::steady_clock::now();
        reserve(input_, input_count * sizeof(double)); reserve(right_, centroid_count * sizeof(double)); reserve(output_, count * sizeof(double)); reserve(terms_, uint64_t(d.rows + d.centroids) * sizeof(double));
        std::memcpy(input_.contents, embeddings, input_count * sizeof(double)); std::memcpy(right_.contents, centroids, centroid_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx cosine distances";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:cosine_norm_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:terms_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(d.rows + d.centroids, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, cosine_norm_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:final_pipeline]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:terms_ offset:0 atIndex:3]; [encoder setBuffer:output_ offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, final_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        if (labels != nullptr) {
            ArgmaxDimensions assignment = {d.rows, d.centroids};
            encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
            [encoder setComputePipelineState:argmax_pipeline_]; [encoder setBuffer:output_ offset:0 atIndex:0]; [encoder setBytes:&assignment length:sizeof(assignment) atIndex:1]; [encoder setBuffer:terms_ offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(d.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, argmax_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        }
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        const double* result = static_cast<const double*>(output_.contents);
        for (uint32_t row = 0; row < d.rows; ++row) for (uint32_t centroid = 0; centroid < d.centroids; ++centroid) {
            double value = result[uint64_t(row) * d.centroids + centroid];
            require(undefined[row] || undefined[d.rows + centroid] ? std::isnan(value) : std::isfinite(value) && value >= 0 && value <= 2, "double_vbx_cosine_result_invalid");
        }
        const int64_t* result_labels = static_cast<const int64_t*>(terms_.contents);
        if (labels != nullptr) for (uint32_t row = 0; row < d.rows; ++row) require(result_labels[row] >= 0 && result_labels[row] < d.centroids, "double_vbx_argmax_result_invalid");
        std::memcpy(output, result, count * sizeof(double));
        if (labels != nullptr) std::memcpy(labels, result_labels, uint64_t(d.rows) * sizeof(int64_t));
        return {1, labels != nullptr ? 3u : 2u, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics one_hot_vbx(const int64_t* labels, OneHotDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(one_hot_pipeline_ != nil, "double_vbx_one_hot_kernel_unavailable");
        require(d.rows > 0 && d.rows <= 4096 && d.columns > 0 && d.columns <= 1024 && labels != nullptr && output != nullptr, "double_vbx_one_hot_shape_or_pointer_invalid");
        for (uint32_t row = 0; row < d.rows; ++row) require(labels[row] >= 0 && labels[row] < d.columns, "double_vbx_one_hot_label_invalid");
        auto started = std::chrono::steady_clock::now();
        uint64_t label_bytes = uint64_t(d.rows) * sizeof(int64_t), count = uint64_t(d.rows) * d.columns, bytes = count * sizeof(double);
        reserve(input_, label_bytes); reserve(output_, bytes); std::memcpy(input_.contents, labels, label_bytes);
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx one-hot initialization";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:one_hot_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:output_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, one_hot_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, bytes);
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
    }

    DoubleMetrics centroids_vbx(const double* weights, const double* features, CentroidDimensions d, double* output) {
        std::lock_guard<std::mutex> guard(mutex_);
        require(centroid_pipeline_ != nil, "double_vbx_centroid_kernel_unavailable");
        require(d.speakers > 0 && d.speakers <= 1024 && d.frames > 0 && d.frames <= 4096 && d.features > 0 && d.features <= 1024 && d.sequential <= 1, "double_vbx_centroid_shape_invalid");
        uint64_t weight_count = uint64_t(d.speakers) * d.frames, feature_count = uint64_t(d.frames) * d.features, output_count = uint64_t(d.speakers) * d.features;
        require(weight_count * d.features <= 32 * 1024 * 1024 && feature_count <= 1024 * 1024 && weights != nullptr && features != nullptr && output != nullptr, "double_vbx_centroid_work_or_pointer_invalid");
        for (uint32_t speaker = 0; speaker < d.speakers; ++speaker) {
            bool nonzero = false;
            for (uint32_t frame = 0; frame < d.frames; ++frame) {
                double value = weights[uint64_t(speaker) * d.frames + frame];
                require(std::isfinite(value) && value >= 0 && value <= 0x1p200 && (value == 0 || value >= 0x1p-200), "double_vbx_centroid_weight_domain_invalid");
                nonzero |= value != 0;
            }
            require(nonzero, "double_vbx_centroid_zero_mass_not_supported");
        }
        for (uint64_t i = 0; i < feature_count; ++i) require(std::isfinite(features[i]) && std::abs(features[i]) <= 0x1p100 && (features[i] == 0 || std::abs(features[i]) >= 0x1p-100), "double_vbx_centroid_feature_domain_invalid");
        auto started = std::chrono::steady_clock::now();
        reserve(input_, weight_count * sizeof(double)); reserve(right_, feature_count * sizeof(double)); reserve(output_, output_count * sizeof(double));
        std::memcpy(input_.contents, weights, weight_count * sizeof(double)); std::memcpy(right_.contents, features, feature_count * sizeof(double));
        id<MTLCommandBuffer> command = [queue_ commandBuffer]; require(command != nil, "double_command_creation_failed");
        command.label = @"WhisperX experimental VBx weighted centroids";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; require(encoder != nil, "double_encoder_creation_failed");
        [encoder setComputePipelineState:centroid_pipeline_]; [encoder setBuffer:input_ offset:0 atIndex:0]; [encoder setBytes:&d length:sizeof(d) atIndex:1]; [encoder setBuffer:right_ offset:0 atIndex:2]; [encoder setBuffer:output_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(output_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(256, centroid_pipeline_.maxTotalThreadsPerThreadgroup), 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted]; require(command.status == MTLCommandBufferStatusCompleted && command.error == nil, "double_command_failed");
        std::memcpy(output, output_.contents, output_count * sizeof(double));
        return {1, 1, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), command.GPUEndTime - command.GPUStartTime, input_.length + output_.length + right_.length + terms_.length};
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
extern "C" int wxdouble_vbx_centroids(void* context, const double* weights, const double* features, uint32_t speakers, uint32_t frames, uint32_t dimensions, uint32_t sequential, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->centroids_vbx(weights, features, {speakers, frames, dimensions, sequential}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_centroid_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_one_hot(void* context, const int64_t* labels, uint32_t rows, uint32_t columns, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->one_hot_vbx(labels, {rows, columns}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_one_hot_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_cosine(void* context, const double* embeddings, const double* centroids, uint32_t rows, uint32_t count, uint32_t features, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->cosine_vbx(embeddings, centroids, {rows, count, features}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_cosine_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_cosine_scores(void* context, const double* embeddings, const double* centroids, uint32_t rows, uint32_t count, uint32_t features, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->cosine_vbx(embeddings, centroids, {rows, count, features}, output, true); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_cosine_scores_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_argmax(void* context, const double* scores, uint32_t rows, uint32_t columns, int64_t* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->argmax_vbx(scores, {rows, columns}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_argmax_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_cosine_scores_argmax(void* context, const double* embeddings, const double* centroids, uint32_t rows, uint32_t count, uint32_t features, double* scores, int64_t* labels, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try {
            require(context != nullptr && metrics != nullptr && scores != nullptr && labels != nullptr, "double_vbx_resident_output_or_context_missing");
            require(rows > 0 && rows <= 4096 && count > 0 && count <= 1024, "double_vbx_cosine_shape_invalid");
            uintptr_t a = reinterpret_cast<uintptr_t>(scores), b = reinterpret_cast<uintptr_t>(labels);
            require(a <= b ? b - a >= uint64_t(rows) * count * sizeof(double) : a - b >= uint64_t(rows) * sizeof(int64_t), "double_vbx_resident_outputs_overlap");
            *metrics = static_cast<DoubleContext*>(context)->cosine_vbx(embeddings, centroids, {rows, count, features}, scores, true, labels); return 0;
        }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_resident_argmax_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_qinit_from_labels(void* context, const int64_t* labels, uint32_t rows, uint32_t columns, double smoothing, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->qinit_from_labels_vbx(labels, rows, columns, smoothing, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_resident_initialization_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_inactive_mask(void* context, const double* scores, const double* segments, uint32_t rows, uint32_t columns, uint32_t frames, double* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->inactive_mask_vbx(scores, segments, rows, columns, frames, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_inactive_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_nan_cleanup(void* context, const double* scores, uint32_t rows, uint32_t columns, double* output, uint32_t* all_nan, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->nan_cleanup_vbx(scores, rows, columns, output, all_nan); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_nan_cleanup_failed"); return 1; }
    }
}
extern "C" int wxdouble_vbx_kmeans_centroids(void* context, const float* features, const int32_t* labels, uint32_t rows, uint32_t dimensions, uint32_t clusters, float* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->kmeans_centroids_vbx(features, labels, {rows, dimensions, clusters}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_vbx_kmeans_centroids_failed"); return 1; }
    }
}
extern "C" int wxdouble_float_shift(void* context, const float* values, const float* mean, uint32_t rows, uint32_t columns, uint32_t add, float* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->float_shift_vbx(values, mean, {rows, columns, add}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_float_shift_failed"); return 1; }
    }
}
extern "C" int wxdouble_kmeans_tolerance(void* context, const float* values, uint32_t rows, uint32_t columns, float tolerance, float* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->kmeans_tolerance(values, {rows, columns, 0, tolerance}, output); return 0; }
        catch (const InventoryError& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_kmeans_tolerance_failed"); return 1; }
    }
}

extern "C" int wxdouble_kmeans_squared_norms(void* context, const float* values, uint32_t rows, uint32_t columns, float* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->kmeans_squared_norms(values, {rows, columns}, output); return 0; }
        catch (const std::exception& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_kmeans_squared_norms_failed"); return 1; }
    }
}
extern "C" void wxdouble_destroy(void* context) { @autoreleasepool { delete static_cast<DoubleContext*>(context); } }

extern "C" int wxdouble_kmeans_weighted_prefix(void* context, const float* weights, const float* distances, uint32_t count, float* output, DoubleMetrics* metrics, char* error, uint32_t capacity) {
    @autoreleasepool {
        try { require(context != nullptr && metrics != nullptr, "double_context_or_metrics_missing"); *metrics = static_cast<DoubleContext*>(context)->kmeans_weighted_prefix(weights, distances, {count}, output); return 0; }
        catch (const std::exception& failure) { double_error(error, capacity, failure.what()); return 1; }
        catch (...) { double_error(error, capacity, "double_kmeans_weighted_prefix_failed"); return 1; }
    }
}

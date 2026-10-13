#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <torch/library.h>
#include <torch/custom_class.h>
#include <torch/version.h>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include "native_ctc_core.hpp"

namespace {
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// Storage is borrowed through an owning Tensor and a strong Metal reference until both queues have completed.
static std::tuple<at::Tensor, at::Tensor, at::Tensor> row_max(const at::Tensor& input) {
    TORCH_CHECK(std::string(TORCH_VERSION) == "2.8.0", "mps_probe_build_version_not_supported");
    TORCH_CHECK(input.device().type() == c10::DeviceType::MPS && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous() && !input.requires_grad() && !input.is_neg() && !input.is_conj(), "mps_probe_tensor_not_supported");
    TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 32768 && input.size(1) > 0 && input.size(1) <= 4096 && input.numel() <= 16777216, "mps_probe_shape_not_supported");
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    @autoreleasepool {
        const at::Tensor owner = input;
        const uint32_t rows = owner.size(0), columns = owner.size(1);
        const uint64_t bytes = owner.numel() * sizeof(float), offset = owner.storage_offset() * sizeof(float);
        TORCH_CHECK(owner.storage_offset() >= 0 && offset <= owner.storage().nbytes() && bytes <= owner.storage().nbytes() - offset, "mps_probe_storage_range_invalid");
        // PyTorch 2.8 MPS stores the Objective-C buffer in Storage.data(), rather than exposing a host float pointer.
        id<MTLBuffer> source = (__bridge id<MTLBuffer>)owner.storage().data();
        TORCH_CHECK(source != nil && offset <= source.length && bytes <= source.length - offset, "mps_probe_metal_range_invalid");
        const auto started = Clock::now();
        auto* stream = at::mps::getCurrentMPSStream();
        TORCH_CHECK(source.device == stream->commandQueue().device, "mps_probe_device_mismatch");
        __block std::exception_ptr sync_error;
        const auto sync_started = Clock::now();
        dispatch_sync(stream->queue(), ^{ try { stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT); } catch (...) { sync_error = std::current_exception(); } });
        if (sync_error) std::rethrow_exception(sync_error);
        const double producer_wait = seconds(sync_started);
        static id<MTLDevice> device;
        static id<MTLCommandQueue> queue;
        static id<MTLComputePipelineState> pipeline;
        bool setup = pipeline == nil;
        if (setup) {
            device = source.device;
            queue = [device newCommandQueue];
            NSError* error = nil;
            NSString* shader = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void mps_probe_row_max(device const float* input [[buffer(0)]], device uint* output [[buffer(1)]], device atomic_uint* invalid [[buffer(2)]], constant uint& columns [[buffer(3)]], uint row [[thread_position_in_grid]]) { float best = -INFINITY; uint index = 0; for (uint column = 0; column < columns; ++column) { float value = input[row * columns + column]; if (isnan(value) || value > 0.0f) atomic_fetch_or_explicit(invalid, 1u, memory_order_relaxed); if (value > best) { best = value; index = column; } } output[2 * row] = as_type<uint>(best); output[2 * row + 1] = index; }";
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.fastMathEnabled = NO;
            id<MTLLibrary> library = [device newLibraryWithSource:shader options:options error:&error];
            TORCH_CHECK(library != nil, "mps_probe_shader_compile_failed");
            pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"mps_probe_row_max"] error:&error];
            TORCH_CHECK(queue != nil && pipeline != nil, "mps_probe_pipeline_failed");
        }
        TORCH_CHECK(source.device == device, "mps_probe_cached_device_mismatch");
        id<MTLBuffer> result = [device newBufferWithLength:rows * 8 + 4 options:MTLResourceStorageModeShared];
        TORCH_CHECK(result != nil, "mps_probe_result_allocation_failed");
        std::memset(static_cast<char*>(result.contents) + rows * 8, 0, 4);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        command.label = @"WhisperX MPS storage diagnostic row maximum";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        TORCH_CHECK(command != nil && encoder != nil, "mps_probe_encoder_failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:source offset:offset atIndex:0];
        [encoder setBuffer:result offset:0 atIndex:1];
        [encoder setBuffer:result offset:rows * 8 atIndex:2];
        [encoder setBytes:&columns length:sizeof(columns) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(rows, 256), 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        TORCH_CHECK(command.status == MTLCommandBufferStatusCompleted, "mps_probe_command_failed");
        const auto* words = static_cast<const uint32_t*>(result.contents);
        TORCH_CHECK(words[rows * 2] == 0, "mps_probe_log_scores_invalid");
        auto maxima = at::empty({rows}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        auto indices = at::empty({rows}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
        for (uint32_t row = 0; row < rows; ++row) {
            std::memcpy(maxima.data_ptr<float>() + row, words + row * 2, 4);
            std::memcpy(indices.data_ptr<int32_t>() + row, words + row * 2 + 1, 4);
        }
        auto metrics = at::empty({7}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
        double* values = metrics.data_ptr<double>();
        values[0] = producer_wait;
        values[1] = seconds(started);
        values[2] = command.GPUEndTime - command.GPUStartTime;
        values[3] = offset;
        values[4] = bytes;
        values[5] = rows * 8 + 4;
        values[6] = setup;
        return {maxima, indices, metrics};
    }
}
}

TORCH_LIBRARY(whisperx_mps_probe, module) {
    module.def("row_max(Tensor input) -> (Tensor, Tensor, Tensor)");
    module.impl("row_max", torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd, TORCH_FN(row_max)));
}

// Intercept lazy view dispatch before Torch's generic fallback can change the borrowed storage semantics.
TORCH_LIBRARY_IMPL(whisperx_mps_probe, Negative, module) { module.impl("row_max", TORCH_FN(row_max)); }
TORCH_LIBRARY_IMPL(whisperx_mps_probe, Conjugate, module) { module.impl("row_max", TORCH_FN(row_max)); }

namespace {
class MPSCTC final : public torch::CustomClassHolder {
    std::unique_ptr<NativeCTCContext> core_;
    std::mutex mutex_;
public:
    explicit MPSCTC(std::string shader) {
        TORCH_CHECK(std::string(TORCH_VERSION) == "2.8.0", "mps_probe_build_version_not_supported");
        @autoreleasepool { core_ = std::make_unique<NativeCTCContext>(shader.data(), shader.size(), true, true); }
    }

    void close() { std::lock_guard<std::mutex> guard(mutex_); core_.reset(); }

    std::tuple<at::Tensor, at::Tensor> compute(const at::Tensor& input, const at::Tensor& tokens, int64_t blank, bool wildcard) {
        std::lock_guard<std::mutex> guard(mutex_);
        TORCH_CHECK(core_ != nullptr, "mps_ctc_context_closed");
        TORCH_CHECK(input.device().type() == c10::DeviceType::MPS && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous() && !input.requires_grad() && !input.is_neg() && !input.is_conj(), "mps_ctc_tensor_not_supported");
        TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 32768 && input.size(1) > 0 && input.size(1) <= 4096 && input.numel() <= 16777216 && blank >= 0 && blank < input.size(1), "mps_ctc_shape_or_blank_invalid");
        TORCH_CHECK(tokens.device().type() == c10::DeviceType::CPU && tokens.scalar_type() == at::kInt && tokens.dim() == 1 && tokens.is_contiguous() && !tokens.is_neg() && tokens.numel() > 0 && tokens.numel() <= input.size(0), "mps_ctc_tokens_not_supported");
        @autoreleasepool {
            const at::Tensor owner = input, token_owner = tokens;
            const uint64_t bytes = owner.numel() * 4, offset = owner.storage_offset() * 4;
            TORCH_CHECK(owner.storage_offset() >= 0 && offset <= owner.storage().nbytes() && bytes <= owner.storage().nbytes() - offset, "mps_ctc_storage_range_invalid");
            id<MTLBuffer> source = (__bridge id<MTLBuffer>)owner.storage().data();
            TORCH_CHECK(source != nil && offset <= source.length && bytes <= source.length - offset, "mps_ctc_metal_range_invalid");
            const auto started = Clock::now();
            auto* stream = at::mps::getCurrentMPSStream();
            TORCH_CHECK(source.device == stream->commandQueue().device, "mps_ctc_device_mismatch");
            __block std::exception_ptr sync_error;
            const auto sync_started = Clock::now();
            dispatch_sync(stream->queue(), ^{ try { stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT); } catch (...) { sync_error = std::current_exception(); } });
            if (sync_error) std::rethrow_exception(sync_error);
            const double producer_wait = seconds(sync_started);
            Dimensions dimensions{static_cast<uint32_t>(owner.size(0)), static_cast<uint32_t>(owner.size(1)), static_cast<uint32_t>(tokens.numel()), static_cast<int32_t>(blank)};
            auto completed = core_->compute(nullptr, owner.numel(), token_owner.data_ptr<int32_t>(), token_owner.numel(), dimensions, wildcard, false, false, source, offset);
            auto path = at::empty({static_cast<int64_t>(completed.path.size()), 3}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
            float* packed = path.data_ptr<float>();
            for (size_t index = 0; index < completed.path.size(); ++index) {
                packed[index * 3] = completed.path[index].token;
                packed[index * 3 + 1] = completed.path[index].frame;
                packed[index * 3 + 2] = completed.path[index].score;
            }
            auto metrics = at::empty({9}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
            double* values = metrics.data_ptr<double>();
            values[0] = completed.found;
            values[1] = producer_wait;
            values[2] = seconds(started);
            values[3] = completed.timings[0].gpu_seconds;
            values[4] = completed.timings[1].gpu_seconds;
            values[5] = completed.dp_bytes;
            values[6] = completed.allocated_bytes;
            values[7] = offset;
            values[8] = 8 + completed.path.size() * 12;
            return {path, metrics};
        }
    }
};
static auto mps_ctc_class = torch::class_<MPSCTC>("whisperx_mps_probe", "CTC").def(torch::init<std::string>()).def("compute", &MPSCTC::compute).def("close", &MPSCTC::close);
}

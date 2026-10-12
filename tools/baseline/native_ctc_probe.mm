#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define NATIVE_NPZ_INVENTORY_LIBRARY
#include "native_npz_inventory.cpp"
#include <chrono>
#include <map>

struct Dimensions { uint32_t frames; uint32_t vocabulary; uint32_t count; int32_t blank; };
static_assert(sizeof(Dimensions) == 16);
struct Timing { uint64_t wall_ns; double gpu_seconds; };

static std::vector<uint8_t> payload(std::ifstream& source, const Tensor& tensor, uint64_t size) { require(tensor.payload <= 64 * 1024 * 1024, "input_budget_invalid"); return read(source, tensor.offset, tensor.payload, size); }

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

int main(int argc, char** argv) {
    @autoreleasepool {
        opterr = 0;
        std::string input, expected, kernels_path, output_path;
        int replicas = 3;
        bool compact = false, batched = false;
        option options[]{{"input", required_argument, nullptr, 'i'}, {"sha256", required_argument, nullptr, 's'}, {"kernels", required_argument, nullptr, 'k'}, {"output", required_argument, nullptr, 'o'}, {"replicas", required_argument, nullptr, 'r'}, {"compact", no_argument, nullptr, 'c'}, {"batched", no_argument, nullptr, 'b'}, {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
        try {
            int flag;
            while ((flag = getopt_long(argc, argv, "", options, nullptr)) != -1) {
                if (flag == 'i') input = optarg; else if (flag == 's') expected = optarg; else if (flag == 'k') kernels_path = optarg; else if (flag == 'o') output_path = optarg; else if (flag == 'c') compact = true; else if (flag == 'b') batched = true;
                else if (flag == 'r') { std::string count = optarg; require(std::regex_match(count, std::regex("[0-9]+")), "arguments_invalid"); replicas = std::stoi(count); }
                else if (flag == 'h') { std::cout << "Usage: native_ctc_probe --input EMISSIONS_NPZ --sha256 HASH --kernels FILE [--replicas N] [--compact] [--batched] [--output NEW_JSON]\nNative Objective-C++/Metal CTC diagnostic; --compact keeps two FP32 rows plus byte predecessor decisions instead of the full trellis. --batched groups prefix, trellis and backtracking in one command buffer after input validation. No Python, UI, neural model loader, production backend or full quality certificate.\n"; return 0; }
                else throw InventoryError("arguments_invalid");
            }
            require(optind == argc && !input.empty() && !kernels_path.empty() && std::regex_match(expected, std::regex("[0-9a-f]{64}")) && replicas >= 1 && replicas <= 20, "arguments_invalid");
            auto size = std::filesystem::file_size(input); require(size <= 64 * 1024 * 1024, "input_budget_invalid");
            std::ifstream source(input, std::ios::binary); require(source.good() && digest(source) == expected, "input_hash_mismatch");
            auto tensors = inventory(source, size); require(tensors.size() == 3, "input_arrays_invalid");
            std::map<std::string, Tensor> arrays;
            for (const auto& tensor : tensors) arrays.emplace(tensor.name, tensor);
            require(arrays.contains("emission") && arrays.contains("tokens") && arrays.contains("blank_id"), "input_arrays_invalid");
            const auto& emission = arrays.at("emission"); const auto& tokens = arrays.at("tokens"); const auto& blank = arrays.at("blank_id");
            require(emission.dtype == "<f4" && tokens.dtype == "<i4" && blank.dtype == "<i4" && emission.shape.size() == 2 && tokens.shape.size() == 1 && blank.shape.empty() && !emission.fortran, "input_format_invalid");
            require(emission.shape[0] > 0 && emission.shape[0] <= 32768 && emission.shape[1] > 0 && emission.shape[1] <= uint64_t(1) << 24 && tokens.shape[0] > 0 && tokens.shape[0] <= emission.shape[0], "input_shape_invalid");
            uint64_t cells = (emission.shape[0] + 1) * (tokens.shape[0] + 1); require(cells <= 16 * 1024 * 1024, "trellis_budget_invalid");
            auto emission_bytes = payload(source, emission, size), token_bytes = payload(source, tokens, size), blank_bytes = payload(source, blank, size);
            int32_t blank_id; std::memcpy(&blank_id, blank_bytes.data(), sizeof(blank_id));
            Dimensions dimensions{static_cast<uint32_t>(emission.shape[0]), static_cast<uint32_t>(emission.shape[1]), static_cast<uint32_t>(tokens.shape[0]), blank_id};
            id<MTLDevice> device = MTLCreateSystemDefaultDevice(); require(device != nil && [device supportsFamily:MTLGPUFamilyApple7], "apple_metal_required");
            auto shader_size = std::filesystem::file_size(kernels_path); require(shader_size <= 1024 * 1024, "shader_budget_invalid");
            std::ifstream shader_file(kernels_path, std::ios::binary); require(shader_file.good(), "shader_read_failed");
            auto shader_hash = digest(shader_file); auto shader_bytes = read(shader_file, 0, shader_size, shader_size);
            NSString* shader = [[NSString alloc] initWithBytes:shader_bytes.data() length:shader_bytes.size() encoding:NSUTF8StringEncoding]; require(shader != nil, "shader_encoding_invalid");
            MTLCompileOptions* compile = [[MTLCompileOptions alloc] init]; compile.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:shader options:compile error:&error]; require(library != nil, "shader_compilation_failed");
            NSMutableArray<id<MTLComputePipelineState>>* pipelines = [[NSMutableArray alloc] init];
            NSArray<NSString*>* kernel_names = compact ? @[@"ctc_validate", @"ctc_prefix", @"ctc_trellis_compact", @"ctc_backtrack_compact"] : @[@"ctc_validate", @"ctc_prefix", @"ctc_trellis", @"ctc_backtrack"];
            for (NSString* name in kernel_names) {
                id<MTLFunction> function = [library newFunctionWithName:name]; require(function != nil, "shader_function_missing");
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error]; require(pipeline != nil && pipeline.maxTotalThreadsPerThreadgroup >= 256, "pipeline_creation_failed");
                [pipelines addObject:pipeline];
            }
            id<MTLCommandQueue> queue = [device newCommandQueue]; require(queue != nil, "metal_queue_creation_failed");
            id<MTLBuffer> e = [device newBufferWithBytes:emission_bytes.data() length:emission_bytes.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> t = [device newBufferWithBytes:token_bytes.data() length:token_bytes.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> invalid = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> prefix = [device newBufferWithLength:(dimensions.frames + 1) * 4 options:MTLResourceStorageModeShared];
            uint64_t trellis_bytes = compact ? 2 * (dimensions.count + 1) * 4 : cells * 4;
            id<MTLBuffer> trellis = [device newBufferWithLength:trellis_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> path = [device newBufferWithLength:(1 + dimensions.frames * 3) * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> decisions = compact ? [device newBufferWithLength:dimensions.frames * dimensions.count options:MTLResourceStorageModeShared] : nil;
            id<MTLBuffer> endpoint = compact ? [device newBufferWithLength:4 options:MTLResourceStorageModeShared] : nil;
            require(e != nil && t != nil && invalid != nil && prefix != nil && trellis != nil && path != nil, "metal_allocation_failed");
            require(!compact || (decisions != nil && endpoint != nil), "metal_allocation_failed");
            std::vector<std::vector<Timing>> timings;
            for (int replica = 0; replica <= replicas; ++replica) {
                *static_cast<uint32_t*>(invalid.contents) = 0;
                std::vector<Timing> stages;
                stages.push_back(execute(queue, pipelines[0], @[e, t, invalid], dimensions, 2, dimensions.frames * dimensions.vocabulary, 256, @"ctc_validate"));
                require(*static_cast<uint32_t*>(invalid.contents) == 0, "input_values_invalid");
                NSArray<id<MTLBuffer>>* trellis_buffers = compact ? @[e, t, prefix, trellis, decisions, endpoint] : @[e, t, prefix, trellis];
                NSArray<id<MTLBuffer>>* backtrack_buffers = compact ? @[e, t, trellis, path, decisions, endpoint] : @[e, t, trellis, path];
                if (batched) {
                    auto started = std::chrono::steady_clock::now();
                    id<MTLCommandBuffer> command = [queue commandBuffer]; require(command != nil, "metal_command_creation_failed");
                    command.label = @"ctc_prefix_trellis_backtrack";
                    // Default tracked Metal buffers preserve dependencies between these sequential encoders.
                    encode(command, pipelines[1], @[e, prefix], dimensions, 1, 1, 1, kernel_names[1]);
                    encode(command, pipelines[2], trellis_buffers, dimensions, 2, 256, 256, kernel_names[2]);
                    encode(command, pipelines[3], backtrack_buffers, dimensions, 2, 1, 1, kernel_names[3]);
                    stages.push_back(complete(command, started));
                } else {
                    stages.push_back(execute(queue, pipelines[1], @[e, prefix], dimensions, 1, 1, 1, kernel_names[1]));
                    stages.push_back(execute(queue, pipelines[2], trellis_buffers, dimensions, 2, 256, 256, kernel_names[2]));
                    stages.push_back(execute(queue, pipelines[3], backtrack_buffers, dimensions, 2, 1, 1, kernel_names[3]));
                }
                timings.push_back(stages);
            }
            const auto* result = static_cast<const float*>(path.contents); int length = static_cast<int>(result[0]);
            require(length >= -1 && length <= static_cast<int>(dimensions.frames), "gpu_path_invalid");
            std::ostringstream output; output << std::setprecision(10);
            output << "{\"status\":\"native_metal_ctc_executed\",\"pid\":" << getpid() << ",\"scope\":\"native_ctc_primitive_only\",\"input_sha256\":\"" << expected << "\",\"shader_sha256\":\"" << shader_hash << "\",\"frames\":" << dimensions.frames << ",\"tokens\":" << dimensions.count << ",\"quality_certified\":false,\"production_backend\":false,\"full_baseline_gate_passed\":false,\"python_required\":false,\"physical_copy_bytes\":null,\"batched\":" << (batched ? "true" : "false") << ",\"command_buffers_per_run\":" << (batched ? 2 : 4) << ",\"timing_kind\":\"" << (batched ? "validation_then_batched_prefix_trellis_backtrack" : "four_waited_command_buffers_validation_prefix_trellis_backtrack") << "\",\"runs\":[";
            for (size_t replica = 0; replica < timings.size(); ++replica) {
                if (replica) output << ','; output << "{\"warm\":" << (replica ? "true" : "false") << ",\"stages\":[";
                const char* names[]{"ctc_validate", "ctc_prefix", compact ? "ctc_trellis_compact" : "ctc_trellis", compact ? "ctc_backtrack_compact" : "ctc_backtrack"};
                for (size_t stage = 0; stage < timings[replica].size(); ++stage) { if (stage) output << ','; output << "{\"" << (batched && stage == 1 ? "kernel_group" : "kernel") << "\":\"" << (batched && stage == 1 ? "ctc_prefix_trellis_backtrack" : names[stage]) << "\",\"dispatch_completed\":true,\"wall_ns\":" << timings[replica][stage].wall_ns << ",\"gpu_command_elapsed_seconds\":" << timings[replica][stage].gpu_seconds << '}'; }
                output << "]}";
            }
            output << "],\"compact\":" << (compact ? "true" : "false") << ",\"dp_logical_storage_bytes\":" << (trellis_bytes + (compact ? dimensions.frames * dimensions.count + 4 : 0)) << ",\"path\":";
            if (length < 0) output << "null";
            else {
                output << '[';
                for (int index = length - 1; index >= 0; --index) { if (index != length - 1) output << ','; output << '[' << static_cast<int>(result[1 + index * 3]) << ',' << static_cast<int>(result[2 + index * 3]) << ',' << result[3 + index * 3] << ']'; }
                output << ']';
            }
            output << "}\n";
            auto document = output.str();
            if (output_path.empty()) std::cout << document;
            else {
                int descriptor = open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600); require(descriptor >= 0, "output_creation_failed");
                size_t written = 0;
                while (written < document.size()) { ssize_t count = write(descriptor, document.data() + written, document.size() - written); if (count <= 0) { close(descriptor); throw InventoryError("output_write_failed"); } written += static_cast<size_t>(count); }
                int synced = fsync(descriptor); close(descriptor); require(synced == 0, "output_write_failed");
                std::cout << "{\"status\":\"native_metal_ctc_executed\",\"python_required\":false,\"full_baseline_gate_passed\":false}\n";
            }
            return 0;
        } catch (const InventoryError& error) { std::cout << "{\"status\":\"unresolved\",\"code\":\"" << error.what() << "\"}\n"; return 2; }
        catch (...) { std::cout << "{\"status\":\"unresolved\",\"code\":\"native_ctc_failed\"}\n"; return 2; }
    }
}

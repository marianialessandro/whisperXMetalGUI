#include "native_ctc_core.hpp"
#define NATIVE_NPZ_INVENTORY_LIBRARY
#include "native_npz_inventory.cpp"
#include <map>

static std::vector<uint8_t> payload(std::ifstream& source, const Tensor& tensor, uint64_t size) { require(tensor.payload <= 64 * 1024 * 1024, "input_budget_invalid"); return read(source, tensor.offset, tensor.payload, size); }

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
            auto shader_size = std::filesystem::file_size(kernels_path); require(shader_size <= 1024 * 1024, "shader_budget_invalid");
            std::ifstream shader_file(kernels_path, std::ios::binary); require(shader_file.good(), "shader_read_failed");
            auto shader_hash = digest(shader_file); auto shader_bytes = read(shader_file, 0, shader_size, shader_size);
            NativeCTCContext context(reinterpret_cast<const char*>(shader_bytes.data()), shader_bytes.size(), compact, batched);
            std::vector<std::vector<Timing>> timings;
            NativeResult result;
            for (int replica = 0; replica <= replicas; ++replica) {
                result = context.compute(reinterpret_cast<const float*>(emission_bytes.data()), emission_bytes.size() / 4, reinterpret_cast<const int32_t*>(token_bytes.data()), token_bytes.size() / 4, dimensions);
                timings.push_back(result.timings);
            }
            std::ostringstream output; output << std::setprecision(10);
            output << "{\"status\":\"native_metal_ctc_executed\",\"pid\":" << getpid() << ",\"scope\":\"native_ctc_primitive_only\",\"input_sha256\":\"" << expected << "\",\"shader_sha256\":\"" << shader_hash << "\",\"frames\":" << dimensions.frames << ",\"tokens\":" << dimensions.count << ",\"quality_certified\":false,\"production_backend\":false,\"full_baseline_gate_passed\":false,\"python_required\":false,\"physical_copy_bytes\":null,\"batched\":" << (batched ? "true" : "false") << ",\"command_buffers_per_run\":" << (batched ? 2 : 4) << ",\"timing_kind\":\"" << (batched ? "validation_then_batched_prefix_trellis_backtrack" : "four_waited_command_buffers_validation_prefix_trellis_backtrack") << "\",\"runs\":[";
            for (size_t replica = 0; replica < timings.size(); ++replica) {
                if (replica) output << ','; output << "{\"warm\":" << (replica ? "true" : "false") << ",\"stages\":[";
                const char* names[]{"ctc_validate", "ctc_prefix", compact ? "ctc_trellis_compact" : "ctc_trellis", compact ? "ctc_backtrack_compact" : "ctc_backtrack"};
                for (size_t stage = 0; stage < timings[replica].size(); ++stage) { if (stage) output << ','; output << "{\"" << (batched && stage == 1 ? "kernel_group" : "kernel") << "\":\"" << (batched && stage == 1 ? "ctc_prefix_trellis_backtrack" : names[stage]) << "\",\"dispatch_completed\":true,\"wall_ns\":" << timings[replica][stage].wall_ns << ",\"gpu_command_elapsed_seconds\":" << timings[replica][stage].gpu_seconds << '}'; }
                output << "]}";
            }
            output << "],\"compact\":" << (compact ? "true" : "false") << ",\"dp_logical_storage_bytes\":" << result.dp_bytes << ",\"path\":";
            if (!result.found) output << "null";
            else {
                output << '[';
                for (size_t index = 0; index < result.path.size(); ++index) { if (index) output << ','; const auto& point = result.path[index]; output << '[' << point.token << ',' << point.frame << ',' << point.score << ']'; }
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

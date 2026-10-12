#include "native_ctc_bridge.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream input(argv[1], std::ios::binary);
    std::string shader((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    wxctc_context* context = nullptr;
    char error[128]{};
    assert(wxctc_abi_version() == 1);
    assert(wxctc_create(nullptr, 0, &context, error, sizeof(error)) != 0 && context == nullptr);
    assert(wxctc_create(shader.data(), shader.size(), &context, error, sizeof(error)) == 0 && context != nullptr);
    float emission[]{-3, -.1f, -.2f, -2, -.1f, -3};
    int32_t tokens[]{1};
    wxctc_point path[3]{};
    wxctc_result result{};
    assert(wxctc_compute(context, emission, 6, tokens, 1, 3, 2, 0, path, 2, &result, error, sizeof(error)) != 0);
    assert(result.path_length == -1);
    assert(wxctc_compute(context, emission, 5, tokens, 1, 3, 2, 0, path, 3, &result, error, sizeof(error)) != 0);
    assert(wxctc_compute(context, emission, 6, tokens, 1, 3, 2, -1, path, 3, &result, error, sizeof(error)) != 0);
    float maxima[3]{123, 123, 123};
    int64_t indices[3]{123, 123, 123};
    wxctc_max_result maximum_result{};
    assert(wxctc_row_max(context, emission, 6, 3, 2, maxima, indices, 2, &maximum_result, error, sizeof(error)) != 0);
    assert(maxima[0] == 123 && indices[0] == 123);
    assert(wxctc_row_max(context, emission, 5, 3, 2, maxima, indices, 3, &maximum_result, error, sizeof(error)) != 0);
    assert(wxctc_row_max(context, emission, 6, 3, 2, maxima, indices, 3, &maximum_result, error, sizeof(error)) == 0);
    assert(maximum_result.abi_version == 1 && maxima[0] == -.1f && indices[0] == 1 && indices[1] == 0);
    uint64_t grown = 0;
    for (int iteration = 0; iteration < 8; ++iteration) {
        std::vector<float> large(256 * 4, -1);
        int32_t text[]{1, 2, 3};
        std::vector<wxctc_point> large_path(256);
        assert(wxctc_compute(context, large.data(), large.size(), text, 3, 256, 4, 0, large_path.data(), large_path.size(), &result, error, sizeof(error)) == 0);
        assert(result.path_length >= 3);
        if (iteration) assert(grown == result.allocated_shared_buffer_bytes);
        grown = result.allocated_shared_buffer_bytes;
        assert(wxctc_compute(context, emission, 6, tokens, 1, 3, 2, 0, path, 3, &result, error, sizeof(error)) == 0);
        assert(result.abi_version == 1 && result.path_length > 0);
        assert(result.allocated_shared_buffer_bytes == grown);
        assert(path[0].token == 0 && path[0].frame == 0);
    }
    wxctc_destroy(context);
    wxctc_destroy(nullptr);
}

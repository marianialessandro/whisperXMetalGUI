#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

struct EmbeddingMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
extern "C" int wxembedding_create(const char*, uint64_t, void**, char*, uint32_t);
extern "C" int wxembedding_normalize(void*, const float*, uint32_t, uint32_t, float*, EmbeddingMetrics*, char*, uint32_t);
extern "C" void wxembedding_destroy(void*);

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream stream(argv[1]);
    std::string shader((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    void* context = nullptr;
    char error[128] = {};
    EmbeddingMetrics metrics = {};
    float input[] = {3.f, 4.f}, output[] = {91.f, 92.f};
    assert(wxembedding_create(shader.data(), shader.size(), nullptr, error, sizeof(error)) != 0);
    assert(wxembedding_create(nullptr, 0, &context, error, sizeof(error)) != 0 && context == nullptr);
    assert(wxembedding_create(shader.data(), shader.size(), &context, error, sizeof(error)) == 0 && context != nullptr);
    assert(wxembedding_normalize(context, input, 1, 2, output, &metrics, error, sizeof(error)) == 0);
    assert(output[0] == .6f && output[1] == .8f && metrics.version == 1 && metrics.kernels == 1);
    output[0] = 91.f; output[1] = 92.f;
    assert(wxembedding_normalize(context, input, 0, 2, output, &metrics, error, sizeof(error)) != 0);
    assert(wxembedding_normalize(context, input, 1, 1025, output, &metrics, error, sizeof(error)) != 0);
    assert(wxembedding_normalize(context, nullptr, 1, 2, output, &metrics, error, sizeof(error)) != 0);
    assert(wxembedding_normalize(context, input, 1, 2, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxembedding_normalize(context, input, 1, 2, output, nullptr, error, sizeof(error)) != 0);
    assert(wxembedding_normalize(nullptr, input, 1, 2, output, &metrics, nullptr, 0) != 0);
    input[0] = input[1] = 0.f;
    assert(wxembedding_normalize(context, input, 1, 2, output, &metrics, error, sizeof(error)) != 0);
    input[0] = std::numeric_limits<float>::infinity();
    assert(wxembedding_normalize(context, input, 1, 2, output, &metrics, error, sizeof(error)) != 0);
    assert(output[0] == 91.f && output[1] == 92.f);
    input[0] = 3.f; input[1] = 4.f;
    assert(wxembedding_normalize(context, input, 1, 2, output, &metrics, error, sizeof(error)) == 0);
    wxembedding_destroy(context);
    wxembedding_destroy(nullptr);
}

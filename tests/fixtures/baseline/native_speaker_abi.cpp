#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>

struct SpeakerMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
extern "C" int wxspeaker_create(const char*, uint64_t, void**, char*, uint32_t);
extern "C" int wxspeaker_assign(void*, const double*, const uint32_t*, const double*, uint32_t, uint32_t, uint32_t, uint32_t, int32_t*, SpeakerMetrics*, char*, uint32_t);
extern "C" void wxspeaker_destroy(void*);

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream input(argv[1]);
    std::string shader((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    void* context = nullptr;
    char error[128] = {};
    assert(wxspeaker_create(nullptr, 0, &context, error, sizeof(error)) == 1 && context == nullptr);
    assert(wxspeaker_create(shader.data(), shader.size(), &context, error, sizeof(error)) == 0 && context != nullptr);
    double turns[] = {2., 4., 0., 2.}, queries[] = {1., 3.};
    uint32_t labels[] = {0, 1};
    int32_t output = -1;
    SpeakerMetrics metrics = {};
    assert(wxspeaker_assign(nullptr, turns, labels, queries, 2, 1, 2, 0, &output, &metrics, error, sizeof(error)) == 1);
    assert(wxspeaker_assign(context, turns, labels, queries, 0, 1, 2, 0, &output, &metrics, error, sizeof(error)) == 1);
    assert(wxspeaker_assign(context, turns, labels, queries, 2, 1, 2, 0, nullptr, &metrics, error, sizeof(error)) == 1);
    labels[1] = 2;
    assert(wxspeaker_assign(context, turns, labels, queries, 2, 1, 2, 0, &output, &metrics, error, sizeof(error)) == 1);
    labels[1] = 1;
    queries[0] = 4.;
    assert(wxspeaker_assign(context, turns, labels, queries, 2, 1, 2, 0, &output, &metrics, error, sizeof(error)) == 1);
    queries[0] = 1.;
    assert(wxspeaker_assign(context, turns, labels, queries, 2, 1, 2, 0, &output, &metrics, error, sizeof(error)) == 0);
    assert(output == 1 && metrics.version == 1 && metrics.kernels == 2 && metrics.gpu_seconds > 0);
    wxspeaker_destroy(context);
    wxspeaker_destroy(nullptr);
}

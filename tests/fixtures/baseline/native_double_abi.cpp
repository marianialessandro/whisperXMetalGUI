#include <cassert>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

struct DoubleMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
extern "C" int wxdouble_create(const char*, uint64_t, void**, char*, uint32_t);
extern "C" int wxdouble_normalize(void*, const double*, uint32_t, uint32_t, uint32_t, double*, DoubleMetrics*, char*, uint32_t);
extern "C" void wxdouble_destroy(void*);

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream stream(argv[1]);
    std::string shader((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    void* context = nullptr;
    char error[128] = {};
    DoubleMetrics metrics = {};
    double input[] = {3., 4.}, output[] = {91., 92.};
    assert(wxdouble_create(shader.data(), shader.size(), nullptr, error, sizeof(error)) != 0);
    assert(wxdouble_create(nullptr, 0, &context, error, sizeof(error)) != 0 && context == nullptr);
    assert(wxdouble_create(shader.data(), shader.size(), &context, error, sizeof(error)) == 0 && context != nullptr);
    assert(wxdouble_normalize(context, input, 1, 2, 0, output, &metrics, error, sizeof(error)) == 0);
    assert(output[0] == .6 && output[1] == .8 && metrics.version == 1 && metrics.kernels == 1);
    output[0] = 91.; output[1] = 92.;
    assert(wxdouble_normalize(context, input, 0, 2, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_normalize(context, input, 1, 1025, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_normalize(context, nullptr, 1, 2, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_normalize(context, input, 1, 2, 0, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_normalize(context, input, 1, 2, 0, output, nullptr, error, sizeof(error)) != 0);
    assert(wxdouble_normalize(nullptr, input, 1, 2, 0, output, &metrics, nullptr, 0) != 0);
    input[0] = input[1] = 0.;
    assert(wxdouble_normalize(context, input, 1, 2, 0, output, &metrics, error, sizeof(error)) != 0);
    input[0] = std::numeric_limits<double>::infinity();
    assert(wxdouble_normalize(context, input, 1, 2, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(output[0] == 91. && output[1] == 92.);
    input[0] = 3.; input[1] = 4.;
    assert(wxdouble_normalize(context, input, 1, 2, 0, output, &metrics, error, sizeof(error)) == 0);
    wxdouble_destroy(context);
    wxdouble_destroy(nullptr);
}

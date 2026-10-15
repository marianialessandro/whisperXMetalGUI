#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

struct DoubleMetrics { uint32_t version, kernels; uint64_t wall_ns; double gpu_seconds; uint64_t allocated_bytes; };
extern "C" int wxdouble_create(const char*, uint64_t, void**, char*, uint32_t);
extern "C" int wxdouble_normalize(void*, const double*, uint32_t, uint32_t, uint32_t, double*, DoubleMetrics*, char*, uint32_t);
extern "C" int wxdouble_matmul(void*, const double*, const double*, uint32_t, uint32_t, uint32_t, double*, DoubleMetrics*, char*, uint32_t);
extern "C" int wxdouble_elementwise(void*, const double*, const double*, uint32_t, uint32_t, uint32_t, uint32_t, double*, DoubleMetrics*, char*, uint32_t);
extern "C" int wxdouble_vbx_precompute(void*, const double*, const double*, uint32_t, uint32_t, uint32_t, double*, double*, double*, DoubleMetrics*, char*, uint32_t);
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
    input[0] = 3.; input[1] = 4.;
    double right[] = {2., -1.};
    assert(wxdouble_matmul(context, input, right, 1, 2, 1, output, &metrics, error, sizeof(error)) == 0 && output[0] == 2.);
    output[0] = 91.;
    assert(wxdouble_matmul(context, input, right, 1, 1025, 1, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_matmul(context, input, right, 4096, 1024, 9, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_matmul(nullptr, input, right, 1, 2, 1, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_matmul(context, nullptr, right, 1, 2, 1, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_matmul(context, input, nullptr, 1, 2, 1, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_matmul(context, input, right, 1, 2, 1, nullptr, &metrics, error, sizeof(error)) != 0);
    assert(output[0] == 91.);
    double mean[] = {1., 2.};
    assert(wxdouble_elementwise(context, input, mean, 1, 2, 0, 0, output, &metrics, error, sizeof(error)) == 0 && output[0] == 2. && output[1] == 2.);
    assert(wxdouble_elementwise(context, input, nullptr, 1, 2, 1, 2, output, &metrics, error, sizeof(error)) == 0 && output[0] == 3. * std::sqrt(2.));
    output[0] = 91.;
    assert(wxdouble_elementwise(context, input, nullptr, 1, 2, 0, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_elementwise(context, input, mean, 1, 2, 2, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_elementwise(context, input, nullptr, 1, 2, 1, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_elementwise(context, input, nullptr, 1, 2, 1, 1025, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_elementwise(context, input, mean, 1, 2, 1, 2, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_elementwise(context, nullptr, mean, 1, 2, 0, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_elementwise(context, input, mean, 1, 1025, 0, 0, output, &metrics, error, sizeof(error)) != 0);
    assert(output[0] == 91.);
    double phi[] = {4., 9.}, g[] = {91.}, v[] = {92., 93.}, rho[] = {94., 95.};
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 0, g, v, rho, &metrics, error, sizeof(error)) == 0);
    assert(v[0] == 2. && v[1] == 3. && rho[0] == 6. && rho[1] == 12. && g[0] < 0 && metrics.kernels == 2);
    g[0] = 91.; v[0] = 92.; rho[0] = 94.;
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 1025, 0, g, v, rho, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 2, g, v, rho, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 0, nullptr, v, rho, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 0, g, nullptr, rho, &metrics, error, sizeof(error)) != 0);
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 0, g, v, nullptr, &metrics, error, sizeof(error)) != 0);
    phi[0] = -1.;
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 0, g, v, rho, &metrics, error, sizeof(error)) != 0);
    phi[0] = 4.;
    assert(g[0] == 91. && v[0] == 92. && rho[0] == 94.);
    wxdouble_destroy(context);
    wxdouble_destroy(nullptr);
    std::string legacy_shader = shader.substr(0, shader.find("\nstruct MatrixDimensions"));
    context = nullptr;
    assert(wxdouble_create(legacy_shader.data(), legacy_shader.size(), &context, error, sizeof(error)) == 0 && context != nullptr);
    assert(wxdouble_normalize(context, input, 1, 2, 0, output, &metrics, error, sizeof(error)) == 0 && output[0] == .6 && output[1] == .8);
    output[0] = 91.;
    assert(wxdouble_matmul(context, input, right, 1, 2, 1, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91.);
    assert(wxdouble_elementwise(context, input, nullptr, 1, 2, 1, 2, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91.);
    wxdouble_destroy(context);
    std::string matrix_shader = shader.substr(0, shader.find("\nstruct ElementDimensions"));
    context = nullptr;
    assert(wxdouble_create(matrix_shader.data(), matrix_shader.size(), &context, error, sizeof(error)) == 0 && context != nullptr);
    assert(wxdouble_matmul(context, input, right, 1, 2, 1, output, &metrics, error, sizeof(error)) == 0 && output[0] == 2.);
    output[0] = 91.;
    assert(wxdouble_elementwise(context, input, nullptr, 1, 2, 1, 2, output, &metrics, error, sizeof(error)) != 0 && output[0] == 91.);
    assert(wxdouble_vbx_precompute(context, input, phi, 1, 2, 0, g, v, rho, &metrics, error, sizeof(error)) != 0 && g[0] == 91.);
    wxdouble_destroy(context);
}

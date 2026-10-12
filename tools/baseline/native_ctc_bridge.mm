#include "native_ctc_core.hpp"
#include "native_ctc_bridge.h"
#include <memory>

struct wxctc_context { NativeCTCContext core; wxctc_context(const char* shader, uint64_t size) : core(shader, size, true, true) {} };
static_assert(sizeof(wxctc_point) == 12);
static_assert(sizeof(wxctc_segment) == 16);
static_assert(sizeof(wxctc_result) == 56);
static_assert(sizeof(wxctc_max_result) == 48);

static void set_error(char* destination, uint32_t capacity, const char* code) {
    if (destination != nullptr && capacity > 0) { std::strncpy(destination, code, capacity - 1); destination[capacity - 1] = 0; }
}

uint32_t wxctc_abi_version(void) { return 1; }

int wxctc_create(const char* shader, uint64_t shader_bytes, wxctc_context** output, char* error_code, uint32_t error_capacity) {
    @autoreleasepool {
        if (output == nullptr) { set_error(error_code, error_capacity, "output_pointer_invalid"); return 1; }
        *output = nullptr;
        set_error(error_code, error_capacity, "");
        try { *output = new wxctc_context(shader, shader_bytes); return 0; }
        catch (const InventoryError& error) { set_error(error_code, error_capacity, error.what()); return 1; }
        catch (...) { set_error(error_code, error_capacity, "native_context_creation_failed"); return 1; }
    }
}

static int compute_result(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, wxctc_point* path, uint32_t path_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity, bool wildcard) {
    @autoreleasepool {
        set_error(error_code, error_capacity, "");
        if (result != nullptr) { *result = {}; result->path_length = -1; result->abi_version = 1; }
        try {
            require(context != nullptr && result != nullptr && path != nullptr && path_capacity >= frames && token_elements <= UINT32_MAX, "output_or_context_invalid");
            auto completed = context->core.compute(emission, emission_elements, tokens, token_elements, {frames, vocabulary, static_cast<uint32_t>(token_elements), blank}, wildcard);
            for (size_t i = 0; i < completed.path.size(); ++i) path[i] = {completed.path[i].token, completed.path[i].frame, completed.path[i].score};
            *result = {completed.found ? static_cast<int32_t>(completed.path.size()) : -1, 1, completed.timings[0].wall_ns, completed.timings[1].wall_ns, completed.timings[0].gpu_seconds, completed.timings[1].gpu_seconds, completed.dp_bytes, completed.allocated_bytes};
            return 0;
        } catch (const InventoryError& error) { set_error(error_code, error_capacity, error.what()); return 1; }
        catch (...) { set_error(error_code, error_capacity, "native_ctc_execution_failed"); return 1; }
    }
}

int wxctc_compute(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, wxctc_point* path, uint32_t path_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity) {
    return compute_result(context, emission, emission_elements, tokens, token_elements, frames, vocabulary, blank, path, path_capacity, result, error_code, error_capacity, false);
}

int wxctc_compute_wildcard(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, wxctc_point* path, uint32_t path_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity) {
    return compute_result(context, emission, emission_elements, tokens, token_elements, frames, vocabulary, blank, path, path_capacity, result, error_code, error_capacity, true);
}

int wxctc_compute_segments(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, uint32_t wildcard, wxctc_segment* segments, uint32_t segment_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity) {
    @autoreleasepool {
        set_error(error_code, error_capacity, "");
        if (result != nullptr) { *result = {}; result->path_length = -1; result->abi_version = 1; }
        try {
            require(context != nullptr && result != nullptr && segments != nullptr && segment_capacity >= token_elements && token_elements <= UINT32_MAX && wildcard <= 1, "output_or_context_invalid");
            auto completed = context->core.compute(emission, emission_elements, tokens, token_elements, {frames, vocabulary, static_cast<uint32_t>(token_elements), blank}, wildcard, true);
            for (size_t i = 0; i < completed.segments.size(); ++i) segments[i] = {completed.segments[i].start, completed.segments[i].end, completed.segments[i].score};
            *result = {completed.path_length, 1, completed.timings[0].wall_ns, completed.timings[1].wall_ns, completed.timings[0].gpu_seconds, completed.timings[1].gpu_seconds, completed.dp_bytes, completed.allocated_bytes};
            return 0;
        } catch (const InventoryError& error) { set_error(error_code, error_capacity, error.what()); return 1; }
        catch (...) { set_error(error_code, error_capacity, "native_character_segment_execution_failed"); return 1; }
    }
}

int wxctc_row_max(wxctc_context* context, const float* emission, uint64_t emission_elements, uint32_t frames, uint32_t vocabulary, float* values, int64_t* indices, uint32_t output_capacity, wxctc_max_result* result, char* error_code, uint32_t error_capacity) {
    @autoreleasepool {
        set_error(error_code, error_capacity, "");
        if (result != nullptr) { *result = {}; result->abi_version = 1; }
        try {
            require(context != nullptr && result != nullptr && values != nullptr && indices != nullptr && output_capacity >= frames, "output_or_context_invalid");
            auto completed = context->core.row_max(emission, emission_elements, frames, vocabulary);
            for (uint32_t i = 0; i < frames; ++i) { values[i] = completed.values[i]; indices[i] = completed.indices[i]; }
            *result = {1, 0, completed.timings[0].wall_ns, completed.timings[1].wall_ns, completed.timings[0].gpu_seconds, completed.timings[1].gpu_seconds, completed.allocated_bytes};
            return 0;
        } catch (const InventoryError& error) { set_error(error_code, error_capacity, error.what()); return 1; }
        catch (...) { set_error(error_code, error_capacity, "native_row_max_execution_failed"); return 1; }
    }
}

int wxctc_timestamp_bounds(wxctc_context* context, const uint64_t* values, uint64_t count, uint64_t* output, uint32_t output_capacity, wxctc_max_result* result, char* error_code, uint32_t error_capacity) {
    @autoreleasepool {
        set_error(error_code, error_capacity, "");
        if (result != nullptr) { *result = {}; result->abi_version = 1; }
        try {
            require(context != nullptr && output != nullptr && result != nullptr && output_capacity >= 2, "output_or_context_invalid");
            auto completed = context->core.timestamp_bounds(values, count);
            output[0] = completed.minimum; output[1] = completed.maximum;
            *result = {1, 0, 0, completed.timing.wall_ns, 0, completed.timing.gpu_seconds, completed.allocated_bytes};
            return 0;
        } catch (const InventoryError& error) { set_error(error_code, error_capacity, error.what()); return 1; }
        catch (...) { set_error(error_code, error_capacity, "native_timestamp_bounds_execution_failed"); return 1; }
    }
}

void wxctc_destroy(wxctc_context* context) { @autoreleasepool { delete context; } }

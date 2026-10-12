#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Development ABI v1: synchronous calls copy input, finish GPU work, then copy the final path into caller-owned memory.
typedef struct wxctc_context wxctc_context;
typedef struct wxctc_point { int32_t token; int32_t frame; float score; } wxctc_point;
typedef struct wxctc_result { int32_t path_length; uint32_t abi_version; uint64_t validation_wall_ns; uint64_t computation_wall_ns; double validation_gpu_seconds; double computation_gpu_seconds; uint64_t dp_logical_bytes; uint64_t allocated_shared_buffer_bytes; } wxctc_result;

// Arrays must have the declared lengths and remain valid until the call returns; path_capacity must be at least frames.
uint32_t wxctc_abi_version(void);
int wxctc_create(const char* shader, uint64_t shader_bytes, wxctc_context** output, char* error_code, uint32_t error_capacity);
int wxctc_compute(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, wxctc_point* path, uint32_t path_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity);
// Calls on one context are serialized; destroy only after all calls have returned, without racing a new call.
void wxctc_destroy(wxctc_context* context);

#ifdef __cplusplus
}
#endif

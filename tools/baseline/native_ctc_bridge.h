#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Development ABI v1: synchronous calls copy input, finish GPU work, then copy the final path into caller-owned memory.
typedef struct wxctc_context wxctc_context;
typedef struct wxctc_point { int32_t token; int32_t frame; float score; } wxctc_point;
typedef struct wxctc_segment { int32_t start; int32_t end; double score; } wxctc_segment;
typedef struct wxctc_result { int32_t path_length; uint32_t abi_version; uint64_t validation_wall_ns; uint64_t computation_wall_ns; double validation_gpu_seconds; double computation_gpu_seconds; uint64_t dp_logical_bytes; uint64_t allocated_shared_buffer_bytes; } wxctc_result;
typedef struct wxctc_max_result { uint32_t abi_version; uint32_t reserved; uint64_t validation_wall_ns; uint64_t computation_wall_ns; double validation_gpu_seconds; double computation_gpu_seconds; uint64_t allocated_shared_buffer_bytes; } wxctc_max_result;

// Arrays must have the declared lengths and remain valid until the call returns; path_capacity must be at least frames.
uint32_t wxctc_abi_version(void);
int wxctc_create(const char* shader, uint64_t shader_bytes, wxctc_context** output, char* error_code, uint32_t error_capacity);
int wxctc_compute(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, wxctc_point* path, uint32_t path_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity);
// Optional ABI v1 extension: input has no wildcard column; tokens equal to vocabulary use the GPU-generated nonblank maximum column.
int wxctc_compute_wildcard(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, wxctc_point* path, uint32_t path_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity);
// Optional ABI v1 extension: returns one frame-bound/score segment per token with CPython 3.12 compensated float-sum semantics, without reading back the full path; wildcard is 0 or 1.
int wxctc_compute_segments(wxctc_context* context, const float* emission, uint64_t emission_elements, const int32_t* tokens, uint64_t token_elements, uint32_t frames, uint32_t vocabulary, int32_t blank, uint32_t wildcard, wxctc_segment* segments, uint32_t segment_capacity, wxctc_result* result, char* error_code, uint32_t error_capacity);
// Optional ABI v1 extension for nonpositive log emissions; -infinity is supported, NaN/+infinity are rejected, and ties select the first column.
int wxctc_row_max(wxctc_context* context, const float* emission, uint64_t emission_elements, uint32_t frames, uint32_t vocabulary, float* values, int64_t* indices, uint32_t output_capacity, wxctc_max_result* result, char* error_code, uint32_t error_capacity);
// Optional ABI v1 extension: binary64 min/max skipping NaN; empty/all-NaN returns NaN; negative zero is rejected to avoid runtime-dependent tie semantics.
int wxctc_timestamp_bounds(wxctc_context* context, const uint64_t* values, uint64_t count, uint64_t* output, uint32_t output_capacity, wxctc_max_result* result, char* error_code, uint32_t error_capacity);
// Optional ABI v1 extension: CSR subsets of a shared binary64 input; output_capacity is measured in uint64 elements and must cover twice the query count.
int wxctc_timestamp_bounds_batch(wxctc_context* context, const uint64_t* values, uint64_t count, const uint32_t* indices, uint64_t index_count, const uint32_t* offsets, uint64_t offset_count, uint64_t* output, uint32_t output_capacity, wxctc_max_result* result, char* error_code, uint32_t error_capacity);
// Optional ABI v1 extension: NumPy 2.2.6 pairwise means for rounded nonnegative scores (zero or 0.001..1) and NaN, using CSR queries of at most 32768 elements.
int wxctc_word_means_batch(wxctc_context* context, const uint64_t* values, uint64_t count, const uint32_t* indices, uint64_t index_count, const uint32_t* offsets, uint64_t offset_count, uint64_t* output, uint32_t output_capacity, wxctc_max_result* result, char* error_code, uint32_t error_capacity);
// Calls on one context are serialized; destroy only after all calls have returned, without racing a new call.
void wxctc_destroy(wxctc_context* context);

#ifdef __cplusplus
}
#endif

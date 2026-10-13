"""Development-only synchronous C ABI bridge to persistent native Metal CTC; no production backend promotion."""

import ctypes as ct
import hashlib
from pathlib import Path
import threading
import time
import sys

from tools.baseline import manifest


class Point(ct.Structure):
    _fields_ = [("token", ct.c_int32), ("frame", ct.c_int32), ("score", ct.c_float)]


class Segment(ct.Structure):
    _fields_ = [("start", ct.c_int32), ("end", ct.c_int32), ("score", ct.c_double)]


class Result(ct.Structure):
    _fields_ = [("path_length", ct.c_int32), ("abi_version", ct.c_uint32), ("validation_wall_ns", ct.c_uint64), ("computation_wall_ns", ct.c_uint64), ("validation_gpu_seconds", ct.c_double), ("computation_gpu_seconds", ct.c_double), ("dp_logical_bytes", ct.c_uint64), ("allocated_shared_buffer_bytes", ct.c_uint64)]


class MaxResult(ct.Structure):
    _fields_ = [("abi_version", ct.c_uint32), ("reserved", ct.c_uint32), ("validation_wall_ns", ct.c_uint64), ("computation_wall_ns", ct.c_uint64), ("validation_gpu_seconds", ct.c_double), ("computation_gpu_seconds", ct.c_double), ("allocated_shared_buffer_bytes", ct.c_uint64)]


class NativeCTC:
    def __init__(self, library, expected_sha256):
        started = time.perf_counter_ns()
        library = Path(library).resolve(strict=True)
        if manifest.sha256(library) != expected_sha256:
            raise ValueError("native_library_hash_mismatch")
        self._lock = threading.RLock()
        self._context = ct.c_void_p()
        self._library = ct.CDLL(str(library))
        self._library.wxctc_abi_version.argtypes = []
        self._library.wxctc_abi_version.restype = ct.c_uint32
        if self._library.wxctc_abi_version() != 1 or ct.sizeof(Point) != 12 or ct.sizeof(Result) != 56:
            raise ValueError("native_abi_version_mismatch")
        self._library.wxctc_create.argtypes = [ct.c_char_p, ct.c_uint64, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_char), ct.c_uint32]
        self._library.wxctc_create.restype = ct.c_int
        self._library.wxctc_compute.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint64, ct.POINTER(ct.c_int32), ct.c_uint64, ct.c_uint32, ct.c_uint32, ct.c_int32, ct.POINTER(Point), ct.c_uint32, ct.POINTER(Result), ct.POINTER(ct.c_char), ct.c_uint32]
        self._library.wxctc_compute.restype = ct.c_int
        self._library.wxctc_destroy.argtypes = [ct.c_void_p]
        self._library.wxctc_destroy.restype = None
        shader_path = Path(__file__).with_name("native_ctc.metal")
        shader = shader_path.read_bytes()
        error = ct.create_string_buffer(128)
        if self._library.wxctc_create(shader, len(shader), ct.byref(self._context), error, len(error)):
            raise ValueError(error.value.decode("ascii"))
        self.identity = {"library_sha256": expected_sha256, "shader_sha256": hashlib.sha256(shader).hexdigest(), "abi_version": 1}
        self.initialization_wall_ns = time.perf_counter_ns() - started

    def compute(self, emission, tokens, blank=0):
        return self._compute(emission, tokens, blank, False)

    def compute_wildcard(self, emission, tokens, blank=0):
        return self._compute(emission, tokens, blank, True)

    def compute_segments(self, emission, tokens, blank=0, wildcard=False):
        if sys.version_info[:2] != (3, 12): raise ValueError("native_character_sum_runtime_not_supported")
        return self._compute(emission, tokens, blank, wildcard, True)

    def _compute(self, emission, tokens, blank, wildcard, segments=False):
        import numpy as np
        if not isinstance(emission, np.ndarray) or emission.dtype != np.float32 or emission.ndim != 2 or not isinstance(tokens, np.ndarray) or tokens.dtype != np.int32 or tokens.ndim != 1:
            raise ValueError("native_input_format_invalid")
        frames, vocabulary = emission.shape
        count = len(tokens)
        if not 0 < count <= frames <= 32768 or not 0 < vocabulary <= 1 << 24 or emission.nbytes > 64 * 1024 * 1024 or (frames + 1) * (count + 1) > 16 * 1024 * 1024 or type(blank) is not int or not -(1 << 31) <= blank < 1 << 31:
            raise ValueError("native_input_shape_invalid")
        if wildcard and (vocabulary < 2 or frames * (vocabulary + 1) * 4 > 64 * 1024 * 1024): raise ValueError("native_wildcard_shape_invalid")
        with self._lock:
            if not self._context.value:
                raise ValueError("native_context_closed")
            started = time.perf_counter_ns()
            emission, tokens = np.ascontiguousarray(emission), np.ascontiguousarray(tokens)
            path, result, error = (Segment * count)() if segments else (Point * frames)(), Result(), ct.create_string_buffer(128)
            if segments:
                if ct.sizeof(Segment) != 16: raise ValueError("native_segment_abi_mismatch")
                function = self._library.wxctc_compute_segments
                function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint64, ct.POINTER(ct.c_int32), ct.c_uint64, ct.c_uint32, ct.c_uint32, ct.c_int32, ct.c_uint32, ct.POINTER(Segment), ct.c_uint32, ct.POINTER(Result), ct.POINTER(ct.c_char), ct.c_uint32]
                function.restype = ct.c_int
                status = function(self._context, emission.ctypes.data_as(ct.POINTER(ct.c_float)), emission.size, tokens.ctypes.data_as(ct.POINTER(ct.c_int32)), tokens.size, frames, vocabulary, blank, int(wildcard), path, count, ct.byref(result), error, len(error))
            else:
                function = self._library.wxctc_compute_wildcard if wildcard else self._library.wxctc_compute
                if wildcard:
                    function.argtypes = self._library.wxctc_compute.argtypes
                    function.restype = ct.c_int
                status = function(self._context, emission.ctypes.data_as(ct.POINTER(ct.c_float)), emission.size, tokens.ctypes.data_as(ct.POINTER(ct.c_int32)), tokens.size, frames, vocabulary, blank, path, frames, ct.byref(result), error, len(error))
            if status:
                raise ValueError(error.value.decode("ascii"))
            if result.abi_version != 1 or not -1 <= result.path_length <= frames:
                raise ValueError("native_result_invalid")
            points = None if result.path_length < 0 else ([(path[i].start, path[i].end, path[i].score) for i in range(count)] if segments else [(path[i].token, path[i].frame, path[i].score) for i in range(result.path_length)])
            kernels = ["ctc_validate_wildcard", "ctc_expand_wildcard", "ctc_prefix", "ctc_trellis_compact", "ctc_backtrack_compact"] if wildcard else ["ctc_validate", "ctc_prefix", "ctc_trellis_compact", "ctc_backtrack_compact"]
            if segments: kernels.append("ctc_merge_repeats")
            return points, {"frames": frames, "tokens": count, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "logical_cpu_to_native_emission_bytes": emission.nbytes, "logical_final_path_readback_bytes": 0 if segments else 4 + (0 if points is None else 12 * len(points)), "logical_character_segment_readback_bytes": 0 if not segments or points is None else 16 * count, "logical_control_path_length_bytes": 4, "character_segments_gpu": segments, "logical_intermediate_wildcard_readback_bytes": 0, "wildcard_fused": wildcard, "gpu_kernels": kernels, "physical_copy_bytes": None, "transport": "synchronous_in_memory_c_abi", "command_buffers_per_run": 2, "validation_wall_ns": result.validation_wall_ns, "computation_wall_ns": result.computation_wall_ns, "validation_gpu_seconds": result.validation_gpu_seconds, "computation_gpu_seconds": result.computation_gpu_seconds, "dp_logical_storage_bytes": result.dp_logical_bytes, "allocated_shared_buffer_bytes": result.allocated_shared_buffer_bytes}

    def row_max(self, emission):
        import numpy as np
        if not isinstance(emission, np.ndarray) or emission.dtype != np.float32 or emission.ndim != 2:
            raise ValueError("native_input_format_invalid")
        frames, vocabulary = emission.shape
        if not 0 < frames <= 32768 or not 0 < vocabulary <= 1 << 24 or emission.nbytes > 64 * 1024 * 1024:
            raise ValueError("native_input_shape_invalid")
        with self._lock:
            if not self._context.value:
                raise ValueError("native_context_closed")
            started = time.perf_counter_ns()
            function = self._library.wxctc_row_max
            function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_uint64, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_float), ct.POINTER(ct.c_int64), ct.c_uint32, ct.POINTER(MaxResult), ct.POINTER(ct.c_char), ct.c_uint32]
            function.restype = ct.c_int
            if ct.sizeof(MaxResult) != 48: raise ValueError("native_row_max_abi_mismatch")
            emission = np.ascontiguousarray(emission)
            values, indices = np.empty(frames, dtype=np.float32), np.empty(frames, dtype=np.int64)
            result, error = MaxResult(), ct.create_string_buffer(128)
            status = function(self._context, emission.ctypes.data_as(ct.POINTER(ct.c_float)), emission.size, frames, vocabulary, values.ctypes.data_as(ct.POINTER(ct.c_float)), indices.ctypes.data_as(ct.POINTER(ct.c_int64)), frames, ct.byref(result), error, len(error))
            if status: raise ValueError(error.value.decode("ascii"))
            if result.abi_version != 1: raise ValueError("native_result_invalid")
            return values, indices, {"scope": "experimental_wildcard_row_max", "frames": frames, "vocabulary": vocabulary, "kernel": "ctc_row_max", "dispatch_completed": True, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "logical_cpu_to_native_emission_bytes": emission.nbytes, "logical_output_readback_bytes": frames * 12, "physical_copy_bytes": None, "transport": "synchronous_in_memory_c_abi", "command_buffers_per_run": 2, "validation_wall_ns": result.validation_wall_ns, "computation_wall_ns": result.computation_wall_ns, "validation_gpu_seconds": result.validation_gpu_seconds, "computation_gpu_seconds": result.computation_gpu_seconds, "allocated_shared_buffer_bytes": result.allocated_shared_buffer_bytes}

    def timestamp_bounds_batch(self, values, indices, offsets):
        import numpy as np
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 1 or values.size > 65536 or not isinstance(indices, np.ndarray) or indices.dtype != np.uint32 or indices.ndim != 1 or indices.size > 4 * 1024 * 1024 or not isinstance(offsets, np.ndarray) or offsets.dtype != np.uint32 or offsets.ndim != 1 or not 2 <= offsets.size <= 32769: raise ValueError("native_timestamp_batch_format_invalid")
        with self._lock:
            if not self._context.value: raise ValueError("native_context_closed")
            started = time.perf_counter_ns()
            values, indices, offsets = np.ascontiguousarray(values), np.ascontiguousarray(indices), np.ascontiguousarray(offsets)
            queries = offsets.size - 1
            output, result, error = np.empty((queries, 2), dtype=np.uint64), MaxResult(), ct.create_string_buffer(128)
            function = self._library.wxctc_timestamp_bounds_batch
            function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_uint64, ct.POINTER(ct.c_uint32), ct.c_uint64, ct.POINTER(ct.c_uint32), ct.c_uint64, ct.POINTER(ct.c_uint64), ct.c_uint32, ct.POINTER(MaxResult), ct.POINTER(ct.c_char), ct.c_uint32]
            function.restype = ct.c_int
            status = function(self._context, values.ctypes.data_as(ct.POINTER(ct.c_uint64)), values.size, indices.ctypes.data_as(ct.POINTER(ct.c_uint32)), indices.size, offsets.ctypes.data_as(ct.POINTER(ct.c_uint32)), offsets.size, output.ctypes.data_as(ct.POINTER(ct.c_uint64)), output.size, ct.byref(result), error, len(error))
            if status: raise ValueError(error.value.decode("ascii"))
            if result.abi_version != 1: raise ValueError("native_result_invalid")
            return output.view(np.float64), {"scope": "experimental_timestamp_bounds_batch", "elements": values.size, "queries": queries, "index_elements": indices.size, "kernel": "ctc_timestamp_bounds_batch", "dispatch_completed": True, "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": result.computation_wall_ns, "computation_gpu_seconds": result.computation_gpu_seconds, "logical_cpu_to_native_bytes": values.nbytes + indices.nbytes + offsets.nbytes, "logical_output_readback_bytes": queries * 16 + 4, "physical_copy_bytes": None, "allocated_shared_buffer_bytes": result.allocated_shared_buffer_bytes}

    def word_means_batch(self, values, indices, offsets):
        import numpy as np
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 1 or values.size > 32768 or not isinstance(indices, np.ndarray) or indices.dtype != np.uint32 or indices.ndim != 1 or indices.size > 4 * 1024 * 1024 or not isinstance(offsets, np.ndarray) or offsets.dtype != np.uint32 or offsets.ndim != 1 or not 2 <= offsets.size <= 32769: raise ValueError("native_word_means_format_invalid")
        if np.__version__ != "2.2.6" or np.getbufsize() != 8192: raise ValueError("native_word_means_runtime_not_supported")
        with self._lock:
            if not self._context.value: raise ValueError("native_context_closed")
            started = time.perf_counter_ns()
            values, indices, offsets = np.ascontiguousarray(values), np.ascontiguousarray(indices), np.ascontiguousarray(offsets)
            queries = offsets.size - 1
            output, result, error = np.empty(queries, dtype=np.uint64), MaxResult(), ct.create_string_buffer(128)
            function = self._library.wxctc_word_means_batch
            function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_uint64, ct.POINTER(ct.c_uint32), ct.c_uint64, ct.POINTER(ct.c_uint32), ct.c_uint64, ct.POINTER(ct.c_uint64), ct.c_uint32, ct.POINTER(MaxResult), ct.POINTER(ct.c_char), ct.c_uint32]
            function.restype = ct.c_int
            status = function(self._context, values.ctypes.data_as(ct.POINTER(ct.c_uint64)), values.size, indices.ctypes.data_as(ct.POINTER(ct.c_uint32)), indices.size, offsets.ctypes.data_as(ct.POINTER(ct.c_uint32)), offsets.size, output.ctypes.data_as(ct.POINTER(ct.c_uint64)), output.size, ct.byref(result), error, len(error))
            if status: raise ValueError(error.value.decode("ascii"))
            if result.abi_version != 1: raise ValueError("native_result_invalid")
            return output.view(np.float64), {"scope": "experimental_word_means_batch", "elements": values.size, "queries": queries, "index_elements": indices.size, "kernel": "ctc_word_means_batch", "dispatch_completed": True, "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": result.computation_wall_ns, "computation_gpu_seconds": result.computation_gpu_seconds, "logical_cpu_to_native_bytes": values.nbytes + indices.nbytes + offsets.nbytes, "logical_output_readback_bytes": queries * 8 + 4, "physical_copy_bytes": None, "allocated_shared_buffer_bytes": result.allocated_shared_buffer_bytes}

    def timestamp_bounds(self, values):
        import numpy as np
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 1 or values.size > 32768: raise ValueError("native_timestamp_format_invalid")
        with self._lock:
            if not self._context.value: raise ValueError("native_context_closed")
            started = time.perf_counter_ns()
            values = np.ascontiguousarray(values)
            output, result, error = np.empty(2, dtype=np.uint64), MaxResult(), ct.create_string_buffer(128)
            function = self._library.wxctc_timestamp_bounds
            function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_uint64, ct.POINTER(ct.c_uint64), ct.c_uint32, ct.POINTER(MaxResult), ct.POINTER(ct.c_char), ct.c_uint32]
            function.restype = ct.c_int
            status = function(self._context, values.ctypes.data_as(ct.POINTER(ct.c_uint64)), values.size, output.ctypes.data_as(ct.POINTER(ct.c_uint64)), 2, ct.byref(result), error, len(error))
            if status: raise ValueError(error.value.decode("ascii"))
            if result.abi_version != 1: raise ValueError("native_result_invalid")
            return output.view(np.float64), {"scope": "experimental_timestamp_bounds", "elements": values.size, "kernel": "ctc_timestamp_bounds", "dispatch_completed": True, "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": result.computation_wall_ns, "computation_gpu_seconds": result.computation_gpu_seconds, "logical_cpu_to_native_bytes": values.nbytes, "logical_output_readback_bytes": 24, "physical_copy_bytes": None, "allocated_shared_buffer_bytes": result.allocated_shared_buffer_bytes}

    def close(self):
        with self._lock:
            if self._context.value:
                self._library.wxctc_destroy(self._context)
                self._context = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

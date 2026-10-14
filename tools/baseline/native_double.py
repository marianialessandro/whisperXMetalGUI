"""Bounded experimental float64 PLDA binary64 normalization on native Metal, without CPU fallback."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import numpy as np


class Metrics(ct.Structure):
    _fields_ = [("version", ct.c_uint32), ("kernels", ct.c_uint32), ("wall_ns", ct.c_uint64), ("gpu_seconds", ct.c_double), ("allocated_bytes", ct.c_uint64)]


def shader_source():
    root = Path(__file__).parent
    return (root / "native_ctc.metal").read_bytes() + b"\n" + (root / "native_double.metal").read_bytes()


def build_library(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("double_build_output_exists")
    source = Path(__file__).with_suffix(".mm")
    with tempfile.TemporaryDirectory(prefix="double-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "double.dylib"
        subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-fobjc-arc", "-dynamiclib", "-Wno-deprecated-declarations", str(source), "-framework", "Foundation", "-framework", "Metal", "-o", str(staged)], check=True, capture_output=True, timeout=60)
        os.link(staged, output)
    return {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "shader_sha256": hashlib.sha256(shader_source()).hexdigest(), "scope": "experimental_native_double_normalization"}


class NativeDoubleNormalization:
    def __init__(self, library, expected_sha256):
        library = Path(library).resolve(strict=True)
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("double_library_hash_mismatch")
        self.lock = threading.RLock()
        self.context = ct.c_void_p()
        self.library = ct.CDLL(str(library))
        self.library.wxdouble_create.argtypes = [ct.c_char_p, ct.c_uint64, ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_uint32]
        self.library.wxdouble_create.restype = ct.c_int
        self.library.wxdouble_normalize.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
        self.library.wxdouble_normalize.restype = ct.c_int
        self.library.wxdouble_destroy.argtypes = [ct.c_void_p]
        self.library.wxdouble_destroy.restype = None
        shader = shader_source()
        error = ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        if self.library.wxdouble_create(shader, len(shader), ct.byref(self.context), error, len(error)):
            raise ValueError(error.value.decode("ascii"))
        self.initialization_wall_ns = time.perf_counter_ns() - started
        self.identity = {"library_sha256": expected_sha256, "shader_sha256": hashlib.sha256(shader).hexdigest()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self.lock:
            if self.context:
                self.library.wxdouble_destroy(self.context)
                self.context = ct.c_void_p()

    def normalize(self, values, reduction_order="pairwise"):
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_input_format_invalid")
        if reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_reduction_order_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024:
            raise ValueError("double_shape_budget_invalid")
        output = np.empty_like(values)
        metrics = Metrics()
        error = ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            started = time.perf_counter_ns()
            if self.library.wxdouble_normalize(self.context, values.ctypes.data, rows, columns, int(reduction_order == "sequential"), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_native_result_invalid")
        return output, {"scope": "experimental_native_double_normalization", "reduction_order": reduction_order, "arithmetic": "bounded_binary64_integer_emulation", "rows": rows, "columns": columns, "gpu_kernels": ["double_normalize"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def matmul(self, left, right):
        for values in (left, right):
            if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
                raise ValueError("double_matrix_input_format_invalid")
        rows, inner = left.shape
        if right.shape[0] != inner or not 0 < rows <= 4096 or not 0 < inner <= 1024 or not 0 < right.shape[1] <= 1024:
            raise ValueError("double_matrix_shape_invalid")
        columns = right.shape[1]
        if rows * inner * columns > 32 * 1024 * 1024:
            raise ValueError("double_matrix_work_budget_invalid")
        output = np.empty((rows, columns), dtype=np.float64)
        metrics = Metrics()
        error = ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                operation = self.library.wxdouble_matmul
            except AttributeError as failure:
                raise ValueError("double_matrix_abi_unavailable") from failure
            operation.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            operation.restype = ct.c_int
            started = time.perf_counter_ns()
            if operation(self.context, left.ctypes.data, right.ctypes.data, rows, inner, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_matrix_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_matrix_product", "rows": rows, "inner": inner, "columns": columns, "accumulation_policy": "sequential_separately_rounded_binary64_multiply_add", "reference_blas_bitwise_certified": False, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_matmul"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": left.nbytes + right.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="Compile a new local dylib at this path; existing paths are never overwritten.")
    args = parser.parse_args()
    print(json.dumps(build_library(args.build)))


if __name__ == "__main__":
    main()

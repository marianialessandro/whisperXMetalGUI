"""Bounded experimental float64 PLDA binary64 normalization on native Metal, without CPU fallback."""

import argparse
import ctypes as ct
import hashlib
import json
import math
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

    def center(self, values, mean):
        if not isinstance(mean, np.ndarray) or mean.dtype != np.float64 or mean.ndim != 1 or not mean.flags.c_contiguous:
            raise ValueError("double_element_mean_format_invalid")
        return self._elementwise(values, mean, 0)

    def scale_dimension(self, values, dimension):
        if type(dimension) is not int or not 0 < dimension <= 1024:
            raise ValueError("double_element_dimension_invalid")
        return self._elementwise(values, None, dimension)

    def _elementwise(self, values, mean, dimension):
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_element_input_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or (mean is not None and mean.shape != (columns,)):
            raise ValueError("double_element_shape_invalid")
        output, metrics, error = np.empty_like(values), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                operation = self.library.wxdouble_elementwise
            except AttributeError as failure:
                raise ValueError("double_element_abi_unavailable") from failure
            operation.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            operation.restype = ct.c_int
            started = time.perf_counter_ns()
            if operation(self.context, values.ctypes.data, mean.ctypes.data if mean is not None else None, rows, columns, int(mean is None), dimension, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_element_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_center_scale", "operation": "center" if mean is not None else "scale_sqrt_dimension", "rows": rows, "columns": columns, "scale_dimension": dimension, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_center_scale"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes + (mean.nbytes if mean is not None else 0), "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def precompute_vbx(self, features, phi, reduction_order="pairwise"):
        if type(features) is not np.ndarray or features.dtype != np.float64 or features.ndim != 2 or not features.flags.c_contiguous or type(phi) is not np.ndarray or phi.dtype != np.float64 or phi.ndim != 1 or not phi.flags.c_contiguous:
            raise ValueError("double_vbx_input_format_invalid")
        rows, columns = features.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or phi.shape != (columns,) or reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_vbx_shape_or_reduction_invalid")
        g, v, rho = np.empty((rows, 1)), np.empty(columns), np.empty_like(features)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                operation = self.library.wxdouble_vbx_precompute
            except AttributeError as failure:
                raise ValueError("double_vbx_precompute_abi_unavailable") from failure
            operation.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            operation.restype = ct.c_int
            started = time.perf_counter_ns()
            if operation(self.context, features.ctypes.data, phi.ctypes.data, rows, columns, int(reduction_order == "sequential"), g.ctypes.data, v.ctypes.data, rho.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or not all(np.isfinite(values).all() for values in (g, v, rho)):
            raise ValueError("double_vbx_precompute_native_result_invalid")
        return (g, v, rho), {"scope": "experimental_native_binary64_vbx_precompute", "rows": rows, "columns": columns, "reduction_order": reduction_order, "log_two_pi_binary64": "0x3ffd67f1c864beb4", "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_terms", "double_vbx_rho"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": features.nbytes + phi.nbytes, "logical_result_readback_bytes": g.nbytes + v.nbytes + rho.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def inv_l_vbx(self, gamma_transposed, phi, fa, fb, reduction_order="sequential"):
        return self._statistics_vbx(gamma_transposed, phi, fa, fb, 0, reduction_order)

    def alpha_vbx(self, inv_l, statistics, fa, fb):
        return self._statistics_vbx(inv_l, statistics, fa, fb, 1, "sequential")

    def _statistics_vbx(self, values, right, fa, fb, mode, reduction_order):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous or type(right) is not np.ndarray or right.dtype != np.float64 or not right.flags.c_contiguous:
            raise ValueError("double_vbx_statistics_input_format_invalid")
        rows, columns = values.shape
        if right.ndim != (1 if mode == 0 else 2) or (mode == 1 and right.shape != values.shape):
            raise ValueError("double_vbx_statistics_shape_invalid")
        features = right.shape[-1]
        if not 0 < rows <= 1024 or not 0 < columns <= 4096 or not 0 < features <= 1024 or reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_vbx_statistics_shape_or_reduction_invalid")
        if type(fa) not in (float, int) or type(fb) not in (float, int):
            raise ValueError("double_vbx_statistics_scalar_format_invalid")
        output, metrics, error = np.empty((rows, features)), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                operation = self.library.wxdouble_vbx_statistics
            except AttributeError as failure:
                raise ValueError("double_vbx_statistics_abi_unavailable") from failure
            operation.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            operation.restype = ct.c_int
            started = time.perf_counter_ns()
            if operation(self.context, values.ctypes.data, right.ctypes.data, rows, columns, features, mode, int(reduction_order == "sequential"), fa, fb, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != (2 if mode == 0 else 1) or not np.isfinite(output).all():
            raise ValueError("double_vbx_statistics_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_statistics", "operation": "invL" if mode == 0 else "alpha", "rows": rows, "columns": columns, "features": features, "reduction_order": reduction_order if mode == 0 else None, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": (["double_vbx_gamma_sum"] if mode == 0 else []) + ["double_vbx_statistics"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes + right.nbytes + 16, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def penalty_terms_vbx(self, inv_l, alpha):
        return self._scores_vbx(inv_l, alpha, 0, 1.)

    def frame_scores_vbx(self, frames, penalty, g, fa):
        if type(frames) is not np.ndarray or frames.dtype != np.float64 or frames.ndim != 2 or not frames.flags.c_contiguous:
            raise ValueError("double_vbx_scores_input_format_invalid")
        if not 0 < frames.shape[0] <= 4096 or not 0 < frames.shape[1] <= 1024:
            raise ValueError("double_vbx_scores_shape_invalid")
        for values in (penalty, g):
            if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 1 or not values.flags.c_contiguous:
                raise ValueError("double_vbx_scores_vector_format_invalid")
        if penalty.shape != (frames.shape[1],) or g.shape != (frames.shape[0],):
            raise ValueError("double_vbx_scores_vector_shape_invalid")
        packed = np.concatenate((penalty, g))
        output, record = self._scores_vbx(frames, packed, 1, fa)
        record["host_vector_packing_copy_bytes"] = packed.nbytes
        return output, record

    def _scores_vbx(self, values, right, mode, fa):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous or type(right) is not np.ndarray or right.dtype != np.float64 or not right.flags.c_contiguous:
            raise ValueError("double_vbx_scores_input_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= (1024 if mode == 0 else 4096) or not 0 < columns <= 1024 or right.shape != (values.shape if mode == 0 else (rows + columns,)):
            raise ValueError("double_vbx_scores_shape_invalid")
        if type(fa) not in (float, int):
            raise ValueError("double_vbx_scores_scalar_format_invalid")
        output, metrics, error = np.empty_like(values), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                operation = self.library.wxdouble_vbx_scores
            except AttributeError as failure:
                raise ValueError("double_vbx_scores_abi_unavailable") from failure
            operation.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            operation.restype = ct.c_int
            started = time.perf_counter_ns()
            if operation(self.context, values.ctypes.data, right.ctypes.data, rows, columns, mode, fa, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_scores_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_scores", "operation": "penalty_terms" if mode == 0 else "frame_scores", "rows": rows, "columns": columns, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_scores"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes + right.nbytes + 8, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def prior_vbx(self, gamma_transposed, reduction_order="sequential"):
        if type(gamma_transposed) is not np.ndarray or gamma_transposed.dtype != np.float64 or gamma_transposed.ndim != 2 or not gamma_transposed.flags.c_contiguous:
            raise ValueError("double_vbx_prior_input_format_invalid")
        speakers, frames = gamma_transposed.shape
        if not 0 < speakers <= 1024 or not 0 < frames <= 4096 or reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_vbx_prior_shape_or_reduction_invalid")
        output, metrics, error = np.empty(speakers), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                operation = self.library.wxdouble_vbx_prior
            except AttributeError as failure:
                raise ValueError("double_vbx_prior_abi_unavailable") from failure
            operation.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            operation.restype = ct.c_int
            started = time.perf_counter_ns()
            if operation(self.context, gamma_transposed.ctypes.data, speakers, frames, int(reduction_order == "sequential"), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or not np.isfinite(output).all():
            raise ValueError("double_vbx_prior_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_prior", "speakers": speakers, "frames": frames, "frame_reduction_order": reduction_order, "speaker_reduction_order": "pairwise", "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_gamma_sum", "double_vbx_prior"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": gamma_transposed.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def posterior_vbx(self, values, vector, operation="log_prior_addition"):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous or type(vector) is not np.ndarray or vector.dtype != np.float64 or vector.ndim != 1 or not vector.flags.c_contiguous:
            raise ValueError("double_vbx_posterior_input_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or operation not in ("log_prior_addition", "posterior_centering") or vector.shape != (columns if operation == "log_prior_addition" else rows,):
            raise ValueError("double_vbx_posterior_shape_or_operation_invalid")
        output, metrics, error = np.empty_like(values), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_posterior
            except AttributeError as failure:
                raise ValueError("double_vbx_posterior_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, vector.ctypes.data, rows, columns, int(operation == "posterior_centering"), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_posterior_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_posterior", "operation": operation, "rows": rows, "columns": columns, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_posterior"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes + vector.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def likelihood_sum_vbx(self, values):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 1 or not values.flags.c_contiguous or not 0 < values.size <= 4096:
            raise ValueError("double_vbx_likelihood_input_invalid")
        return self._elbo_vbx(values, 1, values.size, 0, False, 1., 0.)

    def elbo_vbx(self, log_inv_l, inv_l, alpha, likelihood, fb, column_order=False):
        for values in (log_inv_l, inv_l, alpha):
            if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
                raise ValueError("double_vbx_elbo_input_format_invalid")
        rows, columns = inv_l.shape
        if not 0 < rows <= 1024 or not 0 < columns <= 1024 or log_inv_l.shape != inv_l.shape or alpha.shape != inv_l.shape or type(column_order) is not bool:
            raise ValueError("double_vbx_elbo_shape_invalid")
        if type(likelihood) not in (float, int, np.float64) or type(fb) not in (float, int, np.float64):
            raise ValueError("double_vbx_elbo_scalar_format_invalid")
        packed = np.concatenate((log_inv_l.ravel(), inv_l.ravel(), alpha.ravel()))
        output, record = self._elbo_vbx(packed, rows, columns, 1, column_order, fb, likelihood)
        record["host_matrix_packing_copy_bytes"] = packed.nbytes
        return output, record

    def _elbo_vbx(self, packed, rows, columns, mode, column_order, fb, likelihood):
        if np.__version__ != "2.2.6" or np.getbufsize() != 8192:
            raise ValueError("double_vbx_elbo_reduction_runtime_not_supported")
        output, metrics, error = np.empty(1), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_elbo
            except AttributeError as failure:
                raise ValueError("double_vbx_elbo_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, packed.ctypes.data, rows, columns, mode, int(column_order), fb, likelihood, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != (1 if mode == 0 else 2) or not np.isfinite(output).all():
            raise ValueError("double_vbx_elbo_native_result_invalid")
        return output[0], {"scope": "experimental_native_binary64_vbx_elbo", "operation": "likelihood_sum" if mode == 0 else "elbo", "rows": rows, "columns": columns, "reduction_layout": "F" if column_order else "C", "reduction_order": "buffered_pairwise", "numpy_reduce_buffer_elements": 8192, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_likelihood"] if mode == 0 else ["double_vbx_elbo_terms", "double_vbx_elbo_final"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": packed.nbytes + 16, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def transcendental_vbx(self, values, operation):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim not in (1, 2) or not values.flags.c_contiguous or not 0 < values.size <= 65536 or operation not in ("log", "exp", "log1p"):
            raise ValueError("double_vbx_transcendental_input_format_invalid")
        output, metrics, error = np.empty_like(values), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_transcendental
            except AttributeError as failure:
                raise ValueError("double_vbx_transcendental_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, values.size, {"log": 0, "exp": 1, "log1p": 2}[operation], output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_transcendental_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_log_exp", "operation": operation, "count": values.size, "arithmetic": "bounded_binary64_integer_emulation", "approximation_policy": "openlibm_fdlibm_separately_rounded", "numpy_bitwise_certified": False, "gpu_kernels": ["double_vbx_log1p" if operation == "log1p" else "double_vbx_transcendental"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def logsumexp_vbx(self, values, reduction_order="pairwise"):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_vbx_logsumexp_input_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or values.size > 65536 or reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_vbx_logsumexp_shape_or_reduction_invalid")
        if np.__version__ != "2.2.6" or np.getbufsize() != 8192:
            raise ValueError("double_vbx_logsumexp_runtime_not_supported")
        output, metrics, error = np.empty(rows), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_logsumexp
            except AttributeError as failure:
                raise ValueError("double_vbx_logsumexp_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, columns, int(reduction_order == "sequential"), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or not np.isfinite(output).all():
            raise ValueError("double_vbx_logsumexp_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_finite_logsumexp", "rows": rows, "columns": columns, "reduction_order": reduction_order, "shift_domain": [-60, 0], "approximation_policy": "scipy_1_15_3_separated_maxima_openlibm_math", "scipy_bitwise_certified": False, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_lse_terms", "double_vbx_lse_final"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


    def nan_cleanup_vbx(self, scores):
        if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 2 or not scores.flags.c_contiguous:
            raise ValueError("double_vbx_nan_cleanup_input_format_invalid")
        rows, columns = scores.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024:
            raise ValueError("double_vbx_nan_cleanup_shape_invalid")
        output, all_nan, metrics, error = np.empty_like(scores), ct.c_uint32(), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_nan_cleanup
            except AttributeError as failure:
                raise ValueError("double_vbx_nan_cleanup_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(ct.c_uint32), ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, scores.ctypes.data, rows, columns, output.ctypes.data, ct.byref(all_nan), ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or all_nan.value > 1 or np.isinf(output).any():
            raise ValueError("double_vbx_nan_cleanup_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_constrained_nan_cleanup", "rows": rows, "columns": columns, "all_nan_input": bool(all_nan.value), "replacement_zero_sign": "positive", "gpu_kernels": ["double_vbx_nan_min_partials", "double_vbx_nan_cleanup"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": scores.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "nan_payload_certified": False, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def inactive_mask_vbx(self, scores, segments):
        if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 or not value.flags.c_contiguous for value in (scores, segments)):
            raise ValueError("double_vbx_inactive_input_format_invalid")
        rows, columns = scores.shape
        frames = segments.shape[1]
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or not 0 < frames <= 4096 or segments.shape[0] != rows or segments.size > 1024 * 1024:
            raise ValueError("double_vbx_inactive_shape_invalid")
        output, metrics, error = np.empty_like(scores), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_inactive_mask
            except AttributeError as failure:
                raise ValueError("double_vbx_inactive_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, scores.ctypes.data, segments.ctypes.data, rows, columns, frames, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or np.isinf(output).any():
            raise ValueError("double_vbx_inactive_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_inactive_mask", "rows": rows, "columns": columns, "frames": frames, "gpu_kernels": ["double_vbx_score_min_partials", "double_vbx_inactive_mask"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": scores.nbytes + segments.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "nan_payload_certified": False, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def qinit_from_labels_vbx(self, labels, columns, smoothing):
        if type(labels) is not np.ndarray or labels.dtype != np.int64 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("double_vbx_one_hot_input_format_invalid")
        rows = labels.size
        if not 0 < rows <= 4096 or type(columns) is not int or not 0 < columns <= 1024 or rows * columns > 65536 or type(smoothing) not in (float, int, np.float64):
            raise ValueError("double_vbx_resident_initialization_shape_or_scalar_invalid")
        output, metrics, error = np.empty((rows, columns)), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_qinit_from_labels
            except AttributeError as failure:
                raise ValueError("double_vbx_resident_initialization_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, labels.ctypes.data, rows, columns, smoothing, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 3 or not np.isfinite(output).all():
            raise ValueError("double_vbx_resident_initialization_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_qinit", "operation": "initial_softmax", "resident_initialization": True, "rows": rows, "columns": columns, "smoothing": float(smoothing), "math_policy": "one_hot_shifted_softmax_openlibm_exp", "scipy_bitwise_certified": False, "gpu_kernels": ["double_vbx_one_hot", "double_vbx_qinit_exp", "double_vbx_initialize"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": labels.nbytes, "logical_result_readback_bytes": output.nbytes, "logical_one_hot_boundary_bytes_avoided": 2 * output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def argmax_vbx(self, scores):
        if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 2 or not scores.flags.c_contiguous:
            raise ValueError("double_vbx_argmax_input_format_invalid")
        rows, columns = scores.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024:
            raise ValueError("double_vbx_argmax_shape_invalid")
        output, metrics, error = np.empty(rows, dtype=np.int64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_argmax
            except AttributeError as failure:
                raise ValueError("double_vbx_argmax_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, scores.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or (output < 0).any() or (output >= columns).any():
            raise ValueError("double_vbx_argmax_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_argmax", "rows": rows, "columns": columns, "comparison_policy": "first_nan_else_first_maximum_signed_zero_equal", "gpu_kernels": ["double_vbx_argmax"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": scores.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def cosine_vbx(self, embeddings, centroids, soft_scores=False, return_labels=False):
        if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 or not value.flags.c_contiguous for value in (embeddings, centroids)):
            raise ValueError("double_vbx_cosine_input_format_invalid")
        if type(soft_scores) is not bool or type(return_labels) is not bool or (return_labels and not soft_scores):
            raise ValueError("double_vbx_cosine_score_mode_invalid")
        rows, features = embeddings.shape
        count = centroids.shape[0]
        if not 0 < rows <= 4096 or not 0 < count <= 1024 or not 0 < features <= 1024 or centroids.shape[1] != features or rows * count * features > 32 * 1024 * 1024:
            raise ValueError("double_vbx_cosine_shape_invalid")
        output, metrics, error = np.empty((rows, count)), Metrics(), ct.create_string_buffer(128)
        labels = np.empty(rows, dtype=np.int64) if return_labels else None
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_cosine_scores_argmax if return_labels else self.library.wxdouble_vbx_cosine_scores if soft_scores else self.library.wxdouble_vbx_cosine
            except AttributeError as failure:
                raise ValueError("double_vbx_cosine_argmax_abi_unavailable" if return_labels else "double_vbx_cosine_score_abi_unavailable" if soft_scores else "double_vbx_cosine_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p] + ([ct.c_void_p] if return_labels else []) + [ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            arguments = [self.context, embeddings.ctypes.data, centroids.ctypes.data, rows, count, features, output.ctypes.data] + ([labels.ctypes.data] if return_labels else []) + [ct.byref(metrics), error, len(error)]
            if native(*arguments):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != (3 if return_labels else 2) or np.isinf(output).any():
            raise ValueError("double_vbx_cosine_native_result_invalid")
        if return_labels and ((labels < 0).any() or (labels >= count).any()):
            raise ValueError("double_vbx_argmax_native_result_invalid")
        return (output, labels) if return_labels else output, {"scope": "experimental_native_binary64_vbx_cosine", "resident_argmax": return_labels, "result_kind": "soft_scores" if soft_scores else "distances", "rows": rows, "centroids": count, "features": features, "undefined_pairs": int(np.isnan(output).sum()), "arithmetic_policy": "sequential_separately_rounded_binary64", "scipy_bitwise_certified": False, "gpu_kernels": ["double_vbx_cosine_norms", "double_vbx_cosine_scores" if soft_scores else "double_vbx_cosine_distances"] + (["double_vbx_argmax"] if return_labels else []), "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": embeddings.nbytes + centroids.nbytes, "logical_result_readback_bytes": output.nbytes + (labels.nbytes if return_labels else 0), "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def one_hot_vbx(self, labels, columns):
        if type(labels) is not np.ndarray or labels.dtype != np.int64 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("double_vbx_one_hot_input_format_invalid")
        rows = labels.size
        if not 0 < rows <= 4096 or type(columns) is not int or not 0 < columns <= 1024:
            raise ValueError("double_vbx_one_hot_shape_invalid")
        output, metrics, error = np.empty((rows, columns)), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_one_hot
            except AttributeError as failure:
                raise ValueError("double_vbx_one_hot_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, labels.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_one_hot_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_one_hot", "operation": "initial_one_hot", "rows": rows, "columns": columns, "gpu_kernels": ["double_vbx_one_hot"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": labels.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_distance_finish(self, products, left_norms, right_norms):
        if any(type(value) is not np.ndarray or value.dtype != np.float64 or not value.flags.c_contiguous for value in (products, left_norms, right_norms)) or products.ndim != 2 or left_norms.ndim != 1 or right_norms.ndim != 1:
            raise ValueError("double_kmeans_distance_finish_input_format_invalid")
        rows, columns = products.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 4096 or products.size > 1024 * 1024 or left_norms.shape != (rows,) or right_norms.shape != (columns,):
            raise ValueError("double_kmeans_distance_finish_shape_invalid")
        output, metrics, error = np.empty(products.shape, dtype=np.float32), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_distance_finish
            except AttributeError as failure:
                raise ValueError("double_kmeans_distance_finish_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, products.ctypes.data, left_norms.ctypes.data, right_norms.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_kmeans_distance_finish_native_result_invalid")
        return output, {"scope": "experimental_native_kmeans_distance_finish", "rows": rows, "columns": columns, "gpu_kernels": ["double_kmeans_distance_finish"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": products.nbytes + left_norms.nbytes + right_norms.nbytes, "logical_result_readback_bytes": output.nbytes, "host_numeric_scale_or_norm_addition": False, "host_numeric_dtype_conversion": False, "binary32_subnormals_preserved": True, "distance_clamping_replaced": False, "distance_blas_products_replaced": False, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def kmeans_upcast_norms(self, values):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_kmeans_upcast_norms_input_format_invalid")
        rows, features = values.shape
        if not 0 < rows <= 4096 or not 0 < features <= 1024:
            raise ValueError("double_kmeans_upcast_norms_shape_invalid")
        output, metrics, error = np.empty(rows, dtype=np.float64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_upcast_norms
            except AttributeError as failure:
                raise ValueError("double_kmeans_upcast_norms_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, features, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all() or np.any(output < 0):
            raise ValueError("double_kmeans_upcast_norms_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_kmeans_upcast_norms", "rows": rows, "features": features, "reduction_policy": "numpy_2_2_6_neon_f64_reverse8_fma2", "gpu_kernels": ["double_kmeans_upcast_norms"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "host_numeric_row_norms": False, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def ahc_resident(self, values, cutoff):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_ahc_resident_input_format_invalid")
        rows, features = values.shape
        if not 2 <= rows <= 256 or not 0 < features <= 1024 or type(cutoff) not in (float, int, np.float64, np.float32):
            raise ValueError("double_ahc_resident_shape_or_policy_invalid")
        output, metrics, error = np.empty(rows, dtype=np.int64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_ahc_resident
            except AttributeError as failure:
                raise ValueError("double_ahc_resident_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, features, float(cutoff), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 4 or np.any(output < 0) or np.any(output >= rows):
            raise ValueError("double_ahc_resident_native_result_invalid")
        return output, {"scope": "experimental_native_resident_ahc", "leaves": rows, "features": features, "resident_ahc": True, "resident_reindex": True, "zero_based_output": True, "criterion": "distance", "reindex_policy": "identity_for_contiguous_native_cluster_ids", "gpu_kernels": ["double_pdist_euclidean", "double_resident_centroid_linkage", "double_resident_fcluster", "integer_resident_cut_labels_i64"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes + 4, "logical_label_readback_bytes": output.nbytes, "logical_status_readback_bytes": 4, "intermediate_distance_readback_bytes": 0, "intermediate_tree_readback_bytes": 0, "intermediate_label_readback_bytes": 0, "host_numeric_distance": False, "host_numeric_merge_search": False, "host_numeric_distance_update": False, "host_numeric_tree_cut": False, "host_numeric_label_reindex": False, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def centroid_linkage(self, distances):
        if type(distances) is not np.ndarray or distances.dtype != np.float64 or distances.ndim != 1 or not distances.flags.c_contiguous:
            raise ValueError("double_centroid_linkage_input_format_invalid")
        leaves = (1 + math.isqrt(1 + 8 * distances.size)) // 2
        if not 2 <= leaves <= 256 or leaves * (leaves - 1) // 2 != distances.size:
            raise ValueError("double_centroid_linkage_shape_invalid")
        output, metrics, error = np.empty((leaves - 1, 4), dtype=np.float64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_centroid_linkage
            except AttributeError as failure:
                raise ValueError("double_centroid_linkage_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, distances.ctypes.data, leaves, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_centroid_linkage_native_result_invalid")
        return output, {"scope": "experimental_native_centroid_dendrogram", "leaves": leaves, "pairs": distances.size, "method": "centroid", "algorithm": "pinned_scipy_fast_linkage_heap", "arithmetic": "binary64_emulation_with_pinned_positive_fma", "gpu_kernels": ["double_centroid_linkage"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": distances.nbytes, "logical_result_readback_bytes": output.nbytes + 4, "logical_status_readback_bytes": 4, "private_array_element_bytes": 9216, "physical_copy_bytes": None, "host_numeric_merge_search": False, "host_numeric_distance_update": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def positive_fma(self, values):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_positive_fma_input_format_invalid")
        if values.shape[1] != 3 or not 0 < len(values) <= 4096:
            raise ValueError("double_positive_fma_shape_invalid")
        output, metrics, error = np.empty(len(values), dtype=np.float64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_positive_fma
            except AttributeError as failure:
                raise ValueError("double_positive_fma_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, len(values), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all() or np.any(output < 0):
            raise ValueError("double_positive_fma_native_result_invalid")
        return output, {"scope": "experimental_native_positive_binary64_fma", "count": len(values), "rounding": "single_nearest_even_after_exact_product_and_addend", "gpu_kernels": ["double_positive_fma"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def fcluster_reindex(self, tree, cutoff):
        if type(tree) is not np.ndarray or tree.dtype != np.float64 or tree.ndim != 2 or not tree.flags.c_contiguous:
            raise ValueError("double_fcluster_reindex_input_format_invalid")
        leaves = len(tree) + 1
        if tree.shape[1] != 4 or not 2 <= leaves <= 256 or type(cutoff) not in (float, int, np.float64, np.float32):
            raise ValueError("double_fcluster_reindex_shape_or_policy_invalid")
        output, metrics, error = np.empty(leaves, dtype=np.int64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_fcluster_reindex
            except AttributeError as failure:
                raise ValueError("double_fcluster_reindex_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, tree.ctypes.data, leaves, float(cutoff), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or np.any(output < 0) or np.any(output >= leaves):
            raise ValueError("double_fcluster_reindex_native_result_invalid")
        return output, {"scope": "experimental_native_resident_flat_cut_and_reindex", "leaves": leaves, "resident_reindex": True, "zero_based_output": True, "reindex_policy": "identity_for_contiguous_native_cluster_ids", "criterion": "distance", "gpu_kernels": ["double_fcluster_distance", "integer_cut_labels_i64"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": tree.nbytes, "logical_result_readback_bytes": output.nbytes, "intermediate_label_readback_bytes": 0, "host_label_dtype_conversion": False, "host_label_offset": False, "host_numeric_tree_cut": False, "host_numeric_label_reindex": False, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def fcluster_distance(self, tree, cutoff, zero_based=False):
        if type(tree) is not np.ndarray or tree.dtype != np.float64 or tree.ndim != 2 or not tree.flags.c_contiguous:
            raise ValueError("double_fcluster_input_format_invalid")
        leaves = len(tree) + 1
        if tree.shape[1] != 4 or not 2 <= leaves <= 256 or type(zero_based) is not bool or type(cutoff) not in (float, int, np.float64, np.float32):
            raise ValueError("double_fcluster_shape_or_policy_invalid")
        output, metrics, error = np.empty(leaves, dtype=np.int32), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_fcluster_distance
            except AttributeError as failure:
                raise ValueError("double_fcluster_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_double, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, tree.ctypes.data, leaves, float(cutoff), int(zero_based), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or np.any(output < int(not zero_based)) or np.any(output > leaves - int(zero_based)):
            raise ValueError("double_fcluster_native_result_invalid")
        return output, {"scope": "experimental_native_distance_flat_clustering", "leaves": leaves, "zero_based_output": zero_based, "criterion": "distance", "label_order": "pinned_scipy_cluster_monocrit_traversal", "gpu_kernels": ["double_fcluster_distance"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": tree.nbytes, "logical_result_readback_bytes": output.nbytes, "private_tree_workspace_bound_bytes": 3584, "host_numeric_tree_cut": False, "host_label_offset": False, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def ahc_reindex(self, labels):
        if type(labels) is not np.ndarray or labels.dtype != np.int64 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("double_ahc_reindex_input_format_invalid")
        if not 0 < labels.size <= 4096:
            raise ValueError("double_ahc_reindex_shape_invalid")
        output, metrics, error = np.empty_like(labels), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_ahc_reindex
            except AttributeError as failure:
                raise ValueError("double_ahc_reindex_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, labels.ctypes.data, labels.size, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or np.any(output < 0) or np.any(output >= labels.size):
            raise ValueError("double_ahc_reindex_native_result_invalid")
        return output, {"scope": "experimental_native_integer_ahc_label_reindex", "count": labels.size, "label_domain_size": 4096, "ordering": "ascending_distinct_label_rank", "gpu_kernels": ["integer_ahc_reindex"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": labels.nbytes, "logical_result_readback_bytes": output.nbytes, "private_presence_bytes_per_thread": 512, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def pdist_euclidean(self, values):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_pdist_input_format_invalid")
        rows, features = values.shape
        pairs = rows * (rows - 1) // 2
        if not 2 <= rows <= 256 or not 0 < features <= 1024 or pairs * features > 32 * 1024 * 1024:
            raise ValueError("double_pdist_shape_invalid")
        output, metrics, error = np.empty(pairs, dtype=np.float64), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_pdist_euclidean
            except AttributeError as failure:
                raise ValueError("double_pdist_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, features, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all() or np.any(output < 0):
            raise ValueError("double_pdist_native_result_invalid")
        return output, {"scope": "experimental_native_condensed_euclidean_distances", "rows": rows, "features": features, "pairs": pairs, "arithmetic": "sequential_separately_rounded_binary64_difference_square_sum_sqrt", "gpu_kernels": ["double_pdist_euclidean"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "scipy_bitwise_certified": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def inactive_assignment(self, scores, segments):
        if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 3 or not value.flags.c_contiguous for value in (scores, segments)):
            raise ValueError("double_inactive_assignment_input_format_invalid")
        chunks, rows, columns = scores.shape
        frames = segments.shape[2]
        if not 0 < chunks <= 4096 or not 0 < rows <= 64 or not 0 < columns <= 64 or min(rows, columns) > 16 or chunks * rows > 4096 or segments.shape[:2] != (chunks, rows) or not 0 < frames <= 4096 or segments.size > 1024 * 1024:
            raise ValueError("double_inactive_assignment_shape_invalid")
        masked, labels = np.empty_like(scores), np.empty((chunks, rows), dtype=np.int8)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_inactive_assignment
            except AttributeError as failure:
                raise ValueError("double_inactive_assignment_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, scores.ctypes.data, segments.ctypes.data, chunks, rows, columns, frames, masked.ctypes.data, labels.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 5 or np.isinf(masked).any() or np.any((labels != -2) & ((labels < 0) | (labels >= columns))):
            raise ValueError("double_inactive_assignment_native_result_invalid")
        return (labels, masked), {"scope": "experimental_native_resident_inactive_and_constrained_assignment", "chunks": chunks, "rows": rows, "columns": columns, "frames": frames, "logical_assignment_matrices": chunks, "gpu_kernels": ["double_vbx_score_min_partials", "double_vbx_inactive_mask", "double_vbx_nan_min_partials", "double_vbx_nan_cleanup", "double_inactive_assignment_batch"], "gpu_blit_encoders": 1, "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": scores.nbytes + segments.nbytes, "logical_result_readback_bytes": masked.nbytes + labels.nbytes + chunks * 4, "logical_public_soft_score_readback_bytes": masked.nbytes, "logical_status_readback_bytes": chunks * 4, "intermediate_score_readback_bytes": 0, "assignment_pair_readback_bytes": 0, "host_numeric_label_scatter": False, "physical_copy_bytes": None, "nan_payload_certified": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def assignment_batch(self, scores):
        if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 3 or not scores.flags.c_contiguous:
            raise ValueError("double_assignment_batch_operand_format_invalid")
        chunks, rows, columns = scores.shape
        if not 0 < chunks <= 4096 or not 0 < rows <= 64 or not 0 < columns <= 64 or min(rows, columns) > 16 or chunks * rows > 4096:
            raise ValueError("double_assignment_batch_shape_invalid")
        output = np.empty((chunks, rows), dtype=np.int8)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_assignment_batch
            except AttributeError as failure:
                raise ValueError("double_assignment_batch_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, scores.ctypes.data, chunks, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 3 or ((output != -2) & ((output < 0) | (output >= columns))).any():
            raise ValueError("double_assignment_batch_native_result_invalid")
        return output, {"scope": "experimental_native_resident_constrained_assignment_batch", "chunks": chunks, "rows": rows, "columns": columns, "logical_assignment_matrices": chunks, "arithmetic": "bounded_binary64_integer_emulation", "tie_policy": "pinned_scipy_1_15_3_shortest_augmenting_path", "scipy_assignment_certified": False, "gpu_kernels": ["double_vbx_nan_min_partials", "double_vbx_nan_cleanup", "double_assignment_batch"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": scores.nbytes, "logical_result_readback_bytes": output.nbytes + chunks * 4, "logical_status_readback_bytes": chunks * 4, "intermediate_score_readback_bytes": 0, "assignment_pair_readback_bytes": 0, "host_numeric_label_scatter": False, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def linear_assignment_max(self, scores):
        if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 2 or not scores.flags.c_contiguous:
            raise ValueError("double_assignment_operand_format_invalid")
        rows, columns = scores.shape
        if not 0 < rows <= 64 or not 0 < columns <= 64 or min(rows, columns) > 16:
            raise ValueError("double_assignment_shape_invalid")
        output = np.empty((2, min(rows, columns)), dtype=np.int64)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_linear_assignment_max
            except AttributeError as failure:
                raise ValueError("double_assignment_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, scores.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or (output < 0).any() or (output[0] >= rows).any() or (output[1] >= columns).any():
            raise ValueError("double_assignment_native_result_invalid")
        return (output[0], output[1]), {"scope": "experimental_native_binary64_rectangular_assignment", "rows": rows, "columns": columns, "matched_pairs": output.shape[1], "transposed": columns < rows, "arithmetic": "bounded_binary64_integer_emulation", "tie_policy": "pinned_scipy_1_15_3_shortest_augmenting_path", "scipy_assignment_certified": False, "gpu_kernels": ["double_linear_assignment_max"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": scores.nbytes, "logical_result_readback_bytes": output.nbytes + 4, "logical_status_readback_bytes": 4, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_best_candidate(self, potentials):
        if type(potentials) is not np.ndarray or potentials.dtype != np.float32 or potentials.ndim not in (1, 2) or (potentials.ndim == 2 and potentials.shape[1] != 1) or not potentials.flags.c_contiguous:
            raise ValueError("double_kmeans_best_candidate_operand_format_invalid")
        count = potentials.size
        if not 0 < count <= 1024:
            raise ValueError("double_kmeans_best_candidate_shape_invalid")
        output = np.empty(1, dtype=np.int64)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_best_candidate
            except AttributeError as failure:
                raise ValueError("double_kmeans_best_candidate_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, potentials.ctypes.data, count, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not 0 <= output[0] < count:
            raise ValueError("double_kmeans_best_candidate_native_result_invalid")
        return output[0], {"scope": "experimental_native_kmeans_best_candidate", "count": count, "arithmetic_policy": "nonnegative_float32_order_keys_first_tie_argmin", "gpu_kernels": ["float_kmeans_best_candidate"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": potentials.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_candidate_search(self, cdf, queries):
        if type(cdf) is not np.ndarray or cdf.dtype != np.float32 or cdf.ndim != 1 or not cdf.flags.c_contiguous or type(queries) is not np.ndarray or queries.dtype != np.float64 or queries.ndim != 1 or not queries.flags.c_contiguous:
            raise ValueError("double_kmeans_candidate_search_operand_format_invalid")
        count, query_count = cdf.size, queries.size
        if not 0 < count <= 4096 or not 0 < query_count <= 1024:
            raise ValueError("double_kmeans_candidate_search_shape_invalid")
        output = np.empty(query_count, dtype=np.int64)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_candidate_search
            except AttributeError as failure:
                raise ValueError("double_kmeans_candidate_search_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, cdf.ctypes.data, queries.ctypes.data, count, query_count, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or (output < 0).any() or (output >= count).any():
            raise ValueError("double_kmeans_candidate_search_native_result_invalid")
        return output, {"scope": "experimental_native_kmeans_candidate_search", "count": count, "queries": query_count, "arithmetic_policy": "searchsorted_left_exact_float32_to_binary64_order_keys_then_upper_clip", "gpu_kernels": ["float_kmeans_candidate_search"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": cdf.nbytes + queries.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_seed_probabilities(self, weights):
        if type(weights) is not np.ndarray or weights.dtype != np.float32 or weights.ndim != 1 or not weights.flags.c_contiguous:
            raise ValueError("double_kmeans_seed_probabilities_operand_format_invalid")
        count = weights.size
        if not 0 < count <= 4096:
            raise ValueError("double_kmeans_seed_probabilities_shape_invalid")
        output = np.empty_like(weights)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_seed_probabilities
            except AttributeError as failure:
                raise ValueError("double_kmeans_seed_probabilities_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, weights.ctypes.data, count, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or not np.isfinite(output).all() or (output < 0).any() or (output > 1).any():
            raise ValueError("double_kmeans_seed_probabilities_native_result_invalid")
        return output, {"scope": "experimental_native_float32_kmeans_seed_probabilities", "count": count, "arithmetic_policy": "float32_pairwise_weight_sum_then_float32_division", "numpy_bitwise_certified": False, "gpu_kernels": ["float_kmeans_seed_weight_sum", "float_kmeans_seed_probabilities"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": weights.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_candidate_minimum(self, closest, candidates):
        if type(candidates) is not np.ndarray or candidates.dtype != np.float32 or candidates.ndim != 2 or not candidates.flags.c_contiguous or type(closest) is not np.ndarray or closest.dtype != np.float32 or closest.shape not in ((candidates.shape[1],), (1, candidates.shape[1])) or not closest.flags.c_contiguous:
            raise ValueError("double_kmeans_candidate_minimum_operand_format_invalid")
        rows, columns = candidates.shape
        if not 0 < rows <= 1024 or not 0 < columns <= 4096:
            raise ValueError("double_kmeans_candidate_minimum_shape_invalid")
        output = np.empty_like(candidates)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_candidate_minimum
            except AttributeError as failure:
                raise ValueError("double_kmeans_candidate_minimum_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, closest.ctypes.data, candidates.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("double_kmeans_candidate_minimum_native_result_invalid")
        return output, {"scope": "experimental_native_float32_kmeans_candidate_minimum", "rows": rows, "columns": columns, "arithmetic_policy": "finite_float32_broadcast_minimum_arm_signed_zero", "numpy_bitwise_certified": False, "gpu_kernels": ["float_kmeans_candidate_minimum"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": closest.nbytes + candidates.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_weighted_prefix(self, weights, distances):
        if type(weights) is not np.ndarray or weights.dtype != np.float32 or weights.ndim != 1 or not weights.flags.c_contiguous or type(distances) is not np.ndarray or distances.dtype != np.float32 or distances.shape not in (weights.shape, (1, weights.size)) or not distances.flags.c_contiguous:
            raise ValueError("double_kmeans_weighted_prefix_operand_format_invalid")
        count = weights.size
        if not 0 < count <= 4096:
            raise ValueError("double_kmeans_weighted_prefix_shape_invalid")
        output = np.empty(count, dtype=np.float32)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_weighted_prefix
            except AttributeError as failure:
                raise ValueError("double_kmeans_weighted_prefix_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, weights.ctypes.data, distances.ctypes.data, count, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("double_kmeans_weighted_prefix_native_result_invalid")
        return output, {"scope": "experimental_native_float32_kmeans_weighted_prefix", "count": count, "distance_rows": 1 if distances.ndim == 2 else None, "arithmetic_policy": "float32_multiply_then_sequential_flat_cumsum", "numpy_bitwise_certified": False, "gpu_kernels": ["float_kmeans_weighted_products", "float_kmeans_weighted_prefix"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": weights.nbytes + distances.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_squared_norms(self, values):
        if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_kmeans_squared_norms_operand_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024:
            raise ValueError("double_kmeans_squared_norms_shape_invalid")
        output = np.empty(rows, dtype=np.float32)
        metrics, error = Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_squared_norms
            except AttributeError as failure:
                raise ValueError("double_kmeans_squared_norms_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("double_kmeans_squared_norms_native_result_invalid")
        return output, {"scope": "experimental_native_float32_kmeans_squared_norms", "rows": rows, "columns": columns, "arithmetic_policy": "pinned_numpy_arm_neon_four_lane_fma_reverse_16_blocks_pairwise_horizontal_sum", "numpy_bitwise_certified": False, "gpu_kernels": ["float_kmeans_squared_norms"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_tolerance(self, values, tolerance):
        if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous or type(tolerance) not in (float, int, np.float32):
            raise ValueError("double_kmeans_tolerance_operand_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 2 <= columns <= 1024 or not 2. ** -32 <= tolerance <= 2. ** 16:
            raise ValueError("double_kmeans_tolerance_shape_or_scalar_invalid")
        output, metrics, error = np.empty(1, np.float32), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_kmeans_tolerance
            except AttributeError as failure:
                raise ValueError("double_kmeans_tolerance_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_float, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, columns, tolerance, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 4 or not np.isfinite(output).all():
            raise ValueError("double_kmeans_tolerance_native_result_invalid")
        return output[0], {"scope": "experimental_native_float32_kmeans_tolerance", "rows": rows, "columns": columns, "tolerance_parameter": float(tolerance), "tolerance_float32": float(np.float32(tolerance)), "arithmetic_policy": "float32_variance_ddof0_then_pairwise_feature_mean_and_float32_scale", "numpy_bitwise_certified": False, "gpu_kernels": ["float_kmeans_tolerance_column_mean", "float_kmeans_tolerance_square", "float_kmeans_tolerance_column_mean", "float_kmeans_tolerance_final"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def float_shift_vbx(self, values, mean, add=False):
        if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous or type(mean) is not np.ndarray or mean.dtype != np.float32 or mean.ndim != 1 or not mean.flags.c_contiguous or type(add) is not bool:
            raise ValueError("double_float_shift_operand_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or mean.shape != (columns,):
            raise ValueError("double_float_shift_shape_invalid")
        output, metrics, error = np.empty_like(values), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_float_shift
            except AttributeError as failure:
                raise ValueError("double_float_shift_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, mean.ctypes.data, rows, columns, int(add), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_float_shift_native_result_invalid")
        return output, {"scope": "experimental_native_float32_shift", "operation": "add" if add else "subtract", "rows": rows, "columns": columns, "gpu_kernels": ["float_vbx_shift"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes + mean.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def kmeans_centroids_vbx(self, features, labels, clusters):
        if type(features) is not np.ndarray or features.dtype != np.float32 or features.ndim != 2 or not features.flags.c_contiguous or type(labels) is not np.ndarray or labels.dtype != np.int32 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("double_vbx_kmeans_operand_format_invalid")
        rows, dimensions = features.shape
        if type(clusters) is not int or not 0 < rows <= 4096 or not 2 <= dimensions <= 1024 or not 0 < clusters <= min(rows, 1024) or rows * dimensions * clusters > 32 * 1024 * 1024 or labels.shape != (rows,):
            raise ValueError("double_vbx_kmeans_shape_invalid")
        output, metrics, error = np.empty((clusters, dimensions), np.float32), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_kmeans_centroids
            except AttributeError as failure:
                raise ValueError("double_vbx_kmeans_centroid_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, features.ctypes.data, labels.ctypes.data, rows, dimensions, clusters, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_kmeans_native_result_invalid")
        return output, {"scope": "experimental_native_float32_kmeans_centroids", "rows": rows, "features": dimensions, "clusters": clusters, "arithmetic_policy": "sequential_float32_sum_then_float32_division", "numpy_bitwise_certified": False, "gpu_kernels": ["float_vbx_kmeans_centroids"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": features.nbytes + labels.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def centroids_vbx(self, weights, features, reduction_order="pairwise"):
        if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 or not value.flags.c_contiguous for value in (weights, features)):
            raise ValueError("double_vbx_centroid_input_format_invalid")
        speakers, frames = weights.shape
        dimensions = features.shape[1]
        if not 0 < speakers <= 1024 or not 0 < frames <= 4096 or not 0 < dimensions <= 1024 or features.shape[0] != frames or weights.size * dimensions > 32 * 1024 * 1024 or features.size > 1024 * 1024 or reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_vbx_centroid_shape_or_reduction_invalid")
        output, metrics, error = np.empty((speakers, dimensions)), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_centroids
            except AttributeError as failure:
                raise ValueError("double_vbx_centroid_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, weights.ctypes.data, features.ctypes.data, speakers, frames, dimensions, int(reduction_order == "sequential"), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_centroid_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_centroids", "speakers": speakers, "frames": frames, "features": dimensions, "mass_reduction_order": reduction_order, "product_policy": "sequential_separately_rounded_binary64", "blas_bitwise_certified": False, "gpu_kernels": ["double_vbx_centroids"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": weights.nbytes + features.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def qinit_softmax_vbx(self, values, smoothing):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_vbx_qinit_input_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or values.size > 65536 or type(smoothing) not in (float, int, np.float64):
            raise ValueError("double_vbx_qinit_shape_or_scalar_invalid")
        output, metrics, error = np.empty_like(values), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_qinit
            except AttributeError as failure:
                raise ValueError("double_vbx_qinit_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data, rows, columns, smoothing, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or not np.isfinite(output).all():
            raise ValueError("double_vbx_qinit_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_qinit", "operation": "initial_softmax", "rows": rows, "columns": columns, "smoothing": float(smoothing), "math_policy": "one_hot_shifted_softmax_openlibm_exp", "scipy_bitwise_certified": False, "gpu_kernels": ["double_vbx_qinit_exp", "double_vbx_initialize"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}

    def uniform_prior_vbx(self, speakers):
        if type(speakers) is not int or not 0 < speakers <= 1024:
            raise ValueError("double_vbx_uniform_prior_shape_invalid")
        output, record = self._initialize_vbx(None, 1, speakers, 0, "pairwise")
        return output[0], record

    def normalize_initial_gamma_vbx(self, values, reduction_order="pairwise"):
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("double_vbx_initialization_input_format_invalid")
        rows, columns = values.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024 or reduction_order not in ("pairwise", "sequential"):
            raise ValueError("double_vbx_initialization_shape_or_reduction_invalid")
        return self._initialize_vbx(values, rows, columns, 1, reduction_order)

    def _initialize_vbx(self, values, rows, columns, mode, reduction_order):
        output, metrics, error = np.empty((rows, columns)), Metrics(), ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("double_context_closed")
            try:
                native = self.library.wxdouble_vbx_initialize
            except AttributeError as failure:
                raise ValueError("double_vbx_initialization_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            started = time.perf_counter_ns()
            if native(self.context, values.ctypes.data if values is not None else None, rows, columns, mode, int(reduction_order == "sequential"), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("double_vbx_initialization_native_result_invalid")
        return output, {"scope": "experimental_native_binary64_vbx_initialization", "operation": "uniform_prior" if mode == 0 else "gamma_l1_normalization", "rows": rows, "columns": columns, "reduction_order": reduction_order if mode == 1 else None, "arithmetic": "bounded_binary64_integer_emulation", "gpu_kernels": ["double_vbx_initialize"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": values.nbytes if values is not None else 0, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="Compile a new local dylib at this path; existing paths are never overwritten.")
    args = parser.parse_args()
    print(json.dumps(build_library(args.build)))


if __name__ == "__main__":
    main()

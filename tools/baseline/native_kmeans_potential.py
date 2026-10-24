"""Bounded CPU Accelerate bridge for pinned KMeans++ operations and Lloyd assignment/centroid updates."""

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


class PotentialMetrics(ct.Structure):
    _fields_ = [("version", ct.c_uint32), ("blas_calls", ct.c_uint32), ("wall_ns", ct.c_uint64)]


def build_library(output, with_inertia=False, with_two_chunks=False):
    with_inertia = with_inertia or with_two_chunks
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("kmeans_potential_build_output_exists")
    source = Path(__file__).with_suffix(".cpp")
    with tempfile.TemporaryDirectory(prefix="kmeans-potential-build-", dir=output.parent) as directory:
        staged = Path(directory) / "potential.dylib"
        command = ["xcrun", "clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-dynamiclib", str(source), str(source.with_name("native_kmeans_assignment.cpp")), str(source.with_name("native_kmeans_relocation.cpp")), "-framework", "Accelerate", "-o", str(staged)]
        if with_inertia:
            from tools.baseline.native_kmeans_inertia import runtime_path
            runtime = runtime_path()
            inertia_object = Path(directory) / "inertia.o"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp", "-fPIC", "-c", str(source.with_name("native_kmeans_inertia.cpp")), "-o", str(inertia_object)], check=True, capture_output=True, timeout=60)
            command += [str(inertia_object), str(runtime), "-Wl,-rpath," + str(runtime.parent)]
            if with_two_chunks:
                two_chunks_object = Path(directory) / "two-chunks.o"
                subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp", "-fPIC", "-c", str(source.with_name("native_kmeans_two_chunks.cpp")), "-o", str(two_chunks_object)], check=True, capture_output=True, timeout=60)
                command += [str(two_chunks_object)]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        if with_inertia:
            install_id = subprocess.run(["otool", "-D", str(runtime)], check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()[1].strip()
            subprocess.run(["install_name_tool", "-change", install_id, "@rpath/libomp.dylib", str(staged)], check=True, capture_output=True, timeout=10)
            subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(staged)], check=True, capture_output=True, timeout=10)
        os.link(staged, output)
    identity = {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "assignment_source_sha256": hashlib.sha256(source.with_name("native_kmeans_assignment.cpp").read_bytes()).hexdigest(), "distance_source_sha256": hashlib.sha256(source.with_name("native_kmeans_distance.hpp").read_bytes()).hexdigest(), "scope": "experimental_native_cpu_kmeans_potentials"}
    identity.update({name + "_source_sha256": hashlib.sha256(source.with_name(filename).read_bytes()).hexdigest() for name, filename in (("relocation", "native_kmeans_relocation.cpp"), ("selection", "native_kmeans_selection.hpp"), ("pairwise", "native_kmeans_pairwise.hpp"))})
    if with_inertia:
        identity.update({"inertia_source_sha256": hashlib.sha256(source.with_name("native_kmeans_inertia.cpp").read_bytes()).hexdigest(), "openmp_runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()})
    identity["relocation_core_source_sha256"] = hashlib.sha256(source.with_name("native_kmeans_relocation_core.hpp").read_bytes()).hexdigest()
    if with_two_chunks:
        identity["two_chunks_source_sha256"] = hashlib.sha256(source.with_name("native_kmeans_two_chunks.cpp").read_bytes()).hexdigest()
    return identity


class NativeKMeansPotentials:
    def __init__(self, library, expected_sha256):
        library = Path(library).resolve(strict=True)
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("kmeans_potential_library_hash_mismatch")
        self.library = ct.CDLL(str(library))
        try:
            self.native = self.library.wxkmeans_weighted_potentials
        except AttributeError as failure:
            raise ValueError("kmeans_potential_abi_unavailable") from failure
        self.native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]
        self.native.restype = ct.c_int
        if getattr(self.library, "wxkmeans_inertia_dense", None) is not None:
            from tools.baseline.native_kmeans_inertia import validate_loaded_runtime
            validate_loaded_runtime(self.library)
        self.lock, self.closed = threading.RLock(), False
        self.identity = {"library_sha256": expected_sha256, "backend": "cpu_accelerate_ilp64", "production_backend": False}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self.lock:
            self.closed = True

    def lloyd_relocated_centroids(self, values, weights, labels, previous, *, n_threads=None):
        if any(type(value) is not np.ndarray or value.dtype != np.float32 or not value.flags.c_contiguous for value in (values, weights, previous)) or values.ndim != 2 or previous.ndim != 2 or weights.ndim != 1 or type(labels) is not np.ndarray or labels.dtype != np.int32 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("kmeans_relocation_operand_format_invalid")
        two_chunks = n_threads is not None
        if two_chunks and (type(n_threads) is not int or n_threads not in (1, 2, 12)):
            raise ValueError("kmeans_two_chunks_thread_count_not_supported")
        rows, features = values.shape
        clusters = previous.shape[0]
        if not 0 < rows <= (512 if two_chunks else 256) or not 0 < clusters <= min(rows, 256) or not 0 < features <= 1024 or previous.shape[1] != features or weights.shape != (rows,) or labels.shape != (rows,):
            raise ValueError("kmeans_relocation_shape_invalid")
        centers, totals, distances, order, counts = np.empty_like(previous), np.empty(clusters, np.float32), np.empty(rows, np.float32), np.empty(rows, np.int32), np.empty(2, np.uint32)
        metrics, error = PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_lloyd_centroids_two_chunks if two_chunks else self.library.wxkmeans_lloyd_relocated_centroids
            except AttributeError as failure:
                raise ValueError("kmeans_two_chunks_centroids_abi_unavailable" if two_chunks else "kmeans_relocation_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p] * 4 + [ct.c_uint32] * (4 if two_chunks else 3) + [ct.c_void_p] * 5 + [ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(values.ctypes.data, weights.ctypes.data, labels.ctypes.data, previous.ctypes.data, rows, clusters, features, *([n_threads] if two_chunks else []), centers.ctypes.data, totals.ctypes.data, distances.ctypes.data, order.ctypes.data, counts.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not np.isfinite(centers).all() or not np.isfinite(totals).all() or not np.isfinite(distances).all() or counts[0] > clusters or counts[1] > counts[0] or (order[:int(counts[0])] < 0).any() or (order[:int(counts[0])] >= rows).any():
            raise ValueError("kmeans_relocation_native_result_invalid")
        record = {"scope": "experimental_native_cpu_lloyd_relocated_centroids", "execution_device": "cpu", "backend": "cpp_fma_pairwise_introselect_relocated_centroids", "samples": rows, "clusters": clusters, "features": features, "empty_clusters": int(counts[0]), "relocated_clusters": int(counts[1]), "selection_calls": int(counts[0] > 0), "distance_rows": rows if counts[0] else 0, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": values.nbytes + weights.nbytes + labels.nbytes + previous.nbytes, "host_result_bytes": centers.nbytes + totals.nbytes + distances.nbytes + order.nbytes + counts.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_centroid_update": False, "python_numeric_relocation": False, "python_argpartition": False, "centroid_updates_replaced": True, "empty_cluster_relocation_replaced": True, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

        if two_chunks:
            record.update({"backend": "cpp_openmp_two_chunk_fma_relocated_centroids", "chunk_size": 256, "chunk_count": (rows + 255) // 256, "requested_threads": n_threads, "effective_iteration_threads": min(n_threads, (rows + 255) // 256), "actual_threads": min(n_threads, (rows + 255) // 256), "thread_count_changed": False, "worker_partial_bytes": min(n_threads, (rows + 255) // 256) * (centers.nbytes + totals.nbytes)})
        return centers, totals, distances, order[:int(counts[0])], record

    def inertia_dense(self, values, weights, centers, labels, n_threads, single_label=-1, *, two_chunks=False):
        if any(type(value) is not np.ndarray or value.dtype != np.float32 or not value.flags.c_contiguous for value in (values, weights, centers)) or values.ndim != 2 or centers.ndim != 2 or weights.ndim != 1 or type(labels) is not np.ndarray or labels.dtype != np.int32 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("kmeans_inertia_operand_format_invalid")
        if type(two_chunks) is not bool:
            raise ValueError("kmeans_two_chunks_configuration_invalid")
        rows, features = values.shape
        clusters = centers.shape[0]
        if not 0 < rows <= (512 if two_chunks else 256) or not 0 < clusters <= 256 or not 0 < features <= 1024 or centers.shape[1] != features or weights.shape != (rows,) or labels.shape != (rows,):
            raise ValueError("kmeans_inertia_shape_invalid")
        if type(n_threads) is not int or n_threads not in (1, 2, 12):
            raise ValueError("kmeans_inertia_thread_count_not_supported")
        if type(single_label) is not int or not -1 <= single_label < clusters:
            raise ValueError("kmeans_inertia_single_label_invalid")
        output, metrics, error = np.empty(1, np.float32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_inertia_dense_two_chunks if two_chunks else self.library.wxkmeans_inertia_dense
            except AttributeError as failure:
                raise ValueError("kmeans_two_chunks_inertia_abi_unavailable" if two_chunks else "kmeans_inertia_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p] * 4 + [ct.c_uint32] * 4 + [ct.c_int32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(values.ctypes.data, weights.ctypes.data, centers.ctypes.data, labels.ctypes.data, rows, clusters, features, n_threads, single_label, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not np.isfinite(output[0]) or output[0] < 0:
            raise ValueError("kmeans_inertia_native_result_invalid")
        return float(output[0]), {"scope": "experimental_native_cpu_openmp_inertia", "execution_device": "cpu", "backend": "cpp_pinned_openmp_static_float32_fma", "samples": rows, "clusters": clusters, "features": features, "two_chunks_enabled": two_chunks, "requested_threads": n_threads, "actual_threads": n_threads, "single_label": single_label, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": values.nbytes + weights.nbytes + centers.nbytes + labels.nbytes, "host_result_bytes": 4, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_inertia": False, "thread_count_changed": False, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def lloyd_center_shifts(self, updated, previous):
        if any(type(value) is not np.ndarray or value.dtype != np.float32 or value.ndim != 2 or not value.flags.c_contiguous for value in (updated, previous)):
            raise ValueError("kmeans_lloyd_center_shifts_operand_format_invalid")
        clusters, features = updated.shape
        if not 0 < clusters <= 256 or not 0 < features <= 1024 or previous.shape != updated.shape:
            raise ValueError("kmeans_lloyd_center_shifts_shape_invalid")
        output, metrics, error = np.empty(clusters, np.float32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_lloyd_center_shifts
            except AttributeError as failure:
                raise ValueError("kmeans_lloyd_center_shifts_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(updated.ctypes.data, previous.ctypes.data, clusters, features, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("kmeans_lloyd_center_shifts_native_result_invalid")
        return output, {"scope": "experimental_native_cpu_lloyd_center_shifts", "execution_device": "cpu", "backend": "cpp_cython_float32_four_term_fma_sqrt", "clusters": clusters, "features": features, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": updated.nbytes + previous.nbytes, "host_result_bytes": output.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_center_shift": False, "center_shifts_replaced": True, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def lloyd_centroids(self, values, weights, labels, clusters):
        if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous or type(weights) is not np.ndarray or weights.dtype != np.float32 or weights.ndim != 1 or not weights.flags.c_contiguous or type(labels) is not np.ndarray or labels.dtype != np.int32 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("kmeans_lloyd_centroids_operand_format_invalid")
        rows, features = values.shape
        if type(clusters) is not int or not 0 < rows <= 256 or not 0 < clusters <= 256 or not 0 < features <= 1024 or weights.shape != (rows,) or labels.shape != (rows,):
            raise ValueError("kmeans_lloyd_centroids_shape_invalid")
        centers, totals = np.empty((clusters, features), np.float32), np.empty(clusters, np.float32)
        metrics, error = PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_lloyd_centroids
            except AttributeError as failure:
                raise ValueError("kmeans_lloyd_centroids_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(values.ctypes.data, weights.ctypes.data, labels.ctypes.data, rows, clusters, features, centers.ctypes.data, totals.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not np.isfinite(centers).all() or not np.isfinite(totals).all() or (totals <= 0).any():
            raise ValueError("kmeans_lloyd_centroids_native_result_invalid")
        return centers, totals, {"scope": "experimental_native_cpu_lloyd_weighted_centroids", "execution_device": "cpu", "backend": "cpp_float32_fma_double_reciprocal", "samples": rows, "clusters": clusters, "features": features, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": values.nbytes + weights.nbytes + labels.nbytes, "host_result_bytes": centers.nbytes + totals.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_centroid_update": False, "centroid_updates_replaced": True, "empty_cluster_relocation_replaced": False, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def lloyd_assignment(self, values, centers, norms, *, n_threads=None):
        if any(type(value) is not np.ndarray or value.dtype != np.float32 or not value.flags.c_contiguous for value in (values, centers, norms)) or values.ndim != 2 or centers.ndim != 2 or norms.ndim != 1:
            raise ValueError("kmeans_lloyd_assignment_operand_format_invalid")
        two_chunks = n_threads is not None
        if two_chunks and (type(n_threads) is not int or n_threads not in (1, 2, 12)):
            raise ValueError("kmeans_two_chunks_thread_count_not_supported")
        rows, features = values.shape
        clusters = centers.shape[0]
        if not 0 < rows <= (512 if two_chunks else 256) or not 0 < clusters <= 256 or not 0 < features <= 1024 or centers.shape[1] != features or norms.shape != (clusters,):
            raise ValueError("kmeans_lloyd_assignment_shape_invalid")
        labels, distances = np.empty(rows, np.int32), np.empty((rows, clusters), np.float32)
        metrics, error = PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_lloyd_assignment_two_chunks if two_chunks else self.library.wxkmeans_lloyd_assignment
            except AttributeError as failure:
                raise ValueError("kmeans_two_chunks_assignment_abi_unavailable" if two_chunks else "kmeans_lloyd_assignment_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, *([ct.c_uint32] * (4 if two_chunks else 3)), ct.c_void_p, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(values.ctypes.data, centers.ctypes.data, norms.ctypes.data, rows, clusters, features, *([n_threads] if two_chunks else []), labels.ctypes.data, distances.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != ((rows + 255) // 256 if two_chunks else 1) or not np.isfinite(distances).all() or (labels < 0).any() or (labels >= clusters).any():
            raise ValueError("kmeans_lloyd_assignment_native_result_invalid")
        record = {"scope": "experimental_native_cpu_lloyd_final_assignment", "execution_device": "cpu", "backend": "accelerate_newlapack_lp64_sgemm_first_minimum", "samples": rows, "clusters": clusters, "features": features, "native_blas_calls": 1, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": values.nbytes + centers.nbytes + norms.nbytes, "host_result_bytes": labels.nbytes + distances.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_assignment": False, "lloyd_iteration_replaced": False, "centroid_updates_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}
        if two_chunks:
            record.update({"backend": "accelerate_lp64_openmp_static_two_chunks", "native_blas_calls": (rows + 255) // 256, "chunk_size": 256, "chunk_count": (rows + 255) // 256, "requested_threads": n_threads, "effective_iteration_threads": min(n_threads, (rows + 255) // 256), "actual_threads": min(n_threads, (rows + 255) // 256), "thread_count_changed": False, "host_input_snapshot_bytes": 2 * values.nbytes + (((rows + 255) // 256) + 1) * (centers.nbytes + norms.nbytes)})
        return labels, distances, record

    def local_trials(self, clusters):
        if type(clusters) is not int or not 0 < clusters <= 4096:
            raise ValueError("kmeans_local_trials_cluster_count_invalid")
        output, metrics, error = np.empty(1, np.uint32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_local_trials
            except AttributeError as failure:
                raise ValueError("kmeans_local_trials_abi_unavailable") from failure
            native.argtypes = [ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(clusters, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not 2 <= output[0] <= 10:
            raise ValueError("kmeans_local_trials_native_result_invalid")
        return int(output[0]), {"scope": "experimental_native_cpu_kmeans_local_trials", "execution_device": "cpu", "backend": "cpp_bounded_integer_log_thresholds", "clusters": clusters, "local_trials": int(output[0]), "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": 0, "host_result_bytes": 4, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_logarithm": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def distinct_labels(self, labels):
        if type(labels) is not np.ndarray or labels.dtype != np.int32 or labels.ndim != 1 or not labels.flags.c_contiguous:
            raise ValueError("kmeans_distinct_operand_format_invalid")
        if not 0 < labels.size <= 4096:
            raise ValueError("kmeans_distinct_count_invalid")
        output, metrics, error = np.empty(1, np.uint32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_distinct_labels
            except AttributeError as failure:
                raise ValueError("kmeans_distinct_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(labels.ctypes.data, labels.size, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not 0 < output[0] <= min(labels.size, 4096):
            raise ValueError("kmeans_distinct_native_result_invalid")
        return int(output[0]), {"scope": "experimental_native_cpu_kmeans_distinct_labels", "execution_device": "cpu", "backend": "cpp_int32_label_bitset_cardinality", "samples": labels.size, "distinct_labels": int(output[0]), "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": labels.nbytes, "host_result_bytes": 4, "worker_bitset_bytes": 512, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_set_cardinality": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def same_clustering(self, labels, previous, clusters):
        if any(type(value) is not np.ndarray or value.dtype != np.int32 or value.ndim != 1 or not value.flags.c_contiguous for value in (labels, previous)):
            raise ValueError("kmeans_same_clustering_operand_format_invalid")
        if type(clusters) is not int or not 0 < clusters <= 4096 or not 0 < labels.size <= 4096 or previous.shape != labels.shape:
            raise ValueError("kmeans_same_clustering_shape_invalid")
        output, metrics, error = np.empty(1, dtype=np.uint32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_same_clustering
            except AttributeError as failure:
                raise ValueError("kmeans_same_clustering_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(labels.ctypes.data, previous.ctypes.data, labels.size, clusters, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or output[0] > 1:
            raise ValueError("kmeans_same_clustering_native_result_invalid")
        return bool(output[0]), {"scope": "experimental_native_cpu_kmeans_same_clustering", "execution_device": "cpu", "backend": "cpp_directed_label_mapping", "samples": labels.size, "clusters": clusters, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": labels.nbytes + previous.nbytes, "host_result_bytes": 4, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_clustering_equivalence": False, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def lloyd_labels_equal(self, labels, previous):
        if any(type(value) is not np.ndarray or value.dtype != np.int32 or value.ndim != 1 or not value.flags.c_contiguous for value in (labels, previous)):
            raise ValueError("kmeans_lloyd_labels_operand_format_invalid")
        if not 0 < labels.size <= 4096 or previous.shape != labels.shape:
            raise ValueError("kmeans_lloyd_labels_count_invalid")
        output, metrics, error = np.empty(1, dtype=np.uint32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_lloyd_labels_equal
            except AttributeError as failure:
                raise ValueError("kmeans_lloyd_labels_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(labels.ctypes.data, previous.ctypes.data, labels.size, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or output[0] > 1:
            raise ValueError("kmeans_lloyd_labels_native_result_invalid")
        return bool(output[0]), {"scope": "experimental_native_cpu_kmeans_lloyd_labels_equal", "execution_device": "cpu", "backend": "cpp_int32_label_equality", "samples": labels.size, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": labels.nbytes + previous.nbytes, "host_result_bytes": 4, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_label_comparison": False, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def lloyd_shift_total(self, shifts):
        if type(shifts) is not np.ndarray or shifts.dtype != np.float32 or shifts.ndim != 1 or not shifts.flags.c_contiguous:
            raise ValueError("kmeans_lloyd_shift_operand_format_invalid")
        if not 0 < shifts.size <= 1024:
            raise ValueError("kmeans_lloyd_shift_count_invalid")
        output, metrics, error = np.empty(1, dtype=np.float32), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_lloyd_shift_total
            except AttributeError as failure:
                raise ValueError("kmeans_lloyd_shift_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]; native.restype = ct.c_int
            if native(shifts.ctypes.data, shifts.size, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not np.isfinite(output[0]) or output[0] < 0:
            raise ValueError("kmeans_lloyd_shift_native_result_invalid")
        return output[0], {"scope": "experimental_native_cpu_kmeans_lloyd_shift_total", "execution_device": "cpu", "backend": "cpp_numpy_pairwise_float32", "centers": shifts.size, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": shifts.nbytes, "host_result_bytes": 4, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_shift_reduction": False, "lloyd_iteration_replaced": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def upcast_chunk(self, values, *, extended_rows=False):
        if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous:
            raise ValueError("kmeans_upcast_chunk_operand_format_invalid")
        if type(extended_rows) is not bool:
            raise ValueError("kmeans_upcast_512_configuration_invalid")
        rows, features = values.shape
        if not 0 < rows <= (512 if extended_rows else 256) or not 0 < features <= 1024:
            raise ValueError("kmeans_upcast_chunk_shape_invalid")
        output, metrics, error = np.empty(values.shape, dtype=np.float64), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_upcast_chunk_512 if extended_rows else self.library.wxkmeans_upcast_chunk
            except AttributeError as failure:
                raise ValueError("kmeans_upcast_512_abi_unavailable" if extended_rows else "kmeans_upcast_chunk_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            if native(values.ctypes.data, rows, features, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != 0 or not np.isfinite(output).all():
            raise ValueError("kmeans_upcast_chunk_native_result_invalid")
        return output, {"scope": "experimental_native_cpu_kmeans_upcast_chunk", "execution_device": "cpu", "backend": "cpp_ieee_integer_conversion", "rows": rows, "features": features, "extended_rows_enabled": extended_rows, "maximum_rows": 512 if extended_rows else 256, "native_blas_calls": 0, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": 0, "host_input_bytes_read": values.nbytes, "native_result_staging_bytes": output.nbytes, "host_result_bytes": output.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "python_numeric_dtype_conversion": False, "signed_zero_preserved": True, "binary32_subnormals_preserved": True, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def distance_products(self, left, right, *, extended_shape=False):
        if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 or not value.flags.c_contiguous for value in (left, right)):
            raise ValueError("kmeans_distance_product_operand_format_invalid")
        if type(extended_shape) is not bool:
            raise ValueError("kmeans_product_512_configuration_invalid")
        rows, features = left.shape
        columns = right.shape[0]
        if not 0 < rows <= (512 if extended_shape else 256) or not 0 < columns <= (512 if extended_shape else 256) or not 0 < features <= 1024 or right.shape[1] != features:
            raise ValueError("kmeans_distance_product_shape_invalid")
        output, metrics, error = np.empty((rows, columns), dtype=np.float64), PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            try:
                native = self.library.wxkmeans_distance_products_512 if extended_shape else self.library.wxkmeans_distance_products
            except AttributeError as failure:
                raise ValueError("kmeans_product_512_abi_unavailable" if extended_shape else "kmeans_distance_product_abi_unavailable") from failure
            native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(PotentialMetrics), ct.c_char_p, ct.c_uint32]
            native.restype = ct.c_int
            if native(left.ctypes.data, right.ctypes.data, rows, columns, features, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != (0 if features == 1 and (rows > 1 or columns > 1) else 1) or not np.isfinite(output).all():
            raise ValueError("kmeans_distance_product_native_result_invalid")
        dispatch = "ddot" if rows == columns == 1 else ("scalar_outer" if features == 1 else ("dgemv_colmajor_trans" if rows == 1 or columns == 1 else ("dsyrk_rowmajor_upper" if rows == columns and left.ctypes.data == right.ctypes.data else "dgemm_rowmajor_nt")))
        return output, {"scope": "experimental_native_cpu_kmeans_distance_products", "execution_device": "cpu", "backend": "accelerate_ilp64", "rows": rows, "columns": columns, "features": features, "extended_shape_enabled": extended_shape, "maximum_extent": 512 if extended_shape else 256, "numpy_dispatch": dispatch, "native_blas_calls": metrics.blas_calls, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": left.nbytes + right.nbytes, "host_result_bytes": output.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "cpu_numeric_distance_products": True, "python_numeric_distance_products": False, "numpy_bitwise_certified": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}

    def weighted_potentials(self, distances, weights):
        if type(distances) is not np.ndarray or distances.dtype != np.float32 or distances.ndim != 2 or not distances.flags.c_contiguous or type(weights) is not np.ndarray or weights.dtype != np.float32 or weights.ndim not in (1, 2) or not weights.flags.c_contiguous or (weights.ndim == 2 and weights.shape[1] != 1):
            raise ValueError("kmeans_potential_operand_format_invalid")
        rows, columns = distances.shape
        if not 0 < rows <= 1024 or not 0 < columns <= 4096 or rows * columns > 1048576 or weights.size != columns:
            raise ValueError("kmeans_potential_shape_invalid")
        output = np.empty((rows,) if weights.ndim == 1 else (rows, 1), dtype=np.float32)
        metrics, error = PotentialMetrics(), ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        with self.lock:
            if self.closed:
                raise ValueError("kmeans_potential_context_closed")
            if self.native(distances.ctypes.data, weights.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.blas_calls != (0 if rows > 1 and columns == 1 else 1) or not np.isfinite(output).all() or np.any(output < 0):
            raise ValueError("kmeans_potential_native_result_invalid")
        return output, {"scope": "experimental_native_cpu_kmeans_weighted_potentials", "execution_device": "cpu", "backend": "accelerate_ilp64", "rows": rows, "columns": columns, "numpy_dispatch": "sdot" if rows == 1 else ("scalar_vector" if columns == 1 else "sgemv_colmajor_trans"), "native_blas_calls": metrics.blas_calls, "native_wall_ns": metrics.wall_ns, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "host_input_snapshot_bytes": distances.nbytes + weights.nbytes, "host_result_bytes": output.nbytes, "gpu_kernels": [], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "cpu_numeric_potential_computation": True, "python_numeric_potential_computation": False, "numpy_bitwise_certified": False, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}


def weighted_potentials(context, records, distances, weights):
    from tools.baseline.native_kmeans_prefix import NUMPY_CORE_SHA256, PLUSPLUS_SHA256
    if np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}).get("name") != "accelerate":
        raise ValueError("kmeans_potential_accelerate_numpy_required")
    output, record = context.weighted_potentials(distances, weights)
    record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_weighted_potentials", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_input_staging_copy_bytes": 0, "cpu_residuals": ["distance_blas_products", "weighted_potentials_native_accelerate", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return output


SAFE_SPARSE_DOT_SHA256 = "c8259546787ab8ac484494c57e1e32550a42a55c9f67346521f77f56a0a13185"


def distance_products(context, records, left, transposed_right, dense_output=False, *, extended_shape=False):
    from tools.baseline.native_kmeans_upcast_norms import EUCLIDEAN_SHA256, UPCAST_SHA256, NUMPY_CORE_SHA256
    if dense_output is not True or type(transposed_right) is not np.ndarray or transposed_right.ndim != 2 or not transposed_right.flags.f_contiguous:
        raise ValueError("kmeans_distance_product_transposed_right_required")
    if np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}).get("name") != "accelerate":
        raise ValueError("kmeans_potential_accelerate_numpy_required")
    output, record = context.distance_products(left, transposed_right.T, **({"extended_shape": extended_shape} if extended_shape is not False else {}))
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_distance_chunk_blas", "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "original_safe_sparse_dot_sha256": SAFE_SPARSE_DOT_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_input_staging_copy_bytes": 0, "cpu_residuals": ["float32_to_float64_chunk_conversion", "distance_blas_products_native_accelerate", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return output


def upcast_chunk(context, records, values, role, target_dtype=np.float64, *, extended_rows=False):
    from tools.baseline.native_kmeans_upcast_norms import EUCLIDEAN_SHA256, UPCAST_SHA256, NUMPY_CORE_SHA256
    if target_dtype is not np.float64:
        raise ValueError("kmeans_upcast_chunk_target_dtype_not_supported")
    if role not in ("X_chunk", "Y_chunk"):
        raise ValueError("kmeans_upcast_chunk_role_invalid")
    output, record = context.upcast_chunk(values, **({"extended_rows": True} if extended_rows else {}))
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_chunk_float64_conversion", "operand_role": role, "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_input_staging_copy_bytes": 0, "cpu_residuals": ["chunk_conversion_native_cpp", "distance_blas_products", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New native CPU Accelerate dylib path; existing files are preserved.")
    parser.add_argument("--with-inertia", action="store_true", help="Additionally build the pinned OpenMP inertia ABI using sklearn's existing libomp; default builds retain the previous dependencies.")
    parser.add_argument("--with-two-chunks", action="store_true", help="Additionally build the pinned OpenMP two-chunk Lloyd ABIs; implies --with-inertia and preserves legacy single-chunk ABIs.")
    args = parser.parse_args()
    print(json.dumps(build_library(args.output, with_inertia=args.with_inertia, with_two_chunks=args.with_two_chunks), sort_keys=True))


if __name__ == "__main__":
    main()

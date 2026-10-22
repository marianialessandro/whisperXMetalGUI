"""Bounded CPU Accelerate bridge preserving the pinned NumPy KMeans++ potential dispatch."""

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


def build_library(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("kmeans_potential_build_output_exists")
    source = Path(__file__).with_suffix(".cpp")
    with tempfile.TemporaryDirectory(prefix="kmeans-potential-build-", dir=output.parent) as directory:
        staged = Path(directory) / "potential.dylib"
        subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-dynamiclib", str(source), "-framework", "Accelerate", "-o", str(staged)], check=True, capture_output=True, timeout=60)
        os.link(staged, output)
    return {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "scope": "experimental_native_cpu_kmeans_potentials"}


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
        self.lock, self.closed = threading.RLock(), False
        self.identity = {"library_sha256": expected_sha256, "backend": "cpu_accelerate_ilp64", "production_backend": False}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self.lock:
            self.closed = True

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New native CPU Accelerate dylib path; existing files are preserved.")
    args = parser.parse_args()
    print(json.dumps(build_library(args.output), sort_keys=True))


if __name__ == "__main__":
    main()

"""Bounded experimental float32 speaker-embedding normalization on native Metal, without CPU fallback."""

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


def build_library(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("embedding_build_output_exists")
    source = Path(__file__).with_suffix(".mm")
    shader = Path(__file__).with_suffix(".metal")
    with tempfile.TemporaryDirectory(prefix="embedding-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "embedding.dylib"
        subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-fobjc-arc", "-dynamiclib", "-Wno-deprecated-declarations", str(source), "-framework", "Foundation", "-framework", "Metal", "-o", str(staged)], check=True, capture_output=True, timeout=60)
        os.link(staged, output)
    return {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "shader_sha256": hashlib.sha256(shader.read_bytes()).hexdigest(), "scope": "experimental_native_embedding_normalization"}


class NativeEmbeddingNormalization:
    def __init__(self, library, expected_sha256):
        library = Path(library).resolve(strict=True)
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("embedding_library_hash_mismatch")
        self.lock = threading.RLock()
        self.context = ct.c_void_p()
        self.library = ct.CDLL(str(library))
        self.library.wxembedding_create.argtypes = [ct.c_char_p, ct.c_uint64, ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_uint32]
        self.library.wxembedding_create.restype = ct.c_int
        self.library.wxembedding_normalize.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
        self.library.wxembedding_normalize.restype = ct.c_int
        self.library.wxembedding_destroy.argtypes = [ct.c_void_p]
        self.library.wxembedding_destroy.restype = None
        shader = Path(__file__).with_suffix(".metal").read_bytes()
        error = ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        if self.library.wxembedding_create(shader, len(shader), ct.byref(self.context), error, len(error)):
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
                self.library.wxembedding_destroy(self.context)
                self.context = ct.c_void_p()

    def normalize(self, embeddings):
        if not isinstance(embeddings, np.ndarray) or embeddings.dtype != np.float32 or embeddings.ndim != 2 or not embeddings.flags.c_contiguous:
            raise ValueError("embedding_input_format_invalid")
        rows, columns = embeddings.shape
        if not 0 < rows <= 4096 or not 0 < columns <= 1024:
            raise ValueError("embedding_shape_budget_invalid")
        output = np.empty_like(embeddings)
        metrics = Metrics()
        error = ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("embedding_context_closed")
            started = time.perf_counter_ns()
            if self.library.wxembedding_normalize(self.context, embeddings.ctypes.data, rows, columns, output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 1 or not np.isfinite(output).all():
            raise ValueError("embedding_native_result_invalid")
        return output, {"scope": "experimental_native_embedding_normalization", "rows": rows, "columns": columns, "gpu_kernels": ["embedding_normalize"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": embeddings.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="Compile a new local dylib at this path; existing paths are never overwritten.")
    args = parser.parse_args()
    print(json.dumps(build_library(args.build)))


if __name__ == "__main__":
    main()

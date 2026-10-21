"""Pinned ARM NumPy-policy dense KMeans precomputed squared row norms on Metal."""

import hashlib
import inspect
from pathlib import Path
import platform

import numpy as np
import numpy._core._multiarray_umath as numpy_core


ROW_NORMS_SHA256 = "f9d19a4bcb9966e7e0d2a4785f686f1af9dc310f9137e9656f3ee31f9cf36faa"
EINSUM_SHA256 = "bf169075ee6664590c53fd2ad4c16e914cd0d383f88a0d96dff70c0a44ab0174"
NUMPY_CORE_SHA256 = "cbf15cd30e2672b47a1cfc9aa82449d4eb0597a0d853e8697953568c1dc1468b"


def validate_reference(original):
    if not inspect.isfunction(original) or original.__closure__ or inspect.unwrap(original) is not original or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != ROW_NORMS_SHA256:
        raise ValueError("kmeans_squared_norms_source_not_supported")
    if platform.machine() != "arm64" or np.__version__ != "2.2.6" or not numpy_core.__cpu_features__.get("NEON_VFPV4") or hashlib.sha256(inspect.getsource(np.einsum).encode()).hexdigest() != EINSUM_SHA256 or hashlib.sha256(Path(numpy_core.__file__).read_bytes()).hexdigest() != NUMPY_CORE_SHA256 or original.__globals__.get("np") is not np:
        raise ValueError("kmeans_squared_norms_runtime_not_supported")


def squared_norms(context, records, features):
    result, record = context.kmeans_squared_norms(features)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_precomputed_squared_norms", "original_row_norms_sha256": ROW_NORMS_SHA256, "numpy_einsum_sha256": EINSUM_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["kmeans_initialization_and_its_internal_norms", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result

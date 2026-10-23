"""Pinned OpenMP float32 inertia adapter preserving the original fit thread count."""

import ctypes as ct
import hashlib
from pathlib import Path


COMMON_BINARY_SHA256 = "793cb26715cc85762eb460e29cce45a4a40584ddcaba761bfd69adafd080f3cf"
OPENMP_SHA256 = "7fe04d06d824817d39e4c99ec44e837fd6b5f0f75f2cc3e77a384c298f3f1709"


def runtime_path():
    import sklearn
    path = Path(sklearn.__file__).parent / ".dylibs/libomp.dylib"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != OPENMP_SHA256:
        raise ValueError("kmeans_inertia_openmp_runtime_not_supported")
    return path.resolve()


def validate_loaded_runtime(library):
    try:
        function = library.wxkmeans_inertia_runtime_path
    except AttributeError as error:
        raise ValueError("kmeans_inertia_runtime_abi_unavailable") from error
    function.argtypes = []; function.restype = ct.c_char_p
    path = function()
    if not path or hashlib.sha256(Path(path.decode()).read_bytes()).hexdigest() != OPENMP_SHA256:
        raise ValueError("kmeans_inertia_loaded_runtime_not_supported")


def validate_reference(original):
    import sklearn.cluster._k_means_common as common
    if original is not common._inertia_dense or hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest() != COMMON_BINARY_SHA256:
        raise ValueError("kmeans_inertia_reference_not_supported")
    runtime_path()


def inertia_value(context, records, X, weights, centers, labels, n_threads, single_label=-1):
    output, record = context.inertia_dense(X, weights, centers, labels, n_threads, single_label)
    record.update({"adapter_scope": "sklearn_dense_float32_lloyd_inertia", "original_common_binary_sha256": COMMON_BINARY_SHA256, "openmp_runtime_sha256": OPENMP_SHA256, "cpu_residuals": ["convergence_and_best_initialization_control", "host_staging_and_control"]})
    records.append(record)
    return output

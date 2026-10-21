"""Source-pinned dense KMeans input mean on Metal, retaining the original fit."""

import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading

import numpy as np


FIT_SHA256 = "7d82e93b0e42485df83af8b86b350a1dd1693aed2881dab19a9e6ec53e37c659"
FIT_CONTEXT_SHA256 = "d8352c8507fc8a9d8936bf31df1b02bcf3d2710726f3928039d90c1bc11b88d3"
_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_mean(original, mean):
    inner = inspect.unwrap(original)
    source = inspect.getsource(inner)
    if hashlib.sha256(source.encode()).hexdigest() != FIT_SHA256 or inner.__closure__:
        raise ValueError("kmeans_mean_source_not_supported")
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.parse("X_mean = X.mean(axis=0)").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    decorator = ast.parse("_fit_context(prefer_skip_nested_validation=True)", mode="eval").body
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or len(tree.body[0].decorator_list) != 1 or ast.dump(tree.body[0].decorator_list[0]) != ast.dump(decorator) or len(matches) != 1:
        raise ValueError("kmeans_mean_expression_contract_changed")
    matches[0].value = ast.parse("__wx_native_kmeans_mean(X)", mode="eval").body
    namespace = dict(inner.__globals__)
    namespace["__wx_native_kmeans_mean"] = mean
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native KMeans mean>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_mean_signature_changed")
    return candidate


def feature_mean(context, records, features):
    if type(features) is not np.ndarray or features.dtype != np.float32 or features.ndim != 2 or not features.flags.c_contiguous:
        raise ValueError("kmeans_mean_operand_not_supported")
    rows, dimensions = features.shape
    if not 0 < rows <= 4096 or not 2 <= dimensions <= 1024:
        raise ValueError("kmeans_mean_shape_not_supported")
    labels = np.zeros(rows, dtype=np.int32)
    output, record = context.kmeans_centroids_vbx(features, labels, 1)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_input_mean", "original_fit_sha256": FIT_SHA256, "original_fit_context_sha256": FIT_CONTEXT_SHA256, "host_generated_label_bytes": labels.nbytes, "cpu_residuals": ["feature_subtraction_and_restoration", "kmeans_initialization", "lloyd_or_elkan", "final_centroid_translation", "host_staging_and_control"]})
    records.append(record)
    return output[0]


@contextlib.contextmanager
def native_kmeans_feature_mean(context, records):
    global _ACTIVE
    from sklearn.base import _fit_context
    from sklearn.cluster import KMeans

    with _LOCK:
        if _ACTIVE:
            raise ValueError("kmeans_mean_scope_already_active")
        if not isinstance(records, list):
            raise ValueError("kmeans_mean_records_required")
        if np.__version__ != "2.2.6" or importlib.metadata.version("scikit-learn") != "1.9.1":
            raise ValueError("kmeans_mean_runtime_not_supported")
        if getattr(context.library, "wxdouble_vbx_kmeans_centroids", None) is None:
            raise ValueError("kmeans_mean_abi_unavailable")
        original = KMeans.fit
        inner = inspect.unwrap(original)
        if not inspect.isfunction(original) or not inspect.isfunction(inner):
            raise ValueError("kmeans_mean_fit_wrapper_not_supported")
        reference = _fit_context(prefer_skip_nested_validation=True)(inner)
        closure = dict(zip(original.__code__.co_freevars, (cell.cell_contents for cell in original.__closure__ or ())))
        if original.__code__ is not reference.__code__ or getattr(original, "__wrapped__", None) is not inner or closure.get("fit_method") is not inner or closure.get("prefer_skip_nested_validation") is not True or inner.__globals__.get("_fit_context") is not _fit_context or hashlib.sha256(inspect.getsource(_fit_context).encode()).hexdigest() != FIT_CONTEXT_SHA256:
            raise ValueError("kmeans_mean_fit_wrapper_not_supported")
        candidate = compile_native_mean(original, lambda features: feature_mean(context, records, features))
        owner = threading.get_ident()

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return candidate(*args, **kwargs) if threading.get_ident() == owner else original(*args, **kwargs)

        _ACTIVE = True
        KMeans.fit = wrapped
        try:
            yield
        finally:
            KMeans.fit = original
            _ACTIVE = False

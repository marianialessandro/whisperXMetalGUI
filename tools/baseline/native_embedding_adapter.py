"""Scoped, source-pinned substitution of the VBx training-embedding normalization expression."""

import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading

import numpy as np


VBX_CALL_SHA256 = "8baf21775d7add4f8b666ab480470c2748592074541f1cfe32a400ed118540d7"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_normalization(original, normalize, centroid=None):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != VBX_CALL_SHA256 or original.__closure__:
        raise ValueError("embedding_vbx_source_not_supported")
    tree = ast.parse(source)
    expected = ast.parse("train_embeddings_normed = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    if len(matches) != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("embedding_vbx_normalization_contract_changed")
    if centroid is not None:
        centroid_expected = ast.parse("centroids = W.T @ train_embeddings.reshape(-1, dimension) / W.sum(0, keepdims=True).T").body[0]
        centroid_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(centroid_expected)]
        if len(centroid_matches) != 1:
            raise ValueError("embedding_vbx_centroid_contract_changed")
        centroid_matches[0].value = ast.copy_location(ast.parse("__wx_native_centroids(W, train_embeddings.reshape(-1, dimension))").body[0].value, centroid_matches[0].value)
    if normalize is not None:
        matches[0].value = ast.copy_location(ast.parse("__wx_native_embedding_normalize(train_embeddings)").body[0].value, matches[0].value)
    namespace = dict(original.__globals__)
    namespace["__wx_native_embedding_normalize"] = normalize
    namespace["__wx_native_centroids"] = centroid
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native VBx normalization>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("embedding_vbx_signature_changed")
    return candidate


def weighted_centroids(context, records, weights, features):
    if type(weights) is not np.ndarray or weights.dtype != np.float64 or weights.ndim != 2 or not (weights.flags.c_contiguous or weights.flags.f_contiguous) or type(features) is not np.ndarray or features.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or features.ndim != 2:
        raise ValueError("vbx_centroid_operand_format_not_supported")
    frames, speakers = weights.shape
    dimensions = features.shape[1]
    if not 0 < frames <= 4096 or not 0 < speakers <= 1024 or not 0 < dimensions <= 1024 or features.shape[0] != frames or weights.size * dimensions > 32 * 1024 * 1024 or features.size > 1024 * 1024:
        raise ValueError("vbx_centroid_shape_not_supported")
    column_contiguous = weights.flags.f_contiguous
    staged_weights = np.ascontiguousarray(weights.T)
    staged_features = np.ascontiguousarray(features, dtype=np.float64)
    output, record = context.centroids_vbx(staged_weights, staged_features, "pairwise" if column_contiguous or speakers == 1 else "sequential")
    record = dict(record)
    record.update({"adapter_scope": "vbx_weighted_centroids_after_speaker_selection", "original_call_sha256": VBX_CALL_SHA256, "weight_layout": "F" if column_contiguous else "C" if weights.flags.c_contiguous else "strided", "host_weight_staging_copy_bytes": 0 if np.shares_memory(staged_weights, weights) else staged_weights.nbytes, "host_feature_staging_copy_bytes": 0 if staged_features is features else staged_features.nbytes, "feature_dtype_conversion": str(features.dtype) != "float64", "cpu_residuals": ["speaker_selection", "ahc", "optional_kmeans_centroids", "distance_and_constrained_assignment", "host_staging_and_control"]})
    records.append(record)
    return output


@contextlib.contextmanager
def native_embedding_normalization(context, records, centroid_context=None, centroid_records=None):
    global _ACTIVE
    from pyannote.audio.pipelines.clustering import VBxClustering

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("embedding_normalization_scope_already_active")
        if importlib.metadata.version("pyannote-audio") != "4.0.4":
            raise ValueError("embedding_pyannote_version_not_supported")
        if centroid_context is not None:
            if not isinstance(centroid_records, list):
                raise ValueError("vbx_centroid_records_required")
            if np.__version__ != "2.2.6":
                raise ValueError("vbx_centroid_numpy_not_supported")
            if getattr(centroid_context.library, "wxdouble_vbx_centroids", None) is None:
                raise ValueError("double_vbx_centroid_abi_unavailable")
        if context is None and centroid_context is None:
            raise ValueError("embedding_or_centroid_context_required")
        original = VBxClustering.__call__
        owner_thread = threading.get_ident()

        def normalize(embeddings):
            output, record = context.normalize(embeddings)
            record = dict(record)
            record["adapter_scope"] = "vbx_training_embeddings_before_centroid_linkage"
            record["original_call_sha256"] = VBX_CALL_SHA256
            record["pyannote_audio_version"] = "4.0.4"
            records.append(record)
            return output

        def centroids(weights, features):
            return weighted_centroids(centroid_context, centroid_records, weights, features)

        candidate = compile_native_normalization(original, normalize if context is not None else None, centroids if centroid_context is not None else None)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            return candidate(*args, **kwargs)

        _ACTIVE = True
        VBxClustering.__call__ = wrapped
        try:
            yield
        finally:
            VBxClustering.__call__ = original
            _ACTIVE = False

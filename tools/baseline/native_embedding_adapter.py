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
CDIST_SHA256 = "eb769cabbc7d564024b480d426594d08aff4280675d887ddd5ba6a35b13905fa"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_normalization(original, normalize, centroid=None, distance=None, soft_scores=False, argmax=None, resident_argmax=False, inactive=None):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != VBX_CALL_SHA256 or original.__closure__:
        raise ValueError("embedding_vbx_source_not_supported")
    if resident_argmax and (not soft_scores or argmax is None or distance is None):
        raise ValueError("vbx_resident_argmax_configuration_invalid")
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
    if inactive is not None:
        const_expected = ast.parse("const = soft_clusters.min() - 1.").body[0]
        mask_expected = ast.parse("soft_clusters[segmentations.data.sum(1) == 0] = const").body[0]
        const_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(const_expected)]
        mask_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(mask_expected)]
        const_loads = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "const" and isinstance(node.ctx, ast.Load)]
        if len(const_matches) != 1 or len(mask_matches) != 1 or len(const_loads) != 1:
            raise ValueError("embedding_vbx_inactive_contract_changed")
        const_matches[0].targets = [ast.Name(id="soft_clusters", ctx=ast.Store())]
        const_matches[0].value = ast.parse("__wx_native_inactive(soft_clusters, segmentations.data)").body[0].value
        class RemoveMask(ast.NodeTransformer):
            def visit_Assign(self, node):
                return None if node is mask_matches[0] else self.generic_visit(node)
        tree = RemoveMask().visit(tree)
    if argmax is not None:
        argmax_expected = ast.parse("hard_clusters = np.argmax(soft_clusters, axis=2)").body[0]
        argmax_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(argmax_expected)]
        if len(argmax_matches) != 1:
            raise ValueError("embedding_vbx_argmax_contract_changed")
        argmax_matches[0].value = ast.Call(func=ast.Name(id="__wx_native_argmax", ctx=ast.Load()), args=[ast.Name(id="soft_clusters", ctx=ast.Load())], keywords=[])
    if soft_scores:
        score_expected = ast.parse("soft_clusters = 2 - e2k_distance").body[0]
        score_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(score_expected)]
        distance_loads = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "e2k_distance" and isinstance(node.ctx, ast.Load)]
        if distance is None or len(score_matches) != 1 or len(distance_loads) != 1:
            raise ValueError("embedding_vbx_soft_score_contract_changed")
        score_matches[0].value = ast.Name(id="e2k_distance", ctx=ast.Load())
    if distance is not None:
        distance_expected = ast.parse('cdist(rearrange(embeddings, "c s d -> (c s) d"), centroids, metric=self.metric)').body[0].value
        distance_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(distance_expected)]
        if len(distance_matches) != 1:
            raise ValueError("embedding_vbx_distance_contract_changed")
        if resident_argmax:
            distance_matches[0].keywords.append(ast.keyword(arg="constrained_assignment", value=ast.Name(id="constrained_assignment", ctx=ast.Load())))
        distance_matches[0].func = ast.copy_location(ast.Name(id="__wx_native_cosine", ctx=ast.Load()), distance_matches[0].func)
    if normalize is not None:
        matches[0].value = ast.copy_location(ast.parse("__wx_native_embedding_normalize(train_embeddings)").body[0].value, matches[0].value)
    namespace = dict(original.__globals__)
    namespace["__wx_native_embedding_normalize"] = normalize
    namespace["__wx_native_centroids"] = centroid
    namespace["__wx_native_cosine"] = distance
    namespace["__wx_native_argmax"] = argmax
    namespace["__wx_native_inactive"] = inactive
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native VBx normalization>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("embedding_vbx_signature_changed")
    return candidate


def weighted_centroids(context, records, weights, features, distances=False):
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
    record.update({"adapter_scope": "vbx_weighted_centroids_after_speaker_selection", "original_call_sha256": VBX_CALL_SHA256, "weight_layout": "F" if column_contiguous else "C" if weights.flags.c_contiguous else "strided", "host_weight_staging_copy_bytes": 0 if np.shares_memory(staged_weights, weights) else staged_weights.nbytes, "host_feature_staging_copy_bytes": 0 if staged_features is features else staged_features.nbytes, "feature_dtype_conversion": str(features.dtype) != "float64", "cpu_residuals": ["speaker_selection", "ahc", "optional_kmeans_centroids", "constrained_assignment" if distances else "distance_and_constrained_assignment", "host_staging_and_control"]})
    records.append(record)
    return output


def cosine_distances(context, records, embeddings, centroids, metric="cosine", native_centroids=False, soft_scores=False, return_labels=False):
    if metric != "cosine":
        raise ValueError("vbx_cosine_metric_not_supported")
    if any(type(value) is not np.ndarray or value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or value.ndim != 2 for value in (embeddings, centroids)):
        raise ValueError("vbx_cosine_operand_format_not_supported")
    rows, features = embeddings.shape
    count = centroids.shape[0]
    if not 0 < rows <= 4096 or not 0 < count <= 1024 or not 0 < features <= 1024 or centroids.shape[1] != features or rows * count * features > 32 * 1024 * 1024:
        raise ValueError("vbx_cosine_shape_not_supported")
    staged_embeddings, staged_centroids = np.ascontiguousarray(embeddings, dtype=np.float64), np.ascontiguousarray(centroids, dtype=np.float64)
    output, record = context.cosine_vbx(staged_embeddings, staged_centroids, soft_scores=soft_scores, return_labels=return_labels)
    record = dict(record)
    record.update({"adapter_scope": "vbx_embedding_to_centroid_cosine_distances", "original_call_sha256": VBX_CALL_SHA256, "scipy_cdist_sha256": CDIST_SHA256, "host_embedding_staging_copy_bytes": 0 if staged_embeddings is embeddings else staged_embeddings.nbytes, "host_centroid_staging_copy_bytes": 0 if staged_centroids is centroids else staged_centroids.nbytes, "embedding_dtype_conversion": str(embeddings.dtype) != "float64", "centroid_dtype_conversion": str(centroids.dtype) != "float64", "cpu_residuals": ["ahc", "optional_kmeans", "constrained_assignment", "host_staging_and_control"] + ([] if native_centroids else ["weighted_centroids"]) + ([] if soft_scores else ["soft_score_subtraction"])})
    records.append(record)
    return output


def unconstrained_argmax(context, records, scores):
    if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 3:
        raise ValueError("vbx_argmax_operand_format_not_supported")
    chunks, local_speakers, clusters = scores.shape
    rows = chunks * local_speakers
    if not 0 < rows <= 4096 or not 0 < clusters <= 1024:
        raise ValueError("vbx_argmax_shape_not_supported")
    staged = np.ascontiguousarray(scores).reshape(rows, clusters)
    output, record = context.argmax_vbx(staged)
    record = dict(record)
    record.update({"adapter_scope": "vbx_unconstrained_assignment", "original_call_sha256": VBX_CALL_SHA256, "chunks": chunks, "local_speakers": local_speakers, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, scores) else staged.nbytes, "cpu_residuals": ["constrained_assignment_when_selected", "ahc", "optional_kmeans", "host_staging_and_control"]})
    records.append(record)
    return output.reshape(chunks, local_speakers)


def inactive_speaker_mask(context, records, scores, segments):
    if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 3 or type(segments) is not np.ndarray or segments.ndim != 3 or segments.dtype.kind not in "biuf" or segments.dtype.itemsize > 8:
        raise ValueError("vbx_inactive_operand_format_not_supported")
    chunks, speakers, columns = scores.shape
    rows, frames = chunks * speakers, segments.shape[1]
    if not 0 < rows <= 4096 or not 0 < columns <= 1024 or not 0 < frames <= 4096 or segments.shape[0] != chunks or segments.shape[2] != speakers or rows * frames > 1024 * 1024:
        raise ValueError("vbx_inactive_shape_not_supported")
    staged_scores = np.ascontiguousarray(scores).reshape(rows, columns)
    staged_segments = np.ascontiguousarray(segments.transpose(0, 2, 1), dtype=np.float64).reshape(rows, frames)
    output, record = context.inactive_mask_vbx(staged_scores, staged_segments)
    record = dict(record)
    record.update({"adapter_scope": "vbx_constrained_inactive_speaker_preparation", "original_call_sha256": VBX_CALL_SHA256, "segment_source_dtype": str(segments.dtype), "host_score_staging_copy_bytes": 0 if np.shares_memory(staged_scores, scores) else staged_scores.nbytes, "host_segment_staging_copy_bytes": 0 if np.shares_memory(staged_segments, segments) else staged_segments.nbytes, "cpu_residuals": ["constrained_nan_replacement_and_solver", "ahc", "optional_kmeans", "host_staging_and_control"]})
    records.append(record)
    return output.reshape(scores.shape)


@contextlib.contextmanager
def native_embedding_normalization(context, records, centroid_context=None, centroid_records=None, distance_context=None, distance_records=None, soft_scores=False, argmax_context=None, argmax_records=None, resident_argmax=False, inactive_context=None, inactive_records=None):
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
        if inactive_context is not None:
            if not isinstance(inactive_records, list):
                raise ValueError("vbx_inactive_records_required")
            if np.__version__ != "2.2.6":
                raise ValueError("vbx_inactive_runtime_not_supported")
            if getattr(inactive_context.library, "wxdouble_vbx_inactive_mask", None) is None:
                raise ValueError("double_vbx_inactive_abi_unavailable")
        if type(resident_argmax) is not bool or (resident_argmax and (not soft_scores or argmax_context is None or argmax_context is not distance_context)):
            raise ValueError("vbx_resident_argmax_configuration_invalid")
        if resident_argmax and getattr(distance_context.library, "wxdouble_vbx_cosine_scores_argmax", None) is None:
            raise ValueError("double_vbx_cosine_argmax_abi_unavailable")
        if argmax_context is not None:
            if not isinstance(argmax_records, list):
                raise ValueError("vbx_argmax_records_required")
            if np.__version__ != "2.2.6" or np.dtype(np.intp).itemsize != 8:
                raise ValueError("vbx_argmax_runtime_not_supported")
            if getattr(argmax_context.library, "wxdouble_vbx_argmax", None) is None:
                raise ValueError("double_vbx_argmax_abi_unavailable")
        if type(soft_scores) is not bool or (soft_scores and distance_context is None):
            raise ValueError("vbx_soft_scores_require_cosine")
        if soft_scores and getattr(distance_context.library, "wxdouble_vbx_cosine_scores", None) is None:
            raise ValueError("double_vbx_cosine_score_abi_unavailable")
        if distance_context is not None:
            import scipy.spatial.distance as spatial_distance
            if not isinstance(distance_records, list):
                raise ValueError("vbx_cosine_records_required")
            if np.__version__ != "2.2.6" or importlib.metadata.version("scipy") != "1.15.3":
                raise ValueError("vbx_cosine_runtime_not_supported")
            if getattr(distance_context.library, "wxdouble_vbx_cosine", None) is None:
                raise ValueError("double_vbx_cosine_abi_unavailable")
            if VBxClustering.__call__.__globals__.get("cdist") is not spatial_distance.cdist or hashlib.sha256(textwrap.dedent(inspect.getsource(spatial_distance.cdist)).encode()).hexdigest() != CDIST_SHA256:
                raise ValueError("vbx_cosine_reference_not_supported")
        if context is None and centroid_context is None and distance_context is None and argmax_context is None and inactive_context is None:
            raise ValueError("embedding_or_centroid_context_required")
        original = VBxClustering.__call__
        owner_thread = threading.get_ident()
        pending_labels = None

        def normalize(embeddings):
            output, record = context.normalize(embeddings)
            record = dict(record)
            record["adapter_scope"] = "vbx_training_embeddings_before_centroid_linkage"
            record["original_call_sha256"] = VBX_CALL_SHA256
            record["pyannote_audio_version"] = "4.0.4"
            records.append(record)
            return output

        def centroids(weights, features):
            return weighted_centroids(centroid_context, centroid_records, weights, features, distance_context is not None)

        def distances(embeddings, centroids, metric, constrained_assignment=False):
            nonlocal pending_labels
            resident = resident_argmax and not constrained_assignment
            result = cosine_distances(distance_context, distance_records, embeddings, centroids, metric, centroid_context is not None, soft_scores, resident)
            if resident:
                output, labels = result
                pending_labels = (output, labels, len(distance_records) - 1)
                return output
            pending_labels = None
            return result

        def assign(scores):
            nonlocal pending_labels
            if resident_argmax:
                if pending_labels is None:
                    raise ValueError("vbx_resident_argmax_cache_missing")
                output, labels, dispatch = pending_labels
                if type(scores) is not np.ndarray or scores.ndim != 3 or scores.shape[0] * scores.shape[1] != labels.size or scores.shape[2] != output.shape[1] or not np.shares_memory(scores, output):
                    raise ValueError("vbx_resident_argmax_cache_mismatch")
                pending_labels = None
                argmax_records.append({"adapter_scope": "vbx_unconstrained_assignment", "original_call_sha256": VBX_CALL_SHA256, "resident_label_reuse": True, "shared_cosine_dispatch_index": dispatch, "gpu_kernels": ["double_vbx_argmax"], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "host_input_staging_copy_bytes": 0, "rows": labels.size, "columns": output.shape[1], "quality_certified": False, "full_pipeline_gpu_certified": False})
                return labels.reshape(scores.shape[:2])
            return unconstrained_argmax(argmax_context, argmax_records, scores)

        def mask_inactive(scores, segments):
            return inactive_speaker_mask(inactive_context, inactive_records, scores, segments)

        candidate = compile_native_normalization(original, normalize if context is not None else None, centroids if centroid_context is not None else None, distances if distance_context is not None else None, soft_scores=soft_scores, argmax=assign if argmax_context is not None else None, resident_argmax=resident_argmax, inactive=mask_inactive if inactive_context is not None else None)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            nonlocal pending_labels
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            pending_labels = None
            try:
                return candidate(*args, **kwargs)
            finally:
                pending_labels = None

        _ACTIVE = True
        VBxClustering.__call__ = wrapped
        try:
            yield
        finally:
            VBxClustering.__call__ = original
            _ACTIVE = False

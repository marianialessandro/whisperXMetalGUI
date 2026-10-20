"""Source-pinned native NaN replacement before the unchanged constrained solver."""

import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading
import warnings

import numpy as np


CONSTRAINED_SHA256 = "93d244a6438c3acb9a75f87a3c8d6b61893ded11fdf02a992e4f0d839086f9d6"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_cleanup(original, cleanup):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != CONSTRAINED_SHA256 or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("constrained_cleanup_source_not_supported")
    tree = ast.parse(source)
    expected = ast.parse("soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    if len(matches) != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("constrained_cleanup_expression_contract_changed")
    matches[0].value = ast.parse("__wx_constrained_cleanup(soft_clusters)").body[0].value
    namespace = dict(original.__globals__)
    namespace["__wx_constrained_cleanup"] = cleanup
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native constrained NaN cleanup>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("constrained_cleanup_signature_changed")
    return candidate


def cleanup_scores(context, records, scores):
    if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 3:
        raise ValueError("constrained_cleanup_operand_format_not_supported")
    rows, columns = scores.shape[0] * scores.shape[1], scores.shape[2]
    if not 0 < rows <= 4096 or not 0 < columns <= 1024:
        raise ValueError("constrained_cleanup_shape_not_supported")
    staged = np.ascontiguousarray(scores).reshape(rows, columns)
    output, record = context.nan_cleanup_vbx(staged)
    record = dict(record)
    record.update({"adapter_scope": "base_constrained_nan_replacement", "original_constrained_sha256": CONSTRAINED_SHA256, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, scores) else staged.nbytes, "cpu_residuals": ["initial_output_allocation", "linear_sum_assignment", "label_scatter", "host_staging_and_control"]})
    records.append(record)
    if record["all_nan_input"]:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
    return output.reshape(scores.shape)


@contextlib.contextmanager
def native_constrained_nan_cleanup(context, records):
    global _ACTIVE
    from pyannote.audio.pipelines.clustering import BaseClustering
    from scipy.optimize import linear_sum_assignment

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("constrained_cleanup_scope_already_active")
        if np.__version__ != "2.2.6" or np.dtype(np.intp).itemsize != 8 or importlib.metadata.version("pyannote-audio") != "4.0.4" or importlib.metadata.version("scipy") != "1.15.3":
            raise ValueError("constrained_cleanup_runtime_not_supported")
        if not isinstance(records, list):
            raise ValueError("constrained_cleanup_records_required")
        if getattr(context.library, "wxdouble_vbx_nan_cleanup", None) is None:
            raise ValueError("double_vbx_nan_cleanup_abi_unavailable")
        original = BaseClustering.constrained_argmax
        if original.__globals__.get("linear_sum_assignment") is not linear_sum_assignment:
            raise ValueError("constrained_cleanup_solver_reference_changed")
        owner_thread = threading.get_ident()

        def cleanup(scores):
            return cleanup_scores(context, records, scores)

        candidate = compile_native_cleanup(original, cleanup)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            return candidate(*args, **kwargs)

        _ACTIVE = True
        BaseClustering.constrained_argmax = wrapped
        try:
            yield
        finally:
            BaseClustering.constrained_argmax = original
            _ACTIVE = False

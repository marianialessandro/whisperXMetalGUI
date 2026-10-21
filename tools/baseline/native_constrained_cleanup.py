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


def compile_native_cleanup(original, cleanup, assignment=None, batch=None):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != CONSTRAINED_SHA256 or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("constrained_cleanup_source_not_supported")
    tree = ast.parse(source)
    expected = ast.parse("soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    if len(matches) != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("constrained_cleanup_expression_contract_changed")
    matches[0].value = ast.parse("__wx_constrained_cleanup(soft_clusters)").body[0].value
    if assignment is not None:
        expected_assignment = ast.parse("linear_sum_assignment(cost, maximize=True)", mode="eval").body
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected_assignment)]
        if len(assignments) != 1:
            raise ValueError("constrained_assignment_expression_contract_changed")
        assignments[0].func = ast.Name(id="__wx_constrained_assignment", ctx=ast.Load())
        assignments[0].keywords = []
    if batch is not None:
        if assignment is not None:
            raise ValueError("constrained_batch_configuration_invalid")
        expected_nodes = [ast.parse("num_chunks, num_speakers, num_clusters = soft_clusters.shape").body[0], ast.parse("hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)").body[0], ast.parse("for c, cost in enumerate(soft_clusters):\n    speakers, clusters = linear_sum_assignment(cost, maximize=True)\n    for s, k in zip(speakers, clusters):\n        hard_clusters[c, s] = k").body[0], ast.parse("return hard_clusters").body[0]]
        if len(tree.body[0].body) != 5 or any(sum(ast.dump(node) == ast.dump(expected) for node in ast.walk(tree)) != 1 for expected in expected_nodes):
            raise ValueError("constrained_batch_expression_contract_changed")
        tree.body[0].body = [ast.Return(value=ast.parse("__wx_constrained_batch(soft_clusters)", mode="eval").body)]
    namespace = dict(original.__globals__)
    namespace["__wx_constrained_cleanup"] = cleanup
    namespace["__wx_constrained_assignment"] = assignment
    namespace["__wx_constrained_batch"] = batch
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
def native_constrained_nan_cleanup(context, records, assignment_records=None, batch_records=None):
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
        if assignment_records is not None:
            from tools.baseline.native_assignment import assignment_pairs, validate_assignment_reference
            if not isinstance(assignment_records, list):
                raise ValueError("constrained_assignment_records_required")
            if getattr(context.library, "wxdouble_linear_assignment_max", None) is None:
                raise ValueError("double_assignment_abi_unavailable")
            validate_assignment_reference(linear_sum_assignment)
        if batch_records is not None:
            from tools.baseline.native_assignment import batch_labels, validate_assignment_reference
            if assignment_records is not None:
                raise ValueError("constrained_batch_configuration_invalid")
            if not isinstance(batch_records, list):
                raise ValueError("constrained_batch_records_required")
            if getattr(context.library, "wxdouble_assignment_batch", None) is None:
                raise ValueError("double_assignment_batch_abi_unavailable")
            if original.__globals__.get("np") is not np:
                raise ValueError("constrained_batch_numpy_reference_changed")
            validate_assignment_reference(linear_sum_assignment)
        owner_thread = threading.get_ident()

        def cleanup(scores):
            output = cleanup_scores(context, records, scores)
            if assignment_records is not None:
                records[-1]["cpu_residuals"] = [name for name in records[-1]["cpu_residuals"] if name != "linear_sum_assignment"]
            return output

        def assignment(scores):
            return assignment_pairs(context, assignment_records, scores)

        def batch(scores):
            return batch_labels(context, batch_records, scores)

        candidate = compile_native_cleanup(original, cleanup, assignment=assignment if assignment_records is not None else None, batch=batch if batch_records is not None else None)

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

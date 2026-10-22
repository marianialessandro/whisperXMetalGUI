"""Source-pinned CPU center-shift reduction inside the original Lloyd control loop."""

import ast
import hashlib
import inspect
import textwrap

import numpy as np

from tools.baseline.native_kmeans_prefix import NUMPY_CORE_SHA256


LLOYD_SHA256 = "d4bf3cb3792265da9d5d1550fa0f61fdd389f0302a677f72e761c482a4824880"
THREADPOOL_DECORATOR_SHA256 = "ec72498b68ea605724643838a8bc4c343ef6d150f8d09fd8b0d068d08c6cab60"


def compile_native_lloyd(original, shift_total):
    from sklearn.cluster import _kmeans
    from sklearn.utils.parallel import _threadpool_controller_decorator
    inner = inspect.unwrap(original)
    if original is not _kmeans._kmeans_single_lloyd or not inspect.isfunction(original) or not inspect.isfunction(inner) or inner.__closure__ or hashlib.sha256(inspect.getsource(inner).encode()).hexdigest() != LLOYD_SHA256:
        raise ValueError("kmeans_lloyd_reference_not_supported")
    reference = _threadpool_controller_decorator(limits=1, user_api="blas")(inner)
    closure = dict(zip(original.__code__.co_freevars, (cell.cell_contents for cell in original.__closure__ or ())))
    if original.__code__ is not reference.__code__ or closure.get("func") is not inner or closure.get("limits") != 1 or closure.get("user_api") != "blas" or inner.__globals__.get("_threadpool_controller_decorator") is not _threadpool_controller_decorator or hashlib.sha256(inspect.getsource(_threadpool_controller_decorator).encode()).hexdigest() != THREADPOOL_DECORATOR_SHA256:
        raise ValueError("kmeans_lloyd_wrapper_not_supported")
    tree = ast.parse(textwrap.dedent(inspect.getsource(inner)))
    target = ast.parse("(center_shift**2).sum()", mode="eval").body
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
    decorator = ast.parse('_threadpool_controller_decorator(limits=1, user_api="blas")', mode="eval").body
    if len(tree.body) != 1 or len(tree.body[0].decorator_list) != 1 or ast.dump(tree.body[0].decorator_list[0]) != ast.dump(decorator) or len(matches) != 1:
        raise ValueError("kmeans_lloyd_shift_expression_contract_changed")
    matches[0].func = ast.Name(id="__wx_lloyd_shift_total", ctx=ast.Load())
    matches[0].args = [ast.Name(id="center_shift", ctx=ast.Load())]
    namespace = dict(inner.__globals__); namespace["__wx_lloyd_shift_total"] = shift_total
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native Lloyd shift reduction>", "exec"), namespace)
    candidate = namespace[inner.__name__]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_lloyd_signature_changed")
    return candidate


def shift_total(context, records, shifts):
    if np.getbufsize() != 8192:
        raise ValueError("kmeans_lloyd_shift_numpy_buffer_not_supported")
    output, record = context.lloyd_shift_total(shifts)
    record.update({"adapter_scope": "sklearn_dense_float32_lloyd_center_shift", "original_lloyd_sha256": LLOYD_SHA256, "threadpool_decorator_sha256": THREADPOOL_DECORATOR_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["lloyd_cython_iteration_and_centroid_updates", "inertia", "convergence_comparison_and_control", "host_staging_and_control"]})
    records.append(record)
    return output

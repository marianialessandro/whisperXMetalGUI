"""Source-pinned CPU center-shift reduction inside the original Lloyd control loop."""

import ast
import hashlib
import inspect
import textwrap

import numpy as np

from tools.baseline.native_kmeans_prefix import NUMPY_CORE_SHA256


LLOYD_SHA256 = "d4bf3cb3792265da9d5d1550fa0f61fdd389f0302a677f72e761c482a4824880"
THREADPOOL_DECORATOR_SHA256 = "ec72498b68ea605724643838a8bc4c343ef6d150f8d09fd8b0d068d08c6cab60"


ARRAY_EQUAL_SHA256 = "be4f8df0142c72cbd63a30ed74a48129155a6aa845e5e6d2f2898a2a59d35be6"


def compile_native_lloyd(original, shift_total=None, labels_equal=None, dense_iteration=None, inertia=None, convergence=None):
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
    if shift_total is not None and convergence is None:
        matches[0].func = ast.Name(id="__wx_lloyd_shift_total", ctx=ast.Load())
        matches[0].args = [ast.Name(id="center_shift", ctx=ast.Load())]
    if convergence is not None:
        target_assignment = ast.parse("center_shift_tot = (center_shift**2).sum()").body[0]
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(target_assignment)]
        tests = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(ast.parse("center_shift_tot <= tol", mode="eval").body)]
        if len(assignments) != 1 or len(tests) != 1:
            raise ValueError("kmeans_convergence_expression_contract_changed")
        assignments[0].targets = [ast.Tuple(elts=[ast.Name(id="center_shift_tot", ctx=ast.Store()), ast.Name(id="__wx_within_tolerance", ctx=ast.Store())], ctx=ast.Store())]
        assignments[0].value = ast.parse("__wx_lloyd_convergence(center_shift, tol)", mode="eval").body
        tests[0].test = ast.Name(id="__wx_within_tolerance", ctx=ast.Load())
    if labels_equal is not None:
        if inner.__globals__.get("np") is not np or hashlib.sha256(inspect.getsource(np.array_equal).encode()).hexdigest() != ARRAY_EQUAL_SHA256:
            raise ValueError("kmeans_lloyd_labels_reference_not_supported")
        target = ast.parse("np.array_equal(labels, labels_old)", mode="eval").body
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
        if len(found) != 1:
            raise ValueError("kmeans_lloyd_labels_expression_contract_changed")
        found[0].func = ast.Name(id="__wx_lloyd_labels_equal", ctx=ast.Load())
    namespace = dict(inner.__globals__); namespace["__wx_lloyd_shift_total"] = shift_total
    namespace["__wx_lloyd_convergence"] = convergence
    namespace["__wx_lloyd_labels_equal"] = labels_equal
    if dense_iteration is not None:
        from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
        if namespace.get("lloyd_iter_chunked_dense") is not lloyd_iter_chunked_dense:
            raise ValueError("kmeans_lloyd_norm_iteration_binding_changed")
        namespace["lloyd_iter_chunked_dense"] = dense_iteration
    if inertia is not None:
        from tools.baseline.native_kmeans_inertia import validate_reference
        validate_reference(namespace.get("_inertia_dense"))
        namespace["_inertia_dense"] = inertia
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


def labels_equal(context, records, labels, previous):
    output, record = context.lloyd_labels_equal(labels, previous)
    record.update({"adapter_scope": "sklearn_dense_float32_lloyd_strict_convergence", "original_lloyd_sha256": LLOYD_SHA256, "threadpool_decorator_sha256": THREADPOOL_DECORATOR_SHA256, "original_array_equal_sha256": ARRAY_EQUAL_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["lloyd_cython_iteration_and_centroid_updates", "inertia", "convergence_branch_and_control", "host_staging_and_control"]})
    records.append(record)
    return output


def tolerance_convergence(context, records, shift_records, shifts, tolerance):
    if np.getbufsize() != 8192:
        raise ValueError("kmeans_lloyd_shift_numpy_buffer_not_supported")
    total, converged, record = context.lloyd_convergence(shifts, tolerance)
    record.update({"adapter_scope": "sklearn_dense_float32_lloyd_tolerance_convergence", "original_lloyd_sha256": LLOYD_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["strict_convergence_control", "iteration_control", "host_staging_and_control"]})
    records.append(record)
    shared = dict(record)
    shared.update({"scope": "experimental_native_cpu_kmeans_lloyd_shift_total", "adapter_scope": "sklearn_dense_float32_lloyd_center_shift", "backend": "shared_native_cpu_convergence_reduction", "python_numeric_shift_reduction": False, "lloyd_iteration_replaced": False, "threadpool_decorator_sha256": THREADPOOL_DECORATOR_SHA256, "shared_convergence_index": len(records) - 1, "native_wall_ns": 0, "completion_and_bridge_wall_ns": 0, "host_input_snapshot_bytes": 0, "host_result_bytes": 0})
    shift_records.append(shared)
    return total, converged

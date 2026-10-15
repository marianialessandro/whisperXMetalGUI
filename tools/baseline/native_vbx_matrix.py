"""Scoped native binary64 products inside the source-pinned original VBx iteration loop."""

import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading

import numpy as np


VBX_SHA256 = "1bb9a2b30bd62ad7b4df2c107416d84565df266e80733b0963ea30e38d41d7b3"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False
_OPERATIONS = {"gamma.T.dot(rho)": "speaker_statistics", "rho.dot(alpha.T)": "frame_scores", "(invL + alpha**2).dot(Phi)": "speaker_penalty"}


def compile_native_vbx(original, product):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != VBX_SHA256 or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("vbx_matrix_source_not_supported")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("vbx_matrix_function_contract_changed")
    expected = {ast.dump(ast.parse(expression).body[0].value): operation for expression, operation in _OPERATIONS.items()}
    counts = {operation: 0 for operation in expected.values()}

    class Substitute(ast.NodeTransformer):
        def visit_Call(self, node):
            operation = expected.get(ast.dump(node))
            if operation is None:
                return self.generic_visit(node)
            counts[operation] += 1
            return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_matrix_product", ctx=ast.Load()), args=[node.func.value, node.args[0], ast.Name(id="ii", ctx=ast.Load()), ast.Constant(operation)], keywords=[]), node)

    tree = Substitute().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise ValueError("vbx_matrix_expression_contract_changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_vbx_matrix_product"] = product
    exec(compile(tree, "<WhisperX scoped native VBx matrix products>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("vbx_matrix_signature_changed")
    return candidate


def matrix_product(context, records, left, right, iteration, operation, call_index):
    if type(left) is not np.ndarray or type(right) is not np.ndarray or left.dtype != np.float64 or right.dtype != np.float64 or left.ndim != 2 or right.ndim not in (1, 2):
        raise ValueError("vbx_matrix_operand_format_not_supported")
    vector = right.ndim == 1
    left_stage = np.ascontiguousarray(left)
    right_matrix = right.reshape(-1, 1) if vector else right
    right_stage = np.ascontiguousarray(right_matrix)
    output, record = context.matmul(left_stage, right_stage)
    record = dict(record)
    record.update({"adapter_scope": "vbx_iteration_matrix_products", "operation": operation, "iteration_index": iteration, "vbx_call_index": call_index, "original_vbx_sha256": VBX_SHA256, "vector_result": vector, "host_left_staging_copy_bytes": 0 if left_stage is left else left_stage.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right_matrix else right_stage.nbytes, "cpu_residuals": ["initialization_and_precomputed_terms", "statistics_reductions_and_elementwise_updates", "log_exp_logsumexp", "posterior_and_prior_updates", "elbo_and_stopping"]})
    records.append(record)
    return output[:, 0] if vector else output


@contextlib.contextmanager
def native_vbx_matrix_products(context, records, calls):
    global _ACTIVE
    import pyannote.audio.utils.vbx as vbx

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("vbx_matrix_scope_already_active")
        if importlib.metadata.version("pyannote-audio") != "4.0.4" or np.__version__ != "2.2.6":
            raise ValueError("vbx_matrix_runtime_not_supported")
        if getattr(context.library, "wxdouble_matmul", None) is None:
            raise ValueError("double_matrix_abi_unavailable")
        if not isinstance(records, list) or not isinstance(calls, list):
            raise ValueError("vbx_matrix_records_required")
        original = vbx.VBx
        owner_thread = threading.get_ident()
        call_indices = []

        def product(left, right, iteration, operation):
            return matrix_product(context, records, left, right, iteration, operation, call_indices[-1])

        candidate = compile_native_vbx(original, product)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            call_index, first_record = len(calls), len(records)
            summary = {"scope": "vbx_iteration_matrix_products", "call_index": call_index, "status": "failed", "completed_iterations": None, "native_product_calls": 0, "original_vbx_sha256": VBX_SHA256, "full_vbx_gpu_certified": False, "quality_certified": False}
            calls.append(summary)
            call_indices.append(call_index)
            try:
                result = candidate(*args, **kwargs)
                summary["completed_iterations"] = len(result[2])
                summary["status"] = "completed"
                return result
            finally:
                summary["native_product_calls"] = len(records) - first_record
                call_indices.pop()

        _ACTIVE = True
        vbx.VBx = wrapped
        try:
            yield
        finally:
            vbx.VBx = original
            _ACTIVE = False

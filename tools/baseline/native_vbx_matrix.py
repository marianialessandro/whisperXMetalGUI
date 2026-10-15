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


def compile_native_vbx(original, product, precompute=None, statistics=None, score=None):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != VBX_SHA256 or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("vbx_matrix_source_not_supported")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("vbx_matrix_function_contract_changed")
    expected = {ast.dump(ast.parse(expression).body[0].value): operation for expression, operation in _OPERATIONS.items()}
    counts = {operation: 0 for operation in expected.values()}
    initial_expressions = {ast.dump(ast.parse(expression).body[0]): label for expression, label in [("G = -0.5 * (np.sum(X**2, axis=1, keepdims=True) + D * np.log(2 * np.pi))", "G"), ("V = np.sqrt(Phi)", "V"), ("rho = X * V", "rho")]}
    initial_counts = {label: 0 for label in ("G", "V", "rho")}

    statistic_expressions = {ast.dump(ast.parse(expression).body[0]): label for expression, label in [("invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)", "invL"), ("alpha = Fa / Fb * invL * gamma.T.dot(rho)", "alpha")]}
    statistic_counts = {label: 0 for label in ("invL", "alpha")}

    score_assignment = ast.dump(ast.parse("log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha**2).dot(Phi) + G)").body[0])
    score_counts = {"penalty_terms": 0, "frame_scores": 0}

    class Substitute(ast.NodeTransformer):
        def visit_Assign(self, node):
            if score is not None and ast.dump(node) == score_assignment:
                score_counts["frame_scores"] += 1
                operands = [self.visit(ast.parse(expression).body[0].value) for expression in ("rho.dot(alpha.T)", "(invL + alpha**2).dot(Phi)")]
                node.value = ast.Call(func=ast.Name(id="__wx_vbx_score", ctx=ast.Load()), args=operands + [ast.Name(id=name, ctx=ast.Load()) for name in ("G", "Fa", "ii")] + [ast.Constant("frame_scores")], keywords=[])
                return node
            statistic = statistic_expressions.get(ast.dump(node)) if statistics is not None else None
            if statistic is not None:
                statistic_counts[statistic] += 1
                if statistic == "invL":
                    operands = [ast.Name(id=name, ctx=ast.Load()) for name in ("gamma", "Phi")]
                else:
                    product_node = self.visit(node.value.right)
                    operands = [ast.Name(id="invL", ctx=ast.Load()), product_node]
                node.value = ast.Call(func=ast.Name(id="__wx_vbx_statistics", ctx=ast.Load()), args=operands + [ast.Name(id=name, ctx=ast.Load()) for name in ("Fa", "Fb", "ii")] + [ast.Constant(statistic)], keywords=[])
                return node
            label = initial_expressions.get(ast.dump(node)) if precompute is not None else None
            if label is None:
                return self.generic_visit(node)
            initial_counts[label] += 1
            if label != "G":
                return None
            targets = ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Store()) for name in ("G", "V", "rho")], ctx=ast.Store())
            value = ast.Call(func=ast.Name(id="__wx_vbx_precompute", ctx=ast.Load()), args=[ast.Name(id="X", ctx=ast.Load()), ast.Name(id="Phi", ctx=ast.Load())], keywords=[])
            return ast.copy_location(ast.Assign(targets=[targets], value=value), node)

        def visit_Call(self, node):
            operation = expected.get(ast.dump(node))
            if operation is None:
                return self.generic_visit(node)
            counts[operation] += 1
            left = node.func.value
            if score is not None and operation == "speaker_penalty":
                score_counts["penalty_terms"] += 1
                left = ast.Call(func=ast.Name(id="__wx_vbx_score", ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()) for name in ("invL", "alpha")] + [ast.Constant(None), ast.Constant(1.), ast.Name(id="ii", ctx=ast.Load()), ast.Constant("penalty_terms")], keywords=[])
            return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_matrix_product", ctx=ast.Load()), args=[left, node.args[0], ast.Name(id="ii", ctx=ast.Load()), ast.Constant(operation)], keywords=[]), node)

    tree = Substitute().visit(tree)
    if any(count != 1 for count in counts.values()) or (precompute is not None and any(count != 1 for count in initial_counts.values())):
        raise ValueError("vbx_matrix_expression_contract_changed")
    if statistics is not None and any(count != 1 for count in statistic_counts.values()):
        raise ValueError("vbx_statistics_expression_contract_changed")
    if score is not None and any(count != 1 for count in score_counts.values()):
        raise ValueError("vbx_score_expression_contract_changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_vbx_matrix_product"] = product
    namespace["__wx_vbx_precompute"] = precompute
    namespace["__wx_vbx_statistics"] = statistics
    namespace["__wx_vbx_score"] = score
    exec(compile(tree, "<WhisperX scoped native VBx matrix products>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("vbx_matrix_signature_changed")
    return candidate


def cpu_residuals(precomputed, statistics, scores):
    residuals = ["initialization" if precomputed else "initialization_and_precomputed_terms"]
    if not statistics:
        residuals.append("statistics_reductions_and_elementwise_updates")
    elif not scores:
        residuals.append("score_elementwise_updates")
    return residuals + ["log_exp_logsumexp", "posterior_and_prior_updates", "elbo_and_stopping"]


def matrix_product(context, records, left, right, iteration, operation, call_index, precomputed_terms=False, statistics=False, scores=False):
    if type(left) is not np.ndarray or type(right) is not np.ndarray or left.dtype != np.float64 or right.dtype != np.float64 or left.ndim != 2 or right.ndim not in (1, 2):
        raise ValueError("vbx_matrix_operand_format_not_supported")
    vector = right.ndim == 1
    left_stage = np.ascontiguousarray(left)
    right_matrix = right.reshape(-1, 1) if vector else right
    right_stage = np.ascontiguousarray(right_matrix)
    output, record = context.matmul(left_stage, right_stage)
    record = dict(record)
    record.update({"adapter_scope": "vbx_iteration_matrix_products", "operation": operation, "iteration_index": iteration, "vbx_call_index": call_index, "original_vbx_sha256": VBX_SHA256, "vector_result": vector, "host_left_staging_copy_bytes": 0 if left_stage is left else left_stage.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right_matrix else right_stage.nbytes, "cpu_residuals": cpu_residuals(precomputed_terms, statistics, scores)})
    records.append(record)
    return output[:, 0] if vector else output


def precomputed_terms(context, records, features, phi, call_index, statistics=False, scores=False):
    if type(features) is not np.ndarray or features.dtype != np.float64 or features.ndim != 2 or not (features.flags.c_contiguous or features.flags.f_contiguous) or type(phi) is not np.ndarray or phi.dtype != np.float64 or phi.ndim != 1:
        raise ValueError("vbx_precompute_operand_format_not_supported")
    column_contiguous = features.flags.f_contiguous and not features.flags.c_contiguous
    staged, phi_stage = np.ascontiguousarray(features), np.ascontiguousarray(phi)
    (g, v, rho), record = context.precompute_vbx(staged, phi_stage, "sequential" if column_contiguous else "pairwise")
    if column_contiguous:
        rho = np.asfortranarray(rho)
    record = dict(record)
    record.update({"adapter_scope": "vbx_precomputed_terms", "vbx_call_index": call_index, "original_vbx_sha256": VBX_SHA256, "input_layout": "F" if column_contiguous else "C", "host_input_staging_copy_bytes": 0 if staged is features else staged.nbytes, "host_phi_staging_copy_bytes": 0 if phi_stage is phi else phi_stage.nbytes, "host_rho_layout_copy_bytes": rho.nbytes if column_contiguous else 0, "cpu_residuals": cpu_residuals(True, statistics, scores)})
    records.append(record)
    return g, v, rho


def statistical_update(context, records, left, right, fa, fb, iteration, operation, call_index, precomputed=False, scores=False):
    if type(left) is not np.ndarray or left.dtype != np.float64 or left.ndim != 2 or not (left.flags.c_contiguous or left.flags.f_contiguous) or type(right) is not np.ndarray or right.dtype != np.float64:
        raise ValueError("vbx_statistics_operand_format_not_supported")
    column_contiguous = left.flags.f_contiguous and not left.flags.c_contiguous
    staged = np.ascontiguousarray(left.T if operation == "invL" else left)
    right_stage = np.ascontiguousarray(right)
    if operation == "invL":
        output, record = context.inv_l_vbx(staged, right_stage, fa, fb, "pairwise" if column_contiguous or left.shape[1] == 1 else "sequential")
    else:
        output, record = context.alpha_vbx(staged, right_stage, fa, fb)
    record = dict(record)
    record.update({"adapter_scope": "vbx_statistics_updates", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "input_layout": "F" if column_contiguous else "C", "host_left_staging_copy_bytes": 0 if np.shares_memory(staged, left) else staged.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right else right_stage.nbytes, "cpu_residuals": cpu_residuals(precomputed, True, scores)})
    records.append(record)
    return output


def score_arithmetic(context, records, left, right, g, fa, iteration, operation, call_index, precomputed=False, statistics=False):
    if type(left) is not np.ndarray or left.dtype != np.float64 or left.ndim != 2 or type(right) is not np.ndarray or right.dtype != np.float64:
        raise ValueError("vbx_score_operand_format_not_supported")
    staged, right_stage = np.ascontiguousarray(left), np.ascontiguousarray(right)
    g_copy_bytes = 0
    if operation == "penalty_terms":
        output, record = context.penalty_terms_vbx(staged, right_stage)
    else:
        if type(g) is not np.ndarray or g.dtype != np.float64 or g.shape != (left.shape[0], 1):
            raise ValueError("vbx_score_g_format_not_supported")
        g_stage = np.ascontiguousarray(g[:, 0])
        g_copy_bytes = 0 if np.shares_memory(g_stage, g) else g_stage.nbytes
        output, record = context.frame_scores_vbx(staged, right_stage, g_stage, fa)
    record = dict(record)
    record.update({"adapter_scope": "vbx_score_arithmetic", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "host_left_staging_copy_bytes": 0 if staged is left else staged.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right else right_stage.nbytes, "host_g_staging_copy_bytes": g_copy_bytes, "cpu_residuals": cpu_residuals(precomputed, statistics, True)})
    records.append(record)
    return output


@contextlib.contextmanager
def native_vbx_matrix_products(context, records, calls, precompute=False, precompute_records=None, statistics=False, statistics_records=None, scores=False, score_records=None):
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
        if type(precompute) is not bool or (precompute and not isinstance(precompute_records, list)):
            raise ValueError("vbx_precompute_records_required")
        if precompute and getattr(context.library, "wxdouble_vbx_precompute", None) is None:
            raise ValueError("double_vbx_precompute_abi_unavailable")
        if type(statistics) is not bool or (statistics and not isinstance(statistics_records, list)):
            raise ValueError("vbx_statistics_records_required")
        if statistics and getattr(context.library, "wxdouble_vbx_statistics", None) is None:
            raise ValueError("double_vbx_statistics_abi_unavailable")
        if type(scores) is not bool or (scores and not isinstance(score_records, list)):
            raise ValueError("vbx_score_records_required")
        if scores and getattr(context.library, "wxdouble_vbx_scores", None) is None:
            raise ValueError("double_vbx_scores_abi_unavailable")
        original = vbx.VBx
        owner_thread = threading.get_ident()
        call_indices = []

        def product(left, right, iteration, operation):
            return matrix_product(context, records, left, right, iteration, operation, call_indices[-1], precomputed_terms=precompute, statistics=statistics, scores=scores)

        def prepare(features, phi):
            return precomputed_terms(context, precompute_records, features, phi, call_indices[-1], statistics=statistics, scores=scores)

        def update(left, right, fa, fb, iteration, operation):
            return statistical_update(context, statistics_records, left, right, fa, fb, iteration, operation, call_indices[-1], precomputed=precompute, scores=scores)

        def score(left, right, g, fa, iteration, operation):
            return score_arithmetic(context, score_records, left, right, g, fa, iteration, operation, call_indices[-1], precomputed=precompute, statistics=statistics)

        candidate = compile_native_vbx(original, product, prepare if precompute else None, update if statistics else None, score if scores else None)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            call_index, first_record = len(calls), len(records)
            first_precompute = len(precompute_records) if precompute else 0
            first_statistics = len(statistics_records) if statistics else 0
            first_score = len(score_records) if scores else 0
            summary = {"scope": "vbx_iteration_matrix_products", "call_index": call_index, "status": "failed", "completed_iterations": None, "native_product_calls": 0, "native_precompute_calls": 0, "native_statistics_calls": 0, "native_score_calls": 0, "original_vbx_sha256": VBX_SHA256, "full_vbx_gpu_certified": False, "quality_certified": False}
            calls.append(summary)
            call_indices.append(call_index)
            try:
                result = candidate(*args, **kwargs)
                summary["completed_iterations"] = len(result[2])
                summary["status"] = "completed"
                return result
            finally:
                summary["native_product_calls"] = len(records) - first_record
                summary["native_precompute_calls"] = len(precompute_records) - first_precompute if precompute else 0
                summary["native_statistics_calls"] = len(statistics_records) - first_statistics if statistics else 0
                summary["native_score_calls"] = len(score_records) - first_score if scores else 0
                call_indices.pop()

        _ACTIVE = True
        vbx.VBx = wrapped
        try:
            yield
        finally:
            vbx.VBx = original
            _ACTIVE = False

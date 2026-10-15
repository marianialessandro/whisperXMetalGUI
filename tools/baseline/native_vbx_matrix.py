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
LSE_SOURCE_SHA256 = {"logsumexp": "7cf6d35d2c02de8c33e4c7c70c5413856e6c5d78a43d21f9281d00f1a7f67185", "_logsumexp": "1c7caebc6652e18064278e8e92a3c37ee13c935fc01eca76e5885803aa457cea", "_elements_and_indices_with_max_real": "3514fc6271dac3a23abc3529b661085d9885a9ef3ab8c9fc33578ebb8771a418"}
_HOOK_LOCK = threading.RLock()
_ACTIVE = False
_OPERATIONS = {"gamma.T.dot(rho)": "speaker_statistics", "rho.dot(alpha.T)": "frame_scores", "(invL + alpha**2).dot(Phi)": "speaker_penalty"}


def compile_native_vbx(original, product, precompute=None, statistics=None, score=None, prior=None, posterior=None, elbo=None, transcendental=None, logsumexp_callback=None, scalar=None, initialize=None):
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

    prior_expressions = {ast.dump(ast.parse(expression).body[0]): label for expression, label in [("pi = np.sum(gamma, axis=0)", "mass"), ("pi = pi / pi.sum()", "normalize")]}
    prior_counts = {label: 0 for label in ("mass", "normalize")}

    posterior_expressions = {ast.dump(ast.parse(expression).body[0].value): label for expression, label in [("log_p_ + lpi", "log_prior_addition"), ("log_p_ + lpi - log_p_x[:, None]", "posterior_centering")]}
    posterior_counts = {"log_prior_addition": 0, "posterior_centering": 0}

    elbo_expressions = {ast.dump(ast.parse(expression).body[0]): label for expression, label in [("log_pX_ = np.sum(log_p_x, axis=0)", "likelihood_sum"), ("ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)", "elbo")]}
    elbo_counts = {"likelihood_sum": 0, "elbo": 0}

    transcendental_expressions = {ast.dump(ast.parse(expression).body[0].value): label for expression, label in [("np.log(pi + eps)", "prior_log"), ("np.log(invL)", "inverse_precision_log"), ("np.exp(log_p_ + lpi - log_p_x[:, None])", "posterior_exp")]}
    transcendental_counts = {label: 0 for label in transcendental_expressions.values()}

    logsumexp_expression = ast.dump(ast.parse("logsumexp(log_p_ + lpi, axis=-1)").body[0].value)
    logsumexp_counts = {"call": 0}

    scalar_expressions = {ast.dump(ast.parse(expression).body[0].value): label for expression, label in [("pi + eps", "prior_epsilon_addition"), ("ELBO - Li[-2][0]", "elbo_difference")]}
    scalar_counts = {"prior_epsilon_addition": 0, "elbo_difference": 0}

    initialization_expressions = {ast.dump(ast.parse(expression).body[0]): label for expression, label in [("pi = np.ones(pi) / pi", "uniform_prior"), ("gamma = gamma / gamma.sum(1, keepdims=True)", "gamma_l1_normalization")]}
    initialization_counts = {label: 0 for label in initialization_expressions.values()}

    class Substitute(ast.NodeTransformer):
        def visit_BinOp(self, node):
            scalar_label = scalar_expressions.get(ast.dump(node)) if scalar is not None else None
            if scalar_label is not None:
                scalar_counts[scalar_label] += 1
                return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_scalar", ctx=ast.Load()), args=[node.left, node.right, ast.Name(id="ii", ctx=ast.Load()), ast.Constant(scalar_label)], keywords=[]), node)
            operation = posterior_expressions.get(ast.dump(node)) if posterior is not None else None
            if operation is None:
                return self.generic_visit(node)
            posterior_counts[operation] += 1
            left = self.visit(node.left)
            right = node.right if operation == "log_prior_addition" else ast.Name(id="log_p_x", ctx=ast.Load())
            return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_posterior", ctx=ast.Load()), args=[left, right, ast.Name(id="ii", ctx=ast.Load()), ast.Constant(operation)], keywords=[]), node)

        def visit_Assign(self, node):
            initialization_label = initialization_expressions.get(ast.dump(node)) if initialize is not None else None
            if initialization_label is not None:
                initialization_counts[initialization_label] += 1
                node.value = ast.Call(func=ast.Name(id="__wx_vbx_initialize", ctx=ast.Load()), args=[ast.Name(id="pi" if initialization_label == "uniform_prior" else "gamma", ctx=ast.Load()), ast.Constant(initialization_label)], keywords=[])
                return node
            elbo_label = elbo_expressions.get(ast.dump(node)) if elbo is not None else None
            if elbo_label is not None:
                elbo_counts[elbo_label] += 1
                expressions = ("log_p_x", "None", "None", "0.", "1.") if elbo_label == "likelihood_sum" else ("np.log(invL)", "invL", "alpha", "log_pX_", "Fb")
                node.value = ast.Call(func=ast.Name(id="__wx_vbx_elbo", ctx=ast.Load()), args=[self.visit(ast.parse(expression).body[0].value) for expression in expressions] + [ast.Name(id="ii", ctx=ast.Load()), ast.Constant(elbo_label)], keywords=[])
                return node
            prior_label = prior_expressions.get(ast.dump(node)) if prior is not None else None
            if prior_label is not None:
                prior_counts[prior_label] += 1
                if prior_label == "normalize":
                    return None
                node.value = ast.Call(func=ast.Name(id="__wx_vbx_prior", ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()) for name in ("gamma", "ii")], keywords=[])
                return node
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
            if logsumexp_callback is not None and ast.dump(node) == logsumexp_expression:
                logsumexp_counts["call"] += 1
                return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_logsumexp", ctx=ast.Load()), args=[self.visit(node.args[0]), ast.Name(id="ii", ctx=ast.Load())], keywords=[]), node)
            transcendental_label = transcendental_expressions.get(ast.dump(node)) if transcendental is not None else None
            if transcendental_label is not None:
                transcendental_counts[transcendental_label] += 1
                return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_transcendental", ctx=ast.Load()), args=[self.visit(node.args[0]), ast.Name(id="ii", ctx=ast.Load()), ast.Constant(transcendental_label)], keywords=[]), node)
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
    if prior is not None and any(count != 1 for count in prior_counts.values()):
        raise ValueError("vbx_prior_expression_contract_changed")
    if posterior is not None and posterior_counts != {"log_prior_addition": 2, "posterior_centering": 1}:
        raise ValueError("vbx_posterior_expression_contract_changed")
    if elbo is not None and any(count != 1 for count in elbo_counts.values()):
        raise ValueError("vbx_elbo_expression_contract_changed")
    if transcendental is not None and any(count != 1 for count in transcendental_counts.values()):
        raise ValueError("vbx_transcendental_expression_contract_changed")
    if logsumexp_callback is not None and logsumexp_counts["call"] != 1:
        raise ValueError("vbx_logsumexp_expression_contract_changed")
    if scalar is not None and scalar_counts != {"prior_epsilon_addition": 1, "elbo_difference": 2}:
        raise ValueError("vbx_scalar_expression_contract_changed")
    if initialize is not None and any(count != 1 for count in initialization_counts.values()):
        raise ValueError("vbx_initialization_expression_contract_changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_vbx_matrix_product"] = product
    namespace["__wx_vbx_precompute"] = precompute
    namespace["__wx_vbx_statistics"] = statistics
    namespace["__wx_vbx_score"] = score
    namespace["__wx_vbx_prior"] = prior
    namespace["__wx_vbx_posterior"] = posterior
    namespace["__wx_vbx_elbo"] = elbo
    namespace["__wx_vbx_transcendental"] = transcendental
    namespace["__wx_vbx_logsumexp"] = logsumexp_callback
    namespace["__wx_vbx_scalar"] = scalar
    namespace["__wx_vbx_initialize"] = initialize
    exec(compile(tree, "<WhisperX scoped native VBx matrix products>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("vbx_matrix_signature_changed")
    return candidate


def cpu_residuals(precomputed, statistics, scores, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
    residuals = ["initialization_rng_if_needed"] if initialization else ["initialization"]
    if not precomputed:
        residuals = residuals + ["precomputed_terms"] if initialization else ["initialization_and_precomputed_terms"]
    if not statistics:
        residuals.append("statistics_reductions_and_elementwise_updates")
    elif not scores:
        residuals.append("score_elementwise_updates")
    if not transcendentals or not logsumexp:
        residuals.append("numpy_log_exp" if logsumexp else "scipy_logsumexp" if transcendentals else "log_exp_logsumexp")
    if posterior:
        if not scalars:
            residuals.append("prior_log_epsilon_addition")
        if not prior:
            residuals.append("prior_updates")
    else:
        residuals.append("posterior_updates" if prior else "posterior_and_prior_updates")
    stopping = "stopping_control" if elbo else "elbo_and_stopping_control"
    return residuals + [stopping if scalars else "stopping" if elbo else "elbo_and_stopping"]


def matrix_product(context, records, left, right, iteration, operation, call_index, precomputed_terms=False, statistics=False, scores=False, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
    if type(left) is not np.ndarray or type(right) is not np.ndarray or left.dtype != np.float64 or right.dtype != np.float64 or left.ndim != 2 or right.ndim not in (1, 2):
        raise ValueError("vbx_matrix_operand_format_not_supported")
    vector = right.ndim == 1
    left_stage = np.ascontiguousarray(left)
    right_matrix = right.reshape(-1, 1) if vector else right
    right_stage = np.ascontiguousarray(right_matrix)
    output, record = context.matmul(left_stage, right_stage)
    record = dict(record)
    record.update({"adapter_scope": "vbx_iteration_matrix_products", "operation": operation, "iteration_index": iteration, "vbx_call_index": call_index, "original_vbx_sha256": VBX_SHA256, "vector_result": vector, "host_left_staging_copy_bytes": 0 if left_stage is left else left_stage.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right_matrix else right_stage.nbytes, "cpu_residuals": cpu_residuals(precomputed_terms, statistics, scores, prior, posterior, elbo, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return output[:, 0] if vector else output


def precomputed_terms(context, records, features, phi, call_index, statistics=False, scores=False, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
    if type(features) is not np.ndarray or features.dtype != np.float64 or features.ndim != 2 or not (features.flags.c_contiguous or features.flags.f_contiguous) or type(phi) is not np.ndarray or phi.dtype != np.float64 or phi.ndim != 1:
        raise ValueError("vbx_precompute_operand_format_not_supported")
    column_contiguous = features.flags.f_contiguous and not features.flags.c_contiguous
    staged, phi_stage = np.ascontiguousarray(features), np.ascontiguousarray(phi)
    (g, v, rho), record = context.precompute_vbx(staged, phi_stage, "sequential" if column_contiguous else "pairwise")
    if column_contiguous:
        rho = np.asfortranarray(rho)
    record = dict(record)
    record.update({"adapter_scope": "vbx_precomputed_terms", "vbx_call_index": call_index, "original_vbx_sha256": VBX_SHA256, "input_layout": "F" if column_contiguous else "C", "host_input_staging_copy_bytes": 0 if staged is features else staged.nbytes, "host_phi_staging_copy_bytes": 0 if phi_stage is phi else phi_stage.nbytes, "host_rho_layout_copy_bytes": rho.nbytes if column_contiguous else 0, "cpu_residuals": cpu_residuals(True, statistics, scores, prior, posterior, elbo, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return g, v, rho


def statistical_update(context, records, left, right, fa, fb, iteration, operation, call_index, precomputed=False, scores=False, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
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
    record.update({"adapter_scope": "vbx_statistics_updates", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "input_layout": "F" if column_contiguous else "C", "host_left_staging_copy_bytes": 0 if np.shares_memory(staged, left) else staged.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right else right_stage.nbytes, "cpu_residuals": cpu_residuals(precomputed, True, scores, prior, posterior, elbo, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return output


def score_arithmetic(context, records, left, right, g, fa, iteration, operation, call_index, precomputed=False, statistics=False, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
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
    record.update({"adapter_scope": "vbx_score_arithmetic", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "host_left_staging_copy_bytes": 0 if staged is left else staged.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right else right_stage.nbytes, "host_g_staging_copy_bytes": g_copy_bytes, "cpu_residuals": cpu_residuals(precomputed, statistics, True, prior, posterior, elbo, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return output


def prior_update(context, records, gamma, iteration, call_index, precomputed=False, statistics=False, scores=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
    if type(gamma) is not np.ndarray or gamma.dtype != np.float64 or gamma.ndim != 2 or not (gamma.flags.c_contiguous or gamma.flags.f_contiguous):
        raise ValueError("vbx_prior_operand_format_not_supported")
    if not 0 < gamma.shape[0] <= 4096 or not 0 < gamma.shape[1] <= 1024:
        raise ValueError("vbx_prior_shape_not_supported")
    column_contiguous = gamma.flags.f_contiguous and not gamma.flags.c_contiguous
    staged = np.ascontiguousarray(gamma.T)
    output, record = context.prior_vbx(staged, "pairwise" if column_contiguous or gamma.shape[1] == 1 else "sequential")
    record = dict(record)
    record.update({"adapter_scope": "vbx_prior_updates", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "input_layout": "F" if column_contiguous else "C", "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, gamma) else staged.nbytes, "cpu_residuals": cpu_residuals(precomputed, statistics, scores, True, posterior, elbo, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return output


def posterior_arithmetic(context, records, values, vector, iteration, operation, call_index, precomputed=False, statistics=False, scores=False, prior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
    if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or type(vector) is not np.ndarray or vector.dtype != np.float64 or vector.ndim != 1:
        raise ValueError("vbx_posterior_operand_format_not_supported")
    if not 0 < values.shape[0] <= 4096 or not 0 < values.shape[1] <= 1024:
        raise ValueError("vbx_posterior_shape_not_supported")
    staged, vector_stage = np.ascontiguousarray(values), np.ascontiguousarray(vector)
    output, record = context.posterior_vbx(staged, vector_stage, operation)
    record = dict(record)
    record.update({"adapter_scope": "vbx_posterior_arithmetic", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "host_input_staging_copy_bytes": 0 if staged is values else staged.nbytes, "host_vector_staging_copy_bytes": 0 if vector_stage is vector else vector_stage.nbytes, "cpu_residuals": cpu_residuals(precomputed, statistics, scores, prior, True, elbo, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return output


def elbo_arithmetic(context, records, values, inv_l, alpha, likelihood, fb, iteration, operation, call_index, precomputed=False, statistics=False, scores=False, prior=False, posterior=False, transcendentals=False, logsumexp=False, scalars=False, initialization=False):
    operands = (values,) if operation == "likelihood_sum" else (values, inv_l, alpha)
    if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != (1 if operation == "likelihood_sum" else 2) for value in operands):
        raise ValueError("vbx_elbo_operand_format_not_supported")
    if operation == "likelihood_sum":
        if not 0 < values.size <= 4096:
            raise ValueError("vbx_elbo_shape_not_supported")
        staged = (np.ascontiguousarray(values),)
        output, record = context.likelihood_sum_vbx(staged[0])
    else:
        if any(value.shape != inv_l.shape for value in operands) or not 0 < inv_l.shape[0] <= 1024 or not 0 < inv_l.shape[1] <= 1024:
            raise ValueError("vbx_elbo_shape_not_supported")
        column_order = all(value.flags.f_contiguous for value in operands)
        staged = tuple(np.ascontiguousarray(value) for value in operands)
        output, record = context.elbo_vbx(*staged, likelihood, fb, column_order)
    record = dict(record)
    record.update({"adapter_scope": "vbx_elbo_arithmetic", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "host_input_staging_copy_bytes": sum(value.nbytes for value, original in zip(staged, operands) if value is not original), "cpu_residuals": cpu_residuals(precomputed, statistics, scores, prior, posterior, True, transcendentals, logsumexp, scalars, initialization)})
    records.append(record)
    return output


def transcendental_update(context, records, values, iteration, operation, call_index, precomputed=False, statistics=False, scores=False, prior=False, posterior=False, elbo=False, logsumexp=False, scalars=False, initialization=False):
    if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim not in (1, 2) or not (values.flags.c_contiguous or values.flags.f_contiguous):
        raise ValueError("vbx_transcendental_operand_format_not_supported")
    if not 0 < values.size <= 65536:
        raise ValueError("vbx_transcendental_work_budget_invalid")
    column_order = values.ndim == 2 and values.flags.f_contiguous and not values.flags.c_contiguous
    staged = np.ascontiguousarray(values)
    output, record = context.transcendental_vbx(staged, "exp" if operation == "posterior_exp" else "log")
    if column_order:
        output = np.asfortranarray(output)
    record = dict(record)
    record.update({"adapter_scope": "vbx_transcendentals", "vbx_operation": operation, "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "input_layout": "F" if column_order else "C", "host_input_staging_copy_bytes": 0 if staged is values else staged.nbytes, "host_output_layout_copy_bytes": output.nbytes if column_order else 0, "cpu_residuals": cpu_residuals(precomputed, statistics, scores, prior, posterior, elbo, True, logsumexp, scalars, initialization)})
    records.append(record)
    return output


def verify_logsumexp_reference(vbx):
    import scipy.special._logsumexp as reference
    if importlib.metadata.version("scipy") != "1.15.3" or vbx.logsumexp is not reference.logsumexp or np.getbufsize() != 8192:
        raise ValueError("vbx_logsumexp_runtime_not_supported")
    for name, expected in LSE_SOURCE_SHA256.items():
        if hashlib.sha256(textwrap.dedent(inspect.getsource(getattr(reference, name))).encode()).hexdigest() != expected:
            raise ValueError("vbx_logsumexp_source_not_supported")


def logsumexp_update(context, records, values, iteration, call_index, precomputed=False, statistics=False, scores=False, prior=False, posterior=False, elbo=False, transcendentals=False, scalars=False, initialization=False):
    if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not (values.flags.c_contiguous or values.flags.f_contiguous):
        raise ValueError("vbx_logsumexp_operand_format_not_supported")
    if not 0 < values.shape[0] <= 4096 or not 0 < values.shape[1] <= 1024 or values.size > 65536:
        raise ValueError("vbx_logsumexp_work_budget_invalid")
    column_order = values.flags.f_contiguous and not values.flags.c_contiguous
    staged = np.ascontiguousarray(values)
    output, record = context.logsumexp_vbx(staged, "sequential" if column_order else "pairwise")
    record = dict(record)
    record.update({"adapter_scope": "vbx_logsumexp", "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "scipy_reference_version": "1.15.3", "scipy_reference_source_sha256": dict(LSE_SOURCE_SHA256), "input_layout": "F" if column_order else "C", "host_input_staging_copy_bytes": 0 if staged is values else staged.nbytes, "cpu_residuals": cpu_residuals(precomputed, statistics, scores, prior, posterior, elbo, transcendentals, True, scalars, initialization)})
    records.append(record)
    return output


def scalar_arithmetic(context, records, left, right, iteration, operation, call_index, precomputed=False, statistics=False, scores=False, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, initialization=False):
    if type(right) not in (float, int, np.float64):
        raise ValueError("vbx_scalar_operand_format_not_supported")
    if operation == "prior_epsilon_addition":
        if type(left) is not np.ndarray or left.dtype != np.float64 or left.ndim != 1 or not 0 < left.size <= 1024:
            raise ValueError("vbx_scalar_prior_format_not_supported")
        matrix = np.ascontiguousarray(left.reshape(1, -1))
        vector = np.full(left.size, right, dtype=np.float64)
        input_copy_bytes = 0 if np.shares_memory(matrix, left) else matrix.nbytes
        native_operation = "log_prior_addition"
    elif operation == "elbo_difference":
        if type(left) not in (float, int, np.float64):
            raise ValueError("vbx_scalar_operand_format_not_supported")
        matrix, vector = np.array([[left]], dtype=np.float64), np.array([right], dtype=np.float64)
        input_copy_bytes = matrix.nbytes
        native_operation = "posterior_centering"
    else:
        raise ValueError("vbx_scalar_operation_not_supported")
    output, record = context.posterior_vbx(matrix, vector, native_operation)
    record = dict(record)
    record.update({"adapter_scope": "vbx_scalar_arithmetic", "operation": operation, "native_operation": native_operation, "vbx_call_index": call_index, "iteration_index": iteration, "original_vbx_sha256": VBX_SHA256, "host_input_staging_copy_bytes": input_copy_bytes, "host_scalar_staging_copy_bytes": vector.nbytes, "cpu_residuals": cpu_residuals(precomputed, statistics, scores, prior, posterior, elbo, transcendentals, logsumexp, True, initialization)})
    records.append(record)
    return output[0] if operation == "prior_epsilon_addition" else output[0, 0]


def initialization_update(context, records, values, operation, call_index, precomputed=False, statistics=False, scores=False, prior=False, posterior=False, elbo=False, transcendentals=False, logsumexp=False, scalars=False):
    column_contiguous = False
    staging_bytes = 0
    if operation == "uniform_prior":
        output, record = context.uniform_prior_vbx(values)
    elif operation == "gamma_l1_normalization":
        if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not (values.flags.c_contiguous or values.flags.f_contiguous):
            raise ValueError("vbx_initialization_operand_format_not_supported")
        if not 0 < values.shape[0] <= 4096 or not 0 < values.shape[1] <= 1024:
            raise ValueError("vbx_initialization_shape_not_supported")
        column_contiguous = values.flags.f_contiguous and not values.flags.c_contiguous
        staged = np.ascontiguousarray(values)
        staging_bytes = 0 if staged is values else staged.nbytes
        output, record = context.normalize_initial_gamma_vbx(staged, "sequential" if column_contiguous else "pairwise")
        if column_contiguous:
            output = np.asfortranarray(output)
    else:
        raise ValueError("vbx_initialization_operation_not_supported")
    record = dict(record)
    record.update({"adapter_scope": "vbx_initialization", "vbx_call_index": call_index, "iteration_index": None, "original_vbx_sha256": VBX_SHA256, "host_input_staging_copy_bytes": staging_bytes, "host_output_layout_copy_bytes": output.nbytes if column_contiguous else 0, "cpu_residuals": cpu_residuals(precomputed, statistics, scores, prior, posterior, elbo, transcendentals, logsumexp, scalars, True)})
    records.append(record)
    return output


@contextlib.contextmanager
def native_vbx_matrix_products(context, records, calls, precompute=False, precompute_records=None, statistics=False, statistics_records=None, scores=False, score_records=None, prior=False, prior_records=None, posterior=False, posterior_records=None, elbo=False, elbo_records=None, transcendentals=False, transcendental_records=None, logsumexp=False, logsumexp_records=None, scalars=False, scalar_records=None, initialization=False, initialization_records=None):
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
        if type(prior) is not bool or (prior and not isinstance(prior_records, list)):
            raise ValueError("vbx_prior_records_required")
        if prior and getattr(context.library, "wxdouble_vbx_prior", None) is None:
            raise ValueError("double_vbx_prior_abi_unavailable")
        if type(posterior) is not bool or (posterior and not isinstance(posterior_records, list)):
            raise ValueError("vbx_posterior_records_required")
        if posterior and getattr(context.library, "wxdouble_vbx_posterior", None) is None:
            raise ValueError("double_vbx_posterior_abi_unavailable")
        if type(elbo) is not bool or (elbo and not isinstance(elbo_records, list)):
            raise ValueError("vbx_elbo_records_required")
        if elbo and np.getbufsize() != 8192:
            raise ValueError("vbx_elbo_reduction_runtime_not_supported")
        if elbo and getattr(context.library, "wxdouble_vbx_elbo", None) is None:
            raise ValueError("double_vbx_elbo_abi_unavailable")
        if type(transcendentals) is not bool or (transcendentals and not isinstance(transcendental_records, list)):
            raise ValueError("vbx_transcendental_records_required")
        if transcendentals and getattr(context.library, "wxdouble_vbx_transcendental", None) is None:
            raise ValueError("double_vbx_transcendental_abi_unavailable")
        if type(logsumexp) is not bool or (logsumexp and not isinstance(logsumexp_records, list)):
            raise ValueError("vbx_logsumexp_records_required")
        if logsumexp:
            verify_logsumexp_reference(vbx)
            if getattr(context.library, "wxdouble_vbx_logsumexp", None) is None:
                raise ValueError("double_vbx_logsumexp_abi_unavailable")
        if type(scalars) is not bool or (scalars and not isinstance(scalar_records, list)):
            raise ValueError("vbx_scalar_records_required")
        if scalars and getattr(context.library, "wxdouble_vbx_posterior", None) is None:
            raise ValueError("double_vbx_scalar_abi_unavailable")
        if type(initialization) is not bool or (initialization and not isinstance(initialization_records, list)):
            raise ValueError("vbx_initialization_records_required")
        if initialization and getattr(context.library, "wxdouble_vbx_initialize", None) is None:
            raise ValueError("double_vbx_initialization_abi_unavailable")
        original = vbx.VBx
        owner_thread = threading.get_ident()
        call_indices = []

        def product(left, right, iteration, operation):
            return matrix_product(context, records, left, right, iteration, operation, call_indices[-1], precomputed_terms=precompute, statistics=statistics, scores=scores, prior=prior, posterior=posterior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def prepare(features, phi):
            return precomputed_terms(context, precompute_records, features, phi, call_indices[-1], statistics=statistics, scores=scores, prior=prior, posterior=posterior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def update(left, right, fa, fb, iteration, operation):
            return statistical_update(context, statistics_records, left, right, fa, fb, iteration, operation, call_indices[-1], precomputed=precompute, scores=scores, prior=prior, posterior=posterior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def score(left, right, g, fa, iteration, operation):
            return score_arithmetic(context, score_records, left, right, g, fa, iteration, operation, call_indices[-1], precomputed=precompute, statistics=statistics, prior=prior, posterior=posterior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def update_prior(gamma, iteration):
            return prior_update(context, prior_records, gamma, iteration, call_indices[-1], precomputed=precompute, statistics=statistics, scores=scores, posterior=posterior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def update_posterior(values, vector, iteration, operation):
            return posterior_arithmetic(context, posterior_records, values, vector, iteration, operation, call_indices[-1], precomputed=precompute, statistics=statistics, scores=scores, prior=prior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def update_elbo(values, inv_l, alpha, likelihood, fb, iteration, operation):
            return elbo_arithmetic(context, elbo_records, values, inv_l, alpha, likelihood, fb, iteration, operation, call_indices[-1], precomputed=precompute, statistics=statistics, scores=scores, prior=prior, posterior=posterior, transcendentals=transcendentals, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def update_transcendental(values, iteration, operation):
            return transcendental_update(context, transcendental_records, values, iteration, operation, call_indices[-1], precomputed=precompute, statistics=statistics, scores=scores, prior=prior, posterior=posterior, elbo=elbo, logsumexp=logsumexp, scalars=scalars, initialization=initialization)

        def update_logsumexp(values, iteration):
            return logsumexp_update(context, logsumexp_records, values, iteration, call_indices[-1], precomputed=precompute, statistics=statistics, scores=scores, prior=prior, posterior=posterior, elbo=elbo, transcendentals=transcendentals, scalars=scalars, initialization=initialization)

        def update_scalar(left, right, iteration, operation):
            return scalar_arithmetic(context, scalar_records, left, right, iteration, operation, call_indices[-1], precomputed=precompute, statistics=statistics, scores=scores, prior=prior, posterior=posterior, elbo=elbo, transcendentals=transcendentals, logsumexp=logsumexp, initialization=initialization)

        def initialize(values, operation):
            return initialization_update(context, initialization_records, values, operation, call_indices[-1], precompute, statistics, scores, prior, posterior, elbo, transcendentals, logsumexp, scalars)

        candidate = compile_native_vbx(original, product, prepare if precompute else None, update if statistics else None, score if scores else None, update_prior if prior else None, update_posterior if posterior else None, update_elbo if elbo else None, update_transcendental if transcendentals else None, update_logsumexp if logsumexp else None, update_scalar if scalars else None, initialize if initialization else None)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            call_index, first_record = len(calls), len(records)
            first_precompute = len(precompute_records) if precompute else 0
            first_statistics = len(statistics_records) if statistics else 0
            first_score = len(score_records) if scores else 0
            first_prior = len(prior_records) if prior else 0
            first_posterior = len(posterior_records) if posterior else 0
            first_elbo = len(elbo_records) if elbo else 0
            first_transcendental = len(transcendental_records) if transcendentals else 0
            first_logsumexp = len(logsumexp_records) if logsumexp else 0
            first_scalar = len(scalar_records) if scalars else 0
            first_initialization = len(initialization_records) if initialization else 0
            summary = {"initialization_requested": initialization, "native_initialization_calls": 0, "rng_initialization_requested": kwargs.get("gamma", args[5] if len(args) > 5 else None) is None, "rng_initialization_observed": False, "scope": "vbx_iteration_matrix_products", "call_index": call_index, "status": "failed", "completed_iterations": None, "native_product_calls": 0, "native_precompute_calls": 0, "native_statistics_calls": 0, "native_score_calls": 0, "native_prior_calls": 0, "native_posterior_calls": 0, "native_elbo_calls": 0, "native_transcendental_calls": 0, "native_logsumexp_calls": 0, "native_scalar_calls": 0, "original_vbx_sha256": VBX_SHA256, "full_vbx_gpu_certified": False, "quality_certified": False}
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
                summary["native_prior_calls"] = len(prior_records) - first_prior if prior else 0
                summary["native_posterior_calls"] = len(posterior_records) - first_posterior if posterior else 0
                summary["native_elbo_calls"] = len(elbo_records) - first_elbo if elbo else 0
                summary["native_transcendental_calls"] = len(transcendental_records) - first_transcendental if transcendentals else 0
                summary["native_logsumexp_calls"] = len(logsumexp_records) - first_logsumexp if logsumexp else 0
                summary["native_scalar_calls"] = len(scalar_records) - first_scalar if scalars else 0
                summary["native_initialization_calls"] = len(initialization_records) - first_initialization if initialization else 0
                summary["rng_initialization_observed"] = any(record["operation"] == "gamma_l1_normalization" for record in initialization_records[first_initialization:]) if initialization else False
                call_indices.pop()

        _ACTIVE = True
        vbx.VBx = wrapped
        try:
            yield
        finally:
            vbx.VBx = original
            _ACTIVE = False

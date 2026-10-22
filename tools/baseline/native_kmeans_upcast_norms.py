"""Native binary64 row norms inside source-pinned KMeans++ distance chunks."""

import ast
import hashlib
import importlib.metadata
import inspect
import textwrap

import numpy as np

from tools.baseline.native_kmeans_norms import EINSUM_SHA256, NUMPY_CORE_SHA256, ROW_NORMS_SHA256, validate_reference


EUCLIDEAN_SHA256 = "c4cd3d6f23ad2a343f097128f4f4a0ae374f2150e079d9ace856b278ad980065"
UPCAST_SHA256 = "e29fa6f364b6154c6bca9345bd9df20f5637896284be491c41c0a3451d99d3fb"
EINSUM_REFERENCE_SHA256 = "d1c8e05b55eb31561169558bda7ddfd05e7ab4d7fac27a4123f33a3a494a220b"
NEON_REFERENCE_SHA256 = "1921a3d2b8848bd7f9ad4276301f0acec3def9a62a913c61ce60706ea0b5a0bd"


def squared_upcast_norms(context, records, values, squared=False):
    if squared is not True:
        raise ValueError("kmeans_upcast_norms_squared_policy_not_supported")
    result, record = context.kmeans_upcast_norms(values)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_float64_distance_chunk_norms", "original_row_norms_sha256": ROW_NORMS_SHA256, "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "numpy_einsum_sha256": EINSUM_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "einsum_reference_sha256": EINSUM_REFERENCE_SHA256, "neon_reference_sha256": NEON_REFERENCE_SHA256, "host_input_staging_copy_bytes": 0, "cpu_residuals": ["float32_to_float64_chunk_conversion", "distance_blas_products", "distance_scale_norm_additions_clamp_and_dtype_conversion", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


def compile_upcast_distances(original, norms, finish=None, clamp=None, resident=False, resident_norms=False, product=None, upcast_chunk=None):
    import sklearn.metrics.pairwise as pairwise

    if type(resident) is not bool or (resident and (finish is None or clamp is None)):
        raise ValueError("kmeans_resident_distances_configuration_invalid")
    if type(resident_norms) is not bool or (resident_norms and not resident):
        raise ValueError("kmeans_resident_norms_configuration_invalid")
    upcast = pairwise._euclidean_distances_upcast
    if original is not pairwise._euclidean_distances or original.__closure__ or upcast.__closure__ or inspect.unwrap(original) is not original or inspect.unwrap(upcast) is not upcast or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != EUCLIDEAN_SHA256 or hashlib.sha256(inspect.getsource(upcast).encode()).hexdigest() != UPCAST_SHA256 or original.__globals__.get("_euclidean_distances_upcast") is not upcast or importlib.metadata.version("scikit-learn") != "1.9.1":
        raise ValueError("kmeans_upcast_norms_reference_not_supported")
    validate_reference(upcast.__globals__.get("row_norms"))
    tree = ast.parse(textwrap.dedent(inspect.getsource(upcast)))
    expected = ("row_norms(X_chunk, squared=True)", "row_norms(Y_chunk, squared=True)")
    for expression in expected:
        target = ast.parse(expression, mode="eval").body
        if len([node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]) != 1:
            raise ValueError("kmeans_upcast_norms_expression_contract_changed")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("kmeans_upcast_norms_function_contract_changed")
    if finish is not None:
        expressions = ("d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)", "d += XX_chunk", "d += YY_chunk")
        matches = []
        for expression in expressions:
            target = ast.parse(expression).body[0]
            found = [node for node in ast.walk(tree) if ast.dump(node) == ast.dump(target)]
            if len(found) != 1:
                raise ValueError("kmeans_distance_finish_expression_contract_changed")
            matches.append(found[0])
        finish_statement = matches[0]
        matches[0].value = ast.parse("__wx_distance_finish(safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True), XX_chunk, YY_chunk" + (", X is Y)" if resident else ")"), mode="eval").body
        class RemoveNormAdditions(ast.NodeTransformer):
            def visit_AugAssign(self, node):
                return None if node in matches[1:] else self.generic_visit(node)
        tree = RemoveNormAdditions().visit(tree)
        ast.fix_missing_locations(tree)
    if resident:
        expressions = ("distances = xp.empty((n_samples_X, n_samples_Y), dtype=xp.float32, device=device_)", "distances[x_slice, y_slice] = xp.astype(d, xp.float32, copy=False)")
        matches = []
        for expression in expressions:
            target = ast.parse(expression).body[0]
            found = [node for node in ast.walk(tree) if ast.dump(node) == ast.dump(target)]
            if len(found) != 1:
                raise ValueError("kmeans_resident_distances_expression_contract_changed")
            matches.append(found[0])
        matches[1].targets = [ast.Name(id="distances", ctx=ast.Store())]
        matches[1].value = ast.Name(id="d", ctx=ast.Load())
        class RemoveResidentAllocation(ast.NodeTransformer):
            def visit_Assign(self, node):
                return None if node is matches[0] else self.generic_visit(node)
        tree = RemoveResidentAllocation().visit(tree)
        tree.body[0].body.insert(1, ast.parse("__wx_squared_distance_guard(X, Y, True)").body[0])
        tree.body[0].body.insert(2, ast.parse("__wx_resident_batch_guard(X, Y, batch_size)").body[0])
        ast.fix_missing_locations(tree)
    if resident_norms:
        expressions = ("if XX is None:\n    XX_chunk = row_norms(X_chunk, squared=True)[:, None]\nelse:\n    XX_chunk = XX[x_slice]", "if YY is None:\n    YY_chunk = row_norms(Y_chunk, squared=True)[None, :]\nelse:\n    YY_chunk = YY[:, y_slice]")
        matches = []
        for expression in expressions:
            target = ast.parse(expression).body[0]
            found = [node for node in ast.walk(tree) if ast.dump(node) == ast.dump(target)]
            if len(found) != 1:
                raise ValueError("kmeans_resident_norms_expression_contract_changed")
            matches.append(found[0])
        class RemoveResidentNormConditions(ast.NodeTransformer):
            def visit_If(self, node):
                return None if node in matches else self.generic_visit(node)
        tree = RemoveResidentNormConditions().visit(tree)
        finish_statement.value = ast.parse("__wx_distance_finish(safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True), X_chunk, Y_chunk, X is Y)", mode="eval").body
        tree.body[0].body.insert(3, ast.parse("__wx_no_chunk_norms(XX, YY)").body[0])
        ast.fix_missing_locations(tree)
    if product is not None:
        from tools.baseline.native_kmeans_potential import SAFE_SPARSE_DOT_SHA256
        reference = upcast.__globals__.get("safe_sparse_dot")
        if not inspect.isfunction(reference) or reference.__closure__ or hashlib.sha256(inspect.getsource(reference).encode()).hexdigest() != SAFE_SPARSE_DOT_SHA256:
            raise ValueError("kmeans_distance_product_reference_not_supported")
        target = ast.parse("safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)", mode="eval").body
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
        if len(found) != 1:
            raise ValueError("kmeans_distance_product_expression_contract_changed")
        found[0].func = ast.Name(id="__wx_distance_product", ctx=ast.Load())
        ast.fix_missing_locations(tree)
    if upcast_chunk is not None:
        for role, expression in (("X_chunk", "xp.astype(X[x_slice, :], xp_max_float)"), ("Y_chunk", "xp.astype(Y[y_slice, :], xp_max_float)")):
            target = ast.parse(expression, mode="eval").body
            found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
            if len(found) != 1:
                raise ValueError("kmeans_upcast_chunk_expression_contract_changed")
            found[0].func = ast.Name(id="__wx_upcast_chunk", ctx=ast.Load())
            found[0].args.append(ast.Constant(value=role))
        ast.fix_missing_locations(tree)
    namespace = dict(upcast.__globals__)
    namespace["__wx_upcast_chunk"] = upcast_chunk
    namespace["__wx_distance_product"] = product
    namespace["__wx_squared_distance_guard"] = validate_resident_distance_inputs if resident else validate_squared_distance_inputs
    namespace["__wx_no_chunk_norms"] = validate_no_chunk_norms
    namespace["__wx_resident_batch_guard"] = validate_resident_batch
    namespace["__wx_distance_finish"] = finish
    namespace["row_norms"] = norms
    exec(compile(tree, "<WhisperX scoped native KMeans upcast norms>", "exec"), namespace)
    candidate_upcast = namespace[upcast.__name__]
    outer = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if clamp is not None:
        if finish is None:
            raise ValueError("kmeans_distance_clamp_configuration_invalid")
        expressions = ("xp_zero = xp.asarray(0, device=device_, dtype=distances.dtype)", "distances = _modify_in_place_if_numpy(xp, xp.maximum, distances, xp_zero, out=distances)", "if X is Y:\n    _fill_diagonal(distances, 0, xp=xp)")
        matches = []
        for expression in expressions:
            target = ast.parse(expression).body[0]
            found = [node for node in ast.walk(outer) if ast.dump(node) == ast.dump(target)]
            if len(found) != 1:
                raise ValueError("kmeans_distance_clamp_expression_contract_changed")
            matches.append(found[0])
        matches[1].value = ast.parse("__wx_distance_clamp(distances, X is Y)", mode="eval").body
        class RemoveClampTemporaries(ast.NodeTransformer):
            def visit_Assign(self, node):
                return None if node is matches[0] else self.generic_visit(node)
            def visit_If(self, node):
                return None if node is matches[2] else self.generic_visit(node)
        outer = RemoveClampTemporaries().visit(outer)
        guard = ast.parse("__wx_squared_distance_guard(X, Y, squared)").body[0]
        outer.body[0].body.insert(1, guard)
        ast.fix_missing_locations(outer)
    if resident_norms:
        outer.body[0].body.insert(2, ast.parse("__wx_no_supplied_norms(X_norm_squared, Y_norm_squared)").body[0])
        ast.fix_missing_locations(outer)
    namespace = dict(original.__globals__)
    namespace["_euclidean_distances_upcast"] = candidate_upcast
    namespace["__wx_no_supplied_norms"] = validate_no_supplied_norms
    namespace["__wx_distance_clamp"] = clamp
    namespace["__wx_squared_distance_guard"] = validate_resident_distance_inputs if resident else validate_squared_distance_inputs
    exec(compile(outer, "<WhisperX scoped KMeans Euclidean distance>", "exec"), namespace)
    candidate = namespace[original.__name__]
    if inspect.signature(candidate_upcast) != inspect.signature(upcast) or inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_upcast_norms_signature_changed")
    return candidate


def finish_distances(context, records, products, left_norms, right_norms):
    if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 for value in (products, left_norms, right_norms)) or left_norms.shape != (products.shape[0], 1) or right_norms.shape != (1, products.shape[1]):
        raise ValueError("kmeans_distance_finish_operand_not_supported")
    staged = tuple(np.ascontiguousarray(value) for value in (products, left_norms, right_norms))
    result, record = context.kmeans_distance_finish(staged[0], staged[1].reshape(-1), staged[2].reshape(-1))
    record = dict(record)
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_distance_chunk_finish", "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_input_staging_copy_bytes": sum(value.nbytes for value, original in zip(staged, (products, left_norms, right_norms)) if not np.shares_memory(value, original)), "cpu_residuals": ["float32_to_float64_chunk_conversion", "distance_blas_products", "distance_clamping_and_optional_sqrt_diagonal_fill", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


def validate_squared_distance_inputs(X, Y, squared):
    if squared is not True or any(type(value) is not np.ndarray or value.dtype != np.float32 or value.ndim != 2 for value in (X, Y)):
        raise ValueError("kmeans_distance_clamp_squared_float32_policy_required")


def clamp_distances(context, records, values, zero_diagonal):
    if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2:
        raise ValueError("kmeans_distance_clamp_operand_not_supported")
    staged = np.ascontiguousarray(values)
    result, record = context.kmeans_distance_clamp(staged, zero_diagonal)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_squared_distance_clamp", "original_euclidean_sha256": EUCLIDEAN_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, values) else staged.nbytes, "cpu_residuals": ["float32_to_float64_chunk_conversion", "distance_blas_products", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


def validate_resident_distance_inputs(X, Y, squared):
    validate_squared_distance_inputs(X, Y, squared)
    if not 0 < X.shape[0] <= 256 or not 0 < Y.shape[0] <= 256 or not 0 < X.shape[1] <= 1024 or X.shape[1] != Y.shape[1]:
        raise ValueError("kmeans_resident_distances_single_chunk_shape_required")


def resident_finish_distances(context, records, products, left_norms, right_norms, zero_diagonal):
    if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 for value in (products, left_norms, right_norms)) or left_norms.shape != (products.shape[0], 1) or right_norms.shape != (1, products.shape[1]):
        raise ValueError("kmeans_resident_distances_operand_not_supported")
    staged = tuple(np.ascontiguousarray(value) for value in (products, left_norms, right_norms))
    result, record = context.kmeans_resident_distances(staged[0], staged[1].reshape(-1), staged[2].reshape(-1), zero_diagonal)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_resident_distance_finish", "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_input_staging_copy_bytes": sum(value.nbytes for value, original in zip(staged, (products, left_norms, right_norms)) if not np.shares_memory(value, original)), "cpu_residuals": ["float32_to_float64_chunk_conversion", "distance_blas_products", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


class ResidentDistancePipeline:
    def __init__(self, context, finish_records, clamp_records, norm_records=None, minimum_records=None):
        self.context, self.finish_records, self.clamp_records = context, finish_records, clamp_records
        self.norm_records = norm_records
        self.minimum_records = minimum_records
        self.closest = None
        self.pending_minimum = None
        self.distance = None
        self.pending = None

    def reset(self):
        self.pending = None
        self.closest = None
        self.pending_minimum = None

    def finish(self, products, left, right, diagonal):
        if self.pending is not None:
            raise ValueError("kmeans_resident_distances_previous_result_not_consumed")
        output = resident_finish_distances(self.context, self.finish_records, products, left, right, diagonal) if self.norm_records is None else resident_norm_finish_distances(self.context, self.finish_records, self.norm_records, products, left, right, diagonal, closest=self.closest)
        self.pending = (output, diagonal, len(self.finish_records) - 1)
        return output

    def clamp(self, values, diagonal):
        pending, self.pending = self.pending, None
        if pending is None or values is not pending[0] or diagonal is not pending[1]:
            raise ValueError("kmeans_resident_distances_result_reuse_mismatch")
        self.clamp_records.append({"adapter_scope": "sklearn_kmeans_plusplus_squared_distance_clamp", "original_euclidean_sha256": EUCLIDEAN_SHA256, "resident_clamp_reuse": True, "shared_finish_dispatch_index": pending[2], "zero_diagonal": diagonal, "squared_only": True, "rows": values.shape[0], "columns": values.shape[1], "gpu_kernels": ["double_kmeans_resident_distance_finish"], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "host_input_staging_copy_bytes": 0, "cpu_residuals": ["host_staging_and_control"], "quality_certified": False, "full_pipeline_gpu_certified": False})
        if self.closest is not None:
            self.pending_minimum = (self.closest, values, pending[2])
        return values

    def candidate_distance(self, closest, *args, **kwargs):
        if self.minimum_records is None or self.norm_records is None or self.distance is None or self.closest is not None or self.pending is not None or self.pending_minimum is not None:
            raise ValueError("kmeans_resident_minimum_state_invalid")
        if type(closest) is not np.ndarray or closest.dtype != np.float32 or closest.ndim not in (1, 2) or (closest.ndim == 2 and closest.shape[0] != 1):
            raise ValueError("kmeans_resident_minimum_closest_not_supported")
        self.closest = closest
        try:
            result = self.distance(*args, **kwargs)
            if self.pending_minimum is None or self.pending_minimum[1] is not result or self.pending is not None:
                raise ValueError("kmeans_resident_minimum_distance_not_consumed")
            return result
        except BaseException:
            self.reset()
            raise
        finally:
            self.closest = None

    def minimum(self, closest, candidates):
        if self.pending_minimum is None or closest is not self.pending_minimum[0] or candidates is not self.pending_minimum[1]:
            raise ValueError("kmeans_resident_minimum_result_identity_invalid")
        from tools.baseline.native_kmeans_prefix import PLUSPLUS_SHA256
        dispatch = self.pending_minimum[2]
        self.minimum_records.append({"adapter_scope": "sklearn_dense_kmeans_plusplus_inplace_candidate_minimum", "resident_minimum_reuse": True, "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "shared_finish_dispatch_index": dispatch, "rows": candidates.shape[0], "columns": candidates.shape[1], "gpu_kernels": ["float_kmeans_candidate_minimum"], "command_buffers_per_run": 0, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "host_inplace_copy_bytes": 0, "computation_gpu_seconds": None, "quality_certified": False, "full_pipeline_gpu_certified": False})
        self.pending_minimum = None
        return candidates


def validate_resident_batch(X, Y, batch_size):
    if batch_size is not None and (type(batch_size) is not int or batch_size < max(X.shape[0], Y.shape[0])):
        raise ValueError("kmeans_resident_distances_single_chunk_batch_required")


def validate_no_chunk_norms(XX, YY):
    if XX is not None or YY is not None:
        raise ValueError("kmeans_resident_norms_provided_chunk_norms_not_supported")


def validate_no_supplied_norms(*norms):
    if any(value is not None and (type(value) is not np.ndarray or value.dtype != np.float32) for value in norms):
        raise ValueError("kmeans_resident_norms_provided_binary64_norms_not_supported")


def resident_norm_finish_distances(context, records, norm_records, products, left, right, zero_diagonal, closest=None):
    if any(type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 2 for value in (products, left, right)):
        raise ValueError("kmeans_resident_norms_operand_not_supported")
    staged = tuple(np.ascontiguousarray(value) for value in (products, left, right))
    staged_closest = np.ascontiguousarray(closest) if closest is not None else None
    result, record = context.kmeans_resident_norms(*staged, zero_diagonal) if closest is None else context.kmeans_resident_minimum(*staged, staged_closest, zero_diagonal)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_kmeans_plusplus_resident_norms_and_distances", "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "original_row_norms_sha256": ROW_NORMS_SHA256, "host_input_staging_copy_bytes": sum(value.nbytes for value, original in zip(staged, (products, left, right)) if not np.shares_memory(value, original)), "cpu_residuals": ["float32_to_float64_chunk_conversion", "distance_blas_products", "weighted_potentials", "rng_and_seed_control", "lloyd_or_elkan", "host_staging_and_control"]})
    if closest is not None and not np.shares_memory(staged_closest, closest):
        record["host_input_staging_copy_bytes"] += staged_closest.nbytes
    dispatch = len(records)
    records.append(record)
    for role, values in (("X_chunk", left), ("Y_chunk", right)):
        norm_records.append({"adapter_scope": "sklearn_kmeans_plusplus_float64_distance_chunk_norms", "operand_role": role, "resident_norm_reuse": True, "shared_finish_dispatch_index": dispatch, "rows": values.shape[0], "features": values.shape[1], "original_euclidean_sha256": EUCLIDEAN_SHA256, "original_upcast_sha256": UPCAST_SHA256, "original_row_norms_sha256": ROW_NORMS_SHA256, "gpu_kernels": ["double_kmeans_upcast_norms"], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "host_input_staging_copy_bytes": 0, "cpu_residuals": ["host_staging_and_control"], "quality_certified": False, "full_pipeline_gpu_certified": False})
    return result

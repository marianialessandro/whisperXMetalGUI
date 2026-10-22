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


def compile_upcast_distances(original, norms, finish=None):
    import sklearn.metrics.pairwise as pairwise

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
        matches[0].value = ast.parse("__wx_distance_finish(safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True), XX_chunk, YY_chunk)", mode="eval").body
        class RemoveNormAdditions(ast.NodeTransformer):
            def visit_AugAssign(self, node):
                return None if node in matches[1:] else self.generic_visit(node)
        tree = RemoveNormAdditions().visit(tree)
        ast.fix_missing_locations(tree)
    namespace = dict(upcast.__globals__)
    namespace["__wx_distance_finish"] = finish
    namespace["row_norms"] = norms
    exec(compile(tree, "<WhisperX scoped native KMeans upcast norms>", "exec"), namespace)
    candidate_upcast = namespace[upcast.__name__]
    namespace = dict(original.__globals__)
    namespace["_euclidean_distances_upcast"] = candidate_upcast
    exec(compile(ast.parse(textwrap.dedent(inspect.getsource(original))), "<WhisperX scoped KMeans Euclidean distance>", "exec"), namespace)
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

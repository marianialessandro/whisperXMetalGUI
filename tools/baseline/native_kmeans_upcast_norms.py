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


def compile_upcast_distances(original, norms):
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
    namespace = dict(upcast.__globals__)
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

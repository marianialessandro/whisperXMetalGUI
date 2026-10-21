"""Pinned dense KMeans tolerance from NumPy-policy float32 variance on Metal."""

import ast
import hashlib
import importlib.metadata
import inspect
import textwrap

import numpy as np
from numpy._core import _methods


TOLERANCE_SHA256 = "efe0efc0d127ebd7a004fa08828a5d4547c19e467520d4286e595d8a8ddedda8"
NUMPY_VAR_SHA256 = "90d47f72bffb5ab08e5a35f14b9d34101734dcc04b37e53844cae2b3d29a50b6"
NUMPY_MEAN_SHA256 = "a23498e5cf1962f09cce7fc8a6fa17f59b837bbf5a42c1a505a8f1e08752b339"


def compile_native_tolerance(original, reduce):
    if not inspect.isfunction(original) or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("kmeans_tolerance_reference_not_supported")
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != TOLERANCE_SHA256:
        raise ValueError("kmeans_tolerance_source_not_supported")
    if np.__version__ != "2.2.6" or importlib.metadata.version("scikit-learn") != "1.9.1" or original.__globals__.get("np") is not np or hashlib.sha256(inspect.getsource(_methods._var).encode()).hexdigest() != NUMPY_VAR_SHA256 or hashlib.sha256(inspect.getsource(_methods._mean).encode()).hexdigest() != NUMPY_MEAN_SHA256:
        raise ValueError("kmeans_tolerance_runtime_not_supported")
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.parse("variances = np.var(X, axis=0)").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list or len(matches) != 1:
        raise ValueError("kmeans_tolerance_expression_contract_changed")
    replacement = ast.copy_location(ast.Return(value=ast.parse("__wx_native_kmeans_tolerance(X, tol)", mode="eval").body), matches[0])
    class ReplaceVariance(ast.NodeTransformer):
        def visit_Assign(self, node):
            return replacement if node is matches[0] else self.generic_visit(node)
    tree = ReplaceVariance().visit(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_native_kmeans_tolerance"] = reduce
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native KMeans tolerance>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_tolerance_signature_changed")
    return candidate


def tolerance_value(context, records, features, tolerance):
    result, record = context.kmeans_tolerance(features, tolerance)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_internal_tolerance", "original_tolerance_sha256": TOLERANCE_SHA256, "numpy_var_sha256": NUMPY_VAR_SHA256, "numpy_mean_sha256": NUMPY_MEAN_SHA256, "variance_ddof": 0, "cpu_residuals": ["squared_norms", "kmeans_initialization", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result

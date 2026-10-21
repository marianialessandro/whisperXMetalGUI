"""Source-pinned KMeans++ weighted cumulative candidate distribution on Metal."""

import ast
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import textwrap

import numpy as np
import numpy._core._multiarray_umath as numpy_core

from tools.baseline.native_kmeans_norms import NUMPY_CORE_SHA256


PLUSPLUS_SHA256 = "5ecfa4fcb38b550e535120364d38a8bbb1e9686b32e61a06de03c4537c6f44c1"
CUMSUM_SHA256 = "ec24882ab4ab9cfcf431ba4a96aba567fa07cd9ae6f67f003ee468cf29dcdcb0"


def compile_native_prefix(original, prefix):
    if not inspect.isfunction(original) or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("kmeans_weighted_prefix_reference_not_supported")
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != PLUSPLUS_SHA256:
        raise ValueError("kmeans_weighted_prefix_source_not_supported")
    if np.__version__ != "2.2.6" or importlib.metadata.version("scikit-learn") != "1.9.1" or original.__globals__.get("np") is not np or hashlib.sha256(inspect.getsource(np.cumsum).encode()).hexdigest() != CUMSUM_SHA256 or hashlib.sha256(Path(numpy_core.__file__).read_bytes()).hexdigest() != NUMPY_CORE_SHA256:
        raise ValueError("kmeans_weighted_prefix_runtime_not_supported")
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.parse("np.cumsum(sample_weight * closest_dist_sq)", mode="eval").body
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected)]
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list or len(matches) != 1:
        raise ValueError("kmeans_weighted_prefix_expression_contract_changed")
    replacement = ast.parse("__wx_native_kmeans_weighted_prefix(sample_weight, closest_dist_sq)", mode="eval").body
    class ReplacePrefix(ast.NodeTransformer):
        def visit_Call(self, node):
            return ast.copy_location(replacement, node) if node is matches[0] else self.generic_visit(node)
    tree = ReplacePrefix().visit(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_native_kmeans_weighted_prefix"] = prefix
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native KMeans++ weighted prefix>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_weighted_prefix_signature_changed")
    return candidate


def weighted_prefix(context, records, weights, distances):
    result, record = context.kmeans_weighted_prefix(weights, distances)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_candidate_distribution", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_cumsum_sha256": CUMSUM_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["rng", "candidate_searchsorted_and_clipping", "distance_and_potential_computation", "best_candidate_selection", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result

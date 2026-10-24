"""Source-pinned native probabilities in the original random KMeans initializer."""

import ast
import hashlib
import inspect
import textwrap

import numpy as np
from numpy._core import _methods

from tools.baseline.native_kmeans_prefix import NUMPY_CORE_SHA256, SUM_SHA256


INIT_SHA256 = "3576bb153287f1ed5b48d8f413758f395fe57a77172a3e2545068e56743f3362"


def compile_random_initializer(original, probabilities):
    if not inspect.isfunction(original) or original.__closure__ or inspect.unwrap(original) is not original or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != INIT_SHA256 or original.__globals__.get("np") is not np:
        raise ValueError("kmeans_random_init_reference_not_supported")
    if hashlib.sha256(inspect.getsource(_methods._sum).encode()).hexdigest() != SUM_SHA256:
        raise ValueError("kmeans_random_init_sum_reference_changed")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    target = ast.parse("sample_weight / sample_weight.sum()", mode="eval").body
    found = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and ast.dump(node) == ast.dump(target)]
    if len(found) != 1:
        raise ValueError("kmeans_random_init_probability_expression_changed")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node.func) == ast.dump(ast.parse("random_state.choice", mode="eval").body)]
    if len(calls) != 1 or not any(keyword.arg == "p" and keyword.value is found[0] for keyword in calls[0].keywords) or not any(keyword.arg == "replace" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False for keyword in calls[0].keywords):
        raise ValueError("kmeans_random_init_choice_contract_changed")
    class ReplaceProbabilities(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if node is found[0]:
                return ast.copy_location(ast.parse("__wx_random_probabilities(sample_weight)", mode="eval").body, node)
            return self.generic_visit(node)
    tree = ReplaceProbabilities().visit(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_random_probabilities"] = probabilities
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native random KMeans probabilities>", "exec"), namespace)
    candidate = namespace[original.__name__]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_random_init_signature_changed")
    return candidate


def random_probabilities(context, records, weights):
    result, record = context.kmeans_seed_probabilities(weights)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_float32_random_initialization_probabilities", "original_init_centroids_sha256": INIT_SHA256, "numpy_sum_sha256": SUM_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "rng_replaced": False, "choice_replace": False, "cpu_residuals": ["rng_and_choice_internal_probability_validation", "centroid_indexing", "host_staging_and_control"]})
    records.append(record)
    return result

"""Source-pinned KMeans++ weighted cumulative candidate distribution on Metal."""

import ast
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import textwrap

import numpy as np
import numpy._core._multiarray_umath as numpy_core
from numpy._core import _methods

from tools.baseline.native_kmeans_norms import NUMPY_CORE_SHA256


PLUSPLUS_SHA256 = "5ecfa4fcb38b550e535120364d38a8bbb1e9686b32e61a06de03c4537c6f44c1"
CUMSUM_SHA256 = "ec24882ab4ab9cfcf431ba4a96aba567fa07cd9ae6f67f003ee468cf29dcdcb0"
SUM_SHA256 = "945c5e7a356ce7e229f9d6c60226751036653bf32ca0727056dcfb4023c5d8f3"
SEARCHSORTED_SHA256 = "f44941290b76f7795a08ade371902f589f73745ec5be21240c39dc86cf8210cb"
CLIP_SHA256 = "9a0eb9c2e80a635150f89e1397012b8f775d6abfeebb19e569cede3f23a2f155"
ARGMIN_SHA256 = "7cfd400e4c8292a14b60ec6628a65256c0c3cd130197eee6e2047daa06b84360"


def compile_native_prefix(original, prefix, minimum=None, probabilities=None, search=None, best=None, query_scale=None, upcast_distance=None, resident_candidate_distance=None):
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
    minimum_matches = []
    if minimum is not None:
        expected_minimum = ast.parse("np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)", mode="eval").body
        minimum_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected_minimum)]
        if len(minimum_matches) != 1:
            raise ValueError("kmeans_candidate_minimum_expression_contract_changed")
    minimum_replacement = ast.parse("__wx_native_kmeans_candidate_minimum(closest_dist_sq, distance_to_candidates)", mode="eval").body
    probability_matches = []
    if probabilities is not None:
        if hashlib.sha256(inspect.getsource(_methods._sum).encode()).hexdigest() != SUM_SHA256:
            raise ValueError("kmeans_seed_probabilities_runtime_not_supported")
        expected_probability = ast.parse("sample_weight / sample_weight.sum()", mode="eval").body
        probability_matches = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and ast.dump(node) == ast.dump(expected_probability)]
        if len(probability_matches) != 1:
            raise ValueError("kmeans_seed_probabilities_expression_contract_changed")
    probability_replacement = ast.parse("__wx_native_kmeans_seed_probabilities(sample_weight)", mode="eval").body
    scale_matches = []
    if query_scale is not None:
        expected_scale = ast.parse("random_state.uniform(size=n_local_trials) * current_pot", mode="eval").body
        scale_matches = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and ast.dump(node) == ast.dump(expected_scale)]
        if len(scale_matches) != 1:
            raise ValueError("kmeans_query_scaling_expression_contract_changed")
    scale_replacement = ast.parse("__wx_native_kmeans_query_scale(random_state.uniform(size=n_local_trials), current_pot)", mode="eval").body
    search_matches, clip_matches = [], []
    if search is not None:
        if hashlib.sha256(inspect.getsource(np.searchsorted).encode()).hexdigest() != SEARCHSORTED_SHA256 or hashlib.sha256(inspect.getsource(np.clip).encode()).hexdigest() != CLIP_SHA256:
            raise ValueError("kmeans_candidate_search_runtime_not_supported")
        expected_search = ast.parse("np.searchsorted(np.cumsum(sample_weight * closest_dist_sq), rand_vals)", mode="eval").body
        expected_clip = ast.parse("np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)", mode="eval").body
        search_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected_search)]
        clip_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Expr) and ast.dump(node.value) == ast.dump(expected_clip)]
        if len(search_matches) != 1 or len(clip_matches) != 1:
            raise ValueError("kmeans_candidate_search_expression_contract_changed")
    best_matches = []
    if best is not None:
        if hashlib.sha256(inspect.getsource(np.argmin).encode()).hexdigest() != ARGMIN_SHA256:
            raise ValueError("kmeans_best_candidate_runtime_not_supported")
        expected_best = ast.parse("np.argmin(candidates_pot)", mode="eval").body
        best_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected_best)]
        if len(best_matches) != 1:
            raise ValueError("kmeans_best_candidate_expression_contract_changed")
    resident_matches = []
    if resident_candidate_distance is not None:
        if minimum is None or upcast_distance is None:
            raise ValueError("kmeans_resident_minimum_configuration_invalid")
        expected_distance = ast.parse("_euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)", mode="eval").body
        resident_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected_distance)]
        if len(resident_matches) != 1:
            raise ValueError("kmeans_resident_minimum_expression_contract_changed")
    class ReplacePrefix(ast.NodeTransformer):
        def visit_Expr(self, node):
            return ast.copy_location(ast.Pass(), node) if clip_matches and node is clip_matches[0] else self.generic_visit(node)

        def visit_BinOp(self, node):
            if scale_matches and node is scale_matches[0]:
                return ast.copy_location(scale_replacement, node)
            return ast.copy_location(probability_replacement, node) if probability_matches and node is probability_matches[0] else self.generic_visit(node)

        def visit_Call(self, node):
            if resident_matches and node is resident_matches[0]:
                node.func = ast.Name(id="__wx_resident_candidate_distance", ctx=ast.Load())
                node.args.insert(0, ast.Name(id="closest_dist_sq", ctx=ast.Load()))
                return node
            if best_matches and node is best_matches[0]:
                node.func = ast.Name(id="__wx_native_kmeans_best_candidate", ctx=ast.Load())
                return node
            if search_matches and node is search_matches[0]:
                node = self.generic_visit(node)
                node.func = ast.Name(id="__wx_native_kmeans_candidate_search", ctx=ast.Load())
                return node
            if prefix is not None and node is matches[0]:
                return ast.copy_location(replacement, node)
            if minimum_matches and node is minimum_matches[0]:
                return ast.copy_location(minimum_replacement, node)
            return self.generic_visit(node)
    tree = ReplacePrefix().visit(tree)
    namespace = dict(original.__globals__)
    if upcast_distance is not None:
        namespace["_euclidean_distances"] = upcast_distance
    namespace["__wx_resident_candidate_distance"] = resident_candidate_distance
    namespace["__wx_native_kmeans_weighted_prefix"] = prefix
    namespace["__wx_native_kmeans_candidate_minimum"] = minimum
    namespace["__wx_native_kmeans_seed_probabilities"] = probabilities
    namespace["__wx_native_kmeans_candidate_search"] = search
    namespace["__wx_native_kmeans_best_candidate"] = best
    namespace["__wx_native_kmeans_query_scale"] = query_scale
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


def seed_probabilities(context, records, weights):
    result, record = context.kmeans_seed_probabilities(weights)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_first_seed_probabilities", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_sum_sha256": SUM_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["rng_and_choice_internal_probability_validation", "distance_and_potential_computation", "candidate_searchsorted_and_clipping", "best_candidate_selection", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


def candidate_search(context, records, cdf, queries):
    result, record = context.kmeans_candidate_search(cdf, queries)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_candidate_search_and_clip", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_searchsorted_sha256": SEARCHSORTED_SHA256, "numpy_clip_sha256": CLIP_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["rng_and_query_scaling", "distance_and_potential_computation", "best_candidate_selection", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


def best_candidate(context, records, potentials):
    result, record = context.kmeans_best_candidate(potentials)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_best_candidate", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_argmin_sha256": ARGMIN_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "cpu_residuals": ["rng_and_query_scaling", "distance_and_potential_computation", "candidate_gathering_and_seed_copy", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return result


def scale_queries(context, records, random_values, potential):
    if type(random_values) is not np.ndarray or random_values.dtype != np.float64 or random_values.ndim != 1 or not random_values.flags.c_contiguous or type(potential) is not np.ndarray or potential.dtype != np.float32 or potential.shape != (1,) or not potential.flags.c_contiguous:
        raise ValueError("kmeans_query_scaling_operand_format_invalid")
    if not 0 < random_values.size <= 1024:
        raise ValueError("kmeans_query_scaling_shape_invalid")
    if not np.isfinite(random_values).all() or np.signbit(random_values).any() or (random_values >= 1).any() or ((random_values != 0) & (random_values < 2. ** -53)).any() or not np.isfinite(potential).all() or np.signbit(potential).any() or (potential > 2. ** 76).any() or ((potential != 0) & (potential < 2. ** -100)).any():
        raise ValueError("kmeans_query_scaling_value_domain_invalid")
    promoted = np.ascontiguousarray(potential, dtype=np.float64).reshape(1, 1)
    output, record = context.matmul(random_values.reshape(-1, 1), promoted)
    record = dict(record)
    record.update({"scope": "experimental_native_binary64_kmeans_query_scaling", "adapter_scope": "sklearn_dense_kmeans_plusplus_random_query_scaling", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "one_term_product": True, "host_exact_scalar_promotion_bytes": promoted.nbytes, "host_random_view_copy_bytes": 0, "rng_calls_replaced": False, "cpu_residuals": ["rng_uniform_and_choice", "distance_and_potential_computation", "candidate_gathering_and_seed_copy", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return output[:, 0]

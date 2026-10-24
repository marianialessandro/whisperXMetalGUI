"""Source-pinned dense KMeans input mean on Metal, retaining the original fit."""

import ast
import builtins
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading

import numpy as np


FIT_SHA256 = "7d82e93b0e42485df83af8b86b350a1dd1693aed2881dab19a9e6ec53e37c659"
FIT_CONTEXT_SHA256 = "d8352c8507fc8a9d8936bf31df1b02bcf3d2710726f3928039d90c1bc11b88d3"
_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_mean(original, mean, shift=None, norms=None, equivalence=None, distinct=None, initializer=None):
    inner = inspect.unwrap(original)
    source = inspect.getsource(inner)
    if hashlib.sha256(source.encode()).hexdigest() != FIT_SHA256 or inner.__closure__:
        raise ValueError("kmeans_mean_source_not_supported")
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.parse("X_mean = X.mean(axis=0)").body[0]
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected)]
    decorator = ast.parse("_fit_context(prefer_skip_nested_validation=True)", mode="eval").body
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or len(tree.body[0].decorator_list) != 1 or ast.dump(tree.body[0].decorator_list[0]) != ast.dump(decorator) or len(matches) != 1:
        raise ValueError("kmeans_mean_expression_contract_changed")
    matches[0].value = ast.parse("__wx_native_kmeans_mean(X)", mode="eval").body
    if norms is not None:
        expected_norms = ast.parse("x_squared_norms = row_norms(X, squared=True)").body[0]
        norm_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(expected_norms)]
        if len(norm_matches) != 1:
            raise ValueError("kmeans_squared_norms_expression_contract_changed")
        norm_matches[0].value = ast.parse("__wx_native_kmeans_squared_norms(X)", mode="eval").body
    if shift is not None:
        replacements = {}
        for name, add in (("X", False), ("init", False), ("X", True), ("best_centers", True)):
            expected_shift = ast.parse(name + (" += " if add else " -= ") + "X_mean").body[0]
            shifts = [node for node in ast.walk(tree) if isinstance(node, ast.AugAssign) and ast.dump(node) == ast.dump(expected_shift)]
            if len(shifts) != 1:
                raise ValueError("kmeans_shift_expression_contract_changed")
            expression = ast.Expr(value=ast.Call(func=ast.Name(id="__wx_native_kmeans_shift", ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()), ast.Name(id="X_mean", ctx=ast.Load()), ast.Constant(value=add), ast.Constant(value=name)], keywords=[]))
            replacements[id(shifts[0])] = ast.copy_location(expression, shifts[0])
        class ReplaceShifts(ast.NodeTransformer):
            def visit_AugAssign(self, node):
                return replacements.get(id(node), node)
        tree = ReplaceShifts().visit(tree)
    if equivalence is not None:
        target = ast.parse("_is_same_clustering(labels, best_labels, self.n_clusters)", mode="eval").body
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
        if len(found) != 1:
            raise ValueError("kmeans_same_clustering_expression_contract_changed")
        found[0].func = ast.Name(id="__wx_same_clustering", ctx=ast.Load())
        found[0].args.insert(0, ast.Name(id="X", ctx=ast.Load()))
    if distinct is not None:
        if any(inner.__globals__.get(name, getattr(builtins, name)) is not getattr(builtins, name) for name in ("len", "set")):
            raise ValueError("kmeans_distinct_builtins_not_supported")
        target = ast.parse("len(set(best_labels))", mode="eval").body
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
        if len(found) != 1:
            raise ValueError("kmeans_distinct_expression_contract_changed")
        found[0].func = ast.Name(id="__wx_distinct_labels", ctx=ast.Load())
        found[0].args = [ast.Name(id="X", ctx=ast.Load()), ast.Name(id="best_labels", ctx=ast.Load())]
    if initializer is not None:
        target = ast.parse("self._init_centroids(X, x_squared_norms=x_squared_norms, init=init, random_state=random_state, sample_weight=sample_weight)", mode="eval").body
        found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(target)]
        if len(found) != 1:
            raise ValueError("kmeans_random_init_fit_expression_changed")
        found[0].func = ast.Name(id="__wx_init_centroids", ctx=ast.Load())
        found[0].args.insert(0, ast.Name(id="self", ctx=ast.Load()))
    namespace = dict(inner.__globals__)
    namespace["__wx_init_centroids"] = initializer
    namespace["__wx_distinct_labels"] = distinct
    namespace["__wx_same_clustering"] = equivalence
    namespace["__wx_native_kmeans_mean"] = mean
    namespace["__wx_native_kmeans_shift"] = shift
    namespace["__wx_native_kmeans_squared_norms"] = norms
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<WhisperX scoped native KMeans mean>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("kmeans_mean_signature_changed")
    return candidate


def feature_mean(context, records, features):
    if type(features) is not np.ndarray or features.dtype != np.float32 or features.ndim != 2 or not features.flags.c_contiguous:
        raise ValueError("kmeans_mean_operand_not_supported")
    rows, dimensions = features.shape
    if not 0 < rows <= 4096 or not 2 <= dimensions <= 1024:
        raise ValueError("kmeans_mean_shape_not_supported")
    labels = np.zeros(rows, dtype=np.int32)
    output, record = context.kmeans_centroids_vbx(features, labels, 1)
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_input_mean", "original_fit_sha256": FIT_SHA256, "original_fit_context_sha256": FIT_CONTEXT_SHA256, "host_generated_label_bytes": labels.nbytes, "cpu_residuals": ["tolerance_variance", "squared_norms", "feature_subtraction_and_restoration", "kmeans_initialization", "lloyd_or_elkan", "final_centroid_translation", "host_staging_and_control"]})
    records.append(record)
    return output[0]


def shift_inplace(context, records, values, mean, add, role):
    if type(add) is not bool or (role, add) not in (("X", False), ("X", True), ("init", False), ("best_centers", True)):
        raise ValueError("kmeans_shift_role_not_supported")
    if type(values) is not np.ndarray or values.dtype != np.float32 or values.ndim != 2 or not values.flags.c_contiguous or not values.flags.writeable:
        raise ValueError("kmeans_shift_writable_float32_array_required")
    output, record = context.float_shift_vbx(values, mean, add=add)
    np.copyto(values, output, casting="no")
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_centering", "original_fit_sha256": FIT_SHA256, "operand_role": role, "host_inplace_copy_bytes": output.nbytes, "host_copy_is_numeric_conversion": False})
    records.append(record)
    return values


@contextlib.contextmanager
def native_kmeans_feature_mean(context, records, shift_records=None, tolerance_records=None, norm_records=None, prefix_records=None, minimum_records=None, probability_records=None, search_records=None, best_records=None, query_records=None, upcast_norm_records=None, distance_finish_records=None, distance_clamp_records=None, resident_distances=False, resident_norms=False, resident_minimum=False, potential_context=None, potential_records=None, product_records=None, chunk_records=None, lloyd_shift_records=None, lloyd_label_records=None, equivalence_records=None, lloyd_norm_records=None, assignment_records=None, centroid_records=None, center_shift_records=None, inertia_records=None, relocation_records=None, two_chunks=False, upcast_512=False, products_512=False, resident_512=False, distinct_records=None, local_trial_records=None, random_probability_records=None, convergence_records=None, decision_records=None, fused_centroids=False, fused_assignment=False):
    global _ACTIVE
    from sklearn.base import _fit_context
    from sklearn.cluster import KMeans

    with _LOCK:
        if _ACTIVE:
            raise ValueError("kmeans_mean_scope_already_active")
        if not isinstance(records, list):
            raise ValueError("kmeans_mean_records_required")
        if np.__version__ != "2.2.6" or importlib.metadata.version("scikit-learn") != "1.9.1":
            raise ValueError("kmeans_mean_runtime_not_supported")
        if getattr(context.library, "wxdouble_vbx_kmeans_centroids", None) is None:
            raise ValueError("kmeans_mean_abi_unavailable")
        if shift_records is not None:
            if not isinstance(shift_records, list):
                raise ValueError("kmeans_shift_records_required")
            if getattr(context.library, "wxdouble_float_shift", None) is None:
                raise ValueError("kmeans_shift_abi_unavailable")
        if distance_finish_records is not None and (upcast_norm_records is None or not isinstance(distance_finish_records, list)):
            raise ValueError("kmeans_distance_finish_configuration_invalid")
        if distance_finish_records is not None and getattr(context.library, "wxdouble_kmeans_distance_finish", None) is None:
            raise ValueError("double_kmeans_distance_finish_abi_unavailable")
        if distance_clamp_records is not None and (distance_finish_records is None or not isinstance(distance_clamp_records, list)):
            raise ValueError("kmeans_distance_clamp_configuration_invalid")
        if distance_clamp_records is not None and getattr(context.library, "wxdouble_kmeans_distance_clamp", None) is None:
            raise ValueError("double_kmeans_distance_clamp_abi_unavailable")
        if type(resident_distances) is not bool or (resident_distances and distance_clamp_records is None):
            raise ValueError("kmeans_resident_distances_configuration_invalid")
        if resident_distances and getattr(context.library, "wxdouble_kmeans_resident_distances", None) is None:
            raise ValueError("double_kmeans_resident_distances_abi_unavailable")
        if type(resident_norms) is not bool or (resident_norms and not resident_distances):
            raise ValueError("kmeans_resident_norms_configuration_invalid")
        if resident_norms and getattr(context.library, "wxdouble_kmeans_resident_norms", None) is None:
            raise ValueError("double_kmeans_resident_norms_abi_unavailable")
        if type(resident_minimum) is not bool or (resident_minimum and (not resident_norms or minimum_records is None)):
            raise ValueError("kmeans_resident_minimum_configuration_invalid")
        if resident_minimum and getattr(context.library, "wxdouble_kmeans_resident_minimum", None) is None:
            raise ValueError("double_kmeans_resident_minimum_abi_unavailable")
        if type(resident_512) is not bool or (resident_512 and not resident_distances):
            raise ValueError("kmeans_resident_512_configuration_invalid")
        if resident_512:
            names = ["distances"] + (["norms"] if resident_norms else []) + (["minimum"] if resident_minimum else [])
            if any(getattr(context.library, "wxdouble_kmeans_resident_" + name + "_512", None) is None for name in names):
                raise ValueError("double_kmeans_resident_512_abi_unavailable")
        if (potential_context is None) != (potential_records is None) or (potential_records is not None and not isinstance(potential_records, list)):
            raise ValueError("kmeans_potential_configuration_invalid")
        if potential_context is not None:
            from tools.baseline.native_kmeans_potential import weighted_potentials
            if getattr(potential_context.library, "wxkmeans_weighted_potentials", None) is None:
                raise ValueError("kmeans_potential_abi_unavailable")
            if np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}).get("name") != "accelerate":
                raise ValueError("kmeans_potential_accelerate_numpy_required")
        if product_records is not None:
            from tools.baseline.native_kmeans_potential import distance_products
            if potential_context is None or upcast_norm_records is None or not isinstance(product_records, list):
                raise ValueError("kmeans_distance_product_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_distance_products", None) is None:
                raise ValueError("kmeans_distance_product_abi_unavailable")
        if type(products_512) is not bool or (products_512 and product_records is None):
            raise ValueError("kmeans_product_512_configuration_invalid")
        if products_512 and getattr(potential_context.library, "wxkmeans_distance_products_512", None) is None:
            raise ValueError("kmeans_product_512_abi_unavailable")
        if chunk_records is not None:
            from tools.baseline.native_kmeans_potential import upcast_chunk
            if potential_context is None or upcast_norm_records is None or not isinstance(chunk_records, list):
                raise ValueError("kmeans_upcast_chunk_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_upcast_chunk", None) is None:
                raise ValueError("kmeans_upcast_chunk_abi_unavailable")
        if type(upcast_512) is not bool or (upcast_512 and chunk_records is None):
            raise ValueError("kmeans_upcast_512_configuration_invalid")
        if upcast_512 and getattr(potential_context.library, "wxkmeans_upcast_chunk_512", None) is None:
            raise ValueError("kmeans_upcast_512_abi_unavailable")
        if decision_records is not None:
            if convergence_records is None or lloyd_label_records is None or not isinstance(decision_records, list) or potential_context is None:
                raise ValueError("kmeans_decision_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_lloyd_decision", None) is None:
                raise ValueError("kmeans_decision_abi_unavailable")
        if convergence_records is not None:
            if potential_context is None or lloyd_shift_records is None or not isinstance(convergence_records, list):
                raise ValueError("kmeans_convergence_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_lloyd_convergence", None) is None:
                raise ValueError("kmeans_convergence_abi_unavailable")
        candidate_initializer = None
        if random_probability_records is not None:
            from tools.baseline.native_kmeans_random_init import compile_random_initializer, random_probabilities
            if not isinstance(random_probability_records, list):
                raise ValueError("kmeans_random_init_configuration_invalid")
            if getattr(context.library, "wxdouble_kmeans_seed_probabilities", None) is None:
                raise ValueError("kmeans_random_init_probability_abi_unavailable")
            from tools.baseline.native_kmeans_norms import validate_reference
            validate_reference(inspect.unwrap(KMeans.fit).__globals__["row_norms"])
            original_initializer = KMeans._init_centroids
            candidate_initializer = compile_random_initializer(original_initializer, lambda weights: random_probabilities(context, random_probability_records, weights))
        if local_trial_records is not None:
            if potential_context is None or not isinstance(local_trial_records, list):
                raise ValueError("kmeans_local_trials_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_local_trials", None) is None:
                raise ValueError("kmeans_local_trials_abi_unavailable")
        if distinct_records is not None:
            if potential_context is None or not isinstance(distinct_records, list):
                raise ValueError("kmeans_distinct_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_distinct_labels", None) is None:
                raise ValueError("kmeans_distinct_abi_unavailable")
        if equivalence_records is not None:
            from tools.baseline.native_kmeans_equivalence import same_clustering, validate_reference as validate_equivalence
            if potential_context is None or not isinstance(equivalence_records, list):
                raise ValueError("kmeans_same_clustering_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_same_clustering", None) is None:
                raise ValueError("kmeans_same_clustering_abi_unavailable")
            from scipy import sparse
            original_equivalence = inspect.unwrap(KMeans.fit).__globals__.get("_is_same_clustering")
            validate_equivalence(original_equivalence)
        if type(fused_assignment) is not bool or (fused_assignment and not fused_centroids):
            raise ValueError("kmeans_fused_assignment_configuration_invalid")
        if type(fused_centroids) is not bool or (fused_centroids and not two_chunks):
            raise ValueError("kmeans_fused_centroids_configuration_invalid")
        if type(two_chunks) is not bool or (two_chunks and (relocation_records is None or inertia_records is None)):
            raise ValueError("kmeans_two_chunks_configuration_invalid")
        if relocation_records is not None and (centroid_records is None or not isinstance(relocation_records, list)):
            raise ValueError("kmeans_relocation_configuration_invalid")
        if inertia_records is not None:
            if center_shift_records is None or potential_context is None or not isinstance(inertia_records, list):
                raise ValueError("kmeans_inertia_configuration_invalid")
            if getattr(potential_context.library, "wxkmeans_inertia_dense", None) is None:
                raise ValueError("kmeans_inertia_abi_unavailable")
            from tools.baseline.native_kmeans_inertia import inertia_value
        if center_shift_records is not None and (centroid_records is None or not isinstance(center_shift_records, list)):
            raise ValueError("kmeans_lloyd_center_shifts_configuration_invalid")
        if centroid_records is not None and (assignment_records is None or not isinstance(centroid_records, list)):
            raise ValueError("kmeans_lloyd_centroids_configuration_invalid")
        if assignment_records is not None and (potential_context is None or lloyd_norm_records is None or not isinstance(assignment_records, list)):
            raise ValueError("kmeans_lloyd_assignment_configuration_invalid")
        lloyd_norm_scope = None
        if lloyd_norm_records is not None:
            from tools.baseline.native_kmeans_lloyd_norms import LloydNormScope
            lloyd_norm_scope = LloydNormScope(context, lloyd_norm_records, assignment_context=potential_context if assignment_records is not None else None, assignment_records=assignment_records, centroid_records=centroid_records, center_shift_records=center_shift_records, relocation_records=relocation_records, two_chunks=two_chunks, **({"fused_centroids": True} if fused_centroids else {}), **({"fused_assignment": True} if fused_assignment else {}))
        original_lloyd = candidate_lloyd = None
        if lloyd_shift_records is not None or lloyd_label_records is not None or lloyd_norm_scope is not None:
            import sklearn.cluster._kmeans as kmeans_module
            from tools.baseline.native_kmeans_lloyd import compile_native_lloyd, shift_total, labels_equal, tolerance_convergence, convergence_decision
            from tools.baseline.native_kmeans_norms import validate_reference
            if ((lloyd_shift_records is not None or lloyd_label_records is not None) and potential_context is None) or (lloyd_shift_records is not None and not isinstance(lloyd_shift_records, list)) or (lloyd_label_records is not None and not isinstance(lloyd_label_records, list)):
                raise ValueError("kmeans_lloyd_shift_configuration_invalid")
            if lloyd_shift_records is not None and getattr(potential_context.library, "wxkmeans_lloyd_shift_total", None) is None:
                raise ValueError("kmeans_lloyd_shift_abi_unavailable")
            if lloyd_label_records is not None and getattr(potential_context.library, "wxkmeans_lloyd_labels_equal", None) is None:
                raise ValueError("kmeans_lloyd_labels_abi_unavailable")
            validate_reference(kmeans_module._kmeans_plusplus.__globals__["row_norms"])
            original_lloyd = kmeans_module._kmeans_single_lloyd
            candidate_lloyd = compile_native_lloyd(original_lloyd, (lambda values: shift_total(potential_context, lloyd_shift_records, values)) if lloyd_shift_records is not None else None, labels_equal=(lambda values, previous: labels_equal(potential_context, lloyd_label_records, values, previous)) if lloyd_label_records is not None else None, dense_iteration=lloyd_norm_scope.iteration if lloyd_norm_scope is not None else None, inertia=(lambda X, weights, centers, labels, n_threads, single_label=-1: inertia_value(potential_context, inertia_records, X, weights, centers, labels, n_threads, single_label, **({"two_chunks": True} if two_chunks else {}))) if inertia_records is not None else None, convergence=(lambda values, tolerance: tolerance_convergence(potential_context, convergence_records, lloyd_shift_records, values, tolerance)) if convergence_records is not None else None, decision=(lambda labels, previous, values, tolerance: convergence_decision(potential_context, decision_records, lloyd_label_records, convergence_records, lloyd_shift_records, labels, previous, values, tolerance)) if decision_records is not None else None)
        resident_pipeline = None
        original_tolerance = None
        original_plusplus = None
        if prefix_records is not None or minimum_records is not None or probability_records is not None or search_records is not None or best_records is not None or query_records is not None or upcast_norm_records is not None or potential_records is not None:
            import sklearn.cluster._kmeans as kmeans_module
            from tools.baseline.native_kmeans_prefix import best_candidate, candidate_search, compile_native_prefix, scale_queries, seed_probabilities, weighted_prefix
            if prefix_records is not None:
                if not isinstance(prefix_records, list):
                    raise ValueError("kmeans_weighted_prefix_records_required")
                if getattr(context.library, "wxdouble_kmeans_weighted_prefix", None) is None:
                    raise ValueError("kmeans_weighted_prefix_abi_unavailable")
            if minimum_records is not None:
                from tools.baseline.native_kmeans_minimum import candidate_minimum
                if not isinstance(minimum_records, list):
                    raise ValueError("kmeans_candidate_minimum_records_required")
                if getattr(context.library, "wxdouble_kmeans_candidate_minimum", None) is None:
                    raise ValueError("kmeans_candidate_minimum_abi_unavailable")
            if probability_records is not None:
                if not isinstance(probability_records, list):
                    raise ValueError("kmeans_seed_probabilities_records_required")
                if getattr(context.library, "wxdouble_kmeans_seed_probabilities", None) is None:
                    raise ValueError("kmeans_seed_probabilities_abi_unavailable")
            if search_records is not None:
                if not isinstance(search_records, list):
                    raise ValueError("kmeans_candidate_search_records_required")
                if getattr(context.library, "wxdouble_kmeans_candidate_search", None) is None:
                    raise ValueError("kmeans_candidate_search_abi_unavailable")
            if best_records is not None:
                if not isinstance(best_records, list):
                    raise ValueError("kmeans_best_candidate_records_required")
                if getattr(context.library, "wxdouble_kmeans_best_candidate", None) is None:
                    raise ValueError("kmeans_best_candidate_abi_unavailable")
            if query_records is not None:
                if not isinstance(query_records, list):
                    raise ValueError("kmeans_query_scaling_records_required")
                if getattr(context.library, "wxdouble_matmul", None) is None:
                    raise ValueError("kmeans_query_scaling_abi_unavailable")
            if upcast_norm_records is not None:
                from tools.baseline.native_kmeans_upcast_norms import compile_upcast_distances, squared_upcast_norms
                if not isinstance(upcast_norm_records, list):
                    raise ValueError("kmeans_upcast_norms_records_required")
                if getattr(context.library, "wxdouble_kmeans_upcast_norms", None) is None:
                    raise ValueError("double_kmeans_upcast_norms_abi_unavailable")
                def upcast_norms(values, squared=False):
                    result = squared_upcast_norms(context, upcast_norm_records, values, squared)
                    if distance_finish_records is not None:
                        upcast_norm_records[-1]["cpu_residuals"] = ["distance_clamping_and_optional_sqrt_diagonal_fill" if name == "distance_scale_norm_additions_clamp_and_dtype_conversion" else name for name in upcast_norm_records[-1]["cpu_residuals"]]
                    if distance_clamp_records is not None:
                        upcast_norm_records[-1]["cpu_residuals"] = [name for name in upcast_norm_records[-1]["cpu_residuals"] if name != "distance_clamping_and_optional_sqrt_diagonal_fill"]
                    return result
                if distance_finish_records is not None:
                    from tools.baseline.native_kmeans_upcast_norms import finish_distances
                def finish(products, left, right):
                    result = finish_distances(context, distance_finish_records, products, left, right)
                    if distance_clamp_records is not None:
                        distance_finish_records[-1]["cpu_residuals"] = [name for name in distance_finish_records[-1]["cpu_residuals"] if name != "distance_clamping_and_optional_sqrt_diagonal_fill"]
                    return result
                if distance_clamp_records is not None:
                    from tools.baseline.native_kmeans_upcast_norms import clamp_distances
                clamp = (lambda values, zero_diagonal: clamp_distances(context, distance_clamp_records, values, zero_diagonal)) if distance_clamp_records is not None else None
                if resident_distances:
                    from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline
                    resident_pipeline = ResidentDistancePipeline(context, distance_finish_records, distance_clamp_records, norm_records=upcast_norm_records if resident_norms else None, minimum_records=minimum_records if resident_minimum else None, **({"extended_shape": True} if resident_512 else {}))
                    finish, clamp = resident_pipeline.finish, resident_pipeline.clamp
                upcast_distance = compile_upcast_distances(kmeans_module._kmeans_plusplus.__globals__.get("_euclidean_distances"), upcast_norms, finish=finish if distance_finish_records is not None else None, clamp=clamp, resident=resident_distances, resident_norms=resident_norms, **({"resident_512": True} if resident_512 else {}), product=(lambda left, right, dense_output: distance_products(potential_context, product_records, left, right, dense_output, **({"extended_shape": True} if products_512 else {}))) if product_records is not None else None, upcast_chunk=(lambda values, target_dtype, role: upcast_chunk(potential_context, chunk_records, values, role, target_dtype, **({"extended_rows": True} if upcast_512 else {}))) if chunk_records is not None else None)
            if resident_minimum:
                resident_pipeline.distance = upcast_distance
            original_plusplus = kmeans_module._kmeans_plusplus
            prefix = (lambda weights, distances: weighted_prefix(context, prefix_records, weights, distances)) if prefix_records is not None else None
            minimum = (lambda closest, candidates: candidate_minimum(context, minimum_records, closest, candidates)) if minimum_records is not None else None
            if resident_minimum:
                minimum = resident_pipeline.minimum
            probabilities = (lambda weights: seed_probabilities(context, probability_records, weights)) if probability_records is not None else None
            search = (lambda cdf, queries: candidate_search(context, search_records, cdf, queries)) if search_records is not None else None
            best = (lambda potentials: best_candidate(context, best_records, potentials)) if best_records is not None else None
            query_scale = (lambda random_values, potential: scale_queries(context, query_records, random_values, potential)) if query_records is not None else None
            def local_trials(clusters):
                output, record = potential_context.local_trials(clusters)
                record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_default_local_trials", "original_plusplus_sha256": hashlib.sha256(inspect.getsource(original_plusplus).encode()).hexdigest(), "cpu_residuals": ["rng_and_seed_control", "host_staging_and_control"]})
                local_trial_records.append(record)
                return output
            candidate_plusplus = compile_native_prefix(original_plusplus, prefix, minimum=minimum, probabilities=probabilities, search=search, best=best, query_scale=query_scale, upcast_distance=upcast_distance if upcast_norm_records is not None else None, resident_candidate_distance=resident_pipeline.candidate_distance if resident_minimum else None, potential=(lambda distances, weights: weighted_potentials(potential_context, potential_records, distances, weights)) if potential_context is not None else None, local_trials=local_trials if local_trial_records is not None else None)
        if tolerance_records is not None:
            import sklearn.cluster._kmeans as kmeans_module
            from tools.baseline.native_kmeans_tolerance import compile_native_tolerance, tolerance_value
            if not isinstance(tolerance_records, list):
                raise ValueError("kmeans_tolerance_records_required")
            if getattr(context.library, "wxdouble_kmeans_tolerance", None) is None:
                raise ValueError("kmeans_tolerance_abi_unavailable")
            original_tolerance = kmeans_module._tolerance
            candidate_tolerance = compile_native_tolerance(original_tolerance, lambda features, tolerance: tolerance_value(context, tolerance_records, features, tolerance))
        original = KMeans.fit
        inner = inspect.unwrap(original)
        if not inspect.isfunction(original) or not inspect.isfunction(inner):
            raise ValueError("kmeans_mean_fit_wrapper_not_supported")
        if norm_records is not None:
            from tools.baseline.native_kmeans_norms import squared_norms, validate_reference
            if not isinstance(norm_records, list):
                raise ValueError("kmeans_squared_norms_records_required")
            if getattr(context.library, "wxdouble_kmeans_squared_norms", None) is None:
                raise ValueError("kmeans_squared_norms_abi_unavailable")
            original_norms = inner.__globals__.get("row_norms")
            validate_reference(original_norms)
        reference = _fit_context(prefer_skip_nested_validation=True)(inner)
        closure = dict(zip(original.__code__.co_freevars, (cell.cell_contents for cell in original.__closure__ or ())))
        if original.__code__ is not reference.__code__ or getattr(original, "__wrapped__", None) is not inner or closure.get("fit_method") is not inner or closure.get("prefer_skip_nested_validation") is not True or inner.__globals__.get("_fit_context") is not _fit_context or hashlib.sha256(inspect.getsource(_fit_context).encode()).hexdigest() != FIT_CONTEXT_SHA256:
            raise ValueError("kmeans_mean_fit_wrapper_not_supported")
        def mean(features):
            output = feature_mean(context, records, features)
            if shift_records is not None:
                records[-1]["cpu_residuals"] = [name for name in records[-1]["cpu_residuals"] if name not in ("feature_subtraction_and_restoration", "final_centroid_translation")]
            if tolerance_records is not None:
                records[-1]["cpu_residuals"] = [name for name in records[-1]["cpu_residuals"] if name != "tolerance_variance"]
            if norm_records is not None:
                records[-1]["cpu_residuals"] = [name for name in records[-1]["cpu_residuals"] if name != "squared_norms"]
                records[-1]["cpu_residuals"].append("kmeans_initialization_internal_norms")
            if upcast_norm_records is not None:
                records[-1]["cpu_residuals"] = [name for name in records[-1]["cpu_residuals"] if name != "kmeans_initialization_internal_norms"]
            return output

        def shift(values, offset, add, role):
            return shift_inplace(context, shift_records, values, offset, add, role)

        def norms(features):
            from scipy import sparse
            return original_norms(features, squared=True) if sparse.issparse(features) else squared_norms(context, norm_records, features)

        def distinct(X, labels):
            from scipy import sparse
            if sparse.issparse(X):
                return len(set(labels))
            result, record = potential_context.distinct_labels(labels)
            record.update({"adapter_scope": "sklearn_dense_float32_final_distinct_cluster_count", "original_fit_sha256": FIT_SHA256, "cpu_residuals": ["warning_and_attribute_control", "host_staging_and_control"]})
            distinct_records.append(record)
            return result

        def initialize(instance, X, *args, **kwargs):
            from scipy import sparse
            if type(X) is not np.ndarray or X.dtype != np.float32 or sparse.issparse(X) or getattr(instance._init_centroids, "__func__", None) is not original_initializer:
                return instance._init_centroids(X, *args, **kwargs)
            return candidate_initializer(instance, X, *args, **kwargs)

        candidate = compile_native_mean(original, mean, shift=shift if shift_records is not None else None, norms=norms if norm_records is not None else None, equivalence=(lambda X, labels, previous, clusters: original_equivalence(labels, previous, clusters) if sparse.issparse(X) else same_clustering(potential_context, equivalence_records, labels, previous, clusters)) if equivalence_records is not None else None, distinct=distinct if distinct_records is not None else None, initializer=initialize if candidate_initializer is not None else None)
        owner = threading.get_ident()

        if candidate_lloyd is not None:
            @functools.wraps(original_lloyd)
            def wrapped_lloyd(X, *args, **kwargs):
                return candidate_lloyd(X, *args, **kwargs) if threading.get_ident() == owner and type(X) is np.ndarray and X.dtype == np.float32 else original_lloyd(X, *args, **kwargs)
            if inspect.unwrap(candidate).__globals__.get("_kmeans_single_lloyd") is not original_lloyd:
                raise ValueError("kmeans_lloyd_fit_binding_not_supported")
            inspect.unwrap(candidate).__globals__["_kmeans_single_lloyd"] = wrapped_lloyd

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner:
                return original(*args, **kwargs)
            try:
                return candidate(*args, **kwargs)
            finally:
                if resident_pipeline is not None:
                    resident_pipeline.reset()

        if original_tolerance is not None:
            @functools.wraps(original_tolerance)
            def wrapped_tolerance(*args, **kwargs):
                return candidate_tolerance(*args, **kwargs) if threading.get_ident() == owner else original_tolerance(*args, **kwargs)

        if original_plusplus is not None:
            @functools.wraps(original_plusplus)
            def wrapped_plusplus(X, *args, **kwargs):
                from scipy import sparse
                return candidate_plusplus(X, *args, **kwargs) if threading.get_ident() == owner and not sparse.issparse(X) else original_plusplus(X, *args, **kwargs)

        if candidate_initializer is not None and original_plusplus is not None:
            if candidate_initializer.__globals__.get("_kmeans_plusplus") is not original_plusplus:
                raise ValueError("kmeans_random_init_plusplus_binding_changed")
            candidate_initializer.__globals__["_kmeans_plusplus"] = wrapped_plusplus

        if lloyd_norm_scope is not None:
            lloyd_norm_scope.__enter__()
        _ACTIVE = True
        KMeans.fit = wrapped
        if original_tolerance is not None:
            kmeans_module._tolerance = wrapped_tolerance
        if original_plusplus is not None:
            kmeans_module._kmeans_plusplus = wrapped_plusplus
        try:
            yield
        finally:
            if lloyd_norm_scope is not None:
                lloyd_norm_scope.__exit__()
            KMeans.fit = original
            if original_tolerance is not None:
                kmeans_module._tolerance = original_tolerance
            if original_plusplus is not None:
                kmeans_module._kmeans_plusplus = original_plusplus
            _ACTIVE = False

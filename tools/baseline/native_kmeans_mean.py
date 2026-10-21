"""Source-pinned dense KMeans input mean on Metal, retaining the original fit."""

import ast
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


def compile_native_mean(original, mean, shift=None, norms=None):
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
    namespace = dict(inner.__globals__)
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
def native_kmeans_feature_mean(context, records, shift_records=None, tolerance_records=None, norm_records=None, prefix_records=None):
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
        original_tolerance = None
        original_plusplus = None
        if prefix_records is not None:
            import sklearn.cluster._kmeans as kmeans_module
            from tools.baseline.native_kmeans_prefix import compile_native_prefix, weighted_prefix
            if not isinstance(prefix_records, list):
                raise ValueError("kmeans_weighted_prefix_records_required")
            if getattr(context.library, "wxdouble_kmeans_weighted_prefix", None) is None:
                raise ValueError("kmeans_weighted_prefix_abi_unavailable")
            original_plusplus = kmeans_module._kmeans_plusplus
            candidate_plusplus = compile_native_prefix(original_plusplus, lambda weights, distances: weighted_prefix(context, prefix_records, weights, distances))
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
            return output

        def shift(values, offset, add, role):
            return shift_inplace(context, shift_records, values, offset, add, role)

        def norms(features):
            from scipy import sparse
            return original_norms(features, squared=True) if sparse.issparse(features) else squared_norms(context, norm_records, features)

        candidate = compile_native_mean(original, mean, shift=shift if shift_records is not None else None, norms=norms if norm_records is not None else None)
        owner = threading.get_ident()

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return candidate(*args, **kwargs) if threading.get_ident() == owner else original(*args, **kwargs)

        if original_tolerance is not None:
            @functools.wraps(original_tolerance)
            def wrapped_tolerance(*args, **kwargs):
                return candidate_tolerance(*args, **kwargs) if threading.get_ident() == owner else original_tolerance(*args, **kwargs)

        if original_plusplus is not None:
            @functools.wraps(original_plusplus)
            def wrapped_plusplus(X, *args, **kwargs):
                from scipy import sparse
                return candidate_plusplus(X, *args, **kwargs) if threading.get_ident() == owner and not sparse.issparse(X) else original_plusplus(X, *args, **kwargs)

        _ACTIVE = True
        KMeans.fit = wrapped
        if original_tolerance is not None:
            kmeans_module._tolerance = wrapped_tolerance
        if original_plusplus is not None:
            kmeans_module._kmeans_plusplus = wrapped_plusplus
        try:
            yield
        finally:
            KMeans.fit = original
            if original_tolerance is not None:
                kmeans_module._tolerance = original_tolerance
            if original_plusplus is not None:
                kmeans_module._kmeans_plusplus = original_plusplus
            _ACTIVE = False

"""Scoped optional native softmax for the pinned one-hot cluster initialization."""

import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading

import numpy as np


CLUSTER_SHA256 = "74a83b4dbfa2fe19f144507a9bcd379a10e59d8199665dbaf160f748c561b8b2"
SOFTMAX_SHA256 = "a8fa3344c04966fbda42d5d3d3f80cb82412a25fef28e644729b50f6a479d4e4"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def compile_native_cluster(original, initialize, run_vbx, one_hot=None, resident_initialize=None):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != CLUSTER_SHA256 or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("vbx_qinit_source_not_supported")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("vbx_qinit_function_contract_changed")
    expected = ast.dump(ast.parse("softmax(qinit * init_smoothing, axis=1)").body[0].value)
    counts = []
    one_hot_expressions = {ast.dump(ast.parse(expression).body[0]): label for expression, label in [("qinit = np.zeros((len(ahc_init), ahc_init.max() + 1))", "allocate"), ("qinit[range(len(ahc_init)), ahc_init.astype(int)] = 1.0", "scatter")]}
    one_hot_counts = {"allocate": 0, "scatter": 0}
    resident_expected = ast.dump(ast.parse("qinit = qinit if init_smoothing < 0 else softmax(qinit * init_smoothing, axis=1)").body[0])
    resident_counts = []
    if resident_initialize is not None and (one_hot is None or initialize is None):
        raise ValueError("vbx_resident_initialization_configuration_invalid")

    class Substitute(ast.NodeTransformer):
        def visit_Assign(self, node):
            if resident_initialize is not None and ast.dump(node) == resident_expected:
                resident_counts.append(1)
                node.value = ast.Call(func=ast.Name(id="__wx_vbx_resident_initialize", ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()) for name in ("ahc_init", "init_smoothing")], keywords=[])
                return node
            label = one_hot_expressions.get(ast.dump(node)) if one_hot is not None else None
            if label is None:
                return self.generic_visit(node)
            one_hot_counts[label] += 1
            if label == "scatter" or resident_initialize is not None:
                return None
            node.value = ast.Call(func=ast.Name(id="__wx_vbx_one_hot", ctx=ast.Load()), args=[ast.Name(id="ahc_init", ctx=ast.Load())], keywords=[])
            return node

        def visit_Call(self, node):
            if initialize is None or ast.dump(node) != expected:
                return self.generic_visit(node)
            counts.append(1)
            return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_qinit", ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()) for name in ("qinit", "init_smoothing")], keywords=[]), node)

    tree = Substitute().visit(tree)
    if one_hot is not None and any(count != 1 for count in one_hot_counts.values()):
        raise ValueError("vbx_one_hot_expression_contract_changed")
    if resident_initialize is not None and len(resident_counts) != 1:
        raise ValueError("vbx_resident_initialization_expression_contract_changed")
    if resident_initialize is None and initialize is not None and len(counts) != 1:
        raise ValueError("vbx_qinit_expression_contract_changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_vbx_qinit"] = initialize
    namespace["__wx_vbx_one_hot"] = one_hot
    namespace["__wx_vbx_resident_initialize"] = resident_initialize
    namespace["VBx"] = run_vbx
    exec(compile(tree, "<WhisperX scoped native cluster initial softmax>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("vbx_qinit_signature_changed")
    return candidate


def initial_softmax(context, records, values, smoothing, one_hot=False):
    if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
        raise ValueError("vbx_qinit_operand_format_not_supported")
    if not 0 < values.shape[0] <= 4096 or not 0 < values.shape[1] <= 1024 or values.size > 65536:
        raise ValueError("vbx_qinit_shape_not_supported")
    staged = np.ascontiguousarray(values)
    output, record = context.qinit_softmax_vbx(staged, smoothing)
    record = dict(record)
    record.update({"adapter_scope": "vbx_initial_cluster_softmax", "original_cluster_sha256": CLUSTER_SHA256, "scipy_softmax_sha256": SOFTMAX_SHA256, "host_input_staging_copy_bytes": 0 if staged is values else staged.nbytes, "host_output_layout_copy_bytes": 0 if output.flags.c_contiguous else output.nbytes, "cpu_residuals": ["ahc", "label_shape_metadata" if one_hot else "initial_one_hot_and_label_metadata", "host_staging_and_control"]})
    records.append(record)
    return output


def prepare_labels(labels):
    if type(labels) is not np.ndarray or labels.ndim != 1 or labels.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise ValueError("vbx_one_hot_labels_format_not_supported")
    if not 0 < labels.size <= 4096:
        raise ValueError("vbx_one_hot_shape_not_supported")
    columns = int(labels.max()) + 1
    if not 0 < columns <= 1024:
        raise ValueError("vbx_one_hot_shape_not_supported")
    staged = np.ascontiguousarray(labels, dtype=np.int64)
    return staged, columns


def initial_one_hot(context, records, labels):
    staged, columns = prepare_labels(labels)
    output, record = context.one_hot_vbx(staged, columns)
    record = dict(record)
    record.update({"adapter_scope": "vbx_initial_one_hot", "original_cluster_sha256": CLUSTER_SHA256, "label_dtype_conversion": str(labels.dtype) != "int64", "host_label_staging_copy_bytes": 0 if staged is labels else staged.nbytes, "cpu_residuals": ["ahc", "label_shape_metadata", "host_staging_and_control"]})
    records.append(record)
    return output


def initial_resident_qinit(context, records, one_hot_records, labels, smoothing):
    if smoothing < 0:
        return initial_one_hot(context, one_hot_records, labels)
    staged, columns = prepare_labels(labels)
    output, record = context.qinit_from_labels_vbx(staged, columns, smoothing)
    record = dict(record)
    record.update({"adapter_scope": "vbx_initial_cluster_softmax", "original_cluster_sha256": CLUSTER_SHA256, "scipy_softmax_sha256": SOFTMAX_SHA256, "label_dtype_conversion": str(labels.dtype) != "int64", "host_label_staging_copy_bytes": 0 if staged is labels else staged.nbytes, "cpu_residuals": ["ahc", "label_shape_metadata", "host_staging_and_control"]})
    records.append(record)
    one_hot_records.append({"adapter_scope": "vbx_initial_one_hot", "operation": "initial_one_hot", "original_cluster_sha256": CLUSTER_SHA256, "resident_one_hot_reuse": True, "shared_softmax_dispatch_index": len(records) - 1, "gpu_kernels": ["double_vbx_one_hot"], "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "host_label_staging_copy_bytes": 0, "quality_certified": False, "full_pipeline_gpu_certified": False})
    return output


@contextlib.contextmanager
def native_vbx_qinit(context, records, one_hot=False, one_hot_records=None, softmax=True, resident_initialization=False):
    global _ACTIVE
    import pyannote.audio.utils.vbx as vbx
    import pyannote.audio.pipelines.clustering as clustering
    import scipy.special as special

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("vbx_qinit_scope_already_active")
        if importlib.metadata.version("pyannote-audio") != "4.0.4" or np.__version__ != "2.2.6" or importlib.metadata.version("scipy") != "1.15.3":
            raise ValueError("vbx_qinit_runtime_not_supported")
        if not isinstance(records, list):
            raise ValueError("vbx_qinit_records_required")
        if type(one_hot) is not bool or type(softmax) is not bool or (one_hot and not isinstance(one_hot_records, list)):
            raise ValueError("vbx_one_hot_records_required")
        if type(resident_initialization) is not bool or (resident_initialization and not (one_hot and softmax)):
            raise ValueError("vbx_resident_initialization_configuration_invalid")
        if resident_initialization and getattr(context.library, "wxdouble_vbx_qinit_from_labels", None) is None:
            raise ValueError("double_vbx_resident_initialization_abi_unavailable")
        if not one_hot and not softmax:
            raise ValueError("vbx_qinit_stage_required")
        if one_hot and getattr(context.library, "wxdouble_vbx_one_hot", None) is None:
            raise ValueError("double_vbx_one_hot_abi_unavailable")
        if softmax and getattr(context.library, "wxdouble_vbx_qinit", None) is None:
            raise ValueError("double_vbx_qinit_abi_unavailable")
        original = vbx.cluster_vbx
        if clustering.cluster_vbx is not original or vbx.softmax is not special.softmax or hashlib.sha256(textwrap.dedent(inspect.getsource(special.softmax)).encode()).hexdigest() != SOFTMAX_SHA256:
            raise ValueError("vbx_qinit_reference_not_supported")
        owner_thread = threading.get_ident()

        def initialize(values, smoothing):
            return initial_softmax(context, records, values, smoothing, one_hot)

        def run_vbx(*args, **kwargs):
            return vbx.VBx(*args, **kwargs)

        def construct(labels):
            return initial_one_hot(context, one_hot_records, labels)

        def construct_resident(labels, smoothing):
            return initial_resident_qinit(context, records, one_hot_records, labels, smoothing)

        candidate = compile_native_cluster(original, initialize if softmax else None, run_vbx, construct if one_hot else None, construct_resident if resident_initialization else None)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return original(*args, **kwargs)
            return candidate(*args, **kwargs)

        _ACTIVE = True
        vbx.cluster_vbx = clustering.cluster_vbx = wrapped
        try:
            yield
        finally:
            vbx.cluster_vbx = clustering.cluster_vbx = original
            _ACTIVE = False

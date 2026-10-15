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


def compile_native_cluster(original, initialize, run_vbx):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != CLUSTER_SHA256 or original.__closure__ or inspect.unwrap(original) is not original:
        raise ValueError("vbx_qinit_source_not_supported")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("vbx_qinit_function_contract_changed")
    expected = ast.dump(ast.parse("softmax(qinit * init_smoothing, axis=1)").body[0].value)
    counts = []

    class Substitute(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) != expected:
                return self.generic_visit(node)
            counts.append(1)
            return ast.copy_location(ast.Call(func=ast.Name(id="__wx_vbx_qinit", ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()) for name in ("qinit", "init_smoothing")], keywords=[]), node)

    tree = Substitute().visit(tree)
    if len(counts) != 1:
        raise ValueError("vbx_qinit_expression_contract_changed")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["__wx_vbx_qinit"] = initialize
    namespace["VBx"] = run_vbx
    exec(compile(tree, "<WhisperX scoped native cluster initial softmax>", "exec"), namespace)
    candidate = namespace[tree.body[0].name]
    if inspect.signature(candidate) != inspect.signature(original):
        raise ValueError("vbx_qinit_signature_changed")
    return candidate


def initial_softmax(context, records, values, smoothing):
    if type(values) is not np.ndarray or values.dtype != np.float64 or values.ndim != 2 or not values.flags.c_contiguous:
        raise ValueError("vbx_qinit_operand_format_not_supported")
    if not 0 < values.shape[0] <= 4096 or not 0 < values.shape[1] <= 1024 or values.size > 65536:
        raise ValueError("vbx_qinit_shape_not_supported")
    staged = np.ascontiguousarray(values)
    output, record = context.qinit_softmax_vbx(staged, smoothing)
    record = dict(record)
    record.update({"adapter_scope": "vbx_initial_cluster_softmax", "original_cluster_sha256": CLUSTER_SHA256, "scipy_softmax_sha256": SOFTMAX_SHA256, "host_input_staging_copy_bytes": 0 if staged is values else staged.nbytes, "host_output_layout_copy_bytes": 0 if output.flags.c_contiguous else output.nbytes, "cpu_residuals": ["ahc", "initial_one_hot_and_label_metadata", "host_staging_and_control"]})
    records.append(record)
    return output


@contextlib.contextmanager
def native_vbx_qinit(context, records):
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
        if getattr(context.library, "wxdouble_vbx_qinit", None) is None:
            raise ValueError("double_vbx_qinit_abi_unavailable")
        original = vbx.cluster_vbx
        if clustering.cluster_vbx is not original or vbx.softmax is not special.softmax or hashlib.sha256(textwrap.dedent(inspect.getsource(special.softmax)).encode()).hexdigest() != SOFTMAX_SHA256:
            raise ValueError("vbx_qinit_reference_not_supported")
        owner_thread = threading.get_ident()

        def initialize(values, smoothing):
            return initial_softmax(context, records, values, smoothing)

        def run_vbx(*args, **kwargs):
            return vbx.VBx(*args, **kwargs)

        candidate = compile_native_cluster(original, initialize, run_vbx)

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

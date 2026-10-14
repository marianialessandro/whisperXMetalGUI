"""Scoped experimental native binary64 products in the original PLDA transformation, with an explicit sequential accumulation policy."""

import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading
import types

import numpy as np

from tools.baseline.native_plda_adapter import VBX_SETUP_SHA256


PLDA_CALL_SHA256 = "9d4254a13741aa2f25a2c1aab54f4bcc5faf90ca342193d3fb6c6187ed4bd0a2"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def prepared_parameters(model, vbx):
    codes = [value for value in vbx.vbx_setup.__code__.co_consts if isinstance(value, types.CodeType)]
    parameters = {}
    for function, names in ((model._xvec_tf, {"mean1", "mean2", "lda"}), (model._plda_tf, {"plda_mu", "plda_tr"})):
        if not isinstance(function, types.FunctionType) or function.__globals__ is not vbx.__dict__ or not any(function.__code__ is code for code in codes):
            raise ValueError("plda_matrix_transform_closure_not_supported")
        values = inspect.getclosurevars(function).nonlocals
        if set(values) != names:
            raise ValueError("plda_matrix_parameter_names_not_supported")
        parameters.update(values)
    for name, values in parameters.items():
        expected_dtype = np.float32 if name in ("mean2", "lda") else np.float64
        if not isinstance(values, np.ndarray) or values.dtype != expected_dtype:
            raise ValueError("plda_matrix_parameter_dtype_not_supported")
    lda, transform = parameters["lda"], parameters["plda_tr"]
    if lda.ndim != 2 or transform.ndim != 2 or not 0 < lda.shape[0] <= 1024 or not 0 < lda.shape[1] <= 1024 or transform.shape != (lda.shape[1], lda.shape[1]):
        raise ValueError("plda_matrix_parameter_shape_not_supported")
    if parameters["mean1"].shape != (lda.shape[0],) or parameters["mean2"].shape != (lda.shape[1],) or parameters["plda_mu"].shape != (lda.shape[1],):
        raise ValueError("plda_matrix_mean_shape_not_supported")
    if type(model.lda_dimension) is not int or not 0 < model.lda_dimension <= lda.shape[1]:
        raise ValueError("plda_matrix_dimension_not_supported")
    return parameters


def transform_embeddings(context, records, model, embeddings, vbx):
    parameters = prepared_parameters(model, vbx)
    if not isinstance(embeddings, np.ndarray) or embeddings.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or embeddings.ndim != 2 or embeddings.shape[1] != parameters["lda"].shape[0] or not 0 < embeddings.shape[0] <= 1024:
        raise ValueError("plda_matrix_embedding_shape_or_dtype_not_supported")
    if embeddings.shape[0] * parameters["lda"].shape[0] * parameters["lda"].shape[1] > 32 * 1024 * 1024 or embeddings.shape[0] * parameters["lda"].shape[1] ** 2 > 32 * 1024 * 1024:
        raise ValueError("plda_matrix_transform_work_budget_invalid")
    parameters = {name: np.array(values, copy=True, order="K") for name, values in parameters.items()}
    hashes = {name: hashlib.sha256(values.tobytes(order="C")).hexdigest() for name, values in parameters.items()}
    snapshot_bytes = sum(values.nbytes for values in parameters.values())

    def product(left, right, operation):
        left_stage, right_stage = np.ascontiguousarray(left, dtype=np.float64), np.ascontiguousarray(right, dtype=np.float64)
        output, record = context.matmul(left_stage, right_stage)
        record = dict(record)
        record.update({"adapter_scope": "plda_matrix_products", "operation": operation, "original_plda_call_sha256": PLDA_CALL_SHA256, "original_vbx_setup_sha256": VBX_SETUP_SHA256, "prepared_parameter_sha256": hashes, "parameter_hash_snapshot_bytes": snapshot_bytes if operation == "lda" else 0, "parameter_snapshot_copy_bytes": snapshot_bytes if operation == "lda" else 0, "host_left_staging_copy_bytes": 0 if left_stage is left else left_stage.nbytes, "host_right_staging_copy_bytes": 0 if right_stage is right else right_stage.nbytes, "cpu_residuals": ["centering", "scaling", "model_setup_inverse_and_eigh"], "numpy_version": np.__version__})
        records.append(record)
        return output

    lda = parameters["lda"]
    scaled = np.sqrt(lda.shape[0]) * vbx.l2_norm(embeddings - parameters["mean1"]).T
    projected = product(lda.T, scaled, "lda").T - parameters["mean2"]
    transformed = np.sqrt(lda.shape[1]) * vbx.l2_norm(projected)
    return product(transformed - parameters["plda_mu"], parameters["plda_tr"].T, "plda")[:, :model.lda_dimension]


@contextlib.contextmanager
def native_plda_matrix_products(context, records):
    global _ACTIVE
    from pyannote.audio.core.plda import PLDA
    import pyannote.audio.utils.vbx as vbx

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("plda_matrix_scope_already_active")
        if importlib.metadata.version("pyannote-audio") != "4.0.4" or np.__version__ != "2.2.6":
            raise ValueError("plda_matrix_runtime_not_supported")
        original = PLDA.__call__
        for function, expected in ((original, PLDA_CALL_SHA256), (vbx.vbx_setup, VBX_SETUP_SHA256)):
            if hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest() != expected:
                raise ValueError("plda_matrix_source_not_supported")
        if getattr(context.library, "wxdouble_matmul", None) is None:
            raise ValueError("double_matrix_abi_unavailable")
        owner_thread = threading.get_ident()

        @functools.wraps(original)
        def wrapped(self, embeddings):
            if threading.get_ident() != owner_thread:
                return original(self, embeddings)
            return transform_embeddings(context, records, self, embeddings, vbx)

        _ACTIVE = True
        PLDA.__call__ = wrapped
        try:
            yield
        finally:
            PLDA.__call__ = original
            _ACTIVE = False

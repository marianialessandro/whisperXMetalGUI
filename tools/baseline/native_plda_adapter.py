"""Scoped native binary64 L2 normalization inside the original PLDA preprocessing pipeline."""

import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading

import numpy as np


L2_NORM_SHA256 = "e9dd7f601e8eae93598cea1121fd61d2f88cd0b748f7934b6c5c44973cb3d428"
VBX_SETUP_SHA256 = "97eedb302c2231fc1b2b604b69a29c3de64510992e916a37dd1b99a3bb078a20"
_HOOK_LOCK = threading.RLock()
_ACTIVE = False


def normalize_l2(context, records, values):
    if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2 or not (values.flags.c_contiguous or values.flags.f_contiguous):
        raise ValueError("plda_normalization_layout_or_dtype_not_supported")
    column_contiguous = values.flags.f_contiguous and not values.flags.c_contiguous
    staged = np.ascontiguousarray(values)
    output, record = context.normalize(staged, "sequential" if column_contiguous else "pairwise")
    if column_contiguous:
        output = np.asfortranarray(output)
    record = dict(record)
    record.update({"adapter_scope": "plda_preprocessing_binary64_l2_norm", "original_l2_norm_sha256": L2_NORM_SHA256, "original_vbx_setup_sha256": VBX_SETUP_SHA256, "input_layout": "F" if column_contiguous else "C", "host_input_staging_copy_bytes": values.nbytes if column_contiguous else 0, "host_output_layout_copy_bytes": values.nbytes if column_contiguous else 0, "cpu_residuals": ["centering", "scaling", "lda_and_plda_matrix_products", "model_setup_inverse_and_eigh"], "numpy_version": np.__version__})
    records.append(record)
    return output


@contextlib.contextmanager
def native_plda_normalization(context, records):
    global _ACTIVE
    import pyannote.audio.utils.vbx as vbx

    with _HOOK_LOCK:
        if _ACTIVE:
            raise ValueError("plda_normalization_scope_already_active")
        if importlib.metadata.version("pyannote-audio") != "4.0.4" or np.__version__ != "2.2.6":
            raise ValueError("plda_normalization_runtime_not_supported")
        original = vbx.l2_norm
        for function, expected in ((original, L2_NORM_SHA256), (vbx.vbx_setup, VBX_SETUP_SHA256)):
            if hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest() != expected or function.__closure__:
                raise ValueError("plda_normalization_source_not_supported")
        owner_thread = threading.get_ident()

        @functools.wraps(original)
        def wrapped(vec_or_matrix):
            if threading.get_ident() != owner_thread:
                return original(vec_or_matrix)
            return normalize_l2(context, records, vec_or_matrix)

        _ACTIVE = True
        vbx.l2_norm = wrapped
        try:
            yield
        finally:
            vbx.l2_norm = original
            _ACTIVE = False

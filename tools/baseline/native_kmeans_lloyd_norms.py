"""Owner-scoped Metal centroid norms at the pinned Cython Lloyd callback boundary."""

import hashlib
from pathlib import Path
import threading

import numpy as np

from tools.baseline.native_kmeans_norms import squared_norms, validate_reference


LLOYD_BINARY_SHA256 = "865e1791855baec7f4a8233cc5792a13a04efcdb4daef57d986c10fe22f3203f"


class LloydNormScope:
    def __init__(self, context, records):
        import sklearn.cluster._k_means_lloyd as module
        if not isinstance(records, list):
            raise ValueError("kmeans_lloyd_norm_records_required")
        if getattr(context.library, "wxdouble_kmeans_squared_norms", None) is None:
            raise ValueError("kmeans_lloyd_norm_abi_unavailable")
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != LLOYD_BINARY_SHA256:
            raise ValueError("kmeans_lloyd_norm_binary_not_supported")
        validate_reference(module.row_norms)
        self.context, self.records, self.module = context, records, module
        self.original_norms, self.original_iteration = module.row_norms, module.lloyd_iter_chunked_dense
        self.owner = threading.get_ident()
        self.centers = None
        self.active = False
        self.hook = self.row_norms

    def __enter__(self):
        if self.active or self.module.row_norms is not self.original_norms:
            raise ValueError("kmeans_lloyd_norm_scope_binding_changed")
        self.module.row_norms = self.hook
        self.active = True
        return self

    def __exit__(self, *args):
        self.module.row_norms = self.original_norms
        self.centers = None
        self.active = False

    def iteration(self, X, sample_weight, centers_old, *args, **kwargs):
        if threading.get_ident() != self.owner:
            return self.original_iteration(X, sample_weight, centers_old, *args, **kwargs)
        if not self.active or self.module.row_norms is not self.hook or self.centers is not None:
            raise ValueError("kmeans_lloyd_norm_scope_not_ready")
        self.centers = np.asarray(centers_old)
        try:
            return self.original_iteration(X, sample_weight, centers_old, *args, **kwargs)
        finally:
            self.centers = None

    def row_norms(self, values, squared=False):
        if threading.get_ident() != self.owner or self.centers is None:
            return self.original_norms(values, squared=squared)
        array = np.asarray(values)
        expected = self.centers
        if squared is not True or array.dtype != expected.dtype or array.shape != expected.shape or array.strides != expected.strides or array.ctypes.data != expected.ctypes.data:
            raise ValueError("kmeans_lloyd_norm_operand_binding_changed")
        result = squared_norms(self.context, temporary := [], array)
        record = temporary[0]
        record.update({"adapter_scope": "sklearn_dense_float32_lloyd_centroid_norms", "execution_device": "gpu", "original_lloyd_binary_sha256": LLOYD_BINARY_SHA256, "callback_policy": "temporary_module_hook_owner_thread_and_active_centroid_storage", "host_operand_view_copy_bytes": 0, "python_numeric_norm_computation": False, "lloyd_iteration_replaced": False, "cpu_residuals": ["lloyd_assignments_and_centroid_updates", "empty_cluster_relocation", "inertia", "convergence_control", "host_staging_and_control"]})
        self.records.append(record)
        return result

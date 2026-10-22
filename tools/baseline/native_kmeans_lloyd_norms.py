"""Owner-scoped Metal centroid norms at the pinned Cython Lloyd callback boundary."""

import hashlib
import inspect
from pathlib import Path
import threading

import numpy as np

from tools.baseline.native_kmeans_norms import squared_norms, validate_reference


LLOYD_BINARY_SHA256 = "865e1791855baec7f4a8233cc5792a13a04efcdb4daef57d986c10fe22f3203f"


class LloydNormScope:
    def __init__(self, context, records, assignment_context=None, assignment_records=None):
        import sklearn.cluster._k_means_lloyd as module
        if not isinstance(records, list):
            raise ValueError("kmeans_lloyd_norm_records_required")
        if getattr(context.library, "wxdouble_kmeans_squared_norms", None) is None:
            raise ValueError("kmeans_lloyd_norm_abi_unavailable")
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != LLOYD_BINARY_SHA256:
            raise ValueError("kmeans_lloyd_norm_binary_not_supported")
        validate_reference(module.row_norms)
        if (assignment_context is None) != (assignment_records is None) or (assignment_records is not None and not isinstance(assignment_records, list)):
            raise ValueError("kmeans_lloyd_assignment_configuration_invalid")
        if assignment_context is not None:
            import sklearn.utils._cython_blas as sklearn_blas
            import scipy.linalg.cython_blas as scipy_blas
            if getattr(assignment_context.library, "wxkmeans_lloyd_assignment", None) is None:
                raise ValueError("kmeans_lloyd_assignment_abi_unavailable")
            if hashlib.sha256(Path(sklearn_blas.__file__).read_bytes()).hexdigest() != "3c553a603c9249bfc27ce0c56cd0898609430f46284ea97d605c745e451c902d" or hashlib.sha256(Path(scipy_blas.__file__).read_bytes()).hexdigest() != "e4b6dbd1ae7cdf31b01aecc6320cfa7a6f788ade4f2ec1b4e465adcfe62ab809":
                raise ValueError("kmeans_lloyd_assignment_blas_reference_not_supported")
        self.assignment_context, self.assignment_records = assignment_context, assignment_records
        self.iteration_signature = inspect.signature(module.lloyd_iter_chunked_dense)
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
        bound = None
        if self.assignment_context is not None:
            bound = self.iteration_signature.bind(X, sample_weight, centers_old, *args, **kwargs)
            bound.apply_defaults()
            if not 0 < X.shape[0] <= 256 or not 0 < centers_old.shape[0] <= 256:
                raise ValueError("kmeans_lloyd_assignment_single_chunk_required")
        self.centers = np.asarray(centers_old)
        try:
            if bound is not None and bound.arguments["update_centers"] is False:
                labels = np.asarray(bound.arguments["labels"])
                if labels.dtype != np.int32 or labels.shape != (X.shape[0],) or not labels.flags.c_contiguous or not labels.flags.writeable:
                    raise ValueError("kmeans_lloyd_assignment_label_output_invalid")
                norms = self.row_norms(centers_old, squared=True)
                assigned, distances, record = self.assignment_context.lloyd_assignment(np.asarray(X), self.centers, norms)
                np.copyto(labels, assigned, casting="no")
                record.update({"adapter_scope": "sklearn_dense_float32_lloyd_final_assignment", "original_lloyd_binary_sha256": LLOYD_BINARY_SHA256, "final_assignment_replaced": True, "centroid_norms_device": "gpu", "host_label_copy_bytes": labels.nbytes, "cpu_residuals": ["regular_lloyd_iterations_and_centroid_updates", "empty_cluster_relocation", "inertia", "convergence_control", "host_staging_and_control"]})
                self.assignment_records.append(record)
                return None
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

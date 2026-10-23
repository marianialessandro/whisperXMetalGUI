"""Owner-scoped Metal centroid norms at the pinned Cython Lloyd callback boundary."""

import hashlib
import inspect
from pathlib import Path
import threading
import time

import numpy as np

from tools.baseline.native_kmeans_norms import squared_norms, validate_reference


ARGPARTITION_SHA256 = "d49a7e075e5e7ad2587010a92d97c5b163625e8e487074d7d8359ef4a6f046cd"
LLOYD_BINARY_SHA256 = "865e1791855baec7f4a8233cc5792a13a04efcdb4daef57d986c10fe22f3203f"


class LloydNormScope:
    def __init__(self, context, records, assignment_context=None, assignment_records=None, centroid_records=None, center_shift_records=None, relocation_records=None, two_chunks=False):
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
        if centroid_records is not None:
            import sklearn.cluster._k_means_common as common
            if assignment_context is None or not isinstance(centroid_records, list):
                raise ValueError("kmeans_lloyd_centroids_configuration_invalid")
            if getattr(assignment_context.library, "wxkmeans_lloyd_centroids", None) is None:
                raise ValueError("kmeans_lloyd_centroids_abi_unavailable")
            if hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest() != "793cb26715cc85762eb460e29cce45a4a40584ddcaba761bfd69adafd080f3cf" or "float" not in getattr(common._euclidean_dense_dense_wrapper, "__signatures__", {}):
                raise ValueError("kmeans_lloyd_centroids_shift_reference_not_supported")
            self.original_shift = common._euclidean_dense_dense_wrapper["float"]
        if center_shift_records is not None:
            if centroid_records is None or not isinstance(center_shift_records, list):
                raise ValueError("kmeans_lloyd_center_shifts_configuration_invalid")
            if getattr(assignment_context.library, "wxkmeans_lloyd_center_shifts", None) is None:
                raise ValueError("kmeans_lloyd_center_shifts_abi_unavailable")
        if relocation_records is not None:
            if centroid_records is None or not isinstance(relocation_records, list):
                raise ValueError("kmeans_relocation_configuration_invalid")
            if getattr(assignment_context.library, "wxkmeans_lloyd_relocated_centroids", None) is None:
                raise ValueError("kmeans_relocation_abi_unavailable")
            from tools.baseline.native_kmeans_prefix import NUMPY_CORE_SHA256
            import numpy._core._multiarray_umath as numpy_core
            if hashlib.sha256(Path(numpy_core.__file__).read_bytes()).hexdigest() != NUMPY_CORE_SHA256 or hashlib.sha256(inspect.getsource(np.argpartition).encode()).hexdigest() != ARGPARTITION_SHA256 or np.getbufsize() != 8192:
                raise ValueError("kmeans_relocation_numpy_reference_not_supported")
        self.relocation_records = relocation_records
        if relocation_records is not None:
            self.numpy_core_sha256 = NUMPY_CORE_SHA256
        if type(two_chunks) is not bool or (two_chunks and (relocation_records is None or center_shift_records is None)):
            raise ValueError("kmeans_two_chunks_configuration_invalid")
        if two_chunks:
            if any(getattr(assignment_context.library, name, None) is None for name in ("wxkmeans_lloyd_assignment_two_chunks", "wxkmeans_lloyd_centroids_two_chunks", "wxkmeans_inertia_dense_two_chunks")):
                raise ValueError("kmeans_two_chunks_abi_unavailable")
            from tools.baseline.native_kmeans_inertia import validate_loaded_runtime
            validate_loaded_runtime(assignment_context.library)
        self.two_chunks = two_chunks
        self.center_shift_records = center_shift_records
        self.centroid_records = centroid_records
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
            if not 0 < X.shape[0] <= (512 if self.two_chunks else 256) or not 0 < centers_old.shape[0] <= 256:
                raise ValueError("kmeans_lloyd_two_chunks_sample_limit" if self.two_chunks else "kmeans_lloyd_assignment_single_chunk_required")
        self.centers = np.asarray(centers_old)
        try:
            if bound is not None and bound.arguments["update_centers"] is True and self.centroid_records is not None:
                arguments = bound.arguments
                clusters, features = self.centers.shape
                outputs = [(np.asarray(arguments[name]), shape, dtype) for name, shape, dtype in (("labels", (X.shape[0],), np.int32), ("centers_new", (clusters, features), np.float32), ("weight_in_clusters", (clusters,), np.float32), ("center_shift", (clusters,), np.float32))]
                if any(array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous or not array.flags.writeable for array, shape, dtype in outputs):
                    raise ValueError("kmeans_lloyd_centroids_output_invalid")
                if any(np.shares_memory(left[0], right[0]) for index, left in enumerate(outputs) for right in outputs[index + 1:]):
                    raise ValueError("kmeans_lloyd_centroids_outputs_overlap")
                norms = self.row_norms(centers_old, squared=True)
                started = time.perf_counter_ns()
                assigned, distances, assignment_record = self.assignment_context.lloyd_assignment(np.asarray(X), self.centers, norms, **({"n_threads": bound.arguments["n_threads"]} if self.two_chunks else {}))
                if self.relocation_records is not None:
                    if np.getbufsize() != 8192:
                        raise ValueError("kmeans_relocation_numpy_buffer_not_supported")
                    updated, totals, relocation_distances, selected, record = self.assignment_context.lloyd_relocated_centroids(np.asarray(X), np.asarray(sample_weight), assigned, self.centers, **({"n_threads": bound.arguments["n_threads"]} if self.two_chunks else {}))
                else:
                    updated, totals, record = self.assignment_context.lloyd_centroids(np.asarray(X), np.asarray(sample_weight), assigned, clusters)
                if self.center_shift_records is not None:
                    shifts, shift_record = self.assignment_context.lloyd_center_shifts(updated, self.centers)
                    shift_record.update({"adapter_scope": "sklearn_dense_float32_lloyd_centroid_movement", "original_common_binary_sha256": "793cb26715cc85762eb460e29cce45a4a40584ddcaba761bfd69adafd080f3cf", "cpu_residuals": ["inertia", "convergence_control", "host_staging_and_control"]})
                else:
                    shifts = np.empty(clusters, np.float32)
                    for cluster in range(clusters):
                        shifts[cluster] = self.original_shift(updated[cluster], self.centers[cluster], False)
                if not np.isfinite(shifts).all():
                    raise ValueError("kmeans_lloyd_centroids_shift_result_invalid")
                for (output, shape, dtype), result in zip(outputs, (assigned, updated, totals, shifts)):
                    np.copyto(output, result, casting="no")
                record.update({"adapter_scope": "sklearn_dense_float32_lloyd_regular_assignment_and_centroids", "original_lloyd_binary_sha256": LLOYD_BINARY_SHA256, "assignment_backend": assignment_record["backend"], "assignment_native_wall_ns": assignment_record["native_wall_ns"], "centroid_completion_and_bridge_wall_ns": record["completion_and_bridge_wall_ns"], "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "centroid_host_input_snapshot_bytes": record["host_input_snapshot_bytes"], "host_input_snapshot_bytes": assignment_record["host_input_snapshot_bytes"] + record["host_input_snapshot_bytes"], "centroid_host_result_bytes": record["host_result_bytes"], "host_result_bytes": assignment_record["host_result_bytes"] + record["host_result_bytes"], "centroid_native_wall_ns": record["native_wall_ns"], "native_wall_ns": assignment_record["native_wall_ns"] + record["native_wall_ns"], "native_blas_calls": assignment_record["native_blas_calls"], "assignment_host_input_snapshot_bytes": assignment_record["host_input_snapshot_bytes"], "assignment_host_result_bytes": assignment_record["host_result_bytes"], "center_shift_backend": "original_cython_euclidean_dense_dense_float", "center_shift_calls": clusters, "host_output_copy_bytes": sum(output.nbytes for output, shape, dtype in outputs), "centroid_norms_device": "gpu", "cpu_residuals": ["cython_center_shift", "inertia", "convergence_control", "host_staging_and_control"], "unsupported_operations": ["empty_cluster_relocation", "multiple_chunk_merge"]})
                if self.center_shift_records is not None:
                    record.update({"center_shift_backend": shift_record["backend"], "center_shift_calls": 1, "center_shift_rows": clusters, "cpu_residuals": ["inertia", "convergence_control", "host_staging_and_control"]})
                    self.center_shift_records.append(shift_record)
                if self.relocation_records is not None:
                    record["unsupported_operations"] = ["three_or_more_chunks"] if self.two_chunks else ["multiple_chunk_merge"]
                    record["numpy_core_binary_sha256"] = self.numpy_core_sha256
                    shared = {key: record[key] for key in ("scope", "execution_device", "samples", "clusters", "features", "empty_clusters", "relocated_clusters", "selection_calls", "distance_rows", "python_numeric_relocation", "python_argpartition", "empty_cluster_relocation_replaced", "gpu_kernels", "command_buffers_per_run", "computation_gpu_seconds", "logical_input_upload_bytes", "logical_result_readback_bytes", "quality_certified", "full_pipeline_gpu_certified", "production_backend", "numpy_core_binary_sha256")}
                    shared.update({"adapter_scope": "sklearn_dense_float32_lloyd_empty_relocation", "backend": "shared_native_cpu_relocated_centroid_update", "shared_centroid_update": True, "shared_operation_index": len(self.centroid_records), "native_wall_ns": 0, "completion_and_bridge_wall_ns": 0, "host_input_snapshot_bytes": 0, "host_result_bytes": 0, "native_blas_calls": 0})
                    self.relocation_records.append(shared)
                self.centroid_records.append(record)
                return None
            if bound is not None and bound.arguments["update_centers"] is False:
                labels = np.asarray(bound.arguments["labels"])
                if labels.dtype != np.int32 or labels.shape != (X.shape[0],) or not labels.flags.c_contiguous or not labels.flags.writeable:
                    raise ValueError("kmeans_lloyd_assignment_label_output_invalid")
                norms = self.row_norms(centers_old, squared=True)
                assigned, distances, record = self.assignment_context.lloyd_assignment(np.asarray(X), self.centers, norms, **({"n_threads": bound.arguments["n_threads"]} if self.two_chunks else {}))
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

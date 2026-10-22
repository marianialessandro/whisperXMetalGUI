"""One resident command buffer for bounded Euclidean centroid AHC and inverse labels."""

import numpy as np

from tools.baseline.native_ahc_distances import LINKAGE_SHA256, PDIST_SHA256
from tools.baseline.native_ahc_cut import FCLUSTER_SHA256
from tools.baseline.native_ahc_linkage import HEAP_REFERENCE_SHA256, UPDATE_REFERENCE_SHA256
from tools.baseline.native_ahc_reindex import UNIQUE_SHA256
from tools.baseline.native_embedding_adapter import VBX_CALL_SHA256


def resident_pipeline_labels(context, records, distances, linkage, cuts, reindex, values, cutoff):
    if type(values) is not np.ndarray or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or values.ndim != 2:
        raise ValueError("ahc_resident_operand_not_supported")
    staged = np.ascontiguousarray(values, dtype=np.float64)
    output, record = context.ahc_resident(staged, cutoff)
    record = dict(record)
    record.update({"adapter_scope": "vbx_resident_centroid_ahc", "original_call_sha256": VBX_CALL_SHA256, "pdist_source_sha256": PDIST_SHA256, "linkage_source_sha256": LINKAGE_SHA256, "fcluster_source_sha256": FCLUSTER_SHA256, "numpy_unique_source_sha256": UNIQUE_SHA256, "heap_reference_sha256": HEAP_REFERENCE_SHA256, "update_reference_sha256": UPDATE_REFERENCE_SHA256, "feature_dtype_conversion": str(values.dtype) != "float64", "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, values) else staged.nbytes, "cpu_residuals": ["input_and_final_label_validation", "host_staging_and_control"]})
    dispatch = len(records)
    records.append(record)
    for destination, stage, kernels in ((distances, "euclidean_distances", ["double_pdist_euclidean"]), (linkage, "centroid_linkage", ["double_resident_centroid_linkage"]), (cuts, "distance_cut", ["double_resident_fcluster"]), (reindex, "label_inverse", ["integer_resident_cut_labels_i64"])):
        destination.append({"adapter_scope": "vbx_resident_ahc_stage", "stage": stage, "original_call_sha256": VBX_CALL_SHA256, "resident_pipeline_reuse": True, "shared_ahc_dispatch_index": dispatch, "resident_reindex": stage == "distance_cut", "resident_label_reuse": stage == "label_inverse", "reindex_policy": "identity_for_contiguous_native_cluster_ids", "count": output.size, "gpu_kernels": kernels, "command_buffers_per_run": 0, "computation_gpu_seconds": None, "logical_input_upload_bytes": 0, "logical_result_readback_bytes": 0, "cpu_residuals": ["host_staging_and_control"], "quality_certified": False, "full_pipeline_gpu_certified": False})
    return output

"""In-place bridge for the source-pinned dense KMeans++ candidate minimum."""

import numpy as np

from tools.baseline.native_kmeans_prefix import NUMPY_CORE_SHA256, PLUSPLUS_SHA256


def candidate_minimum(context, records, closest, candidates):
    if type(candidates) is not np.ndarray or candidates.dtype != np.float32 or candidates.ndim != 2 or not candidates.flags.c_contiguous or not candidates.flags.writeable:
        raise ValueError("kmeans_candidate_minimum_writable_array_required")
    output, record = context.kmeans_candidate_minimum(closest, candidates)
    np.copyto(candidates, output, casting="no")
    record = dict(record)
    record.update({"adapter_scope": "sklearn_dense_kmeans_plusplus_inplace_candidate_minimum", "original_plusplus_sha256": PLUSPLUS_SHA256, "numpy_core_binary_sha256": NUMPY_CORE_SHA256, "host_inplace_copy_bytes": output.nbytes, "host_copy_is_numeric_conversion": False, "cpu_residuals": ["rng", "candidate_searchsorted_and_clipping", "distance_and_potential_computation", "best_candidate_selection", "lloyd_or_elkan", "host_staging_and_control"]})
    records.append(record)
    return candidates

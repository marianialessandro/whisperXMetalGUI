"""Source-pinned resident inactive-speaker preparation and constrained assignment."""

import warnings

import numpy as np

from tools.baseline.native_assignment import SOLVER_BINARY_SHA256
from tools.baseline.native_constrained_cleanup import CONSTRAINED_SHA256


def resident_assignment(context, records, scores, segments):
    if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 3 or type(segments) is not np.ndarray or segments.ndim != 3 or segments.dtype.kind not in "biuf" or segments.dtype.itemsize > 8:
        raise ValueError("resident_assignment_operand_format_not_supported")
    chunks, rows, columns = scores.shape
    if segments.shape[0] != chunks or segments.shape[2] != rows:
        raise ValueError("resident_assignment_segment_shape_not_supported")
    staged_scores = np.ascontiguousarray(scores)
    staged_segments = np.ascontiguousarray(segments.transpose(0, 2, 1), dtype=np.float64)
    try:
        result, record = context.inactive_assignment(staged_scores, staged_segments)
    except ValueError as failure:
        if str(failure) == "double_inactive_assignment_all_nan_input":
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        raise
    record = dict(record)
    record.update({"adapter_scope": "vbx_resident_inactive_mask_cleanup_assignment_and_public_soft_scores", "original_constrained_sha256": CONSTRAINED_SHA256, "scipy_solver_binary_sha256": SOLVER_BINARY_SHA256, "segment_source_dtype": str(segments.dtype), "host_score_staging_copy_bytes": 0 if np.shares_memory(staged_scores, scores) else staged_scores.nbytes, "host_segment_staging_copy_bytes": 0 if np.shares_memory(staged_segments, segments) else staged_segments.nbytes, "cpu_residuals": ["input_validation", "host_output_buffer_allocation", "host_staging_and_control", "ahc", "optional_kmeans"]})
    records.append(record)
    return result

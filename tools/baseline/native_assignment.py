"""Bounded SciPy-policy rectangular maximization using native Metal binary64."""

import contextlib
import hashlib
import importlib.metadata
from pathlib import Path
import warnings

import numpy as np
import scipy.optimize._lsap as solver_module


SOLVER_BINARY_SHA256 = "f0a881da43e09c1df23f3761d5e3a7adbcfe3ec6ea60cb7347ff2a69d5154103"
REFERENCE_SOURCE_SHA256 = "73d78155990732311a116bc04f7f64d874c0f3793172e9aab12a476cb3d15c2c"


def validate_assignment_reference(original):
    if importlib.metadata.version("scipy") != "1.15.3" or original is not solver_module.linear_sum_assignment or hashlib.sha256(Path(solver_module.__file__).read_bytes()).hexdigest() != SOLVER_BINARY_SHA256:
        raise ValueError("constrained_assignment_reference_not_supported")


def assignment_pairs(context, records, scores):
    from tools.baseline.native_constrained_cleanup import CONSTRAINED_SHA256

    if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 2:
        raise ValueError("constrained_assignment_operand_format_not_supported")
    staged = np.ascontiguousarray(scores)
    output, record = context.linear_assignment_max(staged)
    record = dict(record)
    record.update({"adapter_scope": "base_constrained_rectangular_assignment", "original_constrained_sha256": CONSTRAINED_SHA256, "scipy_solver_binary_sha256": SOLVER_BINARY_SHA256, "scipy_reference_source_sha256": REFERENCE_SOURCE_SHA256, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, scores) else staged.nbytes, "cpu_residuals": ["initial_output_allocation", "label_scatter", "host_staging_and_control"]})
    records.append(record)
    return output


def batch_labels(context, records, scores):
    from tools.baseline.native_constrained_cleanup import CONSTRAINED_SHA256

    if type(scores) is not np.ndarray or scores.dtype != np.float64 or scores.ndim != 3:
        raise ValueError("constrained_batch_operand_format_not_supported")
    staged = np.ascontiguousarray(scores)
    try:
        output, record = context.assignment_batch(staged)
    except ValueError as failure:
        if str(failure) == "double_assignment_batch_all_nan_input":
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        raise
    record = dict(record)
    record.update({"adapter_scope": "base_resident_nan_cleanup_assignment_and_label_scatter", "original_constrained_sha256": CONSTRAINED_SHA256, "scipy_solver_binary_sha256": SOLVER_BINARY_SHA256, "scipy_reference_source_sha256": REFERENCE_SOURCE_SHA256, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, scores) else staged.nbytes, "cpu_residuals": ["input_validation", "host_output_buffer_allocation", "host_staging_and_control"]})
    records.append(record)
    return output


@contextlib.contextmanager
def native_batched_assignment(context, records):
    from tools.baseline.native_constrained_cleanup import native_constrained_nan_cleanup

    with native_constrained_nan_cleanup(context, [], batch_records=records):
        yield

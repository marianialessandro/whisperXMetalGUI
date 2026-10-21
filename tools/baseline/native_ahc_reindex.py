"""Bounded native integer replacement for the unused-unique AHC inverse mapping."""

import hashlib
import inspect

import numpy as np


UNIQUE_SHA256 = "482674c7bd0083e5334fdbafbf2c3a488df4ecb057a4512fecf6da19740f21a0"


def validate_reindex_reference(original):
    import numpy.lib._arraysetops_impl as arraysetops

    if np.__version__ != "2.2.6" or np.dtype(np.intp).itemsize != 8 or original.__globals__.get("np") is not np or np.unique is not arraysetops.unique or hashlib.sha256(inspect.getsource(inspect.unwrap(np.unique)).encode()).hexdigest() != UNIQUE_SHA256:
        raise ValueError("ahc_reindex_reference_not_supported")


def reindex_labels(context, records, labels):
    if type(labels) is not np.ndarray or labels.ndim != 1 or labels.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise ValueError("ahc_reindex_operand_format_not_supported")
    staged = np.ascontiguousarray(labels, dtype=np.int64)
    output, record = context.ahc_reindex(staged)
    record = dict(record)
    record.update({"adapter_scope": "vbx_ahc_numpy_unique_return_inverse", "numpy_unique_source_sha256": UNIQUE_SHA256, "source_dtype": str(labels.dtype), "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, labels) else staged.nbytes, "cpu_residuals": ["centroid_linkage", "fcluster", "host_staging_and_control"]})
    records.append(record)
    return output

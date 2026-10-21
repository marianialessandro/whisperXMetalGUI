"""Bounded Euclidean condensed distances before the original SciPy AHC algorithm."""

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np


PDIST_SHA256 = "d607d7527f5cb4d15379ae903969bbfe13f95545e96df8f397f23d89b47b1795"
PDIST_BINARY_SHA256 = "d266b23113875b5f72d454ad9b39bd80f91f40cfb0e5ab4d5bd4e2b22991ad03"
LINKAGE_SHA256 = "dc3365e0f8faaf78258b9b38fe98350426c0fe1b2eb74d94f67ada04c5b7dcbd"


def validate_ahc_reference(original):
    import scipy.cluster.hierarchy as hierarchy
    import scipy.spatial.distance as distance
    import scipy.spatial._distance_pybind as native_distance

    if importlib.metadata.version("scipy") != "1.15.3" or np.__version__ != "2.2.6" or original.__globals__.get("linkage") is not hierarchy.linkage or hierarchy.linkage.__globals__.get("distance") is not distance:
        raise ValueError("ahc_distance_runtime_not_supported")
    if hashlib.sha256(inspect.getsource(hierarchy.linkage).encode()).hexdigest() != LINKAGE_SHA256 or hashlib.sha256(inspect.getsource(distance.pdist).encode()).hexdigest() != PDIST_SHA256 or hashlib.sha256(Path(native_distance.__file__).read_bytes()).hexdigest() != PDIST_BINARY_SHA256 or distance._METRIC_ALIAS["euclidean"].pdist_func is not native_distance.pdist_euclidean:
        raise ValueError("ahc_distance_reference_not_supported")


def condensed_distances(context, records, values):
    if type(values) is not np.ndarray or values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or values.ndim != 2:
        raise ValueError("ahc_distance_operand_format_not_supported")
    staged = np.ascontiguousarray(values, dtype=np.float64)
    output, record = context.pdist_euclidean(staged)
    record = dict(record)
    record.update({"adapter_scope": "vbx_euclidean_pairwise_distances_before_original_centroid_linkage", "pdist_source_sha256": PDIST_SHA256, "pdist_binary_sha256": PDIST_BINARY_SHA256, "linkage_source_sha256": LINKAGE_SHA256, "source_dtype": str(values.dtype), "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, values) else staged.nbytes, "cpu_residuals": ["original_centroid_linkage", "original_fcluster", "host_staging_and_control"]})
    records.append(record)
    return output

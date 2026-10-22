"""Bounded source-pinned centroid dendrogram construction on native Metal."""

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np

from tools.baseline.native_ahc_cut import HIERARCHY_BINARY_SHA256, HIERARCHY_REFERENCE_SHA256
from tools.baseline.native_ahc_distances import LINKAGE_SHA256


HEAP_REFERENCE_SHA256 = "1bf594add2b5f92d07a0649cf8fd54ae5b2352a75ce07dcc462d82a821493e3d"
UPDATE_REFERENCE_SHA256 = "64d6c695524fc08fc1d4c9679bd4424b28c15232b9776df4224ce432b8ee725f"


def validate_linkage_reference(original):
    import scipy.cluster.hierarchy as hierarchy
    import scipy.cluster._hierarchy as native_hierarchy

    if importlib.metadata.version("scipy") != "1.15.3" or np.__version__ != "2.2.6" or original.__globals__.get("linkage") is not hierarchy.linkage or hierarchy.linkage.__globals__.get("_hierarchy") is not native_hierarchy or hashlib.sha256(inspect.getsource(hierarchy.linkage).encode()).hexdigest() != LINKAGE_SHA256 or hashlib.sha256(Path(native_hierarchy.__file__).read_bytes()).hexdigest() != HIERARCHY_BINARY_SHA256 or type(native_hierarchy.fast_linkage) is not type(native_hierarchy.cluster_dist) or native_hierarchy.fast_linkage.__module__ != "scipy.cluster._hierarchy" or native_hierarchy.fast_linkage.__name__ != "fast_linkage":
        raise ValueError("ahc_linkage_reference_not_supported")


def centroid_linkage(context, records, distances, method="centroid", metric="euclidean"):
    if method != "centroid" or metric != "euclidean" or type(distances) is not np.ndarray or distances.dtype != np.float64 or distances.ndim != 1:
        raise ValueError("ahc_linkage_operand_or_policy_not_supported")
    staged = np.ascontiguousarray(distances)
    output, record = context.centroid_linkage(staged)
    record = dict(record)
    record.update({"adapter_scope": "vbx_original_centroid_linkage_call", "linkage_source_sha256": LINKAGE_SHA256, "hierarchy_binary_sha256": HIERARCHY_BINARY_SHA256, "hierarchy_reference_sha256": HIERARCHY_REFERENCE_SHA256, "heap_reference_sha256": HEAP_REFERENCE_SHA256, "update_reference_sha256": UPDATE_REFERENCE_SHA256, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, distances) else staged.nbytes, "cpu_residuals": ["input_validation", "output_tree_structure_validation", "host_staging_and_control"]})
    records.append(record)
    return output

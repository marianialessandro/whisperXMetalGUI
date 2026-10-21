"""Source-pinned bounded distance-criterion flat clustering on Metal."""

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np


FCLUSTER_SHA256 = "e997611514ac25f42f2e7ba018a956a4a3b4f8cb2e0377fda5837843474c20e9"
HIERARCHY_REFERENCE_SHA256 = "50141ab68a04ee82d51bbba906afdf29a7a79877223f9e89699500e926dabf9f"
HIERARCHY_BINARY_SHA256 = "ca3839369a95113ae7b6d0da6134f06b70ed8bf819def1a39ce9439ce5b90edf"


def validate_cut_reference(original):
    import scipy.cluster.hierarchy as hierarchy
    import scipy.cluster._hierarchy as native_hierarchy

    if importlib.metadata.version("scipy") != "1.15.3" or np.__version__ != "2.2.6" or original.__globals__.get("fcluster") is not hierarchy.fcluster or hierarchy.fcluster.__globals__.get("_hierarchy") is not native_hierarchy or hashlib.sha256(inspect.getsource(hierarchy.fcluster).encode()).hexdigest() != FCLUSTER_SHA256 or hashlib.sha256(Path(native_hierarchy.__file__).read_bytes()).hexdigest() != HIERARCHY_BINARY_SHA256:
        raise ValueError("ahc_cut_reference_not_supported")


def cut_tree(context, records, tree, cutoff):
    if type(tree) is not np.ndarray or tree.dtype != np.float64 or tree.ndim != 2:
        raise ValueError("ahc_cut_operand_format_not_supported")
    staged = np.ascontiguousarray(tree)
    output, record = context.fcluster_distance(staged, cutoff, zero_based=True)
    record = dict(record)
    record.update({"adapter_scope": "vbx_distance_fcluster_and_zero_based_label_offset", "fcluster_source_sha256": FCLUSTER_SHA256, "hierarchy_binary_sha256": HIERARCHY_BINARY_SHA256, "hierarchy_reference_sha256": HIERARCHY_REFERENCE_SHA256, "host_input_staging_copy_bytes": 0 if np.shares_memory(staged, tree) else staged.nbytes, "cpu_residuals": ["centroid_linkage", "tree_structure_validation", "host_staging_and_control"]})
    records.append(record)
    return output

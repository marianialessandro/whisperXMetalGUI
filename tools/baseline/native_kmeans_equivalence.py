"""Pinned native CPU replacement for sklearn's directed clustering-label mapping."""

import hashlib
import importlib.metadata
from pathlib import Path


COMMON_BINARY_SHA256 = "793cb26715cc85762eb460e29cce45a4a40584ddcaba761bfd69adafd080f3cf"


def validate_reference(original):
    import sklearn.cluster._k_means_common as common
    if importlib.metadata.version("scikit-learn") != "1.9.1" or original is not common._is_same_clustering or hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest() != COMMON_BINARY_SHA256:
        raise ValueError("kmeans_same_clustering_reference_not_supported")


def same_clustering(context, records, labels, previous, clusters):
    import numpy as np
    if isinstance(clusters, np.integer) and not isinstance(clusters, np.bool_):
        clusters = int(clusters)
    output, record = context.same_clustering(labels, previous, clusters)
    record.update({"adapter_scope": "sklearn_dense_kmeans_best_initialization_mapping", "original_common_binary_sha256": COMMON_BINARY_SHA256, "mapping_policy": "directed_label_map_no_added_bijection_requirement", "cpu_residuals": ["inertia_comparison_and_fit_control", "lloyd_or_elkan_iteration_and_centroids", "host_staging_and_control"]})
    records.append(record)
    return output

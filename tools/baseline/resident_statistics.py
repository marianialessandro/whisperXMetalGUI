"""Shared CPU provenance and CSR metadata preparation for resident native statistics."""

import numpy as np


def resident_statistics_plan(values, indices, offsets, modes, snapshot):
    if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 3 or values.shape[0] > 32768 or not isinstance(indices, np.ndarray) or indices.dtype != np.uint32 or indices.ndim != 1 or indices.size > 4 * 1024 * 1024 or not isinstance(offsets, np.ndarray) or offsets.dtype != np.uint32 or offsets.ndim != 1 or not 2 <= offsets.size <= 32769 or not isinstance(modes, np.ndarray) or modes.dtype != np.uint32 or modes.ndim != 1 or modes.size != offsets.size - 1: raise ValueError("native_alignment_stats_format_invalid")
    if np.__version__ != "2.2.6" or np.getbufsize() != 8192: raise ValueError("native_word_means_runtime_not_supported")
    if snapshot is None: raise ValueError("native_resident_statistics_snapshot_missing")
    generation, count, ratio, offset, reference = snapshot
    missing = np.isnan(values).all(axis=1)
    selected = np.ascontiguousarray(values[~missing])
    if selected.shape[0] != count or selected.tobytes() != reference: raise ValueError("native_resident_statistics_snapshot_changed")
    if indices.size and int(indices.max()) >= len(values): raise ValueError("native_resident_statistics_indices_invalid")
    mapping = np.full(len(values), count, dtype=np.uint32)
    mapping[~missing] = np.arange(count, dtype=np.uint32)
    mapped, offsets, modes = np.ascontiguousarray(mapping[indices]), np.ascontiguousarray(offsets), np.ascontiguousarray(modes)
    return generation, count, ratio, offset, reference, selected, mapping, mapped, offsets, modes

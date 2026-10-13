"""Scoped development-only default nearest interpolation on Metal with pinned Pandas/SciPy policy."""

import contextlib
import numpy as np
import pandas as pd
import scipy


@contextlib.contextmanager
def native_nearest_interpolation(alignment, context, records):
    if pd.__version__ != "3.0.6" or scipy.__version__ != "1.15.3" or not isinstance(records, list): raise ValueError("native_nearest_runtime_or_records_not_supported")
    original = alignment.interpolate_nans
    def interpolate(series, method="nearest"):
        if method == "ignore": return series
        if method != "nearest": raise ValueError("native_nearest_method_not_supported")
        if not isinstance(series, pd.Series) or len(series) > 32768 or not series.index.equals(pd.RangeIndex(len(series))): raise ValueError("native_nearest_series_or_index_not_supported")
        if len(series) == 0 or (series.dtype == np.dtype("object") and all(value is None for value in series)):
            records.append({"scope": "nearest_empty_metadata_only", "elements": len(series), "dispatch_completed": False, "command_buffers_per_run": 0, "logical_cpu_to_native_bytes": 0, "logical_output_readback_bytes": 0})
            return series.copy()
        if series.dtype != np.dtype("float64"): raise ValueError("native_nearest_dtype_not_supported")
        values, record = context.interpolate_nearest(series.to_numpy(dtype=np.float64))
        records.append(record)
        return series._constructor(values, index=series.index, name=series.name).__finalize__(series)
    alignment.interpolate_nans = interpolate
    try:
        yield
    finally:
        alignment.interpolate_nans = original

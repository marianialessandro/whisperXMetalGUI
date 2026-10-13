"""Scoped development-only default nearest interpolation on Metal with pinned Pandas/SciPy policy."""

import contextlib
import numpy as np
import pandas as pd
import scipy


@contextlib.contextmanager
def native_nearest_interpolation(alignment, context, records, batched_frames=False):
    if pd.__version__ != "3.0.6" or scipy.__version__ != "1.15.3" or not isinstance(records, list): raise ValueError("native_nearest_runtime_or_records_not_supported")
    original, original_pd = alignment.interpolate_nans, alignment.pd
    def interpolate(series, method="nearest"):
        if method == "ignore": return series
        if method != "nearest": raise ValueError("native_nearest_method_not_supported")
        if not isinstance(series, pd.Series) or len(series) > 32768 or not series.index.equals(pd.RangeIndex(len(series))): raise ValueError("native_nearest_series_or_index_not_supported")
        if len(series) == 0 or (series.dtype == np.dtype("object") and all(value is None for value in series)):
            records.append({"scope": "nearest_empty_metadata_only", "elements": len(series), "dispatch_completed": False, "command_buffers_per_run": 0, "logical_cpu_to_native_bytes": 0, "logical_output_readback_bytes": 0})
            return series.copy()
        if series.dtype != np.dtype("float64"): raise ValueError("native_nearest_dtype_not_supported")
        cache = getattr(series, "_native_nearest_pair_cache", None)
        if batched_frames and cache is not None:
            columns, index, record = cache
            if series.name not in columns or tuple(series.index) != index or series.to_numpy(dtype=np.float64).tobytes() != columns[series.name][0]: raise ValueError("native_nearest_pair_snapshot_changed")
            record["columns_consumed"] += 1
            return series._constructor(columns[series.name][1], index=series.index, name=series.name, copy=True).__finalize__(series)
        values, record = context.interpolate_nearest(series.to_numpy(dtype=np.float64))
        records.append(record)
        return series._constructor(values, index=series.index, name=series.name).__finalize__(series)
    class InterpolationSeries(pd.Series):
        _metadata = [*pd.Series._metadata, "_native_nearest_pair_cache"]

        @property
        def _constructor(self): return InterpolationSeries

    class InterpolationFrame(pd.DataFrame):
        _metadata = [*pd.DataFrame._metadata, "_native_nearest_pair_cache"]

        @property
        def _constructor(self): return InterpolationFrame

        @property
        def _constructor_sliced(self): return InterpolationSeries

    class PandasProxy:
        def DataFrame(self, data=None, *args, **kwargs):
            subsegments = isinstance(data, list) and bool(data) and all(isinstance(row, dict) and {"text", "words", "start", "end"}.issubset(row) for row in data)
            if not subsegments: return original_pd.DataFrame(data, *args, **kwargs)
            if args or kwargs or len(data) > 32768: raise ValueError("native_nearest_pair_frame_not_supported")
            frame = InterpolationFrame(data)
            if any(frame[column].dtype != np.dtype("float64") for column in ("start", "end")): return original_pd.DataFrame(data)
            values = np.column_stack([frame[column].to_numpy(dtype=np.float64) for column in ("start", "end")])
            filled, record = context.interpolate_nearest_pair(values)
            record.update(scope="experimental_nearest_subsegment_pair", columns_consumed=0)
            records.append(record)
            frame._native_nearest_pair_cache = {column: (values[:, i].tobytes(), filled[:, i]) for i, column in enumerate(("start", "end"))}, tuple(frame.index), record
            return frame

        def __getattr__(self, name): return getattr(original_pd, name)

    if batched_frames: alignment.pd = PandasProxy()
    alignment.interpolate_nans = interpolate
    try:
        yield
    finally:
        alignment.interpolate_nans = original
        if batched_frames: alignment.pd = original_pd

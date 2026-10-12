"""Scoped alignment-only Pandas adapter for native binary64 timestamp extrema; means/interpolation remain original."""

import contextlib
import numpy as np


@contextlib.contextmanager
def native_timestamp_bounds(alignment, context, records):
    original = alignment.pd

    class TimestampSeries(original.Series):
        @property
        def _constructor(self): return TimestampSeries

        def _bounds(self, method, axis, skipna, numeric_only, kwargs):
            if self.name not in ("start", "end"):
                return getattr(super(), method)(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
            if axis not in (None, 0, "index") or skipna is not True or numeric_only is not False or kwargs: raise ValueError("native_timestamp_reduction_options_not_supported")
            if self.dtype != np.dtype("float64") and not (self.dtype == np.dtype("object") and all(value is None for value in self)): raise ValueError("native_timestamp_series_dtype_not_supported")
            values, record = context.timestamp_bounds(self.to_numpy(dtype=np.float64))
            records.append({**record, "column": self.name, "reduction": method})
            return values[0 if method == "min" else 1]

        def min(self, axis=None, skipna=True, numeric_only=False, **kwargs): return self._bounds("min", axis, skipna, numeric_only, kwargs)

        def max(self, axis=None, skipna=True, numeric_only=False, **kwargs): return self._bounds("max", axis, skipna, numeric_only, kwargs)

    class CharacterFrame(original.DataFrame):
        @property
        def _constructor(self): return CharacterFrame

        @property
        def _constructor_sliced(self): return TimestampSeries

    class PandasProxy:
        def DataFrame(self, data=None, *args, **kwargs):
            character_rows = isinstance(data, list) and bool(data) and isinstance(data[0], dict) and {"char", "start", "end", "score", "word-idx"}.issubset(data[0])
            return (CharacterFrame if character_rows else original.DataFrame)(data, *args, **kwargs)

        def __getattr__(self, name): return getattr(original, name)

    alignment.pd = PandasProxy()
    try:
        yield
    finally:
        alignment.pd = original

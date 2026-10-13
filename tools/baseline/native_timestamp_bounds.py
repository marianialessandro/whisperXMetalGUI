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


@contextlib.contextmanager
def native_timestamp_bounds_batched(alignment, context, records, fast_frames=False):
    original, original_load = alignment.pd, alignment.nltk_load
    if fast_frames and original.__version__ != "3.0.6": raise ValueError("native_timestamp_fast_frames_runtime_not_supported")
    sentence_spans = {}

    class TokenizerProxy:
        def __init__(self, tokenizer): self.tokenizer = tokenizer

        def span_tokenize(self, text, *args, **kwargs):
            spans = list(self.tokenizer.span_tokenize(text, *args, **kwargs))
            sentence_spans[text] = spans
            return iter(spans)

        def __getattr__(self, name): return getattr(self.tokenizer, name)

    def load_tokenizer(*args, **kwargs): return TokenizerProxy(original_load(*args, **kwargs))

    class BatchSeries(original.Series):
        _metadata = [*original.Series._metadata, "_native_batch_cache"]

        @property
        def _constructor(self): return BatchSeries

        def _constructor_from_mgr(self, mgr, axes):
            if not fast_frames: return super()._constructor_from_mgr(mgr, axes)
            series = BatchSeries._from_mgr(mgr, axes)
            series._name = None
            return series

        def _bounds(self, method, axis, skipna, numeric_only, kwargs):
            if self.name not in ("start", "end"): return getattr(super(), method)(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
            if axis not in (None, 0, "index") or skipna is not True or numeric_only is not False or kwargs: raise ValueError("native_timestamp_reduction_options_not_supported")
            if self.dtype != np.dtype("float64") and not (self.dtype == np.dtype("object") and all(value is None for value in self)): raise ValueError("native_timestamp_series_dtype_not_supported")
            cache = getattr(self, "_native_batch_cache", None)
            key = self.name, tuple(self.index)
            if cache is None or key not in cache[0]: raise ValueError("native_timestamp_batch_subset_not_registered")
            bounds, snapshot = cache[0][key]
            if self.to_numpy(dtype=np.float64).tobytes() != snapshot: raise ValueError("native_timestamp_batch_snapshot_changed")
            cache[1]["queries_consumed"] += 1
            return bounds[0 if method == "min" else 1]

        def min(self, axis=None, skipna=True, numeric_only=False, **kwargs): return self._bounds("min", axis, skipna, numeric_only, kwargs)

        def max(self, axis=None, skipna=True, numeric_only=False, **kwargs): return self._bounds("max", axis, skipna, numeric_only, kwargs)

    class BatchFrame(original.DataFrame):
        _metadata = [*original.DataFrame._metadata, "_native_batch_cache"]

        @property
        def _constructor(self): return BatchFrame

        @property
        def _constructor_sliced(self): return BatchSeries

        def _constructor_from_mgr(self, mgr, axes):
            if not fast_frames: return super()._constructor_from_mgr(mgr, axes)
            return BatchFrame._from_mgr(mgr, axes=axes)

        def _constructor_sliced_from_mgr(self, mgr, axes):
            if not fast_frames: return super()._constructor_sliced_from_mgr(mgr, axes)
            series = BatchSeries._from_mgr(mgr, axes)
            series._name = None
            return series

    def prepare(data):
        text = "".join(row["char"] for row in data)
        if text not in sentence_spans or len(data) > 32768: raise ValueError("native_timestamp_batch_sentence_metadata_missing")
        values = np.asarray([[row["start"], row["end"]] for row in data], dtype=np.float64).reshape(-1)
        queries = {}
        def register(column, rows): queries.setdefault((column, tuple(rows)), None)
        for start, end in sentence_spans[text]:
            rows = [i for i in range(len(data)) if start <= i <= end]
            register("start", rows)
            register("end", [i for i in rows if data[i]["char"] != " "])
            words = {}
            for i in rows: words.setdefault(data[i]["word-idx"], []).append(i)
            for word_rows in words.values():
                if not "".join(data[i]["char"] for i in word_rows).strip(): continue
                selected = [i for i in word_rows if data[i]["char"] != " "]
                register("start", selected); register("end", selected)
        if not queries: raise ValueError("native_timestamp_batch_queries_missing")
        indices, offsets, snapshots = [], [0], []
        for column, rows in queries:
            selection = [row * 2 + (column == "end") for row in rows]
            indices.extend(selection); offsets.append(len(indices))
            snapshots.append(values[selection].tobytes())
        bounds, record = context.timestamp_bounds_batch(values, np.asarray(indices, dtype=np.uint32), np.asarray(offsets, dtype=np.uint32))
        record["queries_consumed"] = 0
        record["fast_frames"] = fast_frames
        records.append(record)
        cache = {key: (bounds[i], snapshots[i]) for i, key in enumerate(queries)}
        frame = BatchFrame(data)
        frame._native_batch_cache = cache, record
        return frame

    class PandasProxy:
        def DataFrame(self, data=None, *args, **kwargs):
            character_rows = isinstance(data, list) and bool(data) and isinstance(data[0], dict) and {"char", "start", "end", "score", "word-idx"}.issubset(data[0])
            if not character_rows: return original.DataFrame(data, *args, **kwargs)
            if args or kwargs: raise ValueError("native_timestamp_batch_frame_options_not_supported")
            return prepare(data)

        def __getattr__(self, name): return getattr(original, name)

    alignment.pd, alignment.nltk_load = PandasProxy(), load_tokenizer
    try:
        yield
    finally:
        alignment.pd, alignment.nltk_load = original, original_load

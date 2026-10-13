"""Scoped alignment-only Pandas adapters for native timestamp extrema and optional binary64 word score means."""

import contextlib
import time
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
def native_timestamp_bounds_batched(alignment, context, records, fast_frames=False, word_means=False, mean_records=None, fused_reductions=False, compact_plan=False):
    original, original_load = alignment.pd, alignment.nltk_load
    if fast_frames and original.__version__ != "3.0.6": raise ValueError("native_timestamp_fast_frames_runtime_not_supported")
    if word_means and (original.__version__ != "3.0.6" or np.__version__ != "2.2.6" or np.getbufsize() != 8192 or not isinstance(mean_records, list)): raise ValueError("native_word_means_runtime_or_records_not_supported")
    if fused_reductions and not word_means: raise ValueError("native_fused_reductions_require_word_means")
    if compact_plan and not fused_reductions: raise ValueError("native_compact_plan_requires_fused_reductions")
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
        _metadata = [*original.Series._metadata, "_native_batch_cache", "_native_word_mean_cache"]

        @property
        def _constructor(self): return BatchSeries

        def _constructor_from_mgr(self, mgr, axes):
            if not fast_frames: return super()._constructor_from_mgr(mgr, axes)
            series = BatchSeries._from_mgr(mgr, axes)
            series._name = None
            return series

        def _bounds(self, method, axis, skipna, numeric_only, kwargs):
            if fused_reductions and self.name in ("start", "end") and (self.name, method) not in (("start", "min"), ("end", "max")): raise ValueError("native_fused_stats_operation_not_requested")
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

        def mean(self, axis=None, skipna=True, numeric_only=False, **kwargs):
            if self.name != "score" or not word_means: return super().mean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
            if axis not in (None, 0, "index") or skipna is not True or numeric_only is not False or kwargs: raise ValueError("native_word_mean_options_not_supported")
            if self.dtype != np.dtype("float64") and not (self.dtype == np.dtype("object") and all(value is None for value in self)): raise ValueError("native_word_mean_dtype_not_supported")
            cache = getattr(self, "_native_word_mean_cache", None)
            key = tuple(self.index)
            if cache is None or key not in cache[0]: raise ValueError("native_word_mean_subset_not_registered")
            mean, snapshot = cache[0][key]
            if self.to_numpy(dtype=np.float64).tobytes() != snapshot: raise ValueError("native_word_mean_snapshot_changed")
            cache[1]["queries_consumed"] += 1
            return mean

        def min(self, axis=None, skipna=True, numeric_only=False, **kwargs): return self._bounds("min", axis, skipna, numeric_only, kwargs)

        def max(self, axis=None, skipna=True, numeric_only=False, **kwargs): return self._bounds("max", axis, skipna, numeric_only, kwargs)

    class BatchFrame(original.DataFrame):
        _metadata = [*original.DataFrame._metadata, "_native_batch_cache", "_native_word_mean_cache"]

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
        preparation_started = time.perf_counter_ns()
        text = "".join(row["char"] for row in data)
        if text not in sentence_spans or len(data) > 32768: raise ValueError("native_timestamp_batch_sentence_metadata_missing")
        values = np.asarray([[row["start"], row["end"], row["score"]] if fused_reductions else [row["start"], row["end"]] for row in data], dtype=np.float64).reshape(-1)
        queries, mean_queries = {}, {}
        non_spaces = tuple(row["char"] != " " for row in data) if compact_plan else None
        def register(column, rows): queries.setdefault((column, tuple(rows)), None)
        for start, end in sentence_spans[text]:
            rows = tuple(range(max(0, start), min(len(data), end + 1))) if compact_plan and type(start) is int and type(end) is int else [i for i in range(len(data)) if start <= i <= end]
            register("start", rows)
            register("end", [i for i in rows if (non_spaces[i] if compact_plan else data[i]["char"] != " ")])
            words = {}
            for i in rows: words.setdefault(data[i]["word-idx"], []).append(i)
            for word_rows in words.values():
                if not "".join(data[i]["char"] for i in word_rows).strip(): continue
                selected = tuple(i for i in word_rows if non_spaces[i]) if compact_plan else [i for i in word_rows if data[i]["char"] != " "]
                register("start", selected); register("end", selected)
                if word_means: mean_queries.setdefault(tuple(selected), None)
        if not queries: raise ValueError("native_timestamp_batch_queries_missing")
        if fused_reductions:
            values = values.reshape(-1, 3)
            query_modes = {}
            for column, rows in queries: query_modes[rows] = query_modes.get(rows, 0) | (1 if column == "start" else 2)
            for rows in mean_queries: query_modes[rows] = query_modes.get(rows, 0) | 4
            indices, offsets = [], [0]
            for rows in query_modes: indices.extend(rows); offsets.append(len(indices))
            plan_ready = time.perf_counter_ns()
            statistics, record = context.alignment_stats_batch(values, np.asarray(indices, dtype=np.uint32), np.asarray(offsets, dtype=np.uint32), np.asarray(list(query_modes.values()), dtype=np.uint32))
            cache_started = time.perf_counter_ns()
            record.update(cpu_query_preparation_wall_ns=plan_ready - preparation_started, compact_plan=compact_plan, queries=len(queries), fused_subset_count=len(query_modes), queries_consumed=0, fast_frames=fast_frames, fused_reductions=True, completion_group=len(records), timing_scope="shared_timestamp_and_mean_group")
            records.append(record)
            positions = {rows: i for i, rows in enumerate(query_modes)}
            if compact_plan:
                snapshots = {}
                snapshot_bytes, gathered_bytes = 0, 0
                for rows, mode in query_modes.items():
                    fields = [i for i, bit in enumerate((1, 2, 4)) if mode & bit]
                    selected = values[list(rows)]
                    gathered_bytes += selected.nbytes
                    snapshots[rows] = {field: selected[:, field].tobytes() for field in fields}
                    snapshot_bytes += sum(len(snapshot) for snapshot in snapshots[rows].values())
                cache = {(column, rows): (statistics[positions[rows], :2], snapshots[rows][0 if column == "start" else 1]) for column, rows in queries}
                mean_cache = {rows: (statistics[positions[rows], 2], snapshots[rows][2]) for rows in mean_queries}
                record.update(cpu_snapshot_subset_copies=len(snapshots), cpu_snapshot_selected_values_bytes=snapshot_bytes, cpu_snapshot_gathered_values_bytes=gathered_bytes)
            else:
                cache = {(column, rows): (statistics[positions[rows], :2], values[list(rows), 0 if column == "start" else 1].tobytes()) for column, rows in queries}
                mean_cache = {rows: (statistics[positions[rows], 2], values[list(rows), 2].tobytes()) for rows in mean_queries}
                record.update(cpu_snapshot_subset_copies=len(queries) + len(mean_queries), cpu_snapshot_selected_values_bytes=8 * (sum(len(rows) for column, rows in queries) + sum(len(rows) for rows in mean_queries)))
                record["cpu_snapshot_gathered_values_bytes"] = record["cpu_snapshot_selected_values_bytes"]
            mean_record = {**record, "scope": "word_mean_component_of_fused_stats", "queries": len(mean_queries), "queries_consumed": 0, "command_buffers_per_run": 0, "separate_dispatch": False, "completion_and_bridge_wall_ns": 0, "computation_wall_ns": 0, "logical_cpu_to_native_bytes": 0, "logical_output_readback_bytes": 0}
            mean_records.append(mean_record)
            frame = BatchFrame(data)
            frame._native_batch_cache = cache, record
            frame._native_word_mean_cache = mean_cache, mean_record
            record["cpu_cache_preparation_wall_ns"] = time.perf_counter_ns() - cache_started
            return frame
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
        if word_means and mean_queries:
            scores = np.asarray([row["score"] for row in data], dtype=np.float64)
            mean_indices, mean_offsets, mean_snapshots = [], [0], []
            for rows in mean_queries:
                mean_indices.extend(rows); mean_offsets.append(len(mean_indices))
                mean_snapshots.append(scores[list(rows)].tobytes())
            means, mean_record = context.word_means_batch(scores, np.asarray(mean_indices, dtype=np.uint32), np.asarray(mean_offsets, dtype=np.uint32))
            mean_record["queries_consumed"] = 0
            mean_records.append(mean_record)
            frame._native_word_mean_cache = {key: (means[i], mean_snapshots[i]) for i, key in enumerate(mean_queries)}, mean_record
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

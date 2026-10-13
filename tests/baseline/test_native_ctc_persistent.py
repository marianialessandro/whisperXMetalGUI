from concurrent.futures import ThreadPoolExecutor
import hashlib
import struct
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import whisperx.alignment as alignment
from tools.baseline.native_ctc import NativeCTC
from tools.baseline.native_character_rounding import rounded_character_segments
from tools.baseline.native_timestamp_bounds import native_timestamp_bounds, native_timestamp_bounds_batched
from tools.baseline.ctc_alignment_trial import persistent_metal_ctc, native_wildcard_max
from tools.baseline.native_ctc_fused import DeferredWildcard, fused_wildcard_preparation


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal hardware required")
class PersistentNativeCTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(__file__).resolve().parents[2]
        cls.library = Path(cls.temporary.name) / "ctc.dylib"
        cls.harness = Path(cls.temporary.name) / "ctc-abi-test"
        bridge = str(cls.root / "tools/baseline/native_ctc_bridge.mm")
        flags = ["xcrun", "clang++", "-std=c++20", "-O1", "-fobjc-arc", "-Wno-unused-function", "-Wno-deprecated-declarations"]
        subprocess.run([*flags, "-dynamiclib", bridge, "-framework", "Foundation", "-framework", "Metal", "-o", str(cls.library)], capture_output=True, check=True, timeout=60)
        subprocess.run([*flags, "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", bridge, str(cls.root / "tests/fixtures/baseline/native_ctc_abi.cpp"), "-I", str(cls.root / "tools/baseline"), "-framework", "Foundation", "-framework", "Metal", "-o", str(cls.harness)], capture_output=True, check=True, timeout=60)
        cls.identity = hashlib.sha256(cls.library.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def compare(self, context, emission, tokens, blank=0):
        expected = alignment.backtrack(alignment.get_trellis(emission, tokens, blank), emission, tokens, blank)
        actual, record = context.compute(emission.numpy(), np.asarray(tokens, dtype=np.int32), blank)
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertEqual([(p[0], p[1]) for p in actual], [(p.token_index, p.time_index) for p in expected])
            for point, reference in zip(actual, expected): self.assertAlmostEqual(point[2], reference.score, delta=16 * np.finfo(np.float32).eps)
        self.assertEqual(record["command_buffers_per_run"], 2)
        self.assertGreater(record["computation_gpu_seconds"], 0)
        return record

    def test_timestamp_bounds_exact_binary64_missing_values_and_scoped_pandas(self):
        import pandas as pd
        generator = np.random.default_rng(923)
        cases = [np.asarray([], dtype=np.float64), np.array([np.nan, np.nan]), np.array([0., np.nan, 1e-300, 1e300, -1e-300, -1e300, np.inf, -np.inf]), generator.normal(size=513), generator.normal(size=32768)[::2], np.array([np.nextafter(1., 2.), 1.])]
        original_pd = alignment.pd
        with NativeCTC(self.library, self.identity) as context:
            for values in cases:
                bounds, record = context.timestamp_bounds(values)
                for actual, expected in zip(bounds, [pd.Series(values).min(), pd.Series(values).max()]):
                    if np.isnan(expected): self.assertTrue(np.isnan(actual))
                    else: self.assertEqual(actual.tobytes(), np.float64(expected).tobytes())
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertEqual(record["command_buffers_per_run"], 1)
            with self.assertRaisesRegex(ValueError, "negative_zero"): context.timestamp_bounds(np.array([-0., 0.]))
            with self.assertRaisesRegex(ValueError, "format_invalid"): context.timestamp_bounds(np.zeros(32769))
            records = []
            rows = [{"char": "a", "start": 1.001, "end": 2.002, "score": .3, "word-idx": 0}, {"char": "?", "start": None, "end": None, "score": None, "word-idx": 1}]
            with native_timestamp_bounds(alignment, context, records):
                frame = alignment.pd.DataFrame(rows)
                self.assertEqual(frame.loc[frame["word-idx"] == 0]["start"].min(), 1.001)
                self.assertEqual(frame[frame["char"] != " "]["end"].max(), 2.002)
                self.assertTrue(np.isnan(frame.loc[frame["word-idx"] == 1]["start"].min()))
                self.assertEqual(frame["score"].mean(), pd.DataFrame(rows)["score"].mean())
                self.assertEqual(frame["word-idx"].max(), 1)
                all_missing = alignment.pd.DataFrame([rows[1]])
                self.assertTrue(np.isnan(all_missing["start"].min()))
                self.assertIs(type(alignment.pd.Series([1.])), pd.Series)
                with self.assertRaisesRegex(ValueError, "options_not_supported"): frame["start"].min(skipna=False)
            self.assertEqual(len(records), 4)
            self.assertIs(alignment.pd, original_pd)
            with self.assertRaisesRegex(ValueError, "negative_zero"):
                with native_timestamp_bounds(alignment, context, []): alignment.pd.DataFrame([{**rows[0], "start": -0.}])["start"].min()
            self.assertIs(alignment.pd, original_pd)
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.timestamp_bounds(np.array([1.]))

    def test_timestamp_batch_csr_matches_pandas_and_rejects_unsafe_metadata(self):
        import pandas as pd
        generator = np.random.default_rng(943)
        values = generator.normal(size=1025)
        values[::17] = np.nan
        values[1:6] = [0., np.inf, -np.inf, np.nextafter(1., 2.), 1e-300]
        groups = [[], [0], [1, 2, 3, 4, 5], list(range(1025)), [17, 34], [4, 4, 1], *[generator.integers(0, len(values), size=int(size)).tolist() for size in generator.integers(0, 130, size=31)]]
        offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.uint32)
        indices = np.asarray([index for group in groups for index in group], dtype=np.uint32)
        with NativeCTC(self.library, self.identity) as context:
            actual, record = context.timestamp_bounds_batch(values, indices, offsets)
            for bounds, group in zip(actual, groups):
                series = pd.Series(values[group])
                for result, reference in zip(bounds, [series.min(), series.max()]):
                    if np.isnan(reference): self.assertTrue(np.isnan(result))
                    else: self.assertEqual(result.tobytes(), np.float64(reference).tobytes())
            self.assertEqual(record["command_buffers_per_run"], 1)
            self.assertEqual(record["queries"], len(groups))
            self.assertGreater(record["computation_gpu_seconds"], 0)
            for bad_indices, bad_offsets in [(np.array([1025], dtype=np.uint32), np.array([0, 1], dtype=np.uint32)), (indices, np.array([1, len(indices)], dtype=np.uint32)), (indices, np.array([0, len(indices) + 1], dtype=np.uint32)), (indices, np.array([0, 2, 1, len(indices)], dtype=np.uint32)), (indices, np.array([0, 0xffffffff, len(indices)], dtype=np.uint32))]:
                with self.assertRaisesRegex(ValueError, "indices_invalid"): context.timestamp_bounds_batch(values, bad_indices, bad_offsets)
            with self.assertRaisesRegex(ValueError, "negative_zero"): context.timestamp_bounds_batch(np.array([-0.]), np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.uint32))
            empty, _ = context.timestamp_bounds_batch(np.array([], dtype=np.float64), np.array([], dtype=np.uint32), np.array([0, 0, 0], dtype=np.uint32))
            self.assertTrue(np.isnan(empty).all())
            with self.assertRaisesRegex(ValueError, "format_invalid"): context.timestamp_bounds_batch(values, indices.astype(np.int32), offsets)
            reused, _ = context.timestamp_bounds_batch(values, indices, offsets)
            np.testing.assert_array_equal(actual.view(np.uint64), reused.view(np.uint64))

    def test_batched_pandas_uses_original_spans_and_restores_after_mutation_error(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        rows = [{"char": char, "start": None if i == 2 else float(i), "end": None if i == 2 else float(i + 1), "score": .3, "word-idx": word} for i, (char, word) in enumerate(zip("a ? b", [0, 1, 1, 2, 2]))]
        original_pd = alignment.pd
        tokenizer = SimpleNamespace(span_tokenize=lambda text: iter([(0, 3), (3, 5)]))
        with NativeCTC(self.library, self.identity) as context, patch.object(alignment, "nltk_load", return_value=tokenizer) as loader:
            records = []
            with native_timestamp_bounds_batched(alignment, context, records):
                self.assertEqual(list(alignment.nltk_load("test").span_tokenize("a ? b")), [(0, 3), (3, 5)])
                frame = alignment.pd.DataFrame(rows)
                self.assertEqual(frame.loc[[0, 1, 2, 3]]["start"].min(), 0.)
                self.assertEqual(frame.loc[[0, 2]]["end"].max(), 1.)
                self.assertTrue(np.isnan(frame.loc[[2]]["start"].min()))
                self.assertEqual(frame.loc[[4]]["end"].max(), 5.)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["queries_consumed"], 4)
                with self.assertRaisesRegex(ValueError, "dtype_not_supported"): frame.loc[[4]]["start"].astype(np.float32).min()
                with self.assertRaisesRegex(ValueError, "subset_not_registered"): frame.loc[[2, 4]]["start"].min()
                with self.assertRaisesRegex(ValueError, "options_not_supported"): frame.loc[[4]]["start"].min(skipna=False)
            self.assertIs(alignment.pd, original_pd)
            self.assertIs(alignment.nltk_load, loader)
            with self.assertRaisesRegex(ValueError, "snapshot_changed"):
                with native_timestamp_bounds_batched(alignment, context, []):
                    list(alignment.nltk_load("test").span_tokenize("a ? b"))
                    frame = alignment.pd.DataFrame(rows)
                    frame.loc[0, "start"] = 9.
                    frame.loc[[0]]["start"].min()
            self.assertIs(alignment.pd, original_pd)
            self.assertIs(alignment.nltk_load, loader)

    def test_fast_batch_constructors_preserve_metadata_dtypes_and_copy_on_write(self):
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import patch
        rows = [{"char": char, "start": None if i == 2 else float(i), "end": None if i == 2 else float(i + 1), "score": .3, "word-idx": word} for i, (char, word) in enumerate(zip("a ? b", [0, 1, 1, 2, 2]))]
        tokenizer = SimpleNamespace(span_tokenize=lambda text: iter([(0, 3), (3, 5)]))
        reference = pd.DataFrame(rows)
        original_pd = alignment.pd
        with NativeCTC(self.library, self.identity) as context, patch.object(alignment, "nltk_load", return_value=tokenizer) as loader:
            for fast in (False, True):
                with self.subTest(fast=fast), native_timestamp_bounds_batched(alignment, context, [], fast_frames=fast):
                    list(alignment.nltk_load("test").span_tokenize("a ? b"))
                    frame = alignment.pd.DataFrame(rows)
                    frame["sentence-idx"] = None
                    expected = reference.assign(**{"sentence-idx": None})
                    subset = frame.loc[frame["word-idx"] == 0]
                    pd.testing.assert_frame_equal(pd.DataFrame(subset), expected.loc[expected["word-idx"] == 0])
                    self.assertEqual(subset["start"].name, "start")
                    self.assertEqual(subset["start"].min(), 0.)
                    self.assertEqual(subset["end"].max(), 1.)
                    self.assertEqual(frame["word-idx"].unique().tolist(), [0, 1, 2])
                    self.assertEqual(frame["score"].mean(), reference["score"].mean())
                    projected = frame.loc[[0]][["char", "start", "end", "score"]]
                    projected.loc[0, "start"] = 9.
                    self.assertEqual(frame.loc[0, "start"], 0.)
                    self.assertEqual(frame.loc[[0]]["start"].min(), 0.)
                    with self.assertRaisesRegex(ValueError, "snapshot_changed"): projected["start"].min()
                    self.assertEqual(frame.loc[[2]][["start", "end"]].fillna(-1).to_dict("records"), [{"start": -1., "end": -1.}])
                    self.assertEqual(len(frame["start"].iloc[:2]), 2)
                    with self.assertRaisesRegex(ValueError, "subset_not_registered"): frame["start"].iloc[:2].min()
                self.assertIs(alignment.pd, original_pd)
                self.assertIs(alignment.nltk_load, loader)
            with patch.object(original_pd, "__version__", "unsupported"):
                with self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                    with native_timestamp_bounds_batched(alignment, context, [], fast_frames=True): pass
            self.assertIs(alignment.pd, original_pd)
            self.assertIs(alignment.nltk_load, loader)

    def test_word_mean_batches_match_pandas_bitwise_across_pairwise_and_buffer_boundaries(self):
        import pandas as pd
        from unittest.mock import patch
        generator = np.random.default_rng(974)
        with NativeCTC(self.library, self.identity) as context:
            for length in (0, 1, 7, 8, 9, 127, 128, 129, 255, 1025, 8191, 8192, 8193, 16383, 16384, 16385, 24577, 32767, 32768):
                for replica in range(3):
                    scores = generator.integers(0, 1001, length).astype(np.float64) / 1000
                    scores[::19] = np.nan
                    groups = [list(range(length)), [], list(range(min(length, 31))), list(range(length))[::-1]]
                    indices = np.asarray([i for group in groups for i in group], dtype=np.uint32)
                    offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.uint32)
                    means, record = context.word_means_batch(scores, indices, offsets)
                    for actual, group in zip(means, groups):
                        expected = pd.Series(scores[group]).mean()
                        if np.isnan(expected): self.assertTrue(np.isnan(actual))
                        else:
                            self.assertEqual(actual.tobytes(), np.float64(expected).tobytes(), (length, replica))
                            self.assertEqual(round(actual, 3).tobytes(), round(np.float64(expected), 3).tobytes())
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(record["command_buffers_per_run"], 1)
            indices, offsets = np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.uint32)
            for value in (-0., -.1, .0001, 1.001, np.inf, -np.inf):
                with self.assertRaisesRegex(ValueError, "values_invalid"): context.word_means_batch(np.array([value]), indices, offsets)
            with self.assertRaisesRegex(ValueError, "indices_invalid"): context.word_means_batch(np.array([.1]), np.array([1], dtype=np.uint32), offsets)
            with self.assertRaisesRegex(ValueError, "indices_invalid"): context.word_means_batch(np.array([.1]), indices, np.array([0, 2], dtype=np.uint32))
            with self.assertRaisesRegex(ValueError, "indices_invalid"): context.word_means_batch(np.array([.1]), np.zeros(32769, dtype=np.uint32), np.array([0, 32769], dtype=np.uint32))
            with patch.object(np, "__version__", "unsupported"):
                with self.assertRaisesRegex(ValueError, "runtime_not_supported"): context.word_means_batch(np.array([.1]), indices, offsets)
            previous = np.getbufsize()
            try:
                np.setbufsize(4096)
                with self.assertRaisesRegex(ValueError, "runtime_not_supported"): context.word_means_batch(np.array([.1]), indices, offsets)
            finally: np.setbufsize(previous)
            means, _ = context.word_means_batch(np.array([.1]), indices, offsets)
            self.assertEqual(means[0], np.float64(.1))

    def test_word_mean_adapter_missing_scores_rounding_and_snapshot_validation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        rows = [{"char": char, "start": float(i), "end": float(i + 1), "score": score, "word-idx": word} for i, (char, word, score) in enumerate(zip("a ? b", [0, 1, 1, 2, 2], [.101, .202, None, .303, .404]))]
        tokenizer = SimpleNamespace(span_tokenize=lambda text: iter([(0, 3), (3, 5)]))
        original_pd = alignment.pd
        with NativeCTC(self.library, self.identity) as context, patch.object(alignment, "nltk_load", return_value=tokenizer) as loader:
            for fast in (False, True):
                means = []
                with native_timestamp_bounds_batched(alignment, context, [], fast_frames=fast, word_means=True, mean_records=means):
                    list(alignment.nltk_load("test").span_tokenize("a ? b"))
                    frame = alignment.pd.DataFrame(rows)
                    self.assertIsInstance(frame.loc[[0]]["score"].mean(), np.float64)
                    self.assertEqual(round(frame.loc[[4]]["score"].mean(), 3), .404)
                    self.assertTrue(np.isnan(frame.loc[[2]]["score"].mean()))
                    self.assertEqual(len(means), 1)
                    self.assertEqual(means[0]["queries_consumed"], 3)
                    self.assertEqual(frame["word-idx"].mean(), 1.2)
                    with self.assertRaisesRegex(ValueError, "subset_not_registered"): frame["score"].mean()
                    with self.assertRaisesRegex(ValueError, "options_not_supported"): frame.loc[[0]]["score"].mean(skipna=False)
                    with self.assertRaisesRegex(ValueError, "dtype_not_supported"): frame.loc[[0]]["score"].astype(np.float32).mean()
                    derived = frame.loc[[0]]
                    derived.loc[0, "score"] = .909
                    self.assertEqual(frame.loc[[0]]["score"].mean(), .101)
                    with self.assertRaisesRegex(ValueError, "snapshot_changed"): derived["score"].mean()
                self.assertIs(alignment.pd, original_pd)
                self.assertIs(alignment.nltk_load, loader)

    def test_fused_alignment_stats_masks_csr_and_bitwise_long_word_means(self):
        import pandas as pd
        generator = np.random.default_rng(995)
        with NativeCTC(self.library, self.identity) as context:
            for length in (0, 1, 9, 129, 8193, 32768):
                values = np.column_stack((generator.normal(size=length), generator.normal(size=length), generator.integers(0, 1001, length).astype(np.float64) / 1000))
                values[::17, 0] = np.nan; values[::19, 1] = np.nan; values[::23, 2] = np.nan
                groups = [list(range(length)), list(range(length))[::-1], [], list(range(min(length, 27))), [], [], []]
                modes = np.arange(1, 8, dtype=np.uint32)
                indices = np.asarray([i for group in groups for i in group], dtype=np.uint32)
                offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.uint32)
                actual, record = context.alignment_stats_batch(values, indices, offsets, modes)
                for i, (group, mode) in enumerate(zip(groups, modes)):
                    expected = [pd.Series(values[group, 0]).min() if mode & 1 else np.nan, pd.Series(values[group, 1]).max() if mode & 2 else np.nan, pd.Series(values[group, 2]).mean() if mode & 4 else np.nan]
                    for result, reference in zip(actual[i], expected):
                        if np.isnan(reference): self.assertTrue(np.isnan(result))
                        else: self.assertEqual(result.tobytes(), np.float64(reference).tobytes())
                self.assertEqual(record["command_buffers_per_run"], 1)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                if length:
                    full, _ = context.alignment_stats_batch(values, np.arange(length, dtype=np.uint32), np.array([0, length], dtype=np.uint32), np.array([7], dtype=np.uint32))
                    self.assertEqual(full[0, 2].tobytes(), np.float64(pd.Series(values[:, 2]).mean()).tobytes())
            values = np.array([[1., 2., .1], [3., 4., .3]])
            indices, offsets = np.array([0, 1], dtype=np.uint32), np.array([0, 2], dtype=np.uint32)
            for mode in (0, 8):
                with self.assertRaisesRegex(ValueError, "metadata_invalid"): context.alignment_stats_batch(values, indices, offsets, np.array([mode], dtype=np.uint32))
            with self.assertRaisesRegex(ValueError, "metadata_invalid"): context.alignment_stats_batch(values, np.array([0, 2], dtype=np.uint32), offsets, np.array([7], dtype=np.uint32))
            with self.assertRaisesRegex(ValueError, "metadata_invalid"): context.alignment_stats_batch(values, indices, np.array([0, 0xffffffff], dtype=np.uint32), np.array([7], dtype=np.uint32))
            values[0, 0] = -0.; values[0, 2] = -1.
            ignored, _ = context.alignment_stats_batch(values, indices, offsets, np.array([2], dtype=np.uint32))
            self.assertEqual(ignored[0, 1], 4.)
            with self.assertRaisesRegex(ValueError, "negative_zero"): context.alignment_stats_batch(values, indices, offsets, np.array([1], dtype=np.uint32))
            with self.assertRaisesRegex(ValueError, "values_invalid"): context.alignment_stats_batch(values, indices, offsets, np.array([4], dtype=np.uint32))

    def test_fused_adapter_uses_shared_rows_and_reports_one_completion(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        rows = [{"char": char, "start": float(i), "end": float(i + 1), "score": None if i == 2 else .3, "word-idx": word} for i, (char, word) in enumerate(zip("a ? b", [0, 1, 1, 2, 2]))]
        tokenizer = SimpleNamespace(span_tokenize=lambda text: iter([(0, 3), (3, 5)]))
        original_pd = alignment.pd
        with NativeCTC(self.library, self.identity) as context, patch.object(alignment, "nltk_load", return_value=tokenizer) as loader:
            bounds, means = [], []
            with native_timestamp_bounds_batched(alignment, context, bounds, fast_frames=True, word_means=True, mean_records=means, fused_reductions=True):
                list(alignment.nltk_load("test").span_tokenize("a ? b"))
                frame = alignment.pd.DataFrame(rows)
                self.assertEqual(frame.loc[[0]]["start"].min(), 0.)
                self.assertEqual(frame.loc[[0]]["end"].max(), 1.)
                self.assertEqual(frame.loc[[0]]["score"].mean(), .3)
                self.assertTrue(np.isnan(frame.loc[[2]]["score"].mean()))
                self.assertEqual(frame.loc[[0, 1, 2, 3]]["start"].min(), 0.)
                self.assertEqual(frame.loc[[0, 2]]["end"].max(), 3.)
                with self.assertRaisesRegex(ValueError, "operation_not_requested"): frame.loc[[0]]["start"].max()
                with self.assertRaisesRegex(ValueError, "subset_not_registered"): frame["score"].mean()
                derived = frame.loc[[0]]
                derived.loc[0, "score"] = .9
                with self.assertRaisesRegex(ValueError, "snapshot_changed"): derived["score"].mean()
            self.assertEqual(len(bounds), 1); self.assertEqual(len(means), 1)
            self.assertEqual(bounds[0]["command_buffers_per_run"] + means[0]["command_buffers_per_run"], 1)
            self.assertFalse(means[0]["separate_dispatch"])
            self.assertEqual(means[0]["logical_cpu_to_native_bytes"], 0)
            self.assertLess(bounds[0]["fused_subset_count"], bounds[0]["queries"] + means[0]["queries"])
            self.assertEqual(bounds[0]["completion_group"], means[0]["completion_group"])
            self.assertIs(alignment.pd, original_pd); self.assertIs(alignment.nltk_load, loader)

    def test_fused_readback_stays_owned_across_reuse_and_concurrent_requests(self):
        values = np.array([[1., 2., .1], [3., 4., .3]])
        indices, offsets, modes = np.array([0, 1], dtype=np.uint32), np.array([0, 2], dtype=np.uint32), np.array([7], dtype=np.uint32)
        with NativeCTC(self.library, self.identity) as context:
            first, record = context.alignment_stats_batch(values, indices, offsets, modes)
            saved = first.copy()
            changed = np.array([[10., 20., .2], [30., 40., .4]])
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(context.alignment_stats_batch, changed if i % 2 else values, indices, offsets, modes) for i in range(12)]
                for i, future in enumerate(futures):
                    actual, _ = future.result()
                    expected = np.array([[10., 40., (.2 + .4) / 2]]) if i % 2 else saved
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            np.testing.assert_array_equal(first.view(np.uint64), saved.view(np.uint64))
            self.assertEqual(record["native_host_result_staging_bytes"], 0)
            self.assertEqual(record["native_host_result_memcpy_calls"], 1)
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.alignment_stats_batch(values, indices, offsets, modes)

    def test_compact_plan_matches_fused_metadata_and_snapshot_bytes_on_boundary_spans(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        text = " α ? b \t."
        rows, word = [], 0
        for i, char in enumerate(text):
            rows.append({"char": char, "start": None if i % 4 == 0 else i + .1, "end": None if i % 4 == 0 else i + .5, "score": None if i % 4 == 0 else .3, "word-idx": word})
            if i == len(text) - 1 or text[i + 1] == " ": word += 1
        tokenizer = SimpleNamespace(span_tokenize=lambda source: iter([(0, 3), (3, 99), (-2, 0), (8, 7)]))
        captured, reference_bounds, reference_means, reports = [], None, None, []
        with NativeCTC(self.library, self.identity) as context, patch.object(alignment, "nltk_load", return_value=tokenizer):
            function = context.alignment_stats_batch
            def capture(*arrays):
                captured.append(tuple(array.copy() for array in arrays))
                return function(*arrays)
            with patch.object(context, "alignment_stats_batch", side_effect=capture):
                for compact in (False, True):
                    bounds, means = [], []
                    with native_timestamp_bounds_batched(alignment, context, bounds, fast_frames=True, word_means=True, mean_records=means, fused_reductions=True, compact_plan=compact):
                        list(alignment.nltk_load("test").span_tokenize(text))
                        frame = alignment.pd.DataFrame(rows)
                        actual_bounds = {key: (value[0].view(np.uint64).tobytes(), value[1]) for key, value in frame._native_batch_cache[0].items()}
                        actual_means = {key: (value[0].tobytes(), value[1]) for key, value in frame._native_word_mean_cache[0].items()}
                        if reference_bounds is None: reference_bounds, reference_means = actual_bounds, actual_means
                        else:
                            self.assertEqual(actual_bounds, reference_bounds)
                            self.assertEqual(actual_means, reference_means)
                        self.assertTrue(np.isnan(frame.loc[[]]["start"].min()))
                    reports.append(bounds[0])
            for original, compact in zip(captured[0], captured[1]): np.testing.assert_array_equal(original.view(np.uint64) if original.dtype == np.float64 else original, compact.view(np.uint64) if compact.dtype == np.float64 else compact)
            self.assertLess(reports[1]["cpu_snapshot_subset_copies"], reports[0]["cpu_snapshot_subset_copies"])
            self.assertEqual(reports[1]["cpu_snapshot_selected_values_bytes"], reports[0]["cpu_snapshot_selected_values_bytes"])
            self.assertGreaterEqual(reports[1]["cpu_snapshot_gathered_values_bytes"], reports[1]["cpu_snapshot_selected_values_bytes"])
            self.assertGreater(reports[1]["cpu_cache_preparation_wall_ns"], 0)
            with self.assertRaisesRegex(ValueError, "requires_fused"):
                with native_timestamp_bounds_batched(alignment, context, [], compact_plan=True): pass

    def test_round3_matches_cpython_for_random_bits_ties_subnormals_and_payloads(self):
        from unittest.mock import patch
        generator = np.random.default_rng(1017)
        random = generator.bit_generator.random_raw(65536).view(np.float64)
        centers = [0., .0005, .0015, .0025, .0625, .1875, .1235, 1.0015, 9999999.9995, float(2 ** 42), float(2 ** 43), float(2 ** 52)]
        boundary = np.array([value for center in centers for sign in (1., -1.) for value in (np.nextafter(sign * center, -np.inf), sign * center, np.nextafter(sign * center, np.inf))])
        special = np.array([0, 0x8000000000000000, 1, 0x8000000000000001, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000001234, 0xfff0000000001234], dtype=np.uint64).view(np.float64)
        with NativeCTC(self.library, self.identity) as context:
            for values in (random, boundary, special, random[::2]):
                actual, record = context.round3(values)
                expected = np.asarray([round(float(value), 3) for value in values], dtype=np.float64)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertEqual(record["command_buffers_per_run"], 1)
            example = np.array([.0005, .0025, 56294995342131.5])
            actual, _ = context.round3(example)
            self.assertFalse(np.array_equal(actual.view(np.uint64), np.round(example, 3).view(np.uint64)))
            for bad in (np.array([], dtype=np.float64), np.zeros(98305), np.zeros(1, dtype=np.float32)):
                with self.assertRaisesRegex(ValueError, "format_invalid"): context.round3(bad)
            with patch.object(sys, "version_info", (3, 12, 11)):
                with self.assertRaisesRegex(ValueError, "runtime_not_supported"): context.round3(example)
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.round3(example)

    def test_deferred_character_rounding_batches_actual_values_and_rejects_changed_scale(self):
        segments = [(0, 1, .0005), (1, 3, .0625), (3, 4, .1875)]
        ratio, offset = .123456789, 10.0005
        with NativeCTC(self.library, self.identity) as context:
            spans = []
            characters = rounded_character_segments(alignment, segments, "aβ?", context, spans)
            with self.assertRaisesRegex(ValueError, "scale_not_ready"): round(characters[0].score, 3)
            for i, character in enumerate(characters):
                actual = round(character.start * ratio + offset, 3), round(character.end * ratio + offset, 3), round(character.score, 3)
                expected = round(segments[i][0] * ratio + offset, 3), round(segments[i][1] * ratio + offset, 3), round(segments[i][2], 3)
                self.assertEqual([struct.pack("d", value) for value in actual], [struct.pack("d", value) for value in expected])
                self.assertTrue(all(type(value) is float for value in actual))
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["elements"], 9)
            with self.assertRaisesRegex(ValueError, "scale_changed"): round(characters[1].start * (ratio * 2) + offset, 3)
            with self.assertRaisesRegex(ValueError, "digits_not_supported"): round(characters[0].score, 2)
            with self.assertRaises(TypeError): float(characters[0].start)
            self.assertEqual([character.label for character in characters], list("aβ?"))
            originals = alignment.get_trellis, alignment.backtrack, alignment.merge_repeats
            with self.assertRaisesRegex(ValueError, "requires_segments"):
                with persistent_metal_ctc(alignment, [], context, character_rounding=True, rounding_records=[]): pass
            self.assertEqual(originals, (alignment.get_trellis, alignment.backtrack, alignment.merge_repeats))

    def test_resident_scaling_and_rounding_match_separate_python_binary64_operations(self):
        from unittest.mock import patch
        generator = np.random.default_rng(2281)
        emission = np.zeros((32768, 2), dtype=np.float32)
        emission[:, 1] = -10.
        anchors = np.linspace(0, 32767, 127, dtype=np.int32)
        emission[anchors, 0], emission[anchors, 1] = -10., 0.
        tokens = np.ones(127, dtype=np.int32)
        ratios = [2. ** -23, .0005, .0015, .01, .0625, .123456789, 1., 1024., np.nextafter(2048., 0.), *[float(value) for value in np.exp2(generator.uniform(-23, 11, size=41))]]
        offsets = [0., 2. ** -23, .0005, .1235, 10.0005, 1024., float(2 ** 40), np.nextafter(float(2 ** 41), 0.), *[float(value) for value in np.exp2(generator.uniform(-23, 41, size=42))]]
        source = (self.root / "tools/baseline/native_ctc.metal").read_bytes()
        start = source.index(b"kernel void ctc_scale_round_segments")
        raw_source = source[:start] + source[start:].replace(b"python_round3(", b"(")
        original_read = Path.read_bytes
        for rounded in (False, True):
            # The test-only shader variant exposes the scaling result before decimal rounding so quantization cannot hide arithmetic errors.
            with patch.object(Path, "read_bytes", lambda path: (source if rounded else raw_source) if path.name == "native_ctc.metal" else original_read(path)), NativeCTC(self.library, self.identity) as context:
                raw, _ = context.compute_segments(emission, tokens)
                self.assertEqual(raw[-1][1], 32768)
                handle, record = context.prepare_segments(emission, tokens)
                self.assertEqual(len(handle), len(raw))
                self.assertEqual(record["logical_character_segment_readback_bytes"], 0)
                self.assertEqual(record["logical_resident_generation_metadata_bytes"], 8)
                for ratio, offset in zip(ratios, offsets):
                    actual, span = context.scale_round_segments(handle, float(ratio), float(offset))
                    values = [(start * ratio + offset, end * ratio + offset, score) for start, end, score in raw]
                    expected = np.asarray([[round(float(value), 3) if rounded else float(value) for value in row] for row in values])
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertGreater(span["computation_gpu_seconds"], 0)
                    self.assertEqual(span["logical_cpu_to_native_bytes"], 16)
                    self.assertEqual(span["logical_character_reupload_bytes"], 0)
                saved = actual.copy()
                context.round3(np.array([.0005]))
                context.row_max(emission[:2])
                context.scale_round_segments(handle, .01, 0)
                np.testing.assert_array_equal(actual.view(np.uint64), saved.view(np.uint64))

    def test_resident_handles_reject_stale_wrong_owner_domain_and_close(self):
        from tools.baseline.native_ctc import ResidentCharacterSegments
        from unittest.mock import patch
        emission, tokens = np.array([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=np.float32), np.array([1], dtype=np.int32)
        with NativeCTC(self.library, self.identity) as context, NativeCTC(self.library, self.identity) as other:
            handle, _ = context.prepare_segments(emission, tokens)
            for ratio, offset in [(0., 0.), (-1., 0.), (2. ** -24, 0.), (2048., 0.), (.01, -0.), (.01, -1.), (.01, 2. ** -24), (.01, 2. ** 41), (float("nan"), 0.), (.01, float("inf"))]:
                with self.assertRaisesRegex(ValueError, "scale_domain_invalid"): context.scale_round_segments(handle, ratio, offset)
            with self.assertRaisesRegex(ValueError, "handle_invalid"): other.scale_round_segments(handle, .01, 0.)
            with self.assertRaisesRegex(ValueError, "generation_invalid"): context.scale_round_segments(ResidentCharacterSegments(context, handle.generation, 2), .01, 0.)
            with self.assertRaisesRegex(ValueError, "scale_type_invalid"): context.scale_round_segments(handle, .01, 2 ** 54)
            with patch.object(sys, "version_info", (3, 12, 11)):
                with self.assertRaisesRegex(ValueError, "runtime_not_supported"): context.scale_round_segments(handle, .01, 0.)
            new, _ = context.prepare_segments(emission, tokens)
            self.assertNotEqual(new.generation, handle.generation)
            with self.assertRaisesRegex(ValueError, "generation_invalid"): context.scale_round_segments(handle, .01, 0.)
            # A GPU validation failure still invalidates the retained buffer generation before another result can be consumed.
            with self.assertRaises(ValueError): context.compute(np.ones((3, 2), dtype=np.float32), tokens)
            with self.assertRaisesRegex(ValueError, "generation_invalid"): context.scale_round_segments(new, .01, 0.)
            failed, record = context.prepare_segments(np.full((5, 3), -np.inf, dtype=np.float32), np.array([1, 2], dtype=np.int32))
            self.assertIsNone(failed)
            self.assertEqual(record["logical_character_segment_readback_bytes"], 0)
            final, _ = context.prepare_segments(emission, tokens)
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.scale_round_segments(final, .01, 0.)

    def test_resident_deferred_adapter_materializes_once_and_restores_scoped_hooks(self):
        emission, tokens = np.array([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=np.float32), np.array([1], dtype=np.int32)
        with NativeCTC(self.library, self.identity) as context:
            raw, _ = context.compute_segments(emission, tokens)
            handle, _ = context.prepare_segments(emission, tokens)
            spans = []
            characters = rounded_character_segments(alignment, handle, "β", context, spans)
            with self.assertRaisesRegex(ValueError, "scale_type_invalid"): round(characters[0].start * .01 + 2 ** 10000, 3)
            result = (round(characters[0].start * .01 + 10.0005, 3), round(characters[0].end * .01 + 10.0005, 3), round(characters[0].score, 3))
            expected = (round(raw[0][0] * .01 + 10.0005, 3), round(raw[0][1] * .01 + 10.0005, 3), round(raw[0][2], 3))
            self.assertEqual([struct.pack("d", value) for value in result], [struct.pack("d", value) for value in expected])
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["scaling"], "metal_binary64_separate_multiply_add")
            originals = alignment.get_trellis, alignment.backtrack, alignment.merge_repeats
            with self.assertRaisesRegex(ValueError, "requires_character_rounding"):
                with persistent_metal_ctc(alignment, [], context, True, resident_character_rounding=True): pass
            with self.assertRaisesRegex(RuntimeError, "forced"):
                with persistent_metal_ctc(alignment, [], context, True, True, [], True): raise RuntimeError("forced")
            self.assertEqual(originals, (alignment.get_trellis, alignment.backtrack, alignment.merge_repeats))

    def test_resident_statistics_match_uploaded_values_with_nan_positions_and_buffer_reuse(self):
        generator = np.random.default_rng(936)
        emission = torch.log_softmax(torch.from_numpy(generator.normal(size=(257, 7)).astype(np.float32)), dim=-1).numpy()
        tokens = generator.integers(1, 7, size=17, dtype=np.int32)
        with NativeCTC(self.library, self.identity) as context:
            handle, _ = context.prepare_segments(emission, tokens)
            rounded, _ = context.scale_round_segments(handle, .0123456789, 10.0005)
            values = np.full((8193, 3), np.nan)
            positions = np.linspace(0, 8192, len(handle), dtype=np.int32)
            values[positions] = rounded
            groups = [[], [1, 3, 5], list(range(8193)), list(reversed(range(8193))), positions.tolist(), [int(positions[0]), int(positions[0]), 1, int(positions[-1])]]
            offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.uint32)
            indices = np.asarray([i for group in groups for i in group], dtype=np.uint32)
            modes = np.asarray([7, 7, 7, 7, 7, 7], dtype=np.uint32)
            expected, previous = context.alignment_stats_batch(values, indices, offsets, modes)
            for iteration in range(3):
                actual, record = context.alignment_stats_resident(values, indices, offsets, modes)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(record["logical_cpu_to_native_bytes"], indices.nbytes + offsets.nbytes + modes.nbytes)
                self.assertEqual(previous["logical_cpu_to_native_bytes"] - record["logical_cpu_to_native_bytes"], values.nbytes)
                self.assertEqual(record["logical_character_values_upload_bytes"], 0)
                self.assertEqual(record["cpu_resident_snapshot_bytes"], rounded.nbytes)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                context.round3(np.array([.0005]))
                context.row_max(emission[:3])
            saved = actual.copy()
            with ThreadPoolExecutor(max_workers=4) as executor:
                outputs = list(executor.map(lambda _: context.alignment_stats_resident(values, indices, offsets, modes)[0], range(12)))
            for output in outputs: np.testing.assert_array_equal(output.view(np.uint64), saved.view(np.uint64))
            outputs[0][0, 0] = 123.
            np.testing.assert_array_equal(outputs[1].view(np.uint64), saved.view(np.uint64))
            rounded[0, 0] += 1.
            with self.assertRaisesRegex(ValueError, "snapshot_changed"): context.alignment_stats_resident(rounded, np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.uint32), np.array([7], dtype=np.uint32))

    def test_resident_statistics_reject_changed_partial_nan_stale_scale_and_invalid_queries(self):
        emission, tokens = np.array([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=np.float32), np.array([1], dtype=np.int32)
        indices, offsets, modes = np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.uint32), np.array([7], dtype=np.uint32)
        with NativeCTC(self.library, self.identity) as context:
            with self.assertRaisesRegex(ValueError, "snapshot_missing"): context.alignment_stats_resident(np.ones((1, 3)), indices, offsets, modes)
            handle, _ = context.prepare_segments(emission, tokens)
            values, _ = context.scale_round_segments(handle, .01, 0.)
            snapshot = context._resident_rounded_snapshot
            for field in range(3):
                changed = values.copy(); changed[0, field] = np.nan
                with self.assertRaisesRegex(ValueError, "snapshot_changed"): context.alignment_stats_resident(changed, indices, offsets, modes)
            for invalid_indices, invalid_offsets, invalid_modes, code in [(np.array([1], dtype=np.uint32), offsets, modes, "indices_invalid"), (indices, np.array([1, 1], dtype=np.uint32), modes, "metadata_invalid"), (indices, offsets, np.array([8], dtype=np.uint32), "metadata_invalid")]:
                with self.assertRaisesRegex(ValueError, code): context.alignment_stats_resident(values, invalid_indices, invalid_offsets, invalid_modes)
            context.scale_round_segments(handle, .02, 0.)
            context._resident_rounded_snapshot = snapshot
            with self.assertRaisesRegex(ValueError, "generation_or_scale_invalid"): context.alignment_stats_resident(values, indices, offsets, modes)
            context.prepare_segments(emission, tokens)
            context._resident_rounded_snapshot = snapshot
            with self.assertRaisesRegex(ValueError, "generation_or_scale_invalid"): context.alignment_stats_resident(values, indices, offsets, modes)
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.alignment_stats_resident(values, indices, offsets, modes)

    def test_resident_statistics_adapter_preserves_original_queries_and_snapshot_checks(self):
        from unittest.mock import patch
        import nltk.tokenize
        emission = torch.log_softmax(torch.tensor([[0., 1., -1.], [1., 0., -1.], [0., -1., 1.], [1., -1., 0.], [0., 1., -1.]]), dim=-1).numpy()
        tokens = np.array([1, 2, 1], dtype=np.int32)
        with NativeCTC(self.library, self.identity) as context:
            handle, _ = context.prepare_segments(emission, tokens)
            values, _ = context.scale_round_segments(handle, .01, 0.)
            data = [{"char": char, "start": None if i == 1 else float(values[i - (i > 1), 0]), "end": None if i == 1 else float(values[i - (i > 1), 1]), "score": None if i == 1 else float(values[i - (i > 1), 2]), "word-idx": 0 if i < 2 else 1} for i, char in enumerate("a β?")]
            originals = alignment.pd, alignment.nltk_load
            frames, reports = [], []
            with patch.object(alignment, "nltk_load", return_value=nltk.tokenize.PunktSentenceTokenizer()):
                for resident, word_rounding in ((False, False), (True, False), (True, True)):
                    spans, means = [], []
                    with native_timestamp_bounds_batched(alignment, context, spans, fast_frames=True, word_means=True, mean_records=means, fused_reductions=True, compact_plan=True, resident_reductions=resident, word_rounding=word_rounding):
                        list(alignment.nltk_load().span_tokenize("a β?"))
                        frame = alignment.pd.DataFrame(data)
                        word = frame.iloc[[2, 3]]
                        frames.append([frame.start.min(), frame[frame.char != " "].end.max(), word.start.min(), word.end.max(), word.score.mean(), round(word.score.mean(), 3)])
                        changed = word.copy(); changed.loc[2, "score"] += .001
                        with self.assertRaisesRegex(ValueError, "snapshot_changed"): changed.score.mean()
                    reports.append(spans[0])
            np.testing.assert_array_equal(np.array(frames[0]).view(np.uint64), np.array(frames[1]).view(np.uint64))
            np.testing.assert_array_equal(np.array(frames[0]).view(np.uint64), np.array(frames[2]).view(np.uint64))
            self.assertLess(reports[1]["logical_cpu_to_native_bytes"], reports[0]["logical_cpu_to_native_bytes"])
            self.assertEqual(originals, (alignment.pd, alignment.nltk_load))
            with self.assertRaisesRegex(ValueError, "require_fused"):
                with native_timestamp_bounds_batched(alignment, context, [], resident_reductions=True): pass

    def test_numpy_word_rounding_helper_matches_scalar_numpy_ties_and_adjacent_values(self):
        from unittest.mock import patch
        generator = np.random.default_rng(687)
        centers = np.arange(1000, dtype=np.float64) / 1000. + .0005
        values = np.concatenate([generator.uniform(0, 1, 32768), centers, np.nextafter(centers, 0), np.nextafter(centers, 1), np.array([0., .0005, .0025, 1., np.nan])])
        source = (self.root / "tools/baseline/native_ctc.metal").read_bytes()
        source = source.replace(b"kernel void ctc_round3", b"ulong numpy_round3_score(ulong bits);\nkernel void ctc_round3", 1).replace(b"output[index] = python_round3(values[index]);", b"output[index] = numpy_round3_score(values[index]);", 1)
        original_read = Path.read_bytes
        with patch.object(Path, "read_bytes", lambda path: source if path.name == "native_ctc.metal" else original_read(path)), NativeCTC(self.library, self.identity) as context:
            actual, _ = context.round3(values)
            expected = np.array([round(np.float64(value), 3) for value in values])
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            self.assertNotEqual(struct.pack("d", float(actual[-4])), struct.pack("d", round(float(values[-4]), 3)))

    def test_word_rounding_preserves_unrounded_gpu_means_and_one_completion_group(self):
        from tools.baseline.native_timestamp_bounds import NativeRoundedWordMean
        generator = np.random.default_rng(499)
        emission = torch.log_softmax(torch.from_numpy(generator.normal(size=(129, 8)).astype(np.float32)), dim=-1).numpy()
        tokens = generator.integers(1, 8, size=19, dtype=np.int32)
        with NativeCTC(self.library, self.identity) as context:
            handle, _ = context.prepare_segments(emission, tokens)
            values, _ = context.scale_round_segments(handle, .01, 0.)
            groups = [[], [0], list(range(19)), list(reversed(range(19))), [0, 0, 18]]
            indices = np.asarray([i for group in groups for i in group], dtype=np.uint32)
            offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.uint32)
            modes = np.array([7, 7, 7, 3, 7], dtype=np.uint32)
            original, _ = context.alignment_stats_resident(values, indices, offsets, modes)
            statistics, rounded, record = context.alignment_stats_resident_rounded(values, indices, offsets, modes)
            np.testing.assert_array_equal(statistics.view(np.uint64), original.view(np.uint64))
            np.testing.assert_array_equal(rounded.view(np.uint64), np.array([round(mean, 3) for mean in original[:, 2]]).view(np.uint64))
            self.assertEqual(record["command_buffers_per_run"], 1)
            self.assertEqual(record["logical_output_readback_bytes"], len(groups) * 32 + 4)
            self.assertEqual(record["gpu_kernels"], ["ctc_alignment_stats_batch", "ctc_round_word_means"])
            for mean, result in zip(statistics[:, 2], rounded):
                native = NativeRoundedWordMean(mean, result)
                self.assertEqual(struct.pack("d", float(native)), struct.pack("d", float(mean)))
                self.assertEqual(struct.pack("d", round(native, 3)), struct.pack("d", result))
                self.assertIs(type(round(native, 3)), np.float64)
                with self.assertRaisesRegex(ValueError, "digits_not_supported"): round(native, 2)
            with self.assertRaisesRegex(ValueError, "requires_resident_reductions"):
                with native_timestamp_bounds_batched(alignment, context, [], word_rounding=True): pass

    def test_nearest_gpu_matches_original_pandas_scipy_random_missing_and_nan_payloads(self):
        import pandas as pd
        from whisperx.utils import interpolate_nans
        generator = np.random.default_rng(301)
        cases = [np.array([np.nan]), np.array([np.nan, 1., np.nan]), np.array([np.nan, 0., np.nan, np.nan, 4., np.nan]), np.array([0., np.nan, 2.]), np.array([-0., np.nan, 0.]), np.array([np.nan, np.inf, np.nan, -np.inf, np.nan]), np.array([0x7ff8000000001234, 0xfff8000000005678, 0x7ff0000000007890], dtype=np.uint64).view(np.float64)]
        for size in [2, 7, 129, 1025, 8193, 32768]:
            values = generator.normal(size=size); values[generator.random(size) < .73] = np.nan
            cases.extend([values, values[::-2]])
        with NativeCTC(self.library, self.identity) as context:
            for values in cases:
                expected = interpolate_nans(pd.Series(values)).to_numpy()
                actual, record = context.interpolate_nearest(values)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(record["command_buffers_per_run"], 1)
                self.assertGreater(record["computation_gpu_seconds"], 0)
            for invalid in [np.array([], dtype=np.float64), np.zeros(32769), np.zeros(1, dtype=np.float32)]:
                with self.assertRaisesRegex(ValueError, "format_invalid"): context.interpolate_nearest(invalid)
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.interpolate_nearest(np.array([1.]))

    def test_nearest_adapter_preserves_series_metadata_and_rejects_unsupported_policy(self):
        from unittest.mock import patch
        import pandas as pd
        from whisperx.utils import interpolate_nans
        from tools.baseline.native_interpolation import native_nearest_interpolation
        with NativeCTC(self.library, self.identity) as context:
            records = []; original = alignment.interpolate_nans
            with native_nearest_interpolation(alignment, context, records):
                for series in [pd.Series([np.nan, 1., np.nan, np.nan, 4., np.nan], name="start"), pd.Series([None, None], dtype=object, name="end"), pd.Series([], dtype=np.float64, name="start")]:
                    series.attrs = {"origin": "test"}
                    expected = interpolate_nans(series)
                    actual = alignment.interpolate_nans(series)
                    pd.testing.assert_series_equal(actual, expected)
                    self.assertEqual(actual.attrs, series.attrs)
                    self.assertIsNot(actual, series)
                    self.assertIs(alignment.interpolate_nans(series, method="ignore"), series)
                for series in [pd.Series([1., np.nan], index=[1, 2]), pd.Series([1, 2], dtype="int64"), pd.Series([pd.NA], dtype="Float64")]:
                    with self.assertRaises(ValueError): alignment.interpolate_nans(series)
                with self.assertRaisesRegex(ValueError, "method_not_supported"): alignment.interpolate_nans(pd.Series([0., np.nan, 1.]), method="linear")
            self.assertIs(alignment.interpolate_nans, original)
            self.assertEqual([record["command_buffers_per_run"] for record in records], [1, 0, 0])
            with self.assertRaisesRegex(RuntimeError, "forced"):
                with native_nearest_interpolation(alignment, context, []): raise RuntimeError("forced")
            self.assertIs(alignment.interpolate_nans, original)
            with patch("tools.baseline.native_interpolation.scipy.__version__", "1.16.0"):
                with self.assertRaisesRegex(ValueError, "runtime_or_records_not_supported"):
                    with native_nearest_interpolation(alignment, context, []): pass

    def test_buffer_growth_reuse_and_sentinel_paths_match_original(self):
        generator = np.random.default_rng(43)
        large = torch.log_softmax(torch.from_numpy(generator.normal(size=(341, 38)).astype(np.float32)), dim=-1)
        small = torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32)
        with NativeCTC(self.library, self.identity) as context:
            first = self.compare(context, small, [1])
            grown = self.compare(context, large, generator.integers(1, 38, size=87).tolist())
            reused = self.compare(context, small, [1])
            self.assertGreater(grown["allocated_shared_buffer_bytes"], first["allocated_shared_buffer_bytes"])
            self.assertEqual(reused["allocated_shared_buffer_bytes"], grown["allocated_shared_buffer_bytes"])
            self.compare(context, torch.full((16, 4), -1.), [1, 1, 3], 2)
            self.compare(context, torch.full((5, 3), float("-inf")), [1, 2])
            for _ in range(5):
                close_scores = torch.from_numpy((-1. + generator.normal(scale=1e-6, size=(80, 8))).astype(np.float32))
                self.compare(context, close_scores, generator.integers(1, 8, size=25).tolist())
            with self.assertRaisesRegex(ValueError, "input_values_invalid"): context.compute(small.numpy(), np.asarray([-1], dtype=np.int32))
            self.compare(context, small, [1])
        context.close()
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.compute(small.numpy(), np.asarray([1], dtype=np.int32))

    def test_calls_serialize_and_alignment_hooks_restore_after_error(self):
        emission = torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32)
        originals = alignment.get_trellis, alignment.backtrack
        with NativeCTC(self.library, self.identity) as context:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(context.compute, emission.numpy(), np.asarray([1], dtype=np.int32)) for _ in range(6)]
                paths = [future.result()[0] for future in futures]
                self.assertTrue(all(path == paths[0] for path in paths))
            spans = []
            with persistent_metal_ctc(alignment, spans, context):
                trellis = alignment.get_trellis(emission, [1])
                self.assertEqual(alignment.backtrack(trellis, emission, [1])[0].time_index, paths[0][0][1])
            with self.assertRaises(ValueError):
                with persistent_metal_ctc(alignment, [], context): alignment.get_trellis(emission, [-1])
            self.assertIs(alignment.get_trellis, originals[0])
            self.assertIs(alignment.backtrack, originals[1])
            self.assertEqual(spans[0]["transport"], "synchronous_in_memory_c_abi")
        with self.assertRaisesRegex(ValueError, "native_library_hash_mismatch"): NativeCTC(self.library, "0" * 64)

    def test_row_max_preserves_exact_values_first_indices_and_buffer_reuse(self):
        generator = np.random.default_rng(57)
        cases = [torch.tensor([[-3., -.1, -.1], [float("-inf"), float("-inf"), float("-inf")], [-0., 0., -1.]], dtype=torch.float32), torch.from_numpy(-np.abs(generator.normal(size=(65, 513)).astype(np.float32))), torch.full((2, 1), -2.)]
        with NativeCTC(self.library, self.identity) as context:
            capacity = 0
            for emission in cases:
                reference = emission.max(dim=1)
                values, indices, record = context.row_max(emission.numpy())
                np.testing.assert_array_equal(values.view(np.uint32), reference.values.numpy().view(np.uint32))
                np.testing.assert_array_equal(indices, reference.indices.numpy())
                self.assertGreaterEqual(record["allocated_shared_buffer_bytes"], capacity)
                capacity = record["allocated_shared_buffer_bytes"]
                self.assertEqual(record["kernel"], "ctc_row_max")
                self.assertGreater(record["computation_gpu_seconds"], 0)
            for value in [float("nan"), float("inf"), 1.]:
                with self.assertRaisesRegex(ValueError, "input_values_invalid"): context.row_max(np.full((3, 2), value, dtype=np.float32))
            self.compare(context, torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32), [1])
            with self.assertRaises(ValueError): context.row_max(np.empty((3, 0), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "native_context_closed"): context.row_max(np.zeros((3, 2), dtype=np.float32))

    def test_scoped_wildcard_dispatch_preserves_gather_and_restores_after_error(self):
        emission = torch.tensor([[-.1, -3., -1., -1.], [-.2, -2., -.1, -.1]], dtype=torch.float32)
        mask = torch.tensor([False, True, True, True])
        original = emission[:, mask].max(dim=1)
        with NativeCTC(self.library, self.identity) as context:
            spans = []
            with native_wildcard_max(context, spans):
                actual = emission[:, mask].max(dim=1)
                torch.testing.assert_close(actual.values, original.values, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual.indices, original.indices))
                emission.max(dim=0)
                emission.max(dim=1, keepdim=True)
                emission.double().max(dim=1)
            self.assertEqual(len(spans), 1)
            with self.assertRaises(ValueError):
                with native_wildcard_max(context, spans): torch.full((3, 2), float("nan")).max(dim=1)
            before = len(spans)
            emission.max(dim=1)
            self.assertEqual(len(spans), before)

    def test_fused_wildcard_matches_explicit_original_expansion_without_intermediate_readback(self):
        generator = np.random.default_rng(61)
        cases = [(torch.full((16, 4), -1.), [1, 4, 3, 4], 2), (torch.log_softmax(torch.from_numpy(generator.normal(size=(80, 513)).astype(np.float32)), dim=-1), [513, 1, 513, 299, 4], 0), (torch.full((5, 3), float("-inf")), [1, 3], 1)]
        with NativeCTC(self.library, self.identity) as context:
            for emission, tokens, blank in cases:
                mask = torch.ones(emission.shape[1], dtype=torch.bool)
                mask[blank] = False
                extended = torch.cat([emission, emission[:, mask].max(dim=1).values[:, None]], dim=1)
                expected = alignment.backtrack(alignment.get_trellis(extended, tokens, blank), extended, tokens, blank)
                actual, record = context.compute_wildcard(emission.numpy(), np.asarray(tokens, dtype=np.int32), blank)
                if expected is None: self.assertIsNone(actual)
                else:
                    self.assertEqual([(p[0], p[1]) for p in actual], [(p.token_index, p.time_index) for p in expected])
                    for point, reference in zip(actual, expected): self.assertAlmostEqual(point[2], reference.score, delta=16 * np.finfo(np.float32).eps)
                self.assertEqual(record["logical_intermediate_wildcard_readback_bytes"], 0)
                self.assertEqual(record["logical_cpu_to_native_emission_bytes"], emission.numel() * 4)
                self.assertEqual(record["command_buffers_per_run"], 2)
                self.assertTrue(record["wildcard_fused"])
            for tokens, blank in [([5], 0), ([-1], 0), ([4], 4)]:
                with self.assertRaisesRegex(ValueError, "input_values_invalid"): context.compute_wildcard(np.zeros((6, 4), dtype=np.float32), np.asarray(tokens, dtype=np.int32), blank)
            self.compare(context, torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32), [1])

    def test_deferred_preparation_runs_fused_ctc_and_refuses_unmaterialized_values(self):
        emission = torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32)
        mask = torch.tensor([False, True])
        explicit = torch.cat([emission, emission[:, mask].max(dim=1).values[:, None]], dim=1)
        expected = alignment.backtrack(alignment.get_trellis(explicit, [2]), explicit, [2])
        with NativeCTC(self.library, self.identity) as context:
            spans = []
            with persistent_metal_ctc(alignment, spans, context), fused_wildcard_preparation(2):
                selected = emission[:, mask]
                self.assertIsInstance(selected, DeferredWildcard)
                maximum = selected.max(dim=1)
                with self.assertRaises(ValueError): maximum.values.sum()
                with self.assertRaises(ValueError): maximum.indices.sum()
                extended = torch.cat([emission, maximum.values.unsqueeze(1)], dim=1)
                self.assertEqual(extended.shape, explicit.shape)
                trellis = alignment.get_trellis(extended, [2])
                actual = alignment.backtrack(trellis, extended, [2])
                self.assertEqual([(p.token_index, p.time_index) for p in actual], [(p.token_index, p.time_index) for p in expected])
                with self.assertRaisesRegex(ValueError, "base_identity_mismatch"): torch.cat([emission.clone(), maximum.values.unsqueeze(1)], dim=1)
            self.assertEqual(spans[0]["logical_intermediate_wildcard_readback_bytes"], 0)
            self.assertTrue(spans[0]["wildcard_fused"])
        self.assertNotIsInstance(emission[:, mask], DeferredWildcard)
        with self.assertRaisesRegex(ValueError, "vocabulary_mismatch"):
            with fused_wildcard_preparation(3): emission[:, mask]
        self.assertNotIsInstance(emission[:, mask], DeferredWildcard)

    def test_gpu_character_segments_match_cpython_compensated_means_bitwise(self):
        generator = np.random.default_rng(71)
        cases = [(torch.log_softmax(torch.from_numpy(generator.normal(size=(240, 32)).astype(np.float32)), dim=-1), generator.integers(1, 32, size=35).tolist(), 0, False), (torch.full((16, 4), -1.), [1, 4, 3, 4], 2, True)]
        stress = torch.full((66, 3), -100000., dtype=torch.float32)
        stress[0, 1] = 0
        stress[1:65, 0] = float(np.log(1e-17))
        stress[65, 2] = 0
        cases.append((stress, [1, 2], 0, False))
        for count in (31, 63, 127):
            varied = torch.full((count + 2, 3), -100000., dtype=torch.float32)
            varied[0, 1] = 0
            varied[1:count + 1, 0] = torch.from_numpy(generator.uniform(-103, 0, size=count).astype(np.float32))
            varied[count + 1, 2] = 0
            cases.insert(0, (varied, [1, 2], 0, False))
        with NativeCTC(self.library, self.identity) as context:
            for emission, tokens, blank, wildcard in cases:
                points, _ = context.compute_wildcard(emission.numpy(), np.asarray(tokens, dtype=np.int32), blank) if wildcard else context.compute(emission.numpy(), np.asarray(tokens, dtype=np.int32), blank)
                original = alignment.merge_repeats([alignment.Point(*point) for point in points], "x" * len(tokens))
                groups, record = context.compute_segments(emission.numpy(), np.asarray(tokens, dtype=np.int32), blank, wildcard)
                self.assertEqual([(start, end) for start, end, score in groups], [(item.start, item.end) for item in original])
                for group, reference in zip(groups, original): self.assertEqual(struct.pack("d", group[2]), struct.pack("d", reference.score))
                self.assertEqual(record["logical_final_path_readback_bytes"], 0)
                self.assertEqual(record["logical_character_segment_readback_bytes"], len(tokens) * 16)
                self.assertEqual(record["gpu_kernels"][-1], "ctc_merge_repeats")
                self.assertEqual(record["command_buffers_per_run"], 2)
            values = [point[2] for point in points if point[0] == 0]
            sequential = 0.
            for value in values: sequential += value
            self.assertNotEqual(sum(values), sequential)
            impossible = np.full((8, 3), float("-inf"), dtype=np.float32)
            impossible[:, 0] = 0
            groups, _ = context.compute_segments(impossible, np.asarray([1, 2], dtype=np.int32))
            self.assertIsNone(groups)

    def test_character_segment_hook_preserves_labels_and_restores_on_failure(self):
        emission = torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32)
        originals = alignment.get_trellis, alignment.backtrack, alignment.merge_repeats
        with NativeCTC(self.library, self.identity) as context:
            records = []
            with persistent_metal_ctc(alignment, records, context, character_segments=True):
                trellis = alignment.get_trellis(emission, [1])
                path = alignment.backtrack(trellis, emission, [1])
                result = alignment.merge_repeats(path, "è")
                self.assertEqual(result[0].label, "è")
                self.assertEqual((result[0].start, result[0].end), (0, 1))
                with self.assertRaises(ValueError): alignment.merge_repeats(path, "wrong")
            self.assertEqual(records[0]["logical_final_path_readback_bytes"], 0)
            with self.assertRaises(ValueError):
                with persistent_metal_ctc(alignment, [], context, character_segments=True): alignment.get_trellis(emission, [-1])
        self.assertIs(alignment.get_trellis, originals[0])
        self.assertIs(alignment.backtrack, originals[1])
        self.assertIs(alignment.merge_repeats, originals[2])

    def test_c_abi_lifetime_lengths_and_buffer_reuse_under_sanitizers(self):
        result = subprocess.run([str(self.harness), str(self.root / "tools/baseline/native_ctc.metal")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("AddressSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)


if __name__ == "__main__":
    unittest.main()

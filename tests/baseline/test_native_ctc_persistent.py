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

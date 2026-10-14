import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_speaker import NativeSpeakerAssignment, assign_word_speakers, build_library, native_speaker_assignment, shader_source
from whisperx.diarize import IntervalTree, assign_word_speakers as original_assignment


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeSpeakerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "speaker.dylib"
        cls.identity = build_library(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeSpeakerAssignment(self.library, self.identity["library_sha256"])

    def reference(self, turns, labels, queries, nearest):
        tree = IntervalTree([(a, b, str(s)) for (a, b), s in zip(turns, labels)])
        results = []
        for start, end in queries:
            totals = {}
            for speaker, overlap in tree.query(start, end):
                totals[speaker] = totals.get(speaker, 0.0) + overlap
            selected = max(totals, key=totals.get) if totals else tree.find_nearest((start + end) / 2) if nearest else None
            results.append(-1 if selected is None else int(selected))
        return results

    def compare(self, context, turns, labels, queries, nearest=False):
        turns, queries = np.ascontiguousarray(turns, dtype=np.float64), np.ascontiguousarray(queries, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.uint32)
        actual, report = context.assign(turns, labels, queries, int(labels.max()) + 1, nearest)
        self.assertEqual(actual.tolist(), self.reference(turns, labels, queries, nearest))
        self.assertGreater(report["computation_gpu_seconds"], 0)
        self.assertEqual(report["logical_result_readback_bytes"], len(queries) * 4)
        self.assertEqual(report["gpu_kernels"], ["speaker_stable_order", "speaker_assign"])
        self.assertFalse(report["full_pipeline_gpu_certified"])

    def test_exact_ties_stable_start_sort_boundaries_zero_length_and_nearest(self):
        turns = [[2., 4.], [0., 2.], [0., 2.], [4., 6.], [1., 3.]]
        labels = [0, 1, 2, 3, 4]
        queries = [[0., 0.], [1., 1.], [1., 3.], [2., 2.], [3., 5.], [6., 8.], [8., 8.]]
        with self.context() as context:
            for nearest in (False, True):
                self.compare(context, turns, labels, queries, nearest)

    def test_binary64_distinctions_survive_float32_collisions(self):
        tiny = np.nextafter(1., 2.)
        turns = [[0., 1.], [1., tiny], [tiny, 2.], [0., 2. ** -850], [2. ** 31, 2. ** 31 + 1.]]
        labels = [0, 1, 2, 3, 4]
        queries = [[1., tiny], [tiny, tiny], [0., 2. ** -851], [2. ** 31, 2. ** 31 + .5]]
        with self.context() as context:
            self.compare(context, turns, labels, queries)
            self.compare(context, turns, labels, queries, True)
            turns = np.array([[0., .3], [0., .1], [0., .2]])
            labels = np.array([0, 1, 1], dtype=np.uint32)
            choices, _ = context.assign(turns, labels, np.array([[0., 1.]]), 2)
            self.assertEqual(choices.tolist(), [1])
            self.compare(context, turns, labels, [[0., 1.]])

    def test_random_overlaps_repeated_speakers_and_exact_accumulation(self):
        rng = np.random.default_rng(8901)
        with self.context() as context:
            for size in (1, 7, 257, 1024):
                starts = rng.uniform(0, 1000, size=size)
                turns = np.column_stack([starts, starts + rng.uniform(0, 50, size=size)])
                labels = rng.integers(0, 13, size=size, dtype=np.uint32)
                starts = rng.uniform(0, 1050, size=513)
                queries = np.column_stack([starts, starts + rng.uniform(0, 80, size=513)])
                for nearest in (False, True):
                    self.compare(context, turns, labels, queries, nearest)

    def test_adapter_preserves_aliases_missing_words_embeddings_and_batching(self):
        frame = pd.DataFrame([{"start": 2., "end": 4., "speaker": "B"}, {"start": 0., "end": 2., "speaker": "A"}])
        class Timestamp(np.float64):
            pass
        shared = {"word": "ciao", "start": Timestamp(1.), "end": Timestamp(3.)}
        original = {"segments": [{"start": 0., "end": 4., "words": [shared, {"word": "?"}, {"start": 8., "speaker": "old"}]}], "word_segments": [shared]}
        embeddings = {"A": [1., 2.]}
        with self.context() as context:
            for nearest in (False, True):
                source = copy.deepcopy(original)
                expected = original_assignment(frame, copy.deepcopy(original), embeddings, nearest)
                records = []
                self.assertIs(assign_word_speakers(context, records, frame, source, embeddings, nearest), source)
                self.assertEqual(source, expected)
                self.assertIs(source["segments"][0]["words"][0], source["word_segments"][0])
                self.assertIs(source["speaker_embeddings"], embeddings)
                self.assertEqual(len(records), 1)
            source = {"segments": [{"start": 0., "end": 1., "words": [{"start": .2, "end": .3} for _ in range(4100)]}]}
            records = []
            expected = original_assignment(frame, copy.deepcopy(source))
            self.assertEqual(assign_word_speakers(context, records, frame, source), expected)
            self.assertEqual(len(records), 2)
            empty = {"segments": []}
            self.assertIs(assign_word_speakers(context, [], frame, empty, embeddings), empty)
            self.assertNotIn("speaker_embeddings", empty)
            with self.assertRaisesRegex(ValueError, "timestamp_type_invalid"):
                assign_word_speakers(context, [], frame, {"segments": [{"start": np.float32(0.), "end": 1.}]})

    def test_invalid_domains_hash_labels_and_close_fail_then_valid_call_recovers(self):
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            NativeSpeakerAssignment(self.library, "0" * 64)
        with self.context() as context:
            turns, labels, queries = np.array([[0., 1.]]), np.array([0], dtype=np.uint32), np.array([[0., 1.]])
            for value in (np.nan, np.inf, -0., -1., 2. ** -1000, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "timestamp_domain"):
                    context.assign(turns, labels, np.array([[value, value]]), 1)
            with self.assertRaisesRegex(ValueError, "reversed"):
                context.assign(turns, labels, np.array([[2., 1.]]), 1)
            with self.assertRaisesRegex(ValueError, "label_out_of_range"):
                context.assign(turns, np.array([2], dtype=np.uint32), queries, 1)
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.assign(turns.astype(np.float32), labels, queries, 1)
            self.compare(context, turns, labels, queries)
        context.close()
        with self.assertRaisesRegex(ValueError, "context_closed"):
            context.assign(turns, labels, queries, 1)

    def test_serialized_threads_and_scoped_hook_restoration(self):
        with self.context() as context:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.compare, context, [[0., 2.]], [0], [[.2, .4]], True) for _ in range(4)]
                for future in futures:
                    future.result()
            task = SimpleNamespace(assign_word_speakers=original_assignment)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with native_speaker_assignment(task, context, []):
                    self.assertIsNot(task.assign_word_speakers, original_assignment)
                    raise RuntimeError("injected")
            self.assertIs(task.assign_word_speakers, original_assignment)

    def test_native_abi_under_address_and_undefined_sanitizers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "speaker-sanitizers"
            shader = Path(directory) / "speaker.metal"
            shader.write_bytes(shader_source())
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-fobjc-arc", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Wno-deprecated-declarations", str(root / "tools/baseline/native_speaker.mm"), str(root / "tests/fixtures/baseline/native_speaker_abi.cpp"), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)], capture_output=True, check=True, timeout=60)
            result = subprocess.run([str(binary), str(shader)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("runtime error:", result.stderr)
            self.assertNotIn("ERROR: AddressSanitizer", result.stderr)


if __name__ == "__main__":
    unittest.main()

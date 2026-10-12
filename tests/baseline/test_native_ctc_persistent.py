from concurrent.futures import ThreadPoolExecutor
import hashlib
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
from tools.baseline.ctc_alignment_trial import persistent_metal_ctc, native_wildcard_max


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

    def test_c_abi_lifetime_lengths_and_buffer_reuse_under_sanitizers(self):
        result = subprocess.run([str(self.harness), str(self.root / "tools/baseline/native_ctc.metal")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("AddressSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)


if __name__ == "__main__":
    unittest.main()

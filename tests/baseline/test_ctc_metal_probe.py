from pathlib import Path
import sys
import unittest
import hashlib
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
import numpy as np
import torch
from tools.baseline import ctc_metal_probe as probe
from whisperx.alignment import get_trellis, backtrack


@unittest.skipUnless(mx.metal.is_available(), "Metal hardware required for actual kernel tests")
class MetalCTCProbeTests(unittest.TestCase):
    def assert_path(self, emission, tokens, blank=0):
        reference_trellis = get_trellis(emission, tokens, blank)
        reference_path = backtrack(reference_trellis, emission, tokens, blank)
        result = probe.compute(mx.array(emission.numpy()), mx.array(tokens, dtype=mx.int32), blank)
        actual_path = probe.materialize_path(result)
        if reference_path is None:
            self.assertIsNone(actual_path)
        else:
            self.assertEqual([(token, frame) for token, frame, _ in actual_path], [(point.token_index, point.time_index) for point in reference_path])
            for (_, _, score), point in zip(actual_path, reference_path): self.assertAlmostEqual(score, point.score, delta=16 * np.finfo(np.float32).eps)
        actual = np.array(result["trellis"])
        np.testing.assert_array_equal(np.array(result["prefix"])[1:], torch.cumsum(emission[:, blank], 0).numpy())
        expected = reference_trellis.numpy()
        np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = np.isfinite(expected)
        bound = 8 * np.finfo(np.float32).eps * emission.shape[0] * max(1., float(emission.abs().max()))
        np.testing.assert_allclose(actual[finite], expected[finite], atol=bound, rtol=0)
        self.assertFalse(result["production_backend"])
        self.assertFalse(result["quality_certified"])

    def test_seeded_inputs_include_strided_threadgroup_tokens(self):
        generator = np.random.default_rng(17)
        for frames, count in [(8, 1), (64, 17), (340, 300)]:
            with self.subTest(frames=frames, count=count):
                emission = torch.log_softmax(torch.from_numpy(generator.normal(size=(frames, 32)).astype(np.float32)), dim=-1)
                self.assert_path(emission, generator.integers(1, 32, size=count).tolist())

    def test_ties_repeats_and_nonzero_blank_preserve_original_path(self):
        self.assert_path(torch.full((16, 4), -1., dtype=torch.float32), [1, 1, 3], blank=2)
        self.assert_path(torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32), [1])

    def test_impossible_path_preserves_failure(self):
        emission = torch.full((8, 3), -float("inf"), dtype=torch.float32)
        emission[:, 0] = 0
        self.assert_path(emission, [1, 2])

    def test_near_ties_use_compensated_prefix_without_parallel_sum_reordering(self):
        generator = np.random.default_rng(41)
        for seed in range(10):
            with self.subTest(seed=seed):
                emission = torch.from_numpy((-1. + generator.normal(scale=1e-6, size=(80, 8))).astype(np.float32))
                self.assert_path(emission, generator.integers(1, 8, size=25).tolist())

    def test_invalid_tokens_nan_empty_input_and_budget_fail_before_kernel(self):
        emission = mx.zeros((4, 3), dtype=mx.float32)
        for tokens in ([], [-1], [3]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError): probe.compute(emission, mx.array(tokens, dtype=mx.int32))
        with self.assertRaises(ValueError): probe.compute(emission, mx.array([1], dtype=mx.int32), max_cells=1)
        with self.assertRaises(ValueError): probe.compute(mx.full((4, 3), float("nan")), mx.array([1], dtype=mx.int32))

    def test_real_input_requires_hash_and_numeric_bounded_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emissions.npz"
            np.savez(path, emission=np.zeros((4, 3), dtype=np.float32), tokens=np.array([1], dtype=np.int32), blank_id=np.array(0, dtype=np.int32))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            emission, tokens, blank = probe.load_inputs(path, digest)
            self.assertEqual(emission.shape, (4, 3))
            self.assertEqual(tokens.tolist(), [1])
            self.assertEqual(blank, 0)
            with self.assertRaises(ValueError): probe.load_inputs(path, "a" * 64)
            np.savez(path, emission=np.zeros((4, 3), dtype=np.float64), tokens=np.array([1], dtype=np.int32), blank_id=np.array(0, dtype=np.int32))
            with self.assertRaises(ValueError): probe.load_inputs(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from whisperx.alignment import get_trellis, backtrack
import whisperx.alignment as alignment
from tools.baseline.ctc_alignment_trial import native_metal_ctc


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal hardware required")
class NativeMetalCTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "native_ctc_probe"
        root = Path(__file__).resolve().parents[2]
        cls.shader = root / "tools/baseline/native_ctc.metal"
        arguments = ["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-fobjc-arc", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Wno-unused-function", "-Wno-deprecated-declarations", str(root / "tools/baseline/native_ctc_probe.mm"), "-framework", "Foundation", "-framework", "Metal", "-o", str(cls.binary)]
        subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, path, compact=False, batched=False):
        return subprocess.run([str(self.binary), "--input", str(path), "--sha256", hashlib.sha256(path.read_bytes()).hexdigest(), "--kernels", str(self.shader), "--replicas", "1", *(["--compact"] if compact else []), *(["--batched"] if batched else [])], capture_output=True, text=True, timeout=30)

    def test_batched_dependencies_and_reused_buffers_preserve_complete_path(self):
        generator = np.random.default_rng(31)
        emission = torch.log_softmax(torch.from_numpy(generator.normal(size=(341, 38)).astype(np.float32)), dim=-1)
        tokens = generator.integers(1, 38, size=87).tolist()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emissions.npz"
            self.save(path, emission, tokens)
            for compact in (False, True):
                with self.subTest(compact=compact):
                    reference = self.invoke(path, compact)
                    candidate = self.invoke(path, compact, batched=True)
                    self.assertEqual(reference.returncode, 0, reference.stderr)
                    self.assertEqual(candidate.returncode, 0, candidate.stderr)
                    original, batched = json.loads(reference.stdout), json.loads(candidate.stdout)
                    self.assertEqual(original["path"], batched["path"])
                    self.assertEqual(batched["command_buffers_per_run"], 2)
                    self.assertEqual(len(batched["runs"]), 2)
                    self.assertTrue(all(len(run["stages"]) == 2 and all(stage["dispatch_completed"] for stage in run["stages"]) for run in batched["runs"]))
                    self.assertIn("kernel_group", batched["runs"][0]["stages"][1])
                    self.assertNotIn("AddressSanitizer", candidate.stderr)
                    self.assertNotIn("runtime error:", candidate.stderr)

    def test_native_alignment_adapter_preserves_points_and_restores_hooks_on_failure(self):
        emission = torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32)
        expected = backtrack(get_trellis(emission, [1]), emission, [1])
        original_trellis, original_backtrack = alignment.get_trellis, alignment.backtrack
        identity = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        spans = []
        with native_metal_ctc(alignment, spans, self.binary, identity):
            trellis = alignment.get_trellis(emission, [1])
            self.assertEqual(trellis.size(0), 4)
            result = alignment.backtrack(trellis, emission, [1])
            self.assertEqual([(p.token_index, p.time_index) for p in result], [(p.token_index, p.time_index) for p in expected])
            for actual, reference in zip(result, expected): self.assertAlmostEqual(actual.score, reference.score, delta=16 * np.finfo(np.float32).eps)
        self.assertIs(alignment.get_trellis, original_trellis)
        self.assertIs(alignment.backtrack, original_backtrack)
        self.assertEqual(spans[0]["command_buffers_per_run"], 2)
        with self.assertRaises(ValueError):
            with native_metal_ctc(alignment, [], self.binary, identity): alignment.get_trellis(emission, [-1])
        self.assertIs(alignment.get_trellis, original_trellis)
        self.assertIs(alignment.backtrack, original_backtrack)
        with self.assertRaises(ValueError):
            with native_metal_ctc(alignment, [], self.binary, "0" * 64): pass
        self.assertIs(alignment.get_trellis, original_trellis)

    def save(self, path, emission, tokens, blank=0):
        np.savez(path, emission=emission.numpy(), tokens=np.asarray(tokens, dtype=np.int32), blank_id=np.asarray(blank, dtype=np.int32))

    def test_native_gpu_path_matches_original_with_ties_and_strided_tokens(self):
        generator = np.random.default_rng(17)
        cases = [(torch.full((16, 4), -1., dtype=torch.float32), [1, 1, 3], 2)]
        emission = torch.log_softmax(torch.from_numpy(generator.normal(size=(340, 32)).astype(np.float32)), dim=-1)
        cases.append((emission, generator.integers(1, 32, size=300).tolist(), 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emissions.npz"
            for emission, tokens, blank in cases:
                for compact in (False, True):
                    with self.subTest(frames=emission.shape[0], tokens=len(tokens), compact=compact):
                        self.save(path, emission, tokens, blank)
                        result = self.invoke(path, compact)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        report = json.loads(result.stdout)
                        expected = backtrack(get_trellis(emission, tokens, blank), emission, tokens, blank)
                        self.assertEqual([(p[0], p[1]) for p in report["path"]], [(p.token_index, p.time_index) for p in expected])
                        for actual, reference in zip(report["path"], expected): self.assertAlmostEqual(actual[2], reference.score, delta=16 * np.finfo(np.float32).eps)
                        self.assertFalse(report["python_required"])
                        self.assertFalse(report["full_baseline_gate_passed"])
                        suffix = "_compact" if compact else ""
                        self.assertEqual([stage["kernel"] for stage in report["runs"][0]["stages"]], ["ctc_validate", "ctc_prefix", "ctc_trellis" + suffix, "ctc_backtrack" + suffix])
                        self.assertTrue(all(stage["dispatch_completed"] for run in report["runs"] for stage in run["stages"]))
                        if compact: self.assertLess(report["dp_logical_storage_bytes"], (emission.shape[0] + 1) * (len(tokens) + 1) * 4)

    def test_invalid_gpu_values_and_shapes_are_rejected_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emissions.npz"
            for emission, tokens in [(torch.zeros((4, 3), dtype=torch.float32), [-1]), (torch.full((4, 3), float("nan"), dtype=torch.float32), [1]), (torch.zeros((4, 3), dtype=torch.float32), [])]:
                with self.subTest(tokens=tokens):
                    self.save(path, emission, tokens)
                    result = self.invoke(path, batched=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(json.loads(result.stdout)["code"], ("input_values_invalid", "input_shape_invalid"))
                    self.assertNotIn("AddressSanitizer", result.stderr)
                    self.assertNotIn("runtime error:", result.stderr)


if __name__ == "__main__":
    unittest.main()

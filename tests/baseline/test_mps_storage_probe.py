from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.mps_storage_probe import MPSStorageProbe, build_library


@unittest.skipUnless(platform.system() == "Darwin" and torch.__version__ == "2.8.0" and torch.backends.mps.is_available(), "Pinned Torch and Apple Metal required")
class MPSStorageProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.library = Path(cls.temporary.name) / "mps.dylib"
        cls.identity = build_library(cls.library)["library_sha256"]
        cls.probe = MPSStorageProbe(cls.library, cls.identity)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def compare(self, tensor):
        actual, indices, record = self.probe.row_max(tensor)
        expected, expected_indices = tensor.cpu().max(dim=1)
        self.assertEqual(actual.numpy().tobytes(), expected.numpy().tobytes())
        self.assertTrue(torch.equal(indices.to(torch.int64), expected_indices))
        self.assertGreater(record["computation_gpu_seconds"], 0)
        self.assertEqual(record["source_storage_offset_bytes"], tensor.storage_offset() * 4)
        self.assertEqual(record["logical_emission_host_readback_bytes"], 0)
        self.assertEqual(record["logical_emission_reupload_bytes"], 0)
        self.assertEqual(record["logical_final_result_readback_bytes"], tensor.size(0) * 8 + 4)
        self.assertFalse(record["production_backend"])
        return actual.clone(), indices.clone()

    def test_pending_mps_producers_nonzero_offsets_and_storage_reuse(self):
        generator = torch.Generator().manual_seed(51)
        for rows, columns in ((1, 1), (7, 38), (341, 38), (1025, 257)):
            base = torch.randn((rows + 3, columns), generator=generator).to("mps")
            produced = base.log_softmax(dim=1)
            view = produced[2:rows + 2]
            self.assertTrue(view.is_contiguous())
            self.assertGreater(view.storage_offset(), 0)
            self.compare(view)
        tensor = torch.full((13, 38), -2., device="mps")
        first, indices = self.compare(tensor)
        tensor.fill_(-3.)
        second, _ = self.compare(tensor)
        self.assertTrue(torch.equal(first, torch.full((13,), -2.)))
        self.assertTrue(torch.equal(second, torch.full((13,), -3.)))
        self.assertTrue(torch.equal(indices, torch.zeros(13, dtype=torch.int32)))

    def test_ties_infinities_and_signed_zero_preserve_first_index_and_bits(self):
        values = np.array([[-np.inf, -np.inf, -np.inf], [-2., -1., -1.], [-0., 0., -1.], [0., -0., -1.]], dtype=np.float32)
        maxima, indices = self.compare(torch.from_numpy(values).to("mps"))
        self.assertEqual(indices.tolist(), [0, 1, 0, 0])
        self.assertEqual(maxima.numpy()[2:].view(np.uint32).tolist(), [0x80000000, 0])

    def test_invalid_tensor_and_gpu_scores_fail_then_recover(self):
        invalid = [torch.zeros((2, 3)), torch.zeros((2, 3), device="mps", dtype=torch.float16), torch.zeros((2, 3), device="mps", requires_grad=True), torch.zeros((2, 3), device="mps").T, torch.zeros((0, 3), device="mps"), torch.zeros((2, 4097), device="mps"), torch.zeros((32769, 1), device="mps"), torch.zeros((2,), device="mps")]
        for tensor in invalid:
            with self.subTest(shape=tuple(tensor.shape)), self.assertRaisesRegex(RuntimeError, "mps_probe_(tensor|shape)_not_supported"): self.probe.row_max(tensor)
        negative = torch._neg_view(torch.ones((2, 3), device="mps"))
        with self.assertRaisesRegex(ValueError, "mps_probe_tensor_not_supported"): self.probe.row_max(negative)
        with self.assertRaisesRegex(RuntimeError, "mps_probe_tensor_not_supported"): torch.ops.whisperx_mps_probe.row_max(negative)
        for value in (np.nan, np.inf, 0.1):
            with self.assertRaisesRegex(RuntimeError, "mps_probe_log_scores_invalid"): self.probe.row_max(torch.tensor([[-1., value]], device="mps"))
        self.compare(torch.tensor([[-1., -2.]], device="mps"))

    def test_serialized_callers_and_temporary_tensor_ownership(self):
        def run(index):
            tensor = torch.full((129, 38), -float(index + 1), device="mps").log_softmax(dim=1)[1:]
            maxima, indices, record = self.probe.row_max(tensor)
            expected = tensor.cpu().max(dim=1)
            return maxima.numpy().tobytes() == expected.values.numpy().tobytes() and torch.equal(indices.to(torch.int64), expected.indices)
        with ThreadPoolExecutor(max_workers=3) as pool: self.assertTrue(all(pool.map(run, range(9))))
        self.compare(torch.full((23, 38), -1., device="mps")[3:])

    def test_library_runtime_guards_and_exclusive_build(self):
        with self.assertRaisesRegex(ValueError, "hash_mismatch"): MPSStorageProbe(self.library, "0" * 64)
        with patch.object(torch, "__version__", "2.9.0"), self.assertRaisesRegex(ValueError, "runtime_not_supported"): self.probe.row_max(torch.zeros((2, 3), device="mps"))
        original = self.library.read_bytes()
        with self.assertRaisesRegex(ValueError, "output_exists"): build_library(self.library)
        self.assertEqual(self.library.read_bytes(), original)
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "other.dylib"
            other.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "different_library_already_loaded"): MPSStorageProbe(other, self.identity)
            link = Path(directory) / "dangling.dylib"
            target = Path(directory) / "absent.dylib"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "output_exists"): build_library(link)
            self.assertTrue(link.is_symlink())
            self.assertFalse(target.exists())

    def test_cli_help_and_captured_emissions_report(self):
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-I", "-B", str(root / "tools/baseline/mps_storage_probe.py")]
        help_result = subprocess.run([*command, "probe", "--help"], capture_output=True, text=True, check=True, timeout=20)
        self.assertIn("--emissions-sha256", help_result.stdout)
        live_help = subprocess.run([*command, "live", "--help"], capture_output=True, text=True, check=True, timeout=20)
        self.assertIn("--torch-specs", live_help.stdout)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "emissions.npz"
            np.savez(source, emission=np.array([[-1., -2.], [-3., -1.]], dtype=np.float32), tokens=np.array([1], dtype=np.int32), blank_id=np.array(0, dtype=np.int32))
            output = Path(directory) / "report.json"
            arguments = ["probe", "--library", str(self.library), "--library-sha256", self.identity, "--emissions", str(source), "--emissions-sha256", hashlib.sha256(source.read_bytes()).hexdigest(), "--replicas", "2", "--out", str(output)]
            result = subprocess.run([*command, *arguments], capture_output=True, text=True, check=True, timeout=30)
            self.assertIn("mps_storage_row_max_exactly_equal", result.stdout)
            original = output.read_bytes()
            repeated = subprocess.run([*command, *arguments], capture_output=True, text=True, timeout=30)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(output.read_bytes(), original)
            specs = Path(directory) / "specs.json"
            specs.write_text(json.dumps([{"role": "alignment_it", "license_verified": False}]))
            live_output = Path(directory) / "live.json"
            live_args = ["live", "--library", str(self.library), "--library-sha256", self.identity, "--audio", str(source), "--audio-sha256", hashlib.sha256(source.read_bytes()).hexdigest(), "--torch-specs", str(specs), "--language", "it", "--out", str(live_output)]
            refused = subprocess.run([*command, *live_args], capture_output=True, text=True, timeout=30)
            self.assertEqual(refused.returncode, 2)
            self.assertFalse(live_output.exists())
            for option, value in (("--start", "nan"), ("--duration", "inf"), ("--replicas", "0")):
                invalid = subprocess.run([*command, *live_args, option, value], capture_output=True, text=True, timeout=30)
                self.assertEqual(invalid.returncode, 2)
                self.assertFalse(live_output.exists())


if __name__ == "__main__": unittest.main()

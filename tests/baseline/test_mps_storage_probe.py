from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import contextlib
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
from tools.baseline.mps_storage_probe import MPSStorageProbe, MPSCTC, build_library
from tools.baseline.native_mps_ctc import native_mps_ctc


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

    def ctc_context(self):
        kernels = Path(__file__).resolve().parents[2] / "tools/baseline/native_ctc.metal"
        return MPSCTC(self.library, self.identity, kernels, hashlib.sha256(kernels.read_bytes()).hexdigest())

    def ctc_compare(self, context, emission, tokens, blank=0, wildcard=False):
        import whisperx.alignment as alignment
        reference = emission.cpu()
        if wildcard:
            mask = torch.ones(reference.shape[1], dtype=torch.bool)
            mask[blank] = False
            reference = torch.cat([reference, reference[:, mask].max(dim=1).values.unsqueeze(1)], dim=1)
        expected = alignment.backtrack(alignment.get_trellis(reference, tokens, blank), reference, tokens, blank)
        actual, record = context.compute(emission, tokens, blank, wildcard=wildcard)
        if expected is None: self.assertIsNone(actual)
        else:
            self.assertEqual([(row[0], row[1]) for row in actual], [(point.token_index, point.time_index) for point in expected])
            for row, point in zip(actual, expected): self.assertAlmostEqual(row[2], point.score, delta=16 * np.finfo(np.float32).eps)
        self.assertEqual(record["borrowed_source_storage_offset_bytes"], emission.storage_offset() * 4)
        self.assertEqual(record["logical_emission_host_readback_bytes"], 0)
        self.assertEqual(record["logical_cpu_to_native_emission_bytes"], 0)
        self.assertEqual(record["command_buffers_per_run"], 2)
        self.assertGreater(record["computation_gpu_seconds"], 0)
        return actual

    def test_borrowed_ctc_offsets_wildcards_growth_and_impossible_paths(self):
        generator = torch.Generator().manual_seed(341)
        with self.ctc_context() as context:
            for rows, columns, tokens, blank, wildcard in ((19, 6, [1, 2, 2, 4], 0, False), (341, 38, [1, 2, 3, 38, 4], 0, True), (127, 7, [0, 1, 7, 0], 3, True), (17, 4, [0, 2], 1, False)):
                emission = torch.randn((rows + 3, columns), generator=generator).to("mps").log_softmax(dim=1)[2:rows + 2]
                self.ctc_compare(context, emission, tokens, blank, wildcard)
            self.ctc_compare(context, torch.full((9, 4), -np.inf, device="mps"), [1, 2])
            with self.assertRaisesRegex(RuntimeError, "input_values_invalid"): context.compute(torch.tensor([[-1., np.nan], [-2., -1.]], device="mps"), [1])
            self.ctc_compare(context, torch.tensor([[-1., -2.], [-2., -1.], [-1., -3.]], device="mps"), [1])
        context.close()
        with self.assertRaisesRegex(ValueError, "context_closed"): context.compute(torch.zeros((2, 3), device="mps"), [1])

    def test_borrowed_ctc_guards_and_native_close_reject_stale_calls(self):
        with self.ctc_context() as context:
            valid = torch.full((7, 4), -1., device="mps")
            for value in (valid.cpu(), valid.T, valid.half(), valid.clone().requires_grad_(), torch._neg_view(valid)):
                with self.assertRaisesRegex(RuntimeError, "mps_ctc_tensor_not_supported"): context.compute(value, [1, 2])
            for tokens in ([True], [2**32], [1.], np.array([1], dtype=np.int32)):
                with self.assertRaisesRegex(ValueError, "token_metadata_invalid"): context.compute(valid, tokens)
            with self.assertRaisesRegex(RuntimeError, "shape_or_blank_invalid"): context.compute(valid, [1], blank=4)
            with self.assertRaisesRegex(RuntimeError, "input_values_invalid"): context.compute(valid, [-1])
            native = context.context
            context.close()
            with self.assertRaisesRegex(RuntimeError, "context_closed"): native.compute(valid, torch.tensor([1], dtype=torch.int32), 0, False)

    def test_scoped_live_emission_transport_and_restore_without_cpu_matrix(self):
        import whisperx.alignment as alignment
        original_trellis, original_backtrack = alignment.get_trellis, alignment.backtrack
        for wildcard, inference in ((False, False), (True, False), (False, True), (True, True)):
            records = []
            logits = torch.randn((1, 37, 6), generator=torch.Generator().manual_seed(915)).to("mps")
            tokens = [1, 2, 6, 3] if wildcard else [1, 2, 3]
            with self.ctc_context() as context:
                with native_mps_ctc(alignment, context, records):
                    with torch.inference_mode() if inference else contextlib.nullcontext(): scores = torch.log_softmax(logits, dim=-1)
                    emission = scores[0].cpu().detach()
                    self.assertEqual(emission.base_emission.device.type, "mps")
                    if wildcard:
                        mask = torch.ones(6, dtype=torch.bool); mask[0] = False
                        emission = torch.cat([emission, emission[:, mask].max(dim=1).values.unsqueeze(1)], dim=1)
                    trellis = alignment.get_trellis(emission, tokens, 0)
                    actual = alignment.backtrack(trellis, emission, tokens, 0)
                    with self.assertRaises((ValueError, RuntimeError)): emission.numpy()
                expected = self.ctc_compare(context, torch.log_softmax(logits, dim=-1)[0], tokens, wildcard=wildcard)
                self.assertEqual([(p.token_index, p.time_index, p.score) for p in actual], expected)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["wildcard_fused"], wildcard)
                with self.assertRaisesRegex(ValueError, "handoff_missing"):
                    with native_mps_ctc(alignment, context, []): alignment.get_trellis(torch.zeros((7, 6)), [1])
            self.assertIs(alignment.get_trellis, original_trellis)
            self.assertIs(alignment.backtrack, original_backtrack)

    def test_native_borrowed_ranges_and_resident_invalidation_under_sanitizers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "borrowed-harness"
            subprocess.run(["xcrun", "clang++", "-std=c++17", "-O1", "-g", "-fobjc-arc", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Wno-deprecated-declarations", str(root / "tests/fixtures/baseline/mps_ctc_storage.mm"), "-I", str(root / "tools/baseline"), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)], capture_output=True, check=True, timeout=60)
            result = subprocess.run([str(binary), str(root / "tools/baseline/native_ctc.metal")], capture_output=True, text=True, check=True, timeout=30)
            self.assertIn("borrowed_storage_sanitizer_verified", result.stdout)
            self.assertNotIn("runtime error", result.stderr)

    def test_resident_mps_characters_rounding_and_statistics_match_original_bits(self):
        import pandas as pd
        import whisperx.alignment as alignment
        generator = torch.Generator().manual_seed(1277)
        emission = torch.randn((127, 12), generator=generator).to("mps").log_softmax(dim=1)
        tokens = [1, 2, 2, 4]
        with self.ctc_context() as context:
            path, _ = context.compute(emission, tokens)
            segments = alignment.merge_repeats([alignment.Point(*point) for point in path], "abcd")
            resident, record = context.prepare_segments(emission, tokens)
            self.assertEqual(record["logical_final_path_and_control_readback_bytes"], 8)
            self.assertTrue(record["character_segments_resident"])
            for ratio, offset in ((0.02, 0.), (0.013123457, 19.123)):
                rounded, scale_record = context.scale_round_segments(resident, ratio, offset)
                expected = np.asarray([[round(segment.start * ratio + offset, 3), round(segment.end * ratio + offset, 3), round(segment.score, 3)] for segment in segments])
                self.assertEqual(rounded.tobytes(), expected.tobytes())
                self.assertEqual(scale_record["logical_character_reupload_bytes"], 0)
                values = np.insert(rounded, 2, [np.nan] * 3, axis=0)
                groups = [[], [2], [0, 1, 2, 3, 4], [1, 1, 4], [4, 0]]
                indices = np.asarray([index for group in groups for index in group], dtype=np.uint32)
                offsets = np.asarray([0, *np.cumsum([len(group) for group in groups])], dtype=np.uint32)
                modes = np.full(len(groups), 7, dtype=np.uint32)
                actual, means, stats_record = context.alignment_stats_resident_rounded(values, indices, offsets, modes)
                for index, group in enumerate(groups):
                    frame = pd.DataFrame(values[group])
                    reference = np.array([frame[0].min(), frame[1].max(), frame[2].mean()])
                    self.assertEqual(actual[index][~np.isnan(reference)].tobytes(), reference[~np.isnan(reference)].tobytes())
                    self.assertTrue(np.isnan(actual[index][np.isnan(reference)]).all())
                    if np.isnan(reference[2]): self.assertTrue(np.isnan(means[index]))
                    else: self.assertEqual(means[index].tobytes(), round(np.float64(reference[2]), 3).tobytes())
                self.assertEqual(stats_record["logical_character_values_upload_bytes"], 0)
                empty, _, _ = context.alignment_stats_resident_rounded(values, np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.uint32), np.array([7], dtype=np.uint32))
                self.assertTrue(np.isnan(empty).all())

    def test_resident_mps_stale_handles_mutation_failed_scale_and_close(self):
        emission = torch.full((19, 6), -1., device="mps")
        with self.ctc_context() as context:
            resident, _ = context.prepare_segments(emission, [1, 2])
            values, _ = context.scale_round_segments(resident, .02, 0.)
            indices, offsets, modes = np.array([0, 1], dtype=np.uint32), np.array([0, 2], dtype=np.uint32), np.array([7], dtype=np.uint32)
            changed = values.copy(); changed[0, 2] = .4
            with self.assertRaisesRegex(ValueError, "snapshot_changed"): context.alignment_stats_resident(changed, indices, offsets, modes)
            with self.assertRaisesRegex(RuntimeError, "indices_invalid|metadata_invalid"): context.alignment_stats_resident(values, indices, np.array([0, 3], dtype=np.uint32), modes)
            context.alignment_stats_resident(values, indices, offsets, modes)
            with self.assertRaisesRegex(RuntimeError, "segment_invalid|domain_invalid"): context.scale_round_segments(resident, -1., 0.)
            with self.assertRaisesRegex(ValueError, "snapshot_missing"): context.alignment_stats_resident(values, indices, offsets, modes)
            context.scale_round_segments(resident, .02, 0.)
            with self.assertRaisesRegex(RuntimeError, "tensor_not_supported"): context.compute(emission.cpu(), [1, 2])
            with self.assertRaisesRegex(RuntimeError, "generation_invalid"): context.scale_round_segments(resident, .02, 0.)
            with self.ctc_context() as other:
                with self.assertRaisesRegex(ValueError, "handle_invalid"): other.scale_round_segments(resident, .02, 0.)
            resident, _ = context.prepare_segments(emission, [1, 2])
            context.close()
            with self.assertRaisesRegex(ValueError, "context_closed"): context.scale_round_segments(resident, .02, 0.)

    def test_mps_nearest_bits_counts_pairs_and_resident_state_survive(self):
        import pandas as pd
        from whisperx.utils import interpolate_nans
        generator = np.random.default_rng(917)
        with self.ctc_context() as context:
            resident, _ = context.prepare_segments(torch.full((19, 6), -1., device="mps"), [1, 2])
            scaled, _ = context.scale_round_segments(resident, .02, 0.)
            for rows in (1, 7, 257, 8193, 32768):
                values = generator.normal(size=(rows, 2))
                values[generator.random(values.shape) < .4] = np.nan
                if rows > 3: values[1:4] = [[-0., np.inf], [np.nan, -np.inf], [0., np.nan]]
                output, record = context.interpolate_nearest_pair(values)
                for column in range(2):
                    expected = interpolate_nans(pd.Series(values[:, column]), "nearest").to_numpy(dtype=np.float64)
                    self.assertEqual(output[:, column].tobytes(), expected.tobytes())
                    missing = int(np.isnan(values[:, column]).sum())
                    self.assertEqual(record["input_missing_counts"][column], missing)
                    self.assertEqual(record["filled_missing_counts"][column], missing if missing < rows else 0)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertEqual(record["command_buffers_per_run"], 1)
            payloads = np.array([0x7ff8000000000051, 0x7ff8000000000062], dtype=np.uint64).view(np.float64)
            all_missing = np.tile(payloads, (7, 1))
            output, record = context.interpolate_nearest_pair(all_missing)
            self.assertEqual(output.tobytes(), all_missing.tobytes())
            self.assertEqual(record["filled_missing_counts"], [0, 0])
            readonly = np.array([np.nan, 1., np.nan, -0.]); readonly.flags.writeable = False
            result, record = context.interpolate_nearest(readonly)
            self.assertEqual(result.tobytes(), interpolate_nans(pd.Series(readonly), "nearest").to_numpy().tobytes())
            self.assertEqual(record["cpu_readonly_input_copy_bytes"], readonly.nbytes)
            context.alignment_stats_resident(scaled, np.array([0, 1], dtype=np.uint32), np.array([0, 2], dtype=np.uint32), np.array([7], dtype=np.uint32))
            for bad in (torch.zeros(2), torch.zeros(2, dtype=torch.float64, requires_grad=True), torch.zeros((2, 1), dtype=torch.float64), torch.zeros(0, dtype=torch.float64)):
                with self.assertRaisesRegex(RuntimeError, "nearest_(tensor|shape)_invalid"): context.context.nearest(bad, 1)
            with self.assertRaisesRegex(ValueError, "format_invalid"): context.interpolate_nearest(np.zeros(32769))
        with self.assertRaisesRegex(ValueError, "context_closed"): context.interpolate_nearest(np.array([1.]))

    def test_mps_paired_interpolation_scoped_restore_and_snapshot_mutation(self):
        import whisperx.alignment as alignment
        import pandas as pd
        from tools.baseline.native_interpolation import native_nearest_interpolation
        original_pd, original_interpolate = alignment.pd, alignment.interpolate_nans
        rows = [{"text": "a", "words": [], "start": 1., "end": 2.}, {"text": "", "words": [], "start": None, "end": None}, {"text": "b", "words": [], "start": 3., "end": 4.}]
        records = []
        with self.ctc_context() as context:
            with native_nearest_interpolation(alignment, context, records, batched_frames=True):
                frame = alignment.pd.DataFrame(rows)
                for column in ("start", "end"):
                    result = alignment.interpolate_nans(frame[column])
                    expected = original_interpolate(pd.DataFrame(rows)[column])
                    self.assertEqual(result.to_numpy().tobytes(), expected.to_numpy().tobytes())
                self.assertEqual(records[0]["columns_consumed"], 2)
                self.assertEqual(records[0]["filled_missing_counts"], [1, 1])
                frame.loc[1, "start"] = 8.
                with self.assertRaisesRegex(ValueError, "snapshot_changed"): alignment.interpolate_nans(frame["start"])
            self.assertIs(alignment.pd, original_pd)
            self.assertIs(alignment.interpolate_nans, original_interpolate)

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

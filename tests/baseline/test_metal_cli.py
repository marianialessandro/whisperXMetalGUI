import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.metal_cli import main, native_alignment


class MetalCLITests(unittest.TestCase):
    def test_speaker_backend_requires_paired_hash_and_diarization(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(Path(directory) / "unused-report.json"), "audio.wav"]
            for extra in [["--speaker-library", "speaker.dylib"], ["--speaker-library-sha256", "0" * 64], ["--speaker-library", "speaker.dylib", "--speaker-library-sha256", "0" * 64]]:
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    main([*common, *extra])
                self.assertEqual(caught.exception.code, 2)

    def test_embedding_backend_requires_paired_hash_and_diarization(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "unused-report.json"
            common = ["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav"]
            for extra in [["--embedding-library", "embedding.dylib"], ["--embedding-library-sha256", "0" * 64], ["--embedding-library", "embedding.dylib", "--embedding-library-sha256", "0" * 64]]:
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    main([*common, *extra])
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(report.exists())

    def test_plda_backend_requires_paired_hash_and_diarization(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "unused-report.json"
            common = ["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav"]
            for extra in [["--plda-normalization-library", "double.dylib"], ["--plda-normalization-library-sha256", "0" * 64], ["--plda-normalization-library", "double.dylib", "--plda-normalization-library-sha256", "0" * 64]]:
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    main([*common, *extra])
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(report.exists())

    def test_plda_matrix_flag_requires_normalization_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--plda-matrix-products"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_plda_center_scale_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--plda-centering-scaling"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_vbx_matrix_flag_requires_native_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-matrix-products"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_vbx_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = object()
                with self.assertRaisesRegex(ValueError, "matrix_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_matrix_products_requested"])
            self.assertFalse(data["vbx_matrix_products_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_statistics_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_matmul=object())
                with self.assertRaisesRegex(ValueError, "statistics_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-statistics", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_statistics_requested"])
            self.assertFalse(data["vbx_statistics_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_scores_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_matmul=object())
                with self.assertRaisesRegex(ValueError, "scores_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-score-arithmetic", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_score_arithmetic_requested"])
            self.assertFalse(data["vbx_score_arithmetic_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_prior_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_matmul=object())
                with self.assertRaisesRegex(ValueError, "prior_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-prior-updates", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_prior_updates_requested"])
            self.assertFalse(data["vbx_prior_updates_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_posterior_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_matmul=object())
                with self.assertRaisesRegex(ValueError, "posterior_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-posterior-arithmetic", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_posterior_arithmetic_requested"])
            self.assertFalse(data["vbx_posterior_arithmetic_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_elbo_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_matmul=object())
                with self.assertRaisesRegex(ValueError, "elbo_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-elbo-arithmetic", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_elbo_arithmetic_requested"])
            self.assertFalse(data["vbx_elbo_arithmetic_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_transcendental_capability_failure_restores_norm_scope_and_report_privacy(self):
        import pyannote.audio.utils.vbx as vbx
        original = vbx.l2_norm
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_matmul=object())
                with self.assertRaisesRegex(ValueError, "transcendental_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-transcendentals", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_transcendentals_requested"])
            self.assertFalse(data["vbx_transcendentals_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_statistics_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-statistics"])
            self.assertFalse(report.exists())

    def test_vbx_scores_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-score-arithmetic"])
            self.assertFalse(report.exists())

    def test_vbx_prior_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-prior-updates"])
            self.assertFalse(report.exists())

    def test_vbx_posterior_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-posterior-arithmetic"])
            self.assertFalse(report.exists())

    def test_vbx_elbo_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-elbo-arithmetic"])
            self.assertFalse(report.exists())

    def test_vbx_transcendental_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-transcendentals"])
            self.assertFalse(report.exists())

    def test_vbx_precompute_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-precomputed-terms"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_experimental_scope_rejects_cpu_and_unsupported_language_before_original(self):
        calls = []
        def original(transcript, model, align_model_metadata, audio, device, interpolate_method="nearest"):
            calls.append(device)
        task = SimpleNamespace(align=original)
        records = {key: [] for key in ("ctc", "bounds", "means", "rounding", "interpolation")}
        with native_alignment(task, object(), records):
            with self.assertRaisesRegex(ValueError, "requires_mps"):
                task.align([], None, {"language": "it"}, None, "cpu")
            with self.assertRaisesRegex(ValueError, "language_not_supported"):
                task.align([], None, {"language": "ja"}, None, "mps")
        self.assertEqual(calls, [])
        self.assertIs(task.align, original)

    def test_native_scope_preserves_result_identity_and_restores_hooks_on_failure(self):
        result = object()
        def original(transcript, model, align_model_metadata, audio, device, interpolate_method="nearest"):
            if transcript == "fail":
                raise RuntimeError("original_failure")
            return result
        task = SimpleNamespace(align=original)
        records = {key: [] for key in ("ctc", "bounds", "means", "rounding", "interpolation")}
        context = object()
        metadata = {"language": "it", "dictionary": {"a": 1, "b": 2}}
        with patch("tools.baseline.native_mps_ctc.native_mps_ctc", side_effect=lambda *a, **k: contextlib.nullcontext()) as ctc, patch("tools.baseline.native_ctc_fused.fused_wildcard_preparation", side_effect=lambda *a: contextlib.nullcontext()) as wildcard:
            with self.assertRaisesRegex(RuntimeError, "original_failure"):
                with native_alignment(task, context, records):
                    self.assertIs(task.align([], None, metadata, None, "mps"), result)
                    task.align("fail", None, metadata, None, "mps")
            self.assertIs(task.align, original)
            self.assertEqual(ctc.call_count, 2)
            wildcard.assert_called_with(3)

    def test_report_on_failed_native_load_has_no_arguments_or_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            arguments = ["--mps-library", "missing.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--hf_token", "private-secret"]
            with patch("tools.baseline.mps_storage_probe.MPSCTC", side_effect=FileNotFoundError), self.assertRaises(FileNotFoundError):
                main(arguments)
            text = report.read_text()
            data = json.loads(text)
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["native_ctc_calls"], 0)
            self.assertFalse(data["full_pipeline_gpu_certified"])
            self.assertNotIn("private-secret", text)
            self.assertNotIn("private.wav", text)
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(report.read_text(), text)

    def test_optional_embedding_load_failure_is_reported_and_does_not_leak_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            original_argv = sys.argv
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_embedding.NativeEmbeddingNormalization", side_effect=ValueError("injected_embedding_failure")):
                native.return_value.__enter__.return_value.identity = {"library_sha256": "0" * 64}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                with self.assertRaisesRegex(ValueError, "injected_embedding_failure"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--embedding-library", "private.dylib", "--embedding-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["embedding_normalization_requested"])
            self.assertFalse(data["embedding_normalization_observed"])
            self.assertEqual(data["native_embedding_normalization_calls"], 0)
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())
            self.assertIs(sys.argv, original_argv)

    def test_optional_plda_load_failure_has_no_success_or_private_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            original_argv = sys.argv
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization", side_effect=ValueError("injected_plda_failure")):
                native.return_value.__enter__.return_value.identity = {"library_sha256": "0" * 64}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                with self.assertRaisesRegex(ValueError, "injected_plda_failure"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "private.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["plda_normalization_requested"])
            self.assertFalse(data["plda_normalization_observed"])
            self.assertEqual(data["native_plda_normalization_calls"], 0)
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())
            self.assertIs(sys.argv, original_argv)

    def test_no_ctc_calls_are_explicit_and_original_argv_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            original_argv = sys.argv
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.metal_cli.native_alignment", side_effect=lambda *a: contextlib.nullcontext()), patch("whisperx.__main__.cli") as cli:
                native.return_value.__enter__.return_value.identity = {"library_sha256": "0" * 64}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                self.assertEqual(main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "silence.wav"]), 0)
                cli.assert_called_once()
            self.assertIs(sys.argv, original_argv)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "completed_without_native_ctc")
            self.assertEqual(data["native_ctc_calls"], 0)
            self.assertFalse(data["full_pipeline_gpu_certified"])


if __name__ == "__main__":
    unittest.main()

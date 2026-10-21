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
    def test_vbx_constrained_assignment_requires_native_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-constrained-assignment"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_vbx_constrained_assignment_capability_failure_restores_scope_and_privacy(self):
        from pyannote.audio.pipelines.clustering import BaseClustering
        original = BaseClustering.constrained_argmax
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_nan_cleanup=object())
                with self.assertRaisesRegex(ValueError, "assignment_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-constrained-nan-cleanup", "--vbx-constrained-assignment", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertTrue(data["vbx_constrained_assignment_requested"])
            self.assertFalse(data["vbx_constrained_assignment_observed"])
            self.assertEqual(data["native_vbx_constrained_assignment_calls"], 0)
            self.assertIs(BaseClustering.constrained_argmax, original)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_query_scaling_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-query-scaling"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_query_scaling_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_query_scaling_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-query-scaling", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_query_scaling_requested"])
            self.assertFalse(data["kmeans_query_scaling_observed"])
            self.assertEqual(data["native_kmeans_query_scaling_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_best_candidate_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-best-candidate"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_best_candidate_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_best_candidate_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-best-candidate", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_best_candidate_requested"])
            self.assertFalse(data["kmeans_best_candidate_observed"])
            self.assertEqual(data["native_kmeans_best_candidate_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_candidate_search_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-candidate-search"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_candidate_search_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_candidate_search_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-candidate-search", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_candidate_search_requested"])
            self.assertFalse(data["kmeans_candidate_search_observed"])
            self.assertEqual(data["native_kmeans_candidate_search_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_seed_probabilities_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-seed-probabilities"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_seed_probabilities_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_seed_probabilities_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-seed-probabilities", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_seed_probabilities_requested"])
            self.assertFalse(data["kmeans_seed_probabilities_observed"])
            self.assertEqual(data["native_kmeans_seed_probabilities_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_candidate_minimum_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-candidate-minimum"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_candidate_minimum_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_candidate_minimum_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-candidate-minimum", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_candidate_minimum_requested"])
            self.assertFalse(data["kmeans_candidate_minimum_observed"])
            self.assertEqual(data["native_kmeans_candidate_minimum_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_weighted_prefix_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-weighted-prefix"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_weighted_prefix_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_weighted_prefix_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-weighted-prefix", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_weighted_prefix_requested"])
            self.assertFalse(data["kmeans_weighted_prefix_observed"])
            self.assertEqual(data["native_kmeans_weighted_prefix_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_squared_norms_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-squared-norms"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_squared_norms_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_squared_norms_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-squared-norms", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_squared_norms_requested"])
            self.assertFalse(data["kmeans_squared_norms_observed"])
            self.assertEqual(data["native_kmeans_squared_norms_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_tolerance_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-tolerance"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_tolerance_capability_failure_restores_bindings_and_privacy(self):
        from sklearn.cluster import KMeans
        import sklearn.cluster._kmeans as kmeans_module
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_tolerance_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-tolerance", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_tolerance_requested"])
            self.assertFalse(data["kmeans_tolerance_observed"])
            self.assertEqual(data["native_kmeans_tolerance_calls"], 0)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(kmeans_module._tolerance, original_tolerance)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_centering_flag_requires_feature_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-centering"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_centering_capability_failure_preserves_fit_and_privacy(self):
        from sklearn.cluster import KMeans
        original = KMeans.fit
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
                with self.assertRaisesRegex(ValueError, "kmeans_shift_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--kmeans-centering", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_centering_requested"])
            self.assertFalse(data["kmeans_centering_observed"])
            self.assertEqual(data["native_kmeans_centering_calls"], 0)
            self.assertIs(KMeans.fit, original)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_feature_mean_flag_requires_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--kmeans-feature-mean"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_feature_mean_capability_failure_preserves_fit_and_privacy(self):
        from sklearn.cluster import KMeans
        original = KMeans.fit
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = object()
                with self.assertRaisesRegex(ValueError, "kmeans_mean_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--kmeans-feature-mean", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["kmeans_feature_mean_requested"])
            self.assertFalse(data["kmeans_feature_mean_observed"])
            self.assertEqual(data["native_kmeans_feature_mean_calls"], 0)
            self.assertIs(KMeans.fit, original)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_kmeans_centroid_flag_requires_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-kmeans-centroids"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_kmeans_centroid_capability_failure_preserves_report_privacy(self):
        from pyannote.audio.pipelines.clustering import VBxClustering
        original = VBxClustering.__call__
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.baseline.native_double.NativeDoubleNormalization") as double:
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.identity = {}
                double.return_value.__enter__.return_value.initialization_wall_ns = 1
                double.return_value.__enter__.return_value.library = object()
                with self.assertRaisesRegex(ValueError, "kmeans_centroid_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-kmeans-centroids", "--hf_token", "private-secret"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_kmeans_centroids_requested"])
            self.assertFalse(data["vbx_kmeans_centroids_observed"])
            self.assertEqual(data["native_vbx_kmeans_centroid_calls"], 0)
            self.assertIs(VBxClustering.__call__, original)
            self.assertNotIn("private.wav", report.read_text())
            self.assertNotIn("private-secret", report.read_text())

    def test_clustering_cpu_audit_requires_diarization(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--clustering-cpu-audit"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(report.exists())

    def test_clustering_cpu_audit_failure_keeps_observations_private_and_restores(self):
        import numpy as np
        import pyannote.audio.pipelines.clustering as clustering

        def cli():
            self.assertNotIn("--clustering-cpu-audit", sys.argv)
            clustering.linear_sum_assignment(np.array([[1., 2.]]), maximize=True)
            raise ValueError("private-secret-error")

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            original_argv = sys.argv
            with patch("tools.baseline.mps_storage_probe.MPSCTC") as native, patch("tools.metal_cli.native_alignment", side_effect=lambda *a: contextlib.nullcontext()), patch("whisperx.__main__.cli", side_effect=cli):
                native.return_value.__enter__.return_value.identity = {}
                native.return_value.__enter__.return_value.initialization_wall_ns = 1
                with self.assertRaisesRegex(ValueError, "private-secret-error"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--clustering-cpu-audit", "--hf_token", "private-token"])
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["clustering_cpu_audit_requested"])
            self.assertTrue(data["clustering_cpu_audit_observed"])
            self.assertTrue(data["clustering_cpu_audit_complete"])
            self.assertEqual(data["clustering_cpu_api_calls"], 1)
            self.assertEqual(len(data["clustering_cpu_api_observations"]), 4)
            for private in ("private.wav", "private-token", "private-secret-error"):
                self.assertNotIn(private, report.read_text())
            self.assertIsNone(sys.getprofile())
            self.assertIs(sys.argv, original_argv)

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

    def test_vbx_logsumexp_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "logsumexp_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-logsumexp", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_logsumexp_requested"])
            self.assertFalse(data["vbx_logsumexp_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_scalar_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "scalar_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-scalar-arithmetic", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_scalar_arithmetic_requested"])
            self.assertFalse(data["vbx_scalar_arithmetic_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_initialization_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "initialization_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-initialization", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_initialization_requested"])
            self.assertFalse(data["vbx_initialization_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_initial_softmax_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "qinit_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-initial-softmax", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_initial_softmax_requested"])
            self.assertFalse(data["vbx_initial_softmax_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_initial_one_hot_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "one_hot_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-initial-one-hot", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_initial_one_hot_requested"])
            self.assertFalse(data["vbx_initial_one_hot_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_resident_initialization_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "resident_initialization_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-initial-one-hot", "--vbx-initial-softmax", "--vbx-resident-initialization", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_resident_initialization_requested"])
            self.assertFalse(data["vbx_resident_initialization_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_centroids_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "centroid_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-centroids", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_centroids_requested"])
            self.assertFalse(data["vbx_centroids_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_cosine_distances_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "cosine_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-cosine-distances", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_cosine_distances_requested"])
            self.assertFalse(data["vbx_cosine_distances_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_unconstrained_argmax_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "argmax_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-unconstrained-argmax", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_unconstrained_argmax_requested"])
            self.assertFalse(data["vbx_unconstrained_argmax_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_inactive_speaker_mask_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "inactive_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-inactive-speaker-mask", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_inactive_speaker_mask_requested"])
            self.assertFalse(data["vbx_inactive_speaker_mask_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_constrained_nan_cleanup_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "nan_cleanup_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-constrained-nan-cleanup", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_constrained_nan_cleanup_requested"])
            self.assertFalse(data["vbx_constrained_nan_cleanup_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_resident_argmax_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "cosine_argmax_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-unconstrained-argmax", "--vbx-cosine-distances", "--vbx-soft-scores", "--vbx-resident-argmax", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_resident_argmax_requested"])
            self.assertFalse(data["vbx_resident_argmax_observed"])
            self.assertEqual(data["vbx_calls"], [])
            self.assertNotIn("private-secret", report.read_text())
            self.assertNotIn("private.wav", report.read_text())

    def test_vbx_soft_scores_capability_failure_restores_norm_scope_and_report_privacy(self):
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
                with self.assertRaisesRegex(ValueError, "cosine_score_abi_unavailable"):
                    main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--plda-normalization-library", "unused.dylib", "--plda-normalization-library-sha256", "0" * 64, "--native-report", str(report), "private.wav", "--diarize", "--vbx-matrix-products", "--vbx-cosine-distances", "--vbx-soft-scores", "--hf_token", "private-secret"])
            self.assertIs(vbx.l2_norm, original)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertTrue(data["vbx_soft_scores_requested"])
            self.assertFalse(data["vbx_soft_scores_observed"])
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

    def test_vbx_logsumexp_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-logsumexp"])
            self.assertFalse(report.exists())

    def test_vbx_scalar_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-scalar-arithmetic"])
            self.assertFalse(report.exists())

    def test_vbx_initialization_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-initialization"])
            self.assertFalse(report.exists())

    def test_vbx_initial_softmax_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-initial-softmax"])
            self.assertFalse(report.exists())

    def test_vbx_initial_one_hot_flag_requires_matrix_products(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-initial-one-hot"])
            self.assertFalse(report.exists())

    def test_vbx_resident_initialization_requires_both_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-resident-initialization"])
            self.assertFalse(report.exists())

    def test_vbx_centroids_requires_double_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-centroids"])
            self.assertFalse(report.exists())

    def test_vbx_cosine_distances_requires_double_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-cosine-distances"])
            self.assertFalse(report.exists())

    def test_vbx_unconstrained_argmax_requires_double_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-unconstrained-argmax"])
            self.assertFalse(report.exists())

    def test_vbx_inactive_speaker_mask_requires_double_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-inactive-speaker-mask"])
            self.assertFalse(report.exists())

    def test_vbx_constrained_nan_cleanup_requires_double_library(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-constrained-nan-cleanup"])
            self.assertFalse(report.exists())

    def test_vbx_resident_argmax_requires_both_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-resident-argmax"])
            self.assertFalse(report.exists())

    def test_vbx_soft_scores_require_cosine(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mps-library", "unused.dylib", "--mps-library-sha256", "0" * 64, "--native-report", str(report), "audio.wav", "--diarize", "--vbx-soft-scores"])
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
            self.assertFalse(data["clustering_cpu_audit_requested"])
            self.assertFalse(data["clustering_cpu_audit_observed"])
            self.assertFalse(data["clustering_cpu_audit_complete"])
            self.assertEqual(data["clustering_cpu_api_observations"], [])
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

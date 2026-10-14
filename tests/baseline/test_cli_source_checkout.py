from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from whisperx.__main__ import package_version
from importlib.metadata import PackageNotFoundError


class SourceCheckoutCLITests(unittest.TestCase):
    def test_mlx_is_opt_in_and_classic_cli_keeps_its_original_backend(self):
        from whisperx.__main__ import cli
        for backend in [None, "mlx"]:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                arguments = ["whisperx", "sample.wav", "--no_align", "--output_dir", directory, "--verbose", "False"]
                if backend:
                    arguments.extend(["--asr_backend", backend, "--device", "mps"])
                with patch.object(sys, "argv", arguments), patch("whisperx.transcribe.load_audio", return_value=object()), patch("whisperx.transcribe.get_writer"), patch("whisperx.transcribe.load_model") as classic, patch("whisperx.mlx_asr.MlxPipeline") as mlx:
                    selected = mlx if backend else classic
                    selected.return_value.transcribe.return_value = {"segments": [], "language": "en"}
                    cli()
                    selected.assert_called_once()
                    (classic if backend else mlx).assert_not_called()
                    selected.return_value.transcribe.assert_called_once()
                    if backend:
                        selected.return_value.unload.assert_called_once()

    def test_installed_distribution_version_is_preserved(self):
        with patch("importlib.metadata.version", return_value="3.8.7rc1"):
            self.assertEqual(package_version(), "3.8.7rc1")

    def test_source_checkout_without_distribution_metadata_is_explicit(self):
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError("whisperx")):
            self.assertEqual(package_version(), "unknown (uninstalled source checkout)")

    def test_classic_cli_help_can_run_from_checkout_without_loading_asr_models(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, "-B", "-m", "whisperx", "--help"], cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model_cache_only", result.stdout)
        self.assertIn("--diarize", result.stdout)
        self.assertIn("--device", result.stdout)
        self.assertIn("--asr_backend", result.stdout)


if __name__ == "__main__":
    unittest.main()

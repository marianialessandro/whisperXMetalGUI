from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from whisperx.__main__ import package_version
from importlib.metadata import PackageNotFoundError


class SourceCheckoutCLITests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

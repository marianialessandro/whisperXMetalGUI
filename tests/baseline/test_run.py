import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import run, corpus


class RunnerTests(unittest.TestCase):
    def test_full_preflight_keeps_missing_corpus_and_models_explicit(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            report = run.preflight({"assets": [], "cases": []}, corpus.expected_coverage(root), {"licenses": []}, Path(directory), [])
        self.assertEqual(report["status"], "prerequisite_missing")
        self.assertIn({"code": "model_inventory_missing"}, report["missing"])
        self.assertFalse(report["gpu_placement_certified"])

    def test_paired_output_comparison_detects_export_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"quality": "Rapida", "language": "it", "voices": False, "hints": ""}
            audio = root / "sample.wav"
            audio.write_bytes(b"synthetic")
            def execute(runtime, arguments, request, timeout, cache_root=None):
                output = Path(request["output"])
                for name in ("original.json", "result.json"):
                    (output / name).write_text(json.dumps({"segments": [], "language": "it"}))
                for extension in ("txt", "srt", "vtt"):
                    (output / ("sample." + extension)).write_text("")
                if "instrumented" in output.parts:
                    (output / "sample.txt").write_text("changed")
                return [{"event": "completed", "output": str(output), "result": str(output / "result.json"), "warnings": []}]
            with patch.object(run, "execute", execute), self.assertRaises(run.RunError) as error:
                run.paired_run(Path("runtime"), Path("worker"), root, audio, settings, root / "pair")
            self.assertEqual(str(error.exception), "paired_semantic_mismatch")

    def test_missing_credential_never_launches_worker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, "execute") as execute:
            root = Path(directory)
            with self.assertRaises(run.RunError):
                run.paired_run(Path("runtime"), Path("worker"), root, root / "sample.wav", {"quality": "Rapida", "language": "it", "voices": True}, root / "pair")
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()

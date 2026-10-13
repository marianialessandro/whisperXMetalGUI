import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import alignment_inventory as inventory


class Bundle:
    _path = "weights.pth"
    _params = {"layers": [[3, 4]], "dropout": 0.0}
    _model_type = "Wav2Vec2"
    _normalize_waveform = False
    _remove_aux_axis = (1, 2, 3)
    sample_rate = 16000

    def get_labels(self): return ("-", "a")

    def get_model(self, *args, **kwargs): raise AssertionError("inventory must not load checkpoints")


class AlignmentInventoryTests(unittest.TestCase):
    def fixture(self, root):
        (root / "whisperx").mkdir()
        (root / "whisperx/alignment.py").write_text("raise RuntimeError('source must not execute')\nDEFAULT_ALIGN_MODELS_TORCH = {'it': 'BUNDLE'}\n")
        module = types.SimpleNamespace(__version__="2.8.0", pipelines=types.SimpleNamespace(__all__=["BUNDLE"], BUNDLE=Bundle()))
        spec = {"transport": "torch_hub", "role": "alignment_it", "repository": "torchaudio/BUNDLE", "files": ["weights.pth"], "source": "https://download.pytorch.org/torchaudio/models/weights.pth", "torchaudio_version": "2.8.0", "bundle_config": copy.deepcopy(Bundle._params), "labels": ["-", "a"]}
        return module, spec

    def test_inventory_reads_defaults_and_config_without_executing_source_or_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, spec = self.fixture(root)
            report = inventory.collect(root, [spec], module)
            record = report["models"][0]
            self.assertEqual(record["sample_rate"], 16000)
            self.assertEqual(record["remove_aux_axes"], [1, 2, 3])
            self.assertEqual(len(record["bundle_source_sha256"]), 64)
            self.assertFalse(report["checkpoint_loaded"])
            self.assertFalse(report["full_baseline_gate_passed"])
            self.assertNotIn(directory, json.dumps(report))
            self.assertTrue(inventory.matches_spec(spec, report))
            self.assertFalse(inventory.matches_spec({**spec, "labels": ["a", "-"]}, report))
            self.assertFalse(inventory.matches_spec(spec, {**report, "models": report["models"] * 2}))
            self.assertFalse(inventory.matches_spec(spec, None))

    def test_inventory_rejects_selection_config_labels_version_and_duplicate_role_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, spec = self.fixture(root)
            mutations = [("repository", "torchaudio/OTHER", "alignment_default_bundle_mismatch"), ("files", ["other.pth"], "alignment_checkpoint_selection_mismatch"), ("source", "https://example.org/weights.pth", "alignment_checkpoint_selection_mismatch"), ("torchaudio_version", "2.9.0", "alignment_runtime_version_mismatch"), ("labels", ["a", "-"], "alignment_bundle_configuration_mismatch"), ("bundle_config", {"layers": [[4, 3]]}, "alignment_bundle_configuration_mismatch")]
            for key, value, code in mutations:
                with self.subTest(key=key), self.assertRaisesRegex(inventory.InventoryError, code):
                    inventory.collect(root, [{**spec, key: value}], module)
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_role_not_supported"):
                inventory.collect(root, [spec, spec], module)

    def test_defaults_must_be_one_literal_mapping_and_remain_within_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, spec = self.fixture(root)
            source = root / "whisperx/alignment.py"
            for text in ("DEFAULT_ALIGN_MODELS_TORCH = dict(it='BUNDLE')", "DEFAULT_ALIGN_MODELS_TORCH = {}", "DEFAULT_ALIGN_MODELS_TORCH = {'it':'BUNDLE'}\nDEFAULT_ALIGN_MODELS_TORCH = {'it':'OTHER'}"):
                source.write_text(text)
                with self.assertRaisesRegex(inventory.InventoryError, "alignment_defaults_not_supported"):
                    inventory.collect(root, [spec], module)

    def test_selected_runtime_inventory_checks_source_identity_and_sanitizes_child_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, spec = self.fixture(root)
            specs = root / "specs.json"
            specs.write_text(json.dumps([spec]))
            report = inventory.collect(root, [spec], module)
            received = []
            def child(arguments, **options):
                received.append((arguments, options))
                Path(arguments[-1]).write_text(json.dumps(report))
                return types.SimpleNamespace(returncode=0)
            with patch.object(inventory.subprocess, "run", child):
                self.assertEqual(inventory.inspect_runtime(Path(sys.executable), root, specs)["status"], "selection_inventory_verified")
                report["alignment_source_sha256"] = "0" * 64
                self.assertEqual(inventory.inspect_runtime(Path(sys.executable), root, specs)["code"], "alignment_runtime_inventory_failed")
            self.assertEqual(received[0][0][1:3], ["-I", "-B"])
            self.assertEqual(received[0][1]["env"]["HF_HUB_OFFLINE"], "1")
            with patch.object(inventory.subprocess, "run", side_effect=inventory.subprocess.TimeoutExpired("private-credential-sentinel", 30)):
                failure = inventory.inspect_runtime(Path(sys.executable), root, specs)
                self.assertEqual(failure["code"], "alignment_runtime_inventory_failed")
                self.assertNotIn("private-credential-sentinel", json.dumps(failure))


if __name__ == "__main__":
    unittest.main()

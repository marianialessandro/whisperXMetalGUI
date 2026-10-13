import copy
import hashlib
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
    def hf_fixture(self, root):
        module, torch_spec = self.fixture(root)
        with (root / "whisperx/alignment.py").open("a") as output: output.write("DEFAULT_ALIGN_MODELS_HF = {'ja': 'org/hf'}\n")
        revision = "a" * 40
        cache = root / "hf"
        prefix = cache / "models--org--hf"
        snapshot = prefix / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (prefix / "blobs").mkdir()
        documents = {"config.json": {"model_type": "wav2vec2", "vocab_size": 3}, "preprocessor_config.json": {"sampling_rate": 16000}, "vocab.json": {"<pad>": 0, "A": 1, "a": 2}}
        hashes = {}
        for name, document in documents.items():
            payload = json.dumps(document).encode()
            digest = hashlib.sha256(payload).hexdigest()
            (prefix / "blobs" / digest).write_bytes(payload)
            (snapshot / name).symlink_to(Path("../../blobs") / digest)
            hashes[name] = digest
        calls = []
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(get_vocab=lambda: documents["vocab.json"]), feature_extractor=types.SimpleNamespace(sampling_rate=16000))
        def load(path, **options):
            calls.append((path, options))
            self.assertEqual(options, {"local_files_only": True})
            return processor
        transformers = types.SimpleNamespace(__version__="4.test", Wav2Vec2Processor=types.SimpleNamespace(from_pretrained=load))
        spec = {"role": "alignment_ja", "repository": "org/hf", "revision": revision, "cache_root": str(cache), "files": list(documents) + ["pytorch_model.bin"], "sha256": hashes, "transformers_version": "4.test"}
        return module, torch_spec, transformers, processor, spec, calls, snapshot

    def test_hf_inventory_follows_in_cache_blob_links_and_preserves_legacy_torch_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, torch_spec, transformers, processor, spec, calls, snapshot = self.hf_fixture(root)
            legacy = inventory.collect(root, [torch_spec, spec], module)
            self.assertEqual(len(legacy["models"]), 1)
            self.assertEqual(calls, [])
            report = inventory.collect(root, [torch_spec, spec], module, include_hf=True, processor_module=transformers)
            record = report["models"][1]
            self.assertEqual(record["vocabulary_count"], 3)
            self.assertEqual(record["dictionary_count"], 2)
            self.assertEqual(record["out_of_range_vocabulary_ids"], [])
            self.assertEqual(record["metadata_sha256"], spec["sha256"])
            self.assertFalse(record["checkpoint_loaded"])
            self.assertEqual(calls[0][0], str(snapshot.resolve()))
            self.assertNotIn(directory, json.dumps(report))
            self.assertTrue(inventory.matches_spec(spec, report))
            self.assertFalse(inventory.matches_spec({**spec, "revision": "b" * 40}, report))
            self.assertFalse(inventory.matches_spec({**spec, "sha256": {}}, report))
            self.assertFalse(inventory.matches_spec(spec, {**report, "models": report["models"] * 2}))

    def test_hf_default_runtime_revision_required_metadata_and_checkpoint_drift_fail_before_processor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, torch_spec, transformers, processor, spec, calls, snapshot = self.hf_fixture(root)
            variants = [({**spec, "repository": "org/other"}, "alignment_default_repository_mismatch"), ({**spec, "transformers_version": "other"}, "alignment_runtime_version_mismatch"), ({**spec, "revision": "main"}, "alignment_revision_invalid"), ({**spec, "files": ["pytorch_model.bin"]}, "alignment_metadata_missing"), ({**spec, "sha256": {}}, "alignment_metadata_hash_mismatch"), ({**spec, "files": spec["files"] + ["model.safetensors"]}, "alignment_checkpoint_selection_mismatch")]
            for variant, code in variants:
                with self.subTest(code=code), self.assertRaisesRegex(inventory.InventoryError, code): inventory.collect(root, [variant], module, include_hf=True, processor_module=transformers)
            (snapshot / "model.safetensors").write_bytes(b"undeclared competing checkpoint")
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_checkpoint_selection_mismatch"): inventory.collect(root, [spec], module, include_hf=True, processor_module=transformers)
            (snapshot / "model.safetensors").unlink()
            (snapshot / "added_tokens.json").write_text('{"untracked": 3}')
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_metadata_undeclared"): inventory.collect(root, [spec], module, include_hf=True, processor_module=transformers)
            self.assertEqual(calls, [])

    def test_hf_metadata_size_config_and_processor_sample_rate_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, torch_spec, transformers, processor, spec, calls, snapshot = self.hf_fixture(root)
            processor.feature_extractor.sampling_rate = 8000
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_processor_config_not_supported"): inventory.collect(root, [spec], module, include_hf=True, processor_module=transformers)
            config = snapshot / "config.json"
            config.unlink()
            payload = json.dumps({"model_type": "other", "vocab_size": 3}).encode()
            config.write_bytes(payload)
            spec["sha256"]["config.json"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_model_config_not_supported"): inventory.collect(root, [spec], module, include_hf=True, processor_module=transformers)
            config.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_metadata_invalid"): inventory.collect(root, [spec], module, include_hf=True, processor_module=transformers)

    def test_hf_metadata_links_cannot_escape_cache_and_duplicate_or_torch_default_roles_fail(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            module, torch_spec, transformers, processor, spec, calls, snapshot = self.hf_fixture(root)
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_default_repository_mismatch"): inventory.collect(root, [spec, spec], module, include_hf=True, processor_module=transformers)
            with self.assertRaisesRegex(inventory.InventoryError, "alignment_default_repository_mismatch"): inventory.collect(root, [{**spec, "role": "alignment_it"}], module, include_hf=True, processor_module=transformers)
            config = snapshot / "config.json"
            config.unlink()
            config.symlink_to(Path(outside) / "config.json")
            with self.assertRaises(inventory.manifest.BaselineError): inventory.collect(root, [spec], module, include_hf=True, processor_module=transformers)

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

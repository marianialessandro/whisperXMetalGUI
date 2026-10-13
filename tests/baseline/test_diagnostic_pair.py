import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
import contextlib
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import diagnostic_pair, provision, run


class DiagnosticPairTests(unittest.TestCase):
    def test_voice_inventory_requires_components_shared_cache_and_matching_default_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, asr, settings, _ = self.fixtures(root)
            spec = {"repository": "pyannote/speaker-diarization-community-1", "revision": "b" * 40, "cache_root": str(root), "license_verified": True, "license_source": "https://example.org/license"}
            reference = root / "models--pyannote--speaker-diarization-community-1/refs/main"
            reference.parent.mkdir(parents=True)
            reference.write_text(spec["revision"])
            identity = {"status": "files_verified", "components_verified": True, "roles": ["speaker_segmentation", "speaker_embedding", "speaker_plda"]}
            with patch.object(provision, "inspect_spec", return_value=identity):
                self.assertEqual(diagnostic_pair.diarization_prerequisites([spec], asr), spec)
                with self.assertRaisesRegex(run.RunError, "diagnostic_diarization_license_or_cache_invalid"):
                    diagnostic_pair.diarization_prerequisites([{**spec, "cache_root": str(root / "other")}], asr)
                with self.assertRaisesRegex(run.RunError, "diagnostic_diarization_inventory_invalid"):
                    diagnostic_pair.diarization_prerequisites([spec, spec], asr)
                reference.write_text("c" * 40)
                with self.assertRaisesRegex(run.RunError, "worker_default_revision_unverified"):
                    diagnostic_pair.diarization_prerequisites([spec], asr)
            reference.write_text(spec["revision"])
            with patch.object(provision, "inspect_spec", return_value={**identity, "roles": ["speaker_segmentation"]}), self.assertRaisesRegex(run.RunError, "diagnostic_diarization_components_not_verified"):
                diagnostic_pair.diarization_prerequisites([spec], asr)

    def test_voice_completion_requires_successful_diarization_assignment_labels_and_no_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "instrumented").mkdir()
            spans = [{"stage": "diarization", "status": "synchronized"}, {"stage": "speaker_assignment", "status": "returned"}]
            (root / "spans.json").write_text(json.dumps({"spans": spans}))
            result = root / "instrumented/result.json"
            result.write_text(json.dumps({"segments": [{"speaker": "SPEAKER_00"}], "word_segments": [{"speaker": "SPEAKER_00"}]}))
            self.assertEqual(diagnostic_pair.voice_completion(root, []), {"voice_pipeline_completed": True, "observed_speaker_count": 1})
            self.assertFalse(diagnostic_pair.voice_completion(root, ["alignment unavailable"])["voice_pipeline_completed"])
            spans[0]["status"] = "failed"
            (root / "spans.json").write_text(json.dumps({"spans": spans}))
            self.assertFalse(diagnostic_pair.voice_completion(root, [])["voice_pipeline_completed"])
            result.write_text(json.dumps({"segments": [], "word_segments": []}))
            self.assertEqual(diagnostic_pair.voice_completion(root, [])["observed_speaker_count"], 0)

    def test_voice_cli_orphans_fail_before_reading_keychain(self):
        arguments = ["--source-root", ".", "--runtime", "python", "--reference-root", ".", "--corpus", "missing", "--licenses", "missing", "--model-specs", "missing", "--evidence-root", ".", "--asset", "clip"]
        with patch.object(provision, "read_credential") as bridge, contextlib.redirect_stderr(io.StringIO()):
            for option in (["--voices"], ["--speaker-count", "2"]):
                with self.assertRaises(SystemExit) as error: diagnostic_pair.main(arguments + option)
                self.assertEqual(error.exception.code, 2)
            bridge.assert_not_called()

    def test_voice_component_preflight_failure_never_reads_credentials_or_launches_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, spec, settings, _ = self.fixtures(root)
            payloads = {"corpus.json": {"assets": [asset]}, "licenses.json": {"licenses": [license]}, "models.json": [spec], "diarization.json": []}
            for name, payload in payloads.items(): (root / name).write_text(json.dumps(payload))
            arguments = ["--source-root", str(root), "--runtime", sys.executable, "--reference-root", str(root), "--corpus", str(root / "corpus.json"), "--licenses", str(root / "licenses.json"), "--model-specs", str(root / "models.json"), "--evidence-root", str(root), "--asset", "clip", "--voices", "--diarization-specs", str(root / "diarization.json"), "--torch-specs", "unused.json", "--bridge", "private-credential-sentinel"]
            output = io.StringIO()
            with patch.object(provision, "read_credential") as bridge, patch.object(run, "paired_run") as worker, contextlib.redirect_stdout(output):
                self.assertEqual(diagnostic_pair.main(arguments), 2)
                bridge.assert_not_called()
                worker.assert_not_called()
            self.assertNotIn("private-credential-sentinel", output.getvalue())
            self.assertNotIn(directory, output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["code"], "diagnostic_diarization_inventory_invalid")

    def fixtures(self, root):
        audio = root / "audio.wav"
        audio.write_bytes(b"licensed-audio-fixture")
        legal = root / "license.txt"
        legal.write_bytes(b"fixture-license")
        asset = {"id": "clip", "path": "audio.wav", "license_id": "license", "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "type": "parity_only"}
        license = {"id": "license", "verified": True, "asset_ids": ["clip"], "rights": {"download": True}, "attribution": "Fixture", "source": "https://example.org/license", "revision": "1", "text_path": "license.txt", "text_sha256": hashlib.sha256(legal.read_bytes()).hexdigest()}
        repository, revision = "mlx-community/whisper-base-mlx", "a" * 40
        model_root = root / ("models--" + repository.replace("/", "--"))
        snapshot = model_root / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "weights.npz").write_bytes(b"fixture-weights")
        reference = model_root / "refs/main"
        reference.parent.mkdir()
        reference.write_text(revision)
        spec = {"cache_root": str(root), "repository": repository, "revision": revision, "files": ["weights.npz"], "sha256": {"weights.npz": hashlib.sha256(b"fixture-weights").hexdigest()}, "license_verified": True, "license_source": "https://example.org/license"}
        settings = {"quality": "Rapida", "language": "it", "voices": False}
        return asset, license, spec, settings, reference

    def test_partial_diagnostic_checks_real_asset_and_default_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, spec, settings, reference = self.fixtures(root)
            self.assertEqual(diagnostic_pair.prerequisites(asset, license, root, spec, settings), (root / "audio.wav").resolve())
            reference.write_text("b" * 40)
            with self.assertRaises(run.RunError) as error: diagnostic_pair.prerequisites(asset, license, root, spec, settings)
            self.assertEqual(str(error.exception), "worker_default_revision_unverified")

    def test_changed_audio_and_profile_substitution_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, spec, settings, _ = self.fixtures(root)
            with self.assertRaises(run.RunError) as error: diagnostic_pair.prerequisites(asset, license, root, spec, dict(settings, quality="Massima"))
            self.assertEqual(str(error.exception), "diagnostic_profile_model_mismatch")
            (root / "audio.wav").write_bytes(b"changed")
            with self.assertRaises(run.RunError) as error: diagnostic_pair.prerequisites(asset, license, root, spec, settings)
            self.assertEqual(str(error.exception), "asset_hash_mismatch")

    def test_voices_require_separate_explicit_model_credential_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, spec, settings, _ = self.fixtures(root)
            with self.assertRaises(run.RunError) as error: diagnostic_pair.prerequisites(asset, license, root, spec, dict(settings, voices=True))
            self.assertEqual(str(error.exception), "diagnostic_voices_not_supported")

    def test_torch_hub_inspection_does_not_load_pickle_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hub/checkpoints/alignment.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture-checkpoint-never-deserialized")
            spec = {"transport": "torch_hub", "cache_root": str(root), "repository": "torchaudio/fixture", "revision": "a" * 40, "files": [path.name], "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
            report = provision.inspect_spec(spec)
            self.assertEqual(report["status"], "files_verified")
            self.assertFalse(report["loadability_verified"])
            path.write_bytes(b"changed")
            self.assertEqual(provision.inspect_spec(spec)["code"], "model_hash_mismatch")
            self.assertEqual(provision.inspect_spec(dict(spec, files=["../escape.pt"]))["code"], "model_path_invalid")
            with self.assertRaises(provision.ProvisionError) as error: provision.provision_spec(spec, downloader=lambda **kwargs: self.fail("torch checkpoint must not use HF download transport"))
            self.assertEqual(str(error.exception), "torch_hub_explicit_acquisition_required")


if __name__ == "__main__":
    unittest.main()

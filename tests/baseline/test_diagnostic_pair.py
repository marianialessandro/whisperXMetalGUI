import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import diagnostic_pair, provision, run


class DiagnosticPairTests(unittest.TestCase):
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

import json
from pathlib import Path
import sys
import subprocess
import os
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import manifest


class ManifestTests(unittest.TestCase):
    def test_environment_is_allowlisted_and_never_persisted(self):
        secret = "sentinel-private-credential"
        inherited = {"HF_TOKEN": secret, "HUGGING_FACE_HUB_TOKEN": secret, "AWS_SECRET_ACCESS_KEY": secret, "PYTHONPATH": secret, "DYLD_INSERT_LIBRARIES": secret, "HF_HOME": "/model cache", "HOME": "/Users/private", "PATH": secret}
        clean = manifest.clean_environment(Path("/runtime"), inherited)
        self.assertNotIn(secret, json.dumps(clean))
        self.assertEqual(clean["HF_HOME"], "/model cache")
        self.assertEqual(clean["HF_HUB_OFFLINE"], "1")
        self.assertEqual(clean["PATH"], "/runtime/bin:/usr/bin:/bin:/usr/sbin:/sbin")
        self.assertEqual(manifest.failure("source_missing", RuntimeError(secret)), {"status": "prerequisite_missing", "code": "source_missing"})

    def test_dirty_and_untracked_sources_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.py").write_text("original")
            first = manifest.source_identity(root, ["worker.py"])
            (root / "worker.py").write_text("dirty")
            second = manifest.source_identity(root, ["worker.py"])
            self.assertNotEqual(first["digest"], second["digest"])
            self.assertNotIn(directory, json.dumps(second))
            (root / "new.py").write_text("untracked")
            self.assertNotEqual(second["digest"], manifest.source_identity(root, ["worker.py", "new.py"])["digest"])

    def test_missing_source_cannot_be_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = manifest.source_identity(Path(directory), ["missing.py"])
            self.assertEqual(identity["status"], "prerequisite_missing")

    def test_model_files_hash_and_deduplicate_snapshot_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "models--org--model" / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            blob = root / "blob"
            blob.write_bytes(b"weights")
            (snapshot / "weights.bin").symlink_to(blob)
            (snapshot / "other.bin").symlink_to(blob)
            (snapshot / "config.json").write_text("{}")
            result = manifest.model_identity(root, "org/model", "a" * 40, ["weights.bin", "other.bin", "config.json"])
            self.assertEqual(result["status"], "identified")
            self.assertEqual(result["unique_bytes"], 9)
            self.assertEqual(result["files"][0]["sha256"], result["files"][1]["sha256"])
            self.assertEqual(manifest.model_identity(root, "org/model", "a" * 40, ["absent.bin"])["status"], "prerequisite_missing")
            self.assertEqual(manifest.model_identity(root, "org/model", "main", ["weights.bin"])["code"], "model_revision_missing")

    def test_model_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            snapshot = cache / "models--org--model" / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            private = root / "private"
            private.write_text("sentinel")
            (snapshot / "weights.bin").symlink_to(private)
            self.assertEqual(manifest.model_identity(cache, "org/model", "a" * 40, ["weights.bin"])["code"], "path_escape")

    def test_atomic_evidence_refuses_overwrites_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            evidence.mkdir()
            target = manifest.write_evidence(evidence, "run.json", {"status": "pending"})
            with self.assertRaises(manifest.BaselineError):
                manifest.write_evidence(evidence, "run.json", {"status": "changed"})
            self.assertEqual(json.loads(target.read_text()), {"status": "pending"})
            with self.assertRaises(manifest.BaselineError):
                manifest.write_evidence(evidence, "../outside.json", {})
            (evidence / "escape").symlink_to(root, target_is_directory=True)
            with self.assertRaises(manifest.BaselineError):
                manifest.write_evidence(evidence, "escape/run.json", {})

    def test_metadata_and_known_secrets_are_removed_from_persistence(self):
        secret = "sentinel-private-credential"
        clean = manifest.sanitize({"status": "pending", "token": secret, "environment": {"HF_TOKEN": secret}, "error": secret, "source": {"digest": "a" * 64, "path": "/Users/private/" + secret}}, secrets=[secret])
        self.assertNotIn(secret, json.dumps(clean))
        self.assertNotIn("/Users/private", json.dumps(clean))
        self.assertEqual(clean["source"]["digest"], "a" * 64)

    def test_cli_errors_never_emit_secret_environment_or_private_paths(self):
        root = Path(__file__).resolve().parents[2]
        secret = "sentinel-private-credential"
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, HF_TOKEN=secret, AWS_SECRET_ACCESS_KEY=secret)
            result = subprocess.run([sys.executable, "-I", "-B", str(root / "tools/baseline/manifest.py"), "--source", "gui-checkout", "--source-root", directory + "/" + secret, "--runtime", directory + "/" + secret, "--reference-root", directory + "/" + secret, "--evidence-root", directory], capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 2)
            reports = "".join(path.read_text() for path in Path(directory).glob("*.json"))
            self.assertNotIn(secret, result.stdout + result.stderr + reports)
            self.assertNotIn(directory, result.stdout + result.stderr + reports)
            self.assertIn("prerequisite_missing", result.stdout)


if __name__ == "__main__":
    unittest.main()

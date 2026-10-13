import hashlib
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import provision, vad_inventory as inventory


@unittest.skipUnless(sys.platform == "darwin", "Exclusive source snapshot publication requires macOS")
class VadInventoryTests(unittest.TestCase):
    def test_source_cli_requires_explicit_installation_and_preflight_never_installs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, archive, source, module, calls = self.fixture(root)
            specs = root / "specs.json"
            specs.write_text(json.dumps([spec]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output): result = provision.main(["--specs", str(specs), "download"])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(output.getvalue())["code"], "torch_hub_source_explicit_installation_required")
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(provision.main(["--specs", str(specs), "preflight"]), 2)
            self.assertFalse(provision.torch_source_target(spec).exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): provision.main(["--specs", str(specs), "preflight", "--install-torch-source", str(archive)])
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(provision.main(["--specs", str(specs), "download", "--install-torch-source", str(archive), "--bind-default-ref"]), 0)
            self.assertFalse((Path(spec["cache_root"]) / "models--snakers4--silero-vad").exists())

    def test_selected_runtime_checks_source_digest_isolation_and_private_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, archive, source, module, calls = self.fixture(root)
            provision.install_torch_source(spec, archive)
            report = inventory.collect(root, [spec], module)
            specs = root / "specs.json"
            specs.write_text(json.dumps([spec]))
            received = []
            def child(arguments, **options):
                received.append((arguments, options))
                Path(arguments[-1]).write_text(json.dumps(report))
                return types.SimpleNamespace(returncode=0)
            with patch.object(inventory.subprocess, "run", child):
                self.assertEqual(inventory.inspect_runtime(Path(sys.executable), root, specs)["status"], "selection_inventory_verified")
                report["vad_source_sha256"] = "b" * 64
                self.assertEqual(inventory.inspect_runtime(Path(sys.executable), root, specs)["code"], "vad_runtime_inventory_failed")
            self.assertEqual(received[0][0][1:3], ["-I", "-B"])
            self.assertEqual(received[0][1]["env"]["HF_HUB_OFFLINE"], "1")
            with patch.object(inventory.subprocess, "run", side_effect=inventory.subprocess.TimeoutExpired("private-credential-sentinel", 30)):
                failure = inventory.inspect_runtime(Path(sys.executable), root, specs)
                self.assertEqual(failure["code"], "vad_runtime_inventory_failed")
                self.assertNotIn("private-credential-sentinel", json.dumps(failure))

    def fixture(self, root):
        cache = root / "torch"
        cache.mkdir()
        revision = "a" * 40
        files = {"LICENSE": b"MIT License", "hubconf.py": b"raise RuntimeError('must not execute source')", "src/silero_vad/model.py": b"model code", "src/silero_vad/utils_vad.py": b"utils code", "src/silero_vad/data/silero_vad.jit": b"opaque checkpoint"}
        archive = root / "source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name, payload in files.items(): output.writestr("silero-vad-" + revision + "/" + name, payload)
        files[".baseline-source-revision"] = (revision + "\n").encode()
        spec = {"transport": "torch_hub_source", "role": "vad", "repository": "snakers4/silero-vad", "revision": revision, "cache_root": str(cache), "files": sorted(files), "sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}, "archive_sha256": provision.sha256(archive), "source": "https://codeload.github.com/snakers4/silero-vad/zip/" + revision, "license_verified": True, "license_source": "https://raw.githubusercontent.com/snakers4/silero-vad/" + revision + "/LICENSE", "torch_version": "2.8.0"}
        source = root / "whisperx/vads/silero.py"
        source.parent.mkdir(parents=True)
        source.write_text("raise RuntimeError('must not execute source')\ntorch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False, trust_repo=True)\n")
        calls = []
        hub_file = root / "hub.py"
        hub_file.write_text("pinned hub source")
        hub = types.SimpleNamespace(__file__=str(hub_file), set_dir=lambda path: calls.append(path), _parse_repo_info=lambda name: ("snakers4", "silero-vad", "master"))
        module = types.SimpleNamespace(__version__="2.8.0", hub=hub)
        return spec, archive, source, module, calls

    def test_source_install_is_atomic_hash_verified_reusable_and_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, archive, source, module, calls = self.fixture(Path(directory))
            report = provision.install_torch_source(spec, archive)
            self.assertEqual(report["status"], "files_verified")
            self.assertEqual(report["roles"], ["vad"])
            self.assertFalse(report["loadability_verified"])
            self.assertEqual(provision.install_torch_source(spec, archive)["acquisition_status"], "already_verified")
            self.assertEqual(list(Path(spec["cache_root"]).glob("hub/.baseline-source-*")), [])
            target = provision.torch_source_target(spec)
            (target / "extra.py").write_text("unexpected executable source")
            self.assertEqual(provision.inspect_spec(spec)["code"], "torch_source_tree_mismatch")
            with self.assertRaisesRegex(provision.ProvisionError, "torch_source_tree_mismatch"): provision.install_torch_source(spec, archive)
            self.assertTrue((target / "extra.py").exists())

    def test_source_archive_or_member_hash_drift_cannot_publish_or_leave_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, archive, source, module, calls = self.fixture(Path(directory))
            with self.assertRaisesRegex(provision.ProvisionError, "torch_source_archive_mismatch"): provision.install_torch_source({**spec, "archive_sha256": "b" * 64}, archive)
            changed = {**spec, "sha256": {**spec["sha256"], "hubconf.py": "b" * 64}}
            with self.assertRaisesRegex(provision.ProvisionError, "model_hash_mismatch"): provision.install_torch_source(changed, archive)
            self.assertFalse(provision.torch_source_target(spec).exists())
            self.assertEqual(list(Path(spec["cache_root"]).glob("hub/.baseline-source-*")), [])

    def test_source_publication_preserves_concurrent_empty_directory_and_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, archive, source, module, calls = self.fixture(Path(directory))
            publish = provision.rename_directory_exclusive
            def conflicting(staging, target):
                target.mkdir()
                return publish(staging, target)
            with patch.object(provision, "rename_directory_exclusive", conflicting), self.assertRaisesRegex(provision.ProvisionError, "model_acquisition_conflict"): provision.install_torch_source(spec, archive)
            target = provision.torch_source_target(spec)
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])
            (target / "other-writer").write_bytes(b"preserved")
            with self.assertRaises(provision.ProvisionError): provision.install_torch_source(spec, archive)
            self.assertEqual((target / "other-writer").read_bytes(), b"preserved")
            self.assertEqual(list(target.parent.glob(".baseline-source-*")), [])

    def test_source_archive_traversal_links_duplicate_members_and_wrong_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, archive, source, module, calls = self.fixture(root)
            variants = [("silero-vad-" + spec["revision"] + "/../escape", False), ("other-revision/LICENSE", False), ("silero-vad-" + spec["revision"] + "/LICENSE", True)]
            for index, (name, link) in enumerate(variants):
                candidate = root / (str(index) + ".zip")
                with zipfile.ZipFile(candidate, "w") as output:
                    entry = zipfile.ZipInfo(name)
                    if link: entry.external_attr = 0o120777 << 16
                    output.writestr(entry, b"malicious")
                changed = {**spec, "archive_sha256": provision.sha256(candidate)}
                with self.assertRaises(provision.ProvisionError): provision.install_torch_source(changed, candidate)
                self.assertFalse(provision.torch_source_target(spec).exists())
            self.assertFalse((root / "escape").exists())

    def test_vad_inventory_checks_literal_worker_and_actual_cached_branch_without_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, archive, source, module, calls = self.fixture(root)
            provision.install_torch_source(spec, archive)
            report = inventory.collect(root, [spec], module)
            self.assertEqual(report["status"], "selection_inventory_verified")
            self.assertFalse(report["checkpoint_loaded"])
            self.assertTrue(inventory.matches_spec(spec, report))
            self.assertFalse(inventory.matches_spec({**spec, "revision": "b" * 40}, report))
            self.assertNotIn(directory, json.dumps(report))
            self.assertEqual(calls, [str((Path(spec["cache_root"]) / "hub").resolve())])
            source.write_text(source.read_text().replace("onnx=False", "onnx=True"))
            with self.assertRaisesRegex(inventory.InventoryError, "vad_default_loader_mismatch"): inventory.collect(root, [spec], module)

    def test_vad_inventory_rejects_main_cache_runtime_and_revision_marker_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, archive, source, module, calls = self.fixture(root)
            provision.install_torch_source(spec, archive)
            with self.assertRaisesRegex(inventory.InventoryError, "vad_runtime_version_mismatch"): inventory.collect(root, [{**spec, "torch_version": "other"}], module)
            conflict = Path(spec["cache_root"]) / "hub/snakers4_silero-vad_main"
            conflict.mkdir()
            with self.assertRaisesRegex(inventory.InventoryError, "vad_default_cache_conflict"): inventory.collect(root, [spec], module)
            conflict.rmdir()
            marker = provision.torch_source_target(spec) / ".baseline-source-revision"
            marker.write_text("b" * 40 + "\n")
            changed = {**spec, "sha256": {**spec["sha256"], ".baseline-source-revision": provision.sha256(marker)}}
            self.assertEqual(provision.inspect_spec(changed)["code"], "torch_source_revision_mismatch")

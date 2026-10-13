import hashlib
import contextlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import types
import io
import os
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import provision


class ProvisionTests(unittest.TestCase):
    def torch_spec(self, root):
        name = "official.pt"
        return {"transport": "torch_hub", "repository": "torchaudio/TEST", "role": "alignment_fr", "revision": "a" * 40, "files": [name], "cache_root": str(root), "source": "https://download.pytorch.org/torchaudio/models/" + name, "sha256": {name: hashlib.sha256(b"opaque-checkpoint").hexdigest()}, "license_verified": True, "license_source": "https://official.test/LICENSE"}

    def torch_opener(self, spec, payload=b"opaque-checkpoint", **options):
        def open_response(request, timeout):
            self.assertEqual(request.full_url, spec["source"])
            self.assertEqual(timeout, 30)
            response = io.BytesIO(payload)
            response.status = options.get("status", 200)
            response.geturl = lambda: options.get("url", spec["source"])
            response.headers = options.get("headers", {"Content-Length": str(len(payload))})
            return response
        return open_response

    def test_torch_download_verifies_opaque_bytes_and_reuses_only_matching_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            report = provision.provision_torch_spec(spec, opener=self.torch_opener(spec))
            self.assertEqual(report["status"], "files_verified")
            self.assertEqual(report["acquisition_status"], "downloaded_and_verified")
            self.assertFalse(report["loadability_verified"])
            with patch.object(provision.urllib.request, "build_opener", side_effect=AssertionError("unexpected network")):
                self.assertEqual(provision.provision_torch_spec(spec)["acquisition_status"], "already_verified")
            target = Path(directory) / "hub/checkpoints/official.pt"
            target.write_bytes(b"existing drift")
            with self.assertRaisesRegex(provision.ProvisionError, "model_hash_mismatch"): provision.provision_torch_spec(spec)
            self.assertEqual(target.read_bytes(), b"existing drift")

    def test_first_torch_acquisition_records_observation_without_verifying_role(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {**self.torch_spec(directory), "sha256": {}}
            report = provision.provision_torch_spec(spec, observe_hash=True, opener=self.torch_opener(spec))
            self.assertEqual(report["status"], "acquired_hash_observed_unpinned")
            self.assertNotIn("roles", report)
            self.assertEqual(report["files"][0]["sha256"], hashlib.sha256(b"opaque-checkpoint").hexdigest())
            self.assertEqual(provision.inspect_spec(spec)["code"], "model_expected_hash_missing")
            with self.assertRaisesRegex(provision.ProvisionError, "model_observation_requires_new_acquisition"): provision.provision_torch_spec(spec, observe_hash=True)
            pinned = {**spec, "sha256": {"official.pt": report["files"][0]["sha256"]}}
            self.assertEqual(provision.inspect_spec(pinned)["status"], "files_verified")

    def test_torch_authorization_source_paths_and_hash_are_checked_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            variants = [{**spec, "license_verified": False}, {**spec, "license_source": None}, {**spec, "source": "https://private.test/secret"}, {**spec, "repository": None}, {**spec, "role": None}, {**spec, "files": ["../escape.pt"]}, {**spec, "sha256": {}}, {**spec, "sha256": []}, {**spec, "sha256": {"official.pt": None}}, {**spec, "revision": "main"}, {**spec, "revision": None}]
            def forbidden(*args, **kwargs): raise AssertionError("unexpected network")
            for variant in variants:
                with self.subTest(variant=variant), self.assertRaises(provision.ProvisionError): provision.provision_torch_spec(variant, opener=forbidden)
            with self.assertRaisesRegex(provision.ProvisionError, "model_observation_requires_unpinned_spec"): provision.provision_torch_spec(spec, observe_hash=True, opener=forbidden)
            self.assertFalse((Path(directory) / "hub").exists())

    def test_torch_bad_http_truncation_hash_and_encoding_leave_no_checkpoint_or_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            variants = [{"status": 206}, {"url": "https://private.test/secret"}, {"headers": {"Content-Length": "999"}}, {"headers": {"Content-Length": str(1024 ** 3 + 1)}}, {"headers": {"Content-Encoding": "gzip"}}, {"payload": b"wrong"}, {"payload": b""}]
            for options in variants:
                with self.subTest(options=options), self.assertRaises(provision.ProvisionError): provision.provision_torch_spec(spec, opener=self.torch_opener(spec, **options))
                self.assertEqual(list((Path(directory) / "hub/checkpoints").iterdir()), [])

    def test_torch_stream_without_length_is_bounded_and_network_errors_are_private(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            report = provision.provision_torch_spec(spec, opener=self.torch_opener(spec, headers={}))
            self.assertEqual(report["status"], "files_verified")
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            def failed(*args, **kwargs): raise RuntimeError("private-credential-sentinel")
            with self.assertRaises(provision.ProvisionError) as caught: provision.provision_torch_spec(spec, opener=failed)
            self.assertEqual(str(caught.exception), "model_download_failed")
            self.assertEqual(list((Path(directory) / "hub/checkpoints").iterdir()), [])

    def test_torch_download_preserves_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            target = Path(directory) / "hub/checkpoints/official.pt"
            def conflict(source, destination):
                Path(destination).write_bytes(b"other writer")
                raise FileExistsError()
            with patch.object(provision.os, "link", conflict), self.assertRaisesRegex(provision.ProvisionError, "model_acquisition_conflict"): provision.provision_torch_spec(spec, opener=self.torch_opener(spec))
            self.assertEqual(target.read_bytes(), b"other writer")
            self.assertEqual(list(target.parent.glob(".baseline-torch-*")), [])

    def test_torch_timeout_and_redirect_policy_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            with patch.object(provision.time, "monotonic", side_effect=[0, 601]), self.assertRaisesRegex(provision.ProvisionError, "model_download_timeout"): provision.provision_torch_spec(spec, opener=self.torch_opener(spec))
            self.assertEqual(list((Path(directory) / "hub/checkpoints").iterdir()), [])
        with self.assertRaisesRegex(provision.ProvisionError, "model_download_redirect_rejected"): provision.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.test/")

    def test_torch_cli_flags_do_not_enable_network_during_preflight_or_legacy_download(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()): provision.main(["--specs", "unused", "--download-torch", "preflight"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()): provision.main(["--specs", "unused", "--observe-torch-hash", "download"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specs.json"
            path.write_text(json.dumps([self.torch_spec(directory)]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output): result = provision.main(["--specs", str(path), "download"])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(output.getvalue())["code"], "torch_hub_explicit_acquisition_required")

    def test_torch_unannounced_stream_size_limit_and_interruption_clean_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.torch_spec(directory)
            with patch.object(provision, "TORCH_MAX_BYTES", 4), self.assertRaisesRegex(provision.ProvisionError, "model_download_size_invalid"): provision.provision_torch_spec(spec, opener=self.torch_opener(spec, headers={}))
            def cancelled(*args, **kwargs): raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt): provision.provision_torch_spec(spec, opener=cancelled)
            self.assertEqual(list((Path(directory) / "hub/checkpoints").iterdir()), [])

    def test_torch_cache_symlink_cannot_escape_declared_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "hub").symlink_to(outside, target_is_directory=True)
            spec = self.torch_spec(root)
            with self.assertRaises(provision.BaselineError): provision.provision_torch_spec(spec, opener=self.torch_opener(spec))
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_torch_cli_observation_exits_two_until_explicit_pin_then_preflight_is_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {**self.torch_spec(directory), "sha256": {}}
            path = Path(directory) / "specs.json"
            path.write_text(json.dumps([spec]))
            output = io.StringIO()
            opener = types.SimpleNamespace(open=self.torch_opener(spec))
            with patch.object(provision.urllib.request, "build_opener", return_value=opener), contextlib.redirect_stdout(output): result = provision.main(["--specs", str(path), "download", "--download-torch", "--observe-torch-hash"])
            report = json.loads(output.getvalue())
            self.assertEqual(result, 2)
            self.assertEqual(report["status"], "prerequisite_missing")
            self.assertEqual(report["models"][0]["status"], "acquired_hash_observed_unpinned")
            self.assertNotIn("roles", report["models"][0])
            self.assertEqual(report["models"][0]["hash_provenance"], "observed_official_https_first_acquisition_not_published_checksum")
            self.assertEqual(report["models"][0]["source_revision_scope"], "upstream_training_repository_not_download_content")
            spec["sha256"] = {"official.pt": report["models"][0]["files"][0]["sha256"]}
            path.write_text(json.dumps([spec]))
            with patch.object(provision.urllib.request, "build_opener", side_effect=AssertionError("unexpected network")), contextlib.redirect_stdout(io.StringIO()): result = provision.main(["--specs", str(path), "preflight"])
            self.assertEqual(result, 0)

    def community_spec(self, root):
        revision = "a" * 40
        prefix = root / "models--pyannote--speaker-diarization-community-1/snapshots" / revision
        files = {"config.yaml": b"pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n  params:\n    segmentation: $model/segmentation\n    embedding: $model/embedding\n    plda: $model/plda\n", "segmentation/pytorch_model.bin": b"segmentation", "embedding/pytorch_model.bin": b"embedding", "plda/plda.npz": b"plda", "plda/xvec_transform.npz": b"transform"}
        for name, payload in files.items():
            target = prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return {"cache_root": str(root), "repository": "pyannote/speaker-diarization-community-1", "revision": revision, "role": "speaker_segmentation", "files": list(files), "sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}, "components": {"speaker_segmentation": ["segmentation/pytorch_model.bin"], "speaker_embedding": ["embedding/pytorch_model.bin"], "speaker_plda": ["plda/plda.npz", "plda/xvec_transform.npz"]}}

    def test_shared_pipeline_roles_require_verified_component_files_and_config_references(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.community_spec(Path(directory))
            report = provision.inspect_spec(spec)
            self.assertEqual(report["status"], "files_verified")
            self.assertEqual(report["roles"], ["speaker_embedding", "speaker_plda", "speaker_segmentation"])
            self.assertTrue(report["components_verified"])
            self.assertEqual(report["component_config_sha256"], spec["sha256"]["config.yaml"])
            self.assertFalse(report["loadability_verified"])
            legacy = dict(spec)
            legacy.pop("components")
            self.assertEqual(provision.inspect_spec(legacy)["roles"], ["speaker_segmentation"])
            snapshot = Path(directory) / "models--pyannote--speaker-diarization-community-1/snapshots" / spec["revision"]
            (snapshot / "embedding/pytorch_model.bin").write_bytes(b"changed")
            self.assertEqual(provision.inspect_spec(spec)["code"], "model_hash_mismatch")

    def test_component_roles_reject_missing_files_wrong_mapping_and_unknown_components(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.community_spec(Path(directory))
            variants = [{**spec, "files": [name for name in spec["files"] if name != "plda/xvec_transform.npz"]}, {**spec, "components": {"speaker_embedding": ["segmentation/pytorch_model.bin"]}}, {**spec, "components": {"vad": ["segmentation/pytorch_model.bin"]}}]
            for variant in variants:
                with self.subTest(variant=variant["components"]):
                    self.assertEqual(provision.inspect_spec(variant)["code"], "model_component_config_mismatch")

    def test_component_config_drift_and_unsafe_yaml_fail_without_loading_or_exposing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.community_spec(Path(directory))
            config = Path(directory) / "models--pyannote--speaker-diarization-community-1/snapshots" / spec["revision"] / "config.yaml"
            original = config.read_bytes()
            for payload in (original.replace(b"$model/embedding", b"external/embedding"), b"!!python/object/apply:builtins.print [private-credential-sentinel]"):
                config.write_bytes(payload)
                spec["sha256"]["config.yaml"] = hashlib.sha256(payload).hexdigest()
                report = provision.inspect_spec(spec)
                self.assertEqual(report["code"], "model_component_config_mismatch")
                self.assertNotIn("private-credential-sentinel", json.dumps(report))

    def test_bind_default_reference_requires_verified_snapshot_and_never_overwrites_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.spec(root)
            reference = root / "models--org--model/refs/main"
            with self.assertRaises(provision.ProvisionError): provision.bind_default_reference({**spec, "sha256": {"weights.bin": "b" * 64}})
            self.assertFalse(reference.exists())
            self.assertEqual(provision.bind_default_reference(spec)["default_reference_status"], "created_from_verified_snapshot")
            self.assertEqual(reference.read_text(), spec["revision"])
            self.assertEqual(provision.bind_default_reference(spec)["default_reference_status"], "already_pinned")
            reference.write_text("b" * 40)
            with self.assertRaisesRegex(provision.ProvisionError, "model_default_reference_conflict"): provision.bind_default_reference(spec)
            self.assertEqual(reference.read_text(), "b" * 40)
            with self.assertRaisesRegex(provision.ProvisionError, "torch_hub_has_no_huggingface_reference"): provision.bind_default_reference({**spec, "transport": "torch_hub"})
            self.assertEqual(list(reference.parent.glob(".baseline-ref-*")), [])

    def test_bind_default_reference_preserves_a_concurrent_writer_and_cleans_its_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.spec(root)
            reference = root / "models--org--model/refs/main"
            def conflicting_link(source, target):
                Path(target).write_text("b" * 40)
                raise FileExistsError()
            with patch.object(provision.os, "link", conflicting_link), self.assertRaisesRegex(provision.ProvisionError, "model_default_reference_conflict"):
                provision.bind_default_reference(spec)
            self.assertEqual(reference.read_text(), "b" * 40)
            self.assertEqual(list(reference.parent.glob(".baseline-ref-*")), [])

    def spec(self, root):
        snapshot = root / "models--org--model/snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        path = snapshot / "weights.bin"
        path.write_bytes(b"weights")
        return {"cache_root": str(root), "repository": "org/model", "revision": "a" * 40, "files": ["weights.bin"], "sha256": {"weights.bin": hashlib.sha256(b"weights").hexdigest()}, "gated": True, "license_verified": True, "license_source": "https://huggingface.co/org/model"}

    def test_expected_hash_missing_and_mismatch_do_not_verify_models(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.spec(Path(directory))
            self.assertEqual(provision.inspect_spec(spec)["status"], "files_verified")
            self.assertFalse(provision.inspect_spec(spec)["loadability_verified"])
            self.assertEqual(provision.inspect_spec(dict(spec, sha256={}))["code"], "model_expected_hash_missing")
            self.assertEqual(provision.inspect_spec(dict(spec, sha256={"weights.bin": "b" * 64}))["code"], "model_hash_mismatch")

    def test_gated_download_requires_private_credential_and_sanitizes_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.spec(Path(directory))
            called = []
            def failing(**options):
                called.append(options)
                raise RuntimeError("sentinel-private-credential")
            with self.assertRaises(provision.ProvisionError):
                provision.provision_spec(spec, downloader=failing)
            self.assertEqual(called, [])
            with self.assertRaises(provision.ProvisionError) as error:
                provision.provision_spec(spec, "sentinel-private-credential", failing)
            self.assertEqual(str(error.exception), "model_download_failed")
            self.assertEqual(called[0]["token"], "sentinel-private-credential")
            self.assertEqual(called[0]["revision"], "a" * 40)

    def test_bridge_credential_only_arrives_from_pipe(self):
        received = []
        child = types.SimpleNamespace(communicate=lambda timeout: (b"sentinel-private-credential", None), returncode=0, poll=lambda: 0)
        def launcher(arguments, **options):
            received.append((arguments, options))
            return child
        self.assertEqual(provision.read_credential(Path("/private/bridge"), launcher), "sentinel-private-credential")
        self.assertEqual(received[0][1]["stdout"], subprocess.PIPE)
        self.assertEqual(received[0][1]["stderr"], subprocess.DEVNULL)
        self.assertNotIn("sentinel-private-credential", str(received))

    def test_safetensors_header_is_read_without_loading_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
            result = provision.tensor_header(path)
            self.assertEqual(result["tensors"][0]["shape"], [2])
            self.assertFalse(result["native_conversion_proven"])
            path.write_bytes(struct.pack("<Q", 99_999_999))
            with self.assertRaises(provision.ProvisionError):
                provision.tensor_header(path)

    def test_bridge_timeout_terminates_child_and_emits_only_fixed_code(self):
        actions = []
        def communicate(timeout):
            raise subprocess.TimeoutExpired("sentinel-private-credential", timeout)
        child = types.SimpleNamespace(communicate=communicate, poll=lambda: None, kill=lambda: actions.append("kill"), wait=lambda: actions.append("wait"))
        with self.assertRaises(provision.ProvisionError) as error:
            provision.read_credential(Path("bridge"), lambda *args, **kwargs: child)
        self.assertEqual(str(error.exception), "keychain_unavailable")
        self.assertEqual(actions, ["kill", "wait"])

    def test_npz_numeric_headers_preserve_shape_dtype_without_loading_weights(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            np.savez(path, weight=np.zeros((2, 3), dtype=np.float16), scalar=np.array(7, dtype=np.int64))
            report = provision.tensor_header(path)
            self.assertEqual(report["tensors"][0]["shape"], [2, 3])
            self.assertEqual(report["tensors"][0]["dtype"], "<f2")
            self.assertEqual(report["tensors"][1]["shape"], [])
            self.assertFalse(report["native_conversion_proven"])

    def test_npz_rejects_pickle_and_invalid_declared_payload(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            np.savez(path, unsafe=np.array([{"secret": 1}], dtype=object))
            with self.assertRaises(provision.ProvisionError): provision.tensor_header(path)
            header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000000,), }\n"
            with zipfile.ZipFile(path, "w") as archive: archive.writestr("weight.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 4)
            with self.assertRaises(provision.ProvisionError): provision.tensor_header(path)

    def test_native_file_descriptor_logging_is_discarded_and_stdout_restored(self):
        root = Path(__file__).resolve().parents[2]
        script = "import os,sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()));from tools.baseline.provision import quiet_download\nwith quiet_download():\n os.write(1,b'sentinel-private-credential');os.write(2,b'sentinel-private-credential')\nos.write(1,b'restored')"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=root, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"restored")
        self.assertEqual(result.stderr, b"")

    def test_actual_hub_transport_flag_is_temporarily_disabled_and_restored(self):
        from huggingface_hub import constants
        prior = constants.HF_HUB_DISABLE_XET
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            spec = self.spec(Path(directory))
            def downloader(**options):
                observed.append(constants.HF_HUB_DISABLE_XET)
            with patch("huggingface_hub.hf_hub_download", downloader):
                result = provision.provision_spec(spec, "sentinel-private-credential")
        self.assertEqual(observed, [True])
        self.assertEqual(constants.HF_HUB_DISABLE_XET, prior)
        self.assertEqual(result["status"], "files_verified")


if __name__ == "__main__":
    unittest.main()

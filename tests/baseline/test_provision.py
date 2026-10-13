import hashlib
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

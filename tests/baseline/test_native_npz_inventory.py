import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
import zipfile
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.provision import tensor_header


@unittest.skipUnless(platform.system() == "Darwin", "Native CommonCrypto inventory targets macOS")
class NativeNPZInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "native_npz_inventory"
        root = Path(__file__).resolve().parents[2]
        command = ["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Wno-deprecated-declarations", str(root / "tools/baseline/native_npz_inventory.cpp"), "-o", str(cls.binary)]
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, path, digest=None, extra=()):
        expected = digest or hashlib.sha256(path.read_bytes()).hexdigest()
        return subprocess.run([str(self.binary), "--model", str(path), "--sha256", expected, *extra], capture_output=True, text=True, timeout=30)

    def test_native_shapes_match_python_and_output_cannot_be_overwritten(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.npz"
            np.savez(path, weight=np.zeros((2, 3), dtype=np.float16), scalar=np.array(7, dtype=np.int64), empty=np.zeros((0, 3), dtype=np.float32))
            result = self.invoke(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report["native_conversion_proven"])
            keys = ("name", "shape", "dtype", "fortran_order")
            self.assertEqual([{key: tensor[key] for key in keys} for tensor in report["tensors"]], tensor_header(path)["tensors"])
            output = root / "output.json"
            self.assertEqual(self.invoke(path, extra=("--output", str(output))).returncode, 0)
            original = output.read_bytes()
            self.assertEqual(self.invoke(path, extra=("--output", str(output))).returncode, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_bad_hash_compression_pickle_and_truncation_are_rejected_with_sanitizers(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(path, weight=np.zeros(2, dtype=np.float32))
            self.assertEqual(json.loads(self.invoke(path, "a" * 64).stdout)["code"], "model_hash_mismatch")
            np.savez_compressed(path, weight=np.zeros(2, dtype=np.float32))
            self.assertEqual(json.loads(self.invoke(path).stdout)["code"], "zip_compression_or_encryption_unsupported")
            np.savez(path, unsafe=np.array([{}], dtype=object))
            self.assertEqual(json.loads(self.invoke(path).stdout)["code"], "npy_dtype_unsupported")
            path.write_bytes(b"invalid")
            result = self.invoke(path)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("AddressSanitizer", result.stderr)
            self.assertNotIn("runtime error:", result.stderr)

    def test_traversal_names_and_secret_paths_are_not_reported(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = root / "array.npy"
            np.save(array, np.zeros(1, dtype=np.float32))
            path = root / "model.npz"
            with zipfile.ZipFile(path, "w") as archive: archive.writestr("../escape.npy", array.read_bytes())
            self.assertEqual(json.loads(self.invoke(path).stdout)["code"], "zip_tensor_name_invalid")
            missing = root / "sentinel-private-credential"
            result = self.invoke(missing, "a" * 64)
            self.assertNotIn("sentinel-private-credential", result.stdout + result.stderr)
            result = self.invoke(path, extra=("--sentinel-private-credential",))
            self.assertNotIn("sentinel-private-credential", result.stdout + result.stderr)

    def test_zip64_directory_and_locator_are_read_without_large_allocations(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(path, weight=np.zeros((2, 3), dtype=np.float16))
            original = path.read_bytes()
            eocd_offset = original.rfind(b"PK\x05\x06")
            eocd = bytearray(original[eocd_offset:])
            count = struct.unpack_from("<H", eocd, 10)[0]
            size, offset = struct.unpack_from("<II", eocd, 12)
            record = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, count, count, size, offset)
            locator = struct.pack("<IIQI", 0x07064B50, 0, eocd_offset, 1)
            struct.pack_into("<I", eocd, 16, 0xFFFFFFFF)
            path.write_bytes(original[:eocd_offset] + record + locator + eocd)
            result = self.invoke(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["tensors"][0]["shape"], [2, 3])
            broken = bytearray(path.read_bytes())
            struct.pack_into("<I", broken, eocd_offset + len(record) + 16, 2)
            path.write_bytes(broken)
            self.assertEqual(json.loads(self.invoke(path).stdout)["code"], "zip64_locator_invalid")


if __name__ == "__main__":
    unittest.main()

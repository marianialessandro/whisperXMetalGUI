from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import sys
import subprocess
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_embedding import NativeEmbeddingNormalization, build_library


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "embedding.dylib"
        cls.identity = build_library(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeEmbeddingNormalization(self.library, self.identity["library_sha256"])

    def compare(self, context, values):
        snapshot = values.copy()
        expected = values / np.linalg.norm(values, axis=1, keepdims=True)
        actual, record = context.normalize(values)
        np.testing.assert_array_equal(values, snapshot)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertEqual(actual.dtype, np.float32)
        self.assertGreater(record["computation_gpu_seconds"], 0)
        self.assertEqual(record["logical_result_readback_bytes"], values.nbytes)
        self.assertFalse(record["full_pipeline_gpu_certified"])
        return actual

    def test_numpy_reduction_boundaries_and_realistic_dimensions(self):
        rng = np.random.default_rng(721)
        with self.context() as context:
            for dimension in (1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 512, 1024):
                self.compare(context, rng.normal(size=(33, dimension)).astype(np.float32))
            self.compare(context, rng.normal(size=(4096, 256)).astype(np.float32))

    def test_domain_signed_zero_readonly_input_and_reuse(self):
        values = np.array([[0., -0., 2. ** -32, -(2. ** -32)], [2. ** 32, -(2. ** 32), 1., -1.]], dtype=np.float32)
        values.flags.writeable = False
        with self.context() as context:
            actual = self.compare(context, values)
            self.assertTrue(np.signbit(actual[0, 1]))
            self.compare(context, np.ones((1, 1), dtype=np.float32))
            for invalid in (0., np.nan, np.inf, 2. ** -33, 2. ** 33):
                with self.assertRaises(ValueError):
                    context.normalize(np.full((2, 8), invalid, dtype=np.float32))
            self.compare(context, values)

    def test_format_shape_hash_close_and_serialized_threads(self):
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            NativeEmbeddingNormalization(self.library, "0" * 64)
        with self.assertRaisesRegex(ValueError, "output_exists"):
            build_library(self.library)
        with self.context() as context:
            for invalid in (np.ones((2, 8)), np.ones(8, dtype=np.float32), np.ones((2, 8), dtype=np.float32)[:, ::2], np.ones((0, 8), dtype=np.float32), np.ones((1, 1025), dtype=np.float32)):
                with self.assertRaises(ValueError):
                    context.normalize(invalid)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda size: self.compare(context, np.ones((size, 256), dtype=np.float32)), range(1, 9)))
            self.assertEqual(len(results), 8)
        context.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            context.normalize(np.ones((1, 8), dtype=np.float32))

    def test_native_abi_under_address_and_undefined_sanitizers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "embedding-sanitizers"
            shader = root / "tools/baseline/native_embedding.metal"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-fobjc-arc", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Wno-deprecated-declarations", str(root / "tools/baseline/native_embedding.mm"), str(root / "tests/fixtures/baseline/native_embedding_abi.cpp"), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)], capture_output=True, check=True, timeout=60)
            result = subprocess.run([str(binary), str(shader)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("runtime error:", result.stderr)
            self.assertNotIn("ERROR: AddressSanitizer", result.stderr)


if __name__ == "__main__":
    unittest.main()

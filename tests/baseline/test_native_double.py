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
from tools.baseline.native_double import NativeDoubleNormalization, build_library, shader_source


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeDoubleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "double.dylib"
        cls.identity = build_library(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeDoubleNormalization(self.library, self.identity["library_sha256"])

    def compare(self, context, values, order="pairwise"):
        reference_values = np.asfortranarray(values) if order == "sequential" else values
        snapshot = values.copy()
        expected = reference_values / np.linalg.norm(reference_values, axis=1, keepdims=True)
        actual, record = context.normalize(values, order)
        np.testing.assert_array_equal(values, snapshot)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        self.assertEqual(actual.dtype, np.float64)
        self.assertGreater(record["computation_gpu_seconds"], 0)
        self.assertEqual(record["logical_result_readback_bytes"], values.nbytes)
        self.assertFalse(record["full_pipeline_gpu_certified"])
        return actual

    def test_numpy_reduction_boundaries_and_realistic_dimensions(self):
        rng = np.random.default_rng(721)
        with self.context() as context:
            for dimension in (1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 512, 1024):
                values = rng.normal(size=(33, dimension)).astype(np.float64)
                for order in ("pairwise", "sequential"):
                    self.compare(context, values, order)
            self.compare(context, rng.normal(size=(4096, 256)).astype(np.float64))

    def test_wide_exponents_and_rounding_neighbors(self):
        rng = np.random.default_rng(559)
        values = np.ldexp(rng.uniform(.5, 1., (1024, 257)), rng.integers(-199, 200, (1024, 257)))
        values[rng.integers(0, 1024, 1000), rng.integers(0, 257, 1000)] *= -1
        neighbors = np.array([[np.nextafter(1., 0.), 1., np.nextafter(1., 2.), -0.], [np.nextafter(2., 0.), 2., np.nextafter(2., 3.), 0.]])
        with self.context() as context:
            for order in ("pairwise", "sequential"):
                self.compare(context, values, order)
                self.compare(context, neighbors, order)

    def test_domain_signed_zero_readonly_input_and_reuse(self):
        values = np.array([[0., -0., 2. ** -200, -(2. ** -200)], [2. ** 200, -(2. ** 200), 1., -1.]], dtype=np.float64)
        values.flags.writeable = False
        with self.context() as context:
            actual = self.compare(context, values)
            self.assertTrue(np.signbit(actual[0, 1]))
            self.compare(context, np.ones((1, 1), dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "reduction_order"):
                context.normalize(np.ones((1, 8)), "unknown")
            for invalid in (0., np.nan, np.inf, 2. ** -201, 2. ** 201):
                with self.assertRaises(ValueError):
                    context.normalize(np.full((2, 8), invalid, dtype=np.float64))
            self.compare(context, values)

    def test_format_shape_hash_close_and_serialized_threads(self):
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            NativeDoubleNormalization(self.library, "0" * 64)
        with self.assertRaisesRegex(ValueError, "output_exists"):
            build_library(self.library)
        with self.context() as context:
            for invalid in (np.ones((2, 8), dtype=np.float32), np.ones(8, dtype=np.float64), np.ones((2, 8), dtype=np.float64)[:, ::2], np.ones((0, 8), dtype=np.float64), np.ones((1, 1025), dtype=np.float64)):
                with self.assertRaises(ValueError):
                    context.normalize(invalid)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda size: self.compare(context, np.ones((size, 256), dtype=np.float64)), range(1, 9)))
            self.assertEqual(len(results), 8)
        context.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            context.normalize(np.ones((1, 8), dtype=np.float64))

    def test_native_abi_under_address_and_undefined_sanitizers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "double-sanitizers"
            shader = Path(directory) / "double.metal"
            shader.write_bytes(shader_source())
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-fobjc-arc", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Wno-deprecated-declarations", str(root / "tools/baseline/native_double.mm"), str(root / "tests/fixtures/baseline/native_double_abi.cpp"), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)], capture_output=True, check=True, timeout=60)
            result = subprocess.run([str(binary), str(shader)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("runtime error:", result.stderr)
            self.assertNotIn("ERROR: AddressSanitizer", result.stderr)


if __name__ == "__main__":
    unittest.main()

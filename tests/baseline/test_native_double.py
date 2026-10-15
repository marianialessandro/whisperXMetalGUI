from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def ordered_product(self, left, right):
        result = np.zeros((len(left), right.shape[1]), dtype=np.float64)
        for k in range(left.shape[1]):
            result = np.add(result, np.multiply(left[:, k, None], right[k, None, :]))
        return result

    def test_matrix_products_match_separately_rounded_sequential_reference(self):
        rng = np.random.default_rng(917)
        with self.context() as context:
            for inner in (1, 7, 8, 31, 128, 256, 1024):
                left, right = rng.normal(size=(17, inner)), rng.normal(size=(inner, 19))
                expected = self.ordered_product(left, right)
                actual, record = context.matmul(left, right)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertFalse(record["reference_blas_bitwise_certified"])
                self.assertEqual(record["logical_result_readback_bytes"], actual.nbytes)
            left = np.array([[2. ** 100, -(2. ** 100), 2. ** -100, 0., -0.], [1., np.nextafter(-1., 0.), -0., 2., -2.]])
            right = np.array([[2. ** 100, 1.], [2. ** 100, 1.], [2. ** -100, 1.], [-0., 1.], [0., 1.]])
            actual, _ = context.matmul(left, right)
            np.testing.assert_array_equal(actual.view(np.uint64), self.ordered_product(left, right).view(np.uint64))
            self.compare(context, np.ones((1, 256)))

    def test_matrix_validation_lifetime_hash_capability_and_work_budget(self):
        with self.context() as context:
            valid = np.ones((2, 8))
            for left, right in ((valid.astype(np.float32), valid.T.copy()), (valid[:, ::2], valid.T.copy()), (valid, np.ones((7, 3))), (np.ones((4096, 1024)), np.ones((1024, 9)))):
                with self.assertRaises(ValueError):
                    context.matmul(left, right)
            for invalid in (np.inf, np.nan, 2. ** 101, 2. ** -101):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.matmul(np.full((2, 8), invalid), valid.T.copy())
            left, right = valid.copy(), valid.T.copy()
            left.flags.writeable = right.flags.writeable = False
            actual, _ = context.matmul(left, right)
            np.testing.assert_array_equal(actual, np.full((2, 2), 8.))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.matmul(left, right)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.matmul, left, right) for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0], actual)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.matmul(left, right)

    def test_centering_and_dimension_scaling_are_bitwise_numpy(self):
        rng = np.random.default_rng(811)
        with self.context() as context:
            values = rng.normal(size=(37, 257))
            mean = rng.normal(size=257)
            actual, record = context.center(values, mean)
            np.testing.assert_array_equal(actual.view(np.uint64), (values - mean).view(np.uint64))
            self.assertEqual(record["logical_input_upload_bytes"], values.nbytes + mean.nbytes)
            self.assertGreater(record["computation_gpu_seconds"], 0)
            for dimension in (1, 2, 3, 7, 16, 31, 128, 256, 1023, 1024):
                actual, record = context.scale_dimension(values, dimension)
                np.testing.assert_array_equal(actual.view(np.uint64), (np.sqrt(dimension) * values).view(np.uint64))
                self.assertEqual(record["scale_dimension"], dimension)
                self.assertGreater(record["computation_gpu_seconds"], 0)
            wide = np.ldexp(rng.uniform(.5, 1., (41, 257)), rng.integers(-449, 450, (41, 257)))
            offsets = np.ldexp(rng.uniform(.5, 1., 257), rng.integers(-449, 450, 257))
            wide[:, ::2] *= -1
            actual, _ = context.center(wide, offsets)
            np.testing.assert_array_equal(actual.view(np.uint64), (wide - offsets).view(np.uint64))
            zeros = np.array([[0., -0., 0., -0.], [1., -1., np.nextafter(1., 2.), np.nextafter(-1., -2.)]])
            offsets = np.array([0., 0., -0., -0.])
            actual, _ = context.center(zeros, offsets)
            np.testing.assert_array_equal(actual.view(np.uint64), (zeros - offsets).view(np.uint64))
            actual, _ = context.scale_dimension(zeros, 3)
            np.testing.assert_array_equal(actual.view(np.uint64), (np.sqrt(3) * zeros).view(np.uint64))

    def test_elementwise_validation_readonly_threads_close_and_missing_abi(self):
        values, mean = np.ones((3, 8)), np.ones(8)
        values.flags.writeable = mean.flags.writeable = False
        with self.context() as context:
            actual, _ = context.center(values, mean)
            np.testing.assert_array_equal(actual, np.zeros_like(values))
            for invalid in (0, 1025, True, np.int64(2), 2.):
                with self.assertRaisesRegex(ValueError, "dimension_invalid"):
                    context.scale_dimension(values, invalid)
            for invalid in (np.ones(7), np.ones(8, dtype=np.float32), np.ones((1, 8)), np.ones(16)[::2]):
                with self.assertRaises(ValueError):
                    context.center(values, invalid)
            for invalid in (np.ones((3, 8), dtype=np.float32), np.ones((3, 16))[:, ::2], np.ones((0, 8))):
                with self.assertRaises(ValueError):
                    context.center(invalid, mean)
            for invalid in (np.nan, np.inf, 2. ** 451, 2. ** -451):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.center(np.full((3, 8), invalid), mean)
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.center(values, np.full(8, invalid))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.center(values, mean)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.scale_dimension, values, 3) for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0], np.sqrt(3) * values)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.center(values, mean)

    def test_vbx_precomputed_terms_bitwise_numpy_orders_and_boundaries(self):
        rng = np.random.default_rng(922)
        with self.context() as context:
            for columns in (1, 7, 8, 127, 128, 129, 256, 1024):
                values, phi = rng.normal(size=(37, columns)), rng.uniform(.1, 5., columns)
                phi[0] = -0.
                for order in ("pairwise", "sequential"):
                    reference = values if order == "pairwise" else np.asfortranarray(values)
                    expected = (-.5 * (np.sum(reference**2, axis=1, keepdims=True) + columns * np.log(2 * np.pi)), np.sqrt(phi), reference * np.sqrt(phi))
                    actual, record = context.precompute_vbx(values, phi, order)
                    for result, original in zip(actual, expected):
                        np.testing.assert_array_equal(result.view(np.uint64), original.view(np.uint64))
                    self.assertEqual(record["gpu_kernels"], ["double_vbx_terms", "double_vbx_rho"])
                    self.assertEqual(record["logical_result_readback_bytes"], sum(a.nbytes for a in actual))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
            values = np.array([[0., -0., 2. ** 200, -(2. ** -200)], [0., 0., 0., 0.]])
            phi = np.array([-0., 0., 2. ** 200, 2. ** -200])
            expected = (-.5 * (np.sum(values**2, axis=1, keepdims=True) + 4 * np.log(2 * np.pi)), np.sqrt(phi), values * np.sqrt(phi))
            actual, _ = context.precompute_vbx(values, phi)
            for result, original in zip(actual, expected):
                np.testing.assert_array_equal(result.view(np.uint64), original.view(np.uint64))
            actual, _ = context.precompute_vbx(np.ones((4096, 1)), np.ones(1))
            self.assertEqual(actual[0].shape, (4096, 1))
            self.compare(context, np.ones((1, 256)))

    def test_vbx_precompute_validation_readonly_lifetime_and_capability(self):
        values, phi = np.ones((3, 8)), np.ones(8)
        values.flags.writeable = phi.flags.writeable = False
        with self.context() as context:
            for a, b in ((values.astype(np.float32), phi), (values, phi.astype(np.float32)), (values[:, ::2], phi), (values, phi[:7]), (np.ones((0, 8)), phi)):
                with self.assertRaises(ValueError):
                    context.precompute_vbx(a, b)
            for invalid in (np.nan, np.inf, -1., 2. ** 201, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "phi_domain_invalid"):
                    context.precompute_vbx(values, np.full(8, invalid))
            for invalid in (np.nan, np.inf, 2. ** 201, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.precompute_vbx(np.full((3, 8), invalid), phi)
            with self.assertRaisesRegex(ValueError, "reduction_invalid"):
                context.precompute_vbx(values, phi, "invalid")
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.precompute_vbx(values, phi)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.precompute_vbx, values, phi) for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0][2], values)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.precompute_vbx(values, phi)

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

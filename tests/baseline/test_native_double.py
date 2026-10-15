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

    def test_vbx_statistics_bitwise_reductions_alpha_and_bounds(self):
        rng = np.random.default_rng(923)
        with self.context() as context:
            for frames in (1, 7, 8, 127, 128, 129, 256, 1024, 4096):
                gamma = rng.uniform(0., 1., (frames, 3))
                phi = rng.uniform(.1, 8., 17)
                for layout in ("C", "F"):
                    source = np.array(gamma, order=layout)
                    expected = 1. / (1 + .07 / .8 * source.sum(axis=0, keepdims=True).T * phi)
                    actual, record = context.inv_l_vbx(np.ascontiguousarray(source.T), phi, .07, .8, "sequential" if layout == "C" else "pairwise")
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    statistics = rng.normal(size=actual.shape)
                    statistics[0, :2] = (0., -0.)
                    alpha, alpha_record = context.alpha_vbx(actual, statistics, .07, .8)
                    reference = .07 / .8 * expected * statistics
                    np.testing.assert_array_equal(alpha.view(np.uint64), reference.view(np.uint64))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(alpha_record["gpu_kernels"], ["double_vbx_statistics"])
            for fa, fb in ((2. ** -20, 2. ** 20), (2. ** 20, 2. ** -20)):
                gamma = np.array([[0., 2. ** -160, 1.]])
                phi = np.array([0., 2. ** -100, 2. ** 100])
                expected = 1. / (1 + fa / fb * gamma.sum(axis=1, keepdims=True) * phi)
                actual, _ = context.inv_l_vbx(gamma, phi, fa, fb)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                stats = np.array([[-0., 2. ** -100, -(2. ** 100)]])
                alpha, _ = context.alpha_vbx(actual, stats, fa, fb)
                np.testing.assert_array_equal(alpha.view(np.uint64), (fa / fb * expected * stats).view(np.uint64))

    def test_vbx_statistics_validation_lifetime_threads_and_capability(self):
        gamma, phi = np.full((3, 19), .2), np.ones(8)
        gamma.flags.writeable = phi.flags.writeable = False
        with self.context() as context:
            for invalid in (np.nan, np.inf, -1., 1.1, 2. ** -161):
                with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                    context.inv_l_vbx(np.full_like(gamma, invalid), phi, .07, .8)
            for invalid in (0., -1., np.inf, 2. ** -21, 2. ** 21):
                with self.assertRaisesRegex(ValueError, "scalar_domain_invalid"):
                    context.inv_l_vbx(gamma, phi, invalid, .8)
            for invalid in (np.nan, np.inf, -1., 2. ** -101, 2. ** 101):
                with self.assertRaisesRegex(ValueError, "right_domain_invalid"):
                    context.inv_l_vbx(gamma, np.full_like(phi, invalid), .07, .8)
            for left, right in ((gamma.astype(np.float32), phi), (gamma[:, ::2], phi), (gamma, phi.astype(np.float32)), (gamma, phi.reshape(1, -1)), (np.empty((0, 19)), phi)):
                with self.assertRaises(ValueError):
                    context.inv_l_vbx(left, right, .07, .8)
            with self.assertRaises(ValueError):
                context.alpha_vbx(np.ones((3, 8)), np.ones((3, 7)), .07, .8)
            with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                context.alpha_vbx(np.zeros((3, 8)), np.ones((3, 8)), .07, .8)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.inv_l_vbx(gamma, phi, .07, .8)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.inv_l_vbx, gamma, phi, .07, .8) for _ in range(4)]
                for future in futures:
                    self.assertEqual(future.result()[0].shape, (3, 8))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.inv_l_vbx(gamma, phi, .07, .8)

    def test_vbx_score_arithmetic_bitwise_order_signed_zero_and_bounds(self):
        rng = np.random.default_rng(924)
        with self.context() as context:
            for rows, columns in ((1, 1), (7, 8), (129, 17), (1024, 1)):
                inv_l, alpha = rng.uniform(.01, 1., (rows, columns)), rng.normal(size=(rows, columns))
                inv_l[0, 0], alpha[0, 0] = -0., -0.
                actual, record = context.penalty_terms_vbx(inv_l, alpha)
                np.testing.assert_array_equal(actual.view(np.uint64), (inv_l + alpha**2).view(np.uint64))
                self.assertEqual(record["gpu_kernels"], ["double_vbx_scores"])
                frames, penalty, g = rng.normal(size=(rows, columns)), rng.normal(size=columns), rng.normal(size=rows)
                for fa in (.07, 2. ** -20, 2. ** 20):
                    actual, record = context.frame_scores_vbx(frames, penalty, g, fa)
                    expected = fa * (frames - .5 * penalty + g[:, None])
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(record["host_vector_packing_copy_bytes"], penalty.nbytes + g.nbytes)
            inv_l = np.array([[0., 2. ** -200, 2. ** 200, -0.]])
            alpha = np.array([[-0., 2. ** -200, -(2. ** 200), 0.]])
            actual, _ = context.penalty_terms_vbx(inv_l, alpha)
            np.testing.assert_array_equal(actual.view(np.uint64), (inv_l + alpha**2).view(np.uint64))
            frames = np.array([[2. ** 199, 0., -0., 2. ** -200], [-(2. ** 199), -0., 0., -(2. ** -200)]])
            penalty, g = np.array([2. ** 200, -0., 0., 2. ** -199]), np.array([-0., 2. ** 200])
            actual, _ = context.frame_scores_vbx(frames, penalty, g, .07)
            np.testing.assert_array_equal(actual.view(np.uint64), (.07 * (frames - .5 * penalty + g[:, None])).view(np.uint64))
            actual, _ = context.frame_scores_vbx(np.zeros((4096, 1)), np.zeros(1), np.zeros(4096), 1.)
            self.assertEqual(actual.shape, (4096, 1))

    def test_vbx_score_validation_lifetime_threads_and_capability(self):
        values, alpha = np.ones((3, 8)), np.ones((3, 8))
        penalty, g = np.ones(8), np.ones(3)
        values.flags.writeable = alpha.flags.writeable = penalty.flags.writeable = g.flags.writeable = False
        with self.context() as context:
            for invalid in (np.nan, np.inf, -1., 2. ** 201, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                    context.penalty_terms_vbx(np.full_like(values, invalid), alpha)
            for invalid in (np.nan, np.inf, 2. ** 201, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "right_domain_invalid"):
                    context.penalty_terms_vbx(values, np.full_like(alpha, invalid))
                with self.assertRaisesRegex(ValueError, "right_domain_invalid"):
                    context.frame_scores_vbx(values, np.full_like(penalty, invalid), g, .07)
            for invalid in (0., -1., np.inf, 2. ** -21, 2. ** 21):
                with self.assertRaisesRegex(ValueError, "scalar_domain_invalid"):
                    context.frame_scores_vbx(values, penalty, g, invalid)
            for left, right in ((values.astype(np.float32), alpha), (values[:, ::2], alpha), (values, alpha.astype(np.float32)), (values, alpha[:, :7]), (np.ones((1025, 1)), np.ones((1025, 1)))):
                with self.assertRaises(ValueError):
                    context.penalty_terms_vbx(left, right)
            with self.assertRaises(ValueError):
                context.frame_scores_vbx(values, penalty[:-1], g, .07)
            with self.assertRaises(ValueError):
                context.frame_scores_vbx(values, penalty, g.astype(np.float32), .07)
            with patch("tools.baseline.native_double.np.concatenate", side_effect=AssertionError("oversized vectors packed")), self.assertRaisesRegex(ValueError, "shape_invalid"):
                context.frame_scores_vbx(np.ones((4097, 1)), np.ones(1), np.ones(4097), .07)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.penalty_terms_vbx(values, alpha)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.frame_scores_vbx, values, penalty, g, .07) for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0], .07 * (values - .5 * penalty + g[:, None]))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.penalty_terms_vbx(values, alpha)

    def test_vbx_prior_bitwise_numpy_reductions_layouts_and_zero_speakers(self):
        rng = np.random.default_rng(925)
        with self.context() as context:
            for frames, speakers in ((1, 1), (7, 3), (8, 7), (127, 8), (128, 127), (129, 128), (257, 129), (4096, 3), (17, 1024)):
                gamma = rng.uniform(0., 1., (frames, speakers))
                if speakers > 1:
                    gamma[:, 0] = -0.
                for layout in ("C", "F"):
                    source = np.array(gamma, order=layout)
                    masses = np.sum(source, axis=0)
                    expected = masses / masses.sum()
                    order = "pairwise" if layout == "F" or speakers == 1 else "sequential"
                    actual, record = context.prior_vbx(np.ascontiguousarray(source.T), order)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(record["gpu_kernels"], ["double_vbx_gamma_sum", "double_vbx_prior"])
                    self.assertEqual(record["logical_result_readback_bytes"], speakers * 8)
                    self.assertGreater(record["computation_gpu_seconds"], 0)
            for gamma in (np.array([[0., -0.], [2. ** -200, 2. ** -200]]), np.array([[2. ** -200, 0.], [1., 1.]])):
                masses = gamma.sum(axis=1)
                actual, _ = context.prior_vbx(gamma)
                np.testing.assert_array_equal(actual.view(np.uint64), (masses / masses.sum()).view(np.uint64))

    def test_vbx_prior_validation_lifetime_threads_and_capability(self):
        gamma = np.full((3, 19), .2)
        gamma.flags.writeable = False
        with self.context() as context:
            for invalid in (np.nan, np.inf, -1., 1.01, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                    context.prior_vbx(np.full_like(gamma, invalid))
            with self.assertRaisesRegex(ValueError, "zero_mass_not_supported"):
                context.prior_vbx(np.zeros_like(gamma))
            for invalid in (gamma.astype(np.float32), gamma[:, ::2], np.ones((1025, 1)), np.ones((1, 4097)), np.empty((0, 19)), gamma.ravel()):
                with self.assertRaises(ValueError):
                    context.prior_vbx(invalid)
            with self.assertRaisesRegex(ValueError, "reduction_invalid"):
                context.prior_vbx(gamma, "invalid")
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.prior_vbx(gamma)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.prior_vbx, gamma) for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0], np.full(3, 1. / 3))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.prior_vbx(gamma)

    def test_vbx_posterior_bitwise_numpy_broadcasting_and_cancellation(self):
        rng = np.random.default_rng(926)
        with self.context() as context:
            for rows, columns in ((1, 1), (19, 3), (129, 17), (4096, 1), (1, 1024)):
                values = rng.normal(size=(rows, columns))
                for operation in ("log_prior_addition", "posterior_centering"):
                    vector = rng.normal(size=columns if operation == "log_prior_addition" else rows)
                    actual, record = context.posterior_vbx(values, vector, operation)
                    expected = values + vector if operation == "log_prior_addition" else values - vector[:, None]
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(record["gpu_kernels"], ["double_vbx_posterior"])
                    self.assertGreater(record["computation_gpu_seconds"], 0)
            values = np.array([[0., -0., 2. ** 200, 2. ** -200], [-0., 0., -(2. ** 200), -(2. ** -200)]])
            vector = np.array([-0., 0., -(2. ** 200), -(2. ** -200)])
            actual, _ = context.posterior_vbx(values, vector)
            np.testing.assert_array_equal(actual.view(np.uint64), (values + vector).view(np.uint64))
            for vector in (np.array([0., -0.]), np.array([2. ** 200, -(2. ** 200)])):
                actual, _ = context.posterior_vbx(values, vector, "posterior_centering")
                np.testing.assert_array_equal(actual.view(np.uint64), (values - vector[:, None]).view(np.uint64))

    def test_vbx_posterior_validation_lifetime_threads_and_capability(self):
        values, vector = np.ones((3, 8)), np.ones(8)
        values.flags.writeable = vector.flags.writeable = False
        with self.context() as context:
            for invalid in (np.nan, np.inf, 2. ** 201, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                    context.posterior_vbx(np.full_like(values, invalid), vector)
                with self.assertRaisesRegex(ValueError, "right_domain_invalid"):
                    context.posterior_vbx(values, np.full_like(vector, invalid))
            for left, right in ((values.astype(np.float32), vector), (values[:, ::2], vector), (values, vector.astype(np.float32)), (values, vector[:-1]), (np.ones((4097, 1)), np.ones(1))):
                with self.assertRaises(ValueError):
                    context.posterior_vbx(left, right)
            with self.assertRaises(ValueError):
                context.posterior_vbx(values, vector, "invalid")
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.posterior_vbx(values, vector)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.posterior_vbx, values, vector) for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0], values + vector)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.posterior_vbx(values, vector)

    def test_vbx_signed_likelihood_and_elbo_bitwise_numpy_layouts(self):
        rng = np.random.default_rng(928)
        with self.context() as context:
            for count in (1, 7, 8, 127, 128, 129, 1024, 4096):
                values = rng.normal(size=count)
                actual, record = context.likelihood_sum_vbx(values)
                np.testing.assert_array_equal(actual.view(np.uint64), values.sum().view(np.uint64))
                self.assertEqual(record["gpu_kernels"], ["double_vbx_likelihood"])
                self.assertGreater(record["computation_gpu_seconds"], 0)
            for values in (np.array([-0.]), np.full(129, -0.), np.array([2. ** 200, -(2. ** 200), 2. ** -200, -(2. ** -200)])):
                actual, _ = context.likelihood_sum_vbx(values)
                self.assertEqual(actual.view(np.uint64), values.sum().view(np.uint64))
            for rows, columns in ((1, 1), (3, 8), (17, 129), (129, 17), (65, 129), (257, 129), (128, 128), (1024, 1024)):
                inv_l, alpha = rng.uniform(.1, 1., (rows, columns)), rng.normal(size=(rows, columns))
                for order in ("C", "F"):
                    source, model = np.array(inv_l, order=order), np.array(alpha, order=order)
                    log = np.log(source)
                    expected = np.float64(-32.) + .8 * .5 * np.sum(log - source - model**2 + 1)
                    actual, record = context.elbo_vbx(np.ascontiguousarray(log), np.ascontiguousarray(source), np.ascontiguousarray(model), np.float64(-32.), .8, order == "F")
                    self.assertEqual(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(record["reduction_layout"], order)
                    self.assertEqual(record["logical_result_readback_bytes"], 8)
            inv_l = np.array([[2. ** -200, 1., 2. ** 200]])
            alpha = np.array([[2. ** -200, -0., -(2. ** 200)]])
            for fb in (2. ** -20, 2. ** 20):
                actual, _ = context.elbo_vbx(np.log(inv_l), inv_l, alpha, 0., fb)
                expected = np.float64(0.) + fb * .5 * np.sum(np.log(inv_l) - inv_l - alpha**2 + 1)
                self.assertEqual(actual.view(np.uint64), expected.view(np.uint64))

    def test_vbx_elbo_validation_lifetime_threads_and_capability(self):
        inv_l, alpha = np.ones((3, 8)), np.zeros((3, 8))
        log = np.log(inv_l)
        log.flags.writeable = inv_l.flags.writeable = alpha.flags.writeable = False
        with self.context() as context:
            for invalid in (np.nan, np.inf, 2. ** 201, 2. ** -201):
                with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                    context.likelihood_sum_vbx(np.full(8, invalid))
            for invalid in (np.zeros_like(inv_l), -inv_l):
                with self.assertRaisesRegex(ValueError, "input_domain_invalid"):
                    context.elbo_vbx(log, invalid, alpha, -1., .8)
            for invalid in (0., -1., np.inf, 2. ** -21, 2. ** 21):
                with self.assertRaisesRegex(ValueError, "scalar_domain_invalid"):
                    context.elbo_vbx(log, inv_l, alpha, -1., invalid)
            with self.assertRaises(ValueError):
                context.elbo_vbx(log, inv_l, alpha[:, :-1], -1., .8)
            with self.assertRaises(ValueError):
                context.likelihood_sum_vbx(np.ones(4097))
            with patch("tools.baseline.native_double.np.getbufsize", return_value=4096), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                context.likelihood_sum_vbx(np.ones(8))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.likelihood_sum_vbx(np.ones(8))
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.elbo_vbx, log, inv_l, alpha, -1., .8) for _ in range(4)]
                for future in futures:
                    self.assertEqual(future.result()[0], -1.)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.elbo_vbx(log, inv_l, alpha, -1., .8)

    def test_vbx_transcendentals_bounded_accuracy_numpy_and_decimal(self):
        from decimal import Decimal, localcontext
        rng = np.random.default_rng(930)
        log_values = np.concatenate((np.exp(rng.uniform(-130., 130., 20000)), np.ldexp(np.ones(401), np.arange(-200, 201)), np.array([np.nextafter(1., 0.), 1., np.nextafter(1., 2.)])))
        exp_values = np.concatenate((rng.uniform(-60., 0., 20000), np.array([-60., -0., 0., -(2. ** -200)]), -np.arange(1, 80) * np.log(2)))
        with self.context() as context:
            for operation, values in (("log", log_values), ("exp", exp_values)):
                actual, record = context.transcendental_vbx(values, operation)
                expected = getattr(np, operation)(values)
                a, b = actual.view(np.uint64), expected.view(np.uint64)
                self.assertLessEqual(int(np.max(np.maximum(a, b) - np.minimum(a, b))), 1)
                self.assertFalse(record["numpy_bitwise_certified"])
                self.assertGreater(record["computation_gpu_seconds"], 0)
                with localcontext() as precision:
                    precision.prec = 90
                    for index in np.linspace(0, len(values) - 1, 80, dtype=int):
                        value = Decimal.from_float(float(values[index]))
                        reference = np.float64(float(value.ln() if operation == "log" else value.exp()))
                        self.assertLessEqual(abs(int(actual[index].view(np.uint64)) - int(reference.view(np.uint64))), 1)
            result, _ = context.transcendental_vbx(np.array([1.]), "log")
            self.assertEqual(result[0].view(np.uint64), np.uint64(0))
            result, _ = context.transcendental_vbx(np.array([-0., 0.]), "exp")
            np.testing.assert_array_equal(result, np.ones(2))
            result, _ = context.transcendental_vbx(np.zeros(65536), "exp")
            np.testing.assert_array_equal(result, np.ones(65536))

    def test_vbx_transcendental_domain_lifetime_threads_and_capability(self):
        values = np.ones((3, 8))
        values.flags.writeable = False
        with self.context() as context:
            for operation, invalids in (("log", (0., -1., np.nan, np.inf, 2. ** -201, 2. ** 201)), ("exp", (-61., 1., np.nan, np.inf, -(2. ** -201)))):
                for invalid in invalids:
                    with self.assertRaisesRegex(ValueError, "domain_invalid"):
                        context.transcendental_vbx(np.full_like(values, invalid), operation)
            for invalid in (values.astype(np.float32), values[:, ::2], values.reshape(1, 3, 8), np.ones(65537), np.empty(0)):
                with self.assertRaises(ValueError):
                    context.transcendental_vbx(invalid, "log")
            with self.assertRaises(ValueError):
                context.transcendental_vbx(values, "invalid")
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.transcendental_vbx(values, "log")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.transcendental_vbx, values, "log") for _ in range(4)]
                for future in futures:
                    np.testing.assert_array_equal(future.result()[0], np.zeros_like(values))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.transcendental_vbx(values, "log")

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

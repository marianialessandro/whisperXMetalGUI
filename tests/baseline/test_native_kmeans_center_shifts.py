from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_common import _euclidean_dense_dense_wrapper
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
import sklearn.cluster._k_means_lloyd as cython_lloyd

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd_norms import LloydNormScope


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansCenterShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.library)
        cls.double = Path(cls.directory.name) / "gpu.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def gpu(self):
        return NativeDoubleNormalization(self.double, self.double_identity["library_sha256"])

    def test_every_feature_count_group_tail_and_subnormal_matches_cython_bits(self):
        rng = np.random.default_rng(2601)
        with self.context() as native:
            for features in range(1, 1025):
                a = np.ldexp(rng.uniform(-1., 1., (5, features)).astype(np.float32), rng.integers(-149, 33, (5, features))).astype(np.float32)
                b = np.ldexp(rng.uniform(-1., 1., (5, features)).astype(np.float32), rng.integers(-149, 33, (5, features))).astype(np.float32)
                a[0] = -0.; b[0] = 0.
                a[1] = np.nextafter(np.float32(0), np.float32(1)); b[1] = 0.
                a[2] = np.nextafter(np.float32(1), np.float32(2)); b[2] = 1.
                expected = np.array([_euclidean_dense_dense_wrapper["float"](left, right, False) for left, right in zip(a, b)], np.float32)
                actual, record = native.lloyd_center_shifts(a, b)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(record["execution_device"], "cpu")
                self.assertEqual(record["native_blas_calls"], 0)
                self.assertEqual(record["gpu_kernels"], [])
                self.assertEqual(record["command_buffers_per_run"], 0)
                self.assertIsNone(record["computation_gpu_seconds"])
                self.assertFalse(record["python_numeric_center_shift"])
                self.assertFalse(record["lloyd_iteration_replaced"])
                self.assertFalse(np.signbit(actual[0]))

    def test_maximum_budget_unaligned_readonly_threads_and_large_values(self):
        rng = np.random.default_rng(2603)
        with self.context() as native:
            a = np.ndarray((256, 1024), np.float32, buffer=bytearray(256 * 1024 * 4 + 1), offset=1)
            a[:] = rng.normal(size=a.shape)
            b = rng.normal(size=a.shape).astype(np.float32)
            a[0] = np.float32(2. ** 32); b[0] = np.float32(-2. ** 32)
            expected = np.array([_euclidean_dense_dense_wrapper["float"](left, right, False) for left, right in zip(a, b)], np.float32)
            a.flags.writeable = b.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                for actual, record in executor.map(lambda _: native.lloyd_center_shifts(a, b), range(4)):
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(record["host_result_bytes"], 256 * 4)

    def test_domains_shapes_abi_and_closed_context(self):
        a = np.ones((2, 7), np.float32)
        with self.context() as native:
            for value in (np.nan, np.inf, -np.inf, 2. ** 33):
                bad = a.copy(); bad[-1, -1] = value
                for left, right in ((bad, a), (a, bad)):
                    with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                        native.lloyd_center_shifts(left, right)
            for shape in ((0, 7), (257, 7), (2, 0), (2, 1025), (3, 7)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    native.lloyd_center_shifts(a, np.ones(shape, np.float32))
            for bad in (a.astype(np.float64), a[:, ::2], a[0]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    native.lloyd_center_shifts(bad, a)
            with patch.object(native, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                native.lloyd_center_shifts(a, a)
        with self.assertRaisesRegex(ValueError, "closed"):
            native.lloyd_center_shifts(a, a)

    def test_native_regular_iteration_never_invokes_original_dense_or_shift(self):
        x = np.random.default_rng(2605).normal(size=(42, 17)).astype(np.float32)
        centers, weights = x[:3].copy(), np.linspace(.5, 2., len(x), dtype=np.float32)
        reference = (np.empty_like(centers), np.empty(3, np.float32), np.empty(42, np.int32), np.empty(3, np.float32))
        lloyd_iter_chunked_dense(x, weights, centers, *reference, 12)
        outputs = tuple(np.full_like(array, -1) for array in reference)
        with self.context() as native, self.gpu() as gpu:
            scope = LloydNormScope(gpu, [], native, [], [], records := [])
            with scope, patch.object(scope, "original_iteration", side_effect=AssertionError("Cython iteration invoked")), patch.object(scope, "original_shift", side_effect=AssertionError("Cython shift invoked")):
                scope.iteration(x, weights, centers, *outputs, 12)
            for actual, expected in zip(outputs, reference):
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(len(records), 1)
            self.assertIsNone(scope.centers)

    def test_composed_fits_preserve_attributes_inputs_and_rng(self):
        x = np.random.default_rng(2607).normal(size=(42, 17)).astype(np.float32)
        with self.context() as native, self.gpu() as gpu:
            for weighted in (False, True):
                for clusters in (1, 2, 4, 8):
                    for maximum in (1, 300):
                        weights = np.linspace(.5, 2., len(x), dtype=np.float32) if weighted else None
                        left_rng, right_rng = np.random.RandomState(17), np.random.RandomState(17)
                        options = dict(n_clusters=clusters, n_init=3, max_iter=maximum, copy_x=False, tol=0. if maximum == 1 else 1e-4)
                        before, after = x.copy(), x.copy()
                        reference = KMeans(**options, random_state=left_rng).fit(before, sample_weight=weights)
                        with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=[], centroid_records=(centroid_records := []), center_shift_records=(records := []), lloyd_shift_records=[], lloyd_label_records=[]):
                            actual = KMeans(**options, random_state=right_rng).fit(after, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), reference.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, reference.labels_)
                        np.testing.assert_array_equal(before.view(np.uint32), after.view(np.uint32))
                        np.testing.assert_array_equal(left_rng.rand(32), right_rng.rand(32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (reference.inertia_, reference.n_iter_, reference._tol))
                        self.assertTrue(records)
                        self.assertEqual(len(records), len(centroid_records))
                        self.assertTrue(all(row["center_shift_calls"] == 1 and row["center_shift_rows"] == clusters and row["center_shift_backend"] == "cpp_cython_float32_four_term_fma_sqrt" for row in centroid_records))

    def test_configuration_fault_keeps_output_buffers_and_restores_hooks(self):
        original_fit, original_norm = KMeans.fit, cython_lloyd.row_norms
        x = np.random.default_rng(2609).normal(size=(42, 7)).astype(np.float32)
        with self.context() as native, self.gpu() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(gpu, [], center_shift_records=[]):
                    pass
            with patch.object(native, "library", SimpleNamespace(wxkmeans_lloyd_assignment=object(), wxkmeans_lloyd_centroids=object())), self.assertRaisesRegex(ValueError, "center_shifts_abi_unavailable"):
                LloydNormScope(gpu, [], native, [], [], [])
            scope = LloydNormScope(gpu, [], native, [], [], records := [])
            outputs = (np.full((3, 7), 91, np.float32), np.full(3, 92, np.float32), np.full(42, -1, np.int32), np.full(3, 93, np.float32))
            saved = [array.copy() for array in outputs]
            with self.assertRaisesRegex(ValueError, "injected_shift_failure"):
                with scope, patch.object(native, "lloyd_center_shifts", side_effect=ValueError("injected_shift_failure")):
                    scope.iteration(x, np.ones(42, np.float32), x[:3].copy(), *outputs, 12)
            for actual, expected in zip(outputs, saved):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(records, [])
            self.assertIsNone(scope.centers)
            with self.assertRaisesRegex(ValueError, "injected_shift_failure"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=[], centroid_records=[], center_shift_records=[]), patch.object(native, "lloyd_center_shifts", side_effect=ValueError("injected_shift_failure")):
                    KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(cython_lloyd.row_norms, original_norm)


if __name__ == "__main__":
    unittest.main()

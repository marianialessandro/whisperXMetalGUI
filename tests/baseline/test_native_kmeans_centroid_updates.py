from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
import sklearn.cluster._k_means_lloyd as cython_lloyd

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd_norms import LloydNormScope


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansCentroidUpdateTests(unittest.TestCase):
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

    def test_raw_fma_centroids_and_cluster_weights_match_original_cython(self):
        rng = np.random.default_rng(2401)
        with self.context() as native:
            for rows in (1, 7, 42, 129, 256):
                for features in (1, 3, 4, 7, 16, 17, 129, 256, 257, 1024):
                    x = rng.normal(size=(rows, features)).astype(np.float32)
                    for clusters in (1, 2, 4, 8):
                        if clusters > rows:
                            continue
                        for weighted in (False, True):
                            weights = np.linspace(.5, 2., rows, dtype=np.float32) if weighted else np.ones(rows, np.float32)
                            centers, totals, labels, shifts = np.empty((clusters, features), np.float32), np.empty(clusters, np.float32), np.empty(rows, np.int32), np.empty(clusters, np.float32)
                            lloyd_iter_chunked_dense(x, weights, x[:clusters].copy(), centers, totals, labels, shifts, 12)
                            actual, actual_totals, record = native.lloyd_centroids(x, weights, labels, clusters)
                            np.testing.assert_array_equal(actual.view(np.uint32), centers.view(np.uint32))
                            np.testing.assert_array_equal(actual_totals.view(np.uint32), totals.view(np.uint32))
                            self.assertEqual(record["execution_device"], "cpu")
                            self.assertEqual(record["native_blas_calls"], 0)
                            self.assertEqual(record["command_buffers_per_run"], 0)
                            self.assertIsNone(record["computation_gpu_seconds"])
                            self.assertFalse(record["empty_cluster_relocation_replaced"])

    def test_raw_normal_bounds_zero_unaligned_readonly_and_threads(self):
        with self.context() as native:
            for magnitude in (0., 2. ** -60, 2. ** 32):
                for weight in (2. ** -32, 1., 2. ** 32):
                    x, weights, labels = np.full((256, 17), -magnitude, np.float32), np.full(256, weight, np.float32), np.zeros(256, np.int32)
                    actual, totals, unused = native.lloyd_centroids(x, weights, labels, 1)
                    np.testing.assert_array_equal(actual, np.full((1, 17), -magnitude, np.float32))
                    self.assertEqual(totals[0], np.float32(weight * 256))
            x = np.ndarray((42, 7), np.float32, buffer=bytearray(42 * 7 * 4 + 1), offset=1)
            x[:] = np.random.default_rng(2403).normal(size=x.shape)
            weights, labels = np.ones(42, np.float32), np.arange(42, dtype=np.int32) % 3
            x.flags.writeable = weights.flags.writeable = labels.flags.writeable = False
            expected = native.lloyd_centroids(x, weights, labels, 3)[:2]
            with ThreadPoolExecutor(max_workers=2) as executor:
                for centers, totals, unused in executor.map(lambda _: native.lloyd_centroids(x, weights, labels, 3), range(4)):
                    np.testing.assert_array_equal(centers.view(np.uint32), expected[0].view(np.uint32))
                    np.testing.assert_array_equal(totals.view(np.uint32), expected[1].view(np.uint32))

    def test_domains_empty_clusters_closed_and_missing_abi(self):
        x, weights, labels = np.ones((7, 3), np.float32), np.ones(7, np.float32), np.zeros(7, np.int32)
        with self.context() as native:
            for value in (np.nan, np.inf, 2. ** -61, 2. ** 33):
                bad = x.copy(); bad[-1, -1] = value
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    native.lloyd_centroids(bad, weights, labels, 1)
            for value in (-1., np.nan, np.inf, 2. ** -33, 2. ** 33):
                bad = weights.copy(); bad[-1] = value
                with self.assertRaisesRegex(ValueError, "weight_domain_invalid"):
                    native.lloyd_centroids(x, bad, labels, 1)
            for value in (-1, 1):
                bad = labels.copy(); bad[-1] = value
                with self.assertRaisesRegex(ValueError, "label_domain_invalid"):
                    native.lloyd_centroids(x, weights, bad, 1)
            for clusters in (0, 257, True, np.int64(1)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    native.lloyd_centroids(x, weights, labels, clusters)
            for bad in (x.astype(np.float64), x[:, ::2], x[0]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    native.lloyd_centroids(bad, weights, labels, 1)
            with self.assertRaisesRegex(ValueError, "empty_cluster_not_supported"):
                native.lloyd_centroids(x, weights, labels, 2)
            with self.assertRaisesRegex(ValueError, "empty_cluster_not_supported"):
                native.lloyd_centroids(x, np.zeros(7, np.float32), labels, 1)
            with patch.object(native, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                native.lloyd_centroids(x, weights, labels, 1)
        with self.assertRaisesRegex(ValueError, "closed"):
            native.lloyd_centroids(x, weights, labels, 1)

    def test_composed_fits_preserve_centers_labels_inertia_rng_and_input(self):
        x = np.random.default_rng(2405).normal(size=(42, 17)).astype(np.float32)
        with self.context() as native, self.gpu() as gpu:
            for weighted in (False, True):
                for clusters in (1, 2, 4, 8):
                    for maximum in (1, 300):
                        weights = np.linspace(.5, 2., len(x), dtype=np.float32) if weighted else None
                        left_rng, right_rng = np.random.RandomState(17), np.random.RandomState(17)
                        options = dict(n_clusters=clusters, n_init=3, max_iter=maximum, tol=0. if maximum == 1 else 1e-4, copy_x=False)
                        before, after = x.copy(), x.copy()
                        expected = KMeans(**options, random_state=left_rng).fit(before, sample_weight=weights)
                        with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=[], centroid_records=(records := []), lloyd_shift_records=[], lloyd_label_records=[]):
                            actual = KMeans(**options, random_state=right_rng).fit(after, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(after.view(np.uint32), before.view(np.uint32))
                        np.testing.assert_array_equal(left_rng.rand(32), right_rng.rand(32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertTrue(records)
                        self.assertTrue(all(row["center_shift_backend"] == "original_cython_euclidean_dense_dense_float" and row["native_blas_calls"] == 1 for row in records))

    def test_regular_iteration_skips_original_and_commits_outputs_only_on_success(self):
        x = np.random.default_rng(2407).normal(size=(42, 7)).astype(np.float32)
        centers, weights = x[:3].copy(), np.ones(42, np.float32)
        expected_centers, expected_totals, expected_labels, expected_shifts = np.empty_like(centers), np.empty(3, np.float32), np.empty(42, np.int32), np.empty(3, np.float32)
        lloyd_iter_chunked_dense(x, weights, centers, expected_centers, expected_totals, expected_labels, expected_shifts, 12)
        new, totals, labels, shifts = np.full_like(centers, 91), np.full(3, 92, np.float32), np.full(42, -1, np.int32), np.full(3, 93, np.float32)
        with self.context() as native, self.gpu() as gpu:
            scope = LloydNormScope(gpu, [], native, [], records := [])
            with scope, patch.object(scope, "original_iteration", side_effect=AssertionError("Original dense iteration invoked")):
                scope.iteration(x, weights, centers, new, totals, labels, shifts, 12)
                for actual, expected in ((new, expected_centers), (totals, expected_totals), (labels, expected_labels), (shifts, expected_shifts)):
                    np.testing.assert_array_equal(actual, expected)
                new.fill(91); totals.fill(92); labels.fill(-1); shifts.fill(93)
                with self.assertRaisesRegex(ValueError, "empty_cluster_not_supported"):
                    scope.iteration(np.zeros_like(x), weights, np.zeros_like(centers), new, totals, labels, shifts, 12)
                np.testing.assert_array_equal(new, np.full_like(new, 91))
                np.testing.assert_array_equal(totals, np.full(3, 92, np.float32))
                np.testing.assert_array_equal(labels, np.full(42, -1, np.int32))
                np.testing.assert_array_equal(shifts, np.full(3, 93, np.float32))
            self.assertEqual(len(records), 1)
            self.assertIsNone(scope.centers)

    def test_configuration_bypass_fault_and_restore(self):
        original_fit, original_norms = KMeans.fit, cython_lloyd.row_norms
        x = np.random.default_rng(2409).normal(size=(42, 7)).astype(np.float32)
        with self.context() as native, self.gpu() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(gpu, [], centroid_records=[]):
                    pass
            with patch.object(native, "library", SimpleNamespace(wxkmeans_lloyd_assignment=object())), self.assertRaisesRegex(ValueError, "centroids_abi_unavailable"):
                LloydNormScope(gpu, [], native, [], [])
            with self.assertRaisesRegex(ValueError, "injected_centroid_failure"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=[], centroid_records=[]), patch.object(native, "lloyd_centroids", side_effect=ValueError("injected_centroid_failure")):
                    KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(cython_lloyd.row_norms, original_norms)
            with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=[], centroid_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17).fit, x).result()
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                KMeans(n_clusters=3, n_init=1, random_state=17, algorithm="elkan").fit(x)
                self.assertEqual(records, [])
            self.assertIs(cython_lloyd.row_norms, original_norms)


if __name__ == "__main__":
    unittest.main()

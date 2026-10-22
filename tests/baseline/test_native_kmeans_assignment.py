from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.linalg.blas import sgemm
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
import sklearn.cluster._k_means_lloyd as cython_lloyd
from sklearn.utils._cython_blas import _gemm_memview, BLAS_Trans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd_norms import LloydNormScope


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Accelerate required")
class NativeKMeansAssignmentTests(unittest.TestCase):
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

    def test_lp64_matrix_bits_and_cython_labels_boundaries_and_ties(self):
        rng = np.random.default_rng(2201)
        with self.context() as native:
            for rows in (1, 2, 7, 42, 256):
                for clusters in (1, 2, 7, 256):
                    for features in (1, 2, 7, 16, 17, 256, 1024):
                        values = rng.normal(size=(rows, features)).astype(np.float32)
                        centers = rng.normal(size=(clusters, features)).astype(np.float32)
                        norms = np.einsum("ij,ij->i", centers, centers)
                        expected = np.tile(norms, (rows, 1))
                        if features > 1:
                            _gemm_memview["float"](BLAS_Trans.NoTrans, BLAS_Trans.Trans, -2., values, centers, 1., expected)
                        else:
                            expected = sgemm(-2., centers.T, values.T, beta=1., c=expected.T, trans_a=1).T.copy()
                        expected_labels = np.empty(rows, np.int32)
                        lloyd_iter_chunked_dense(values, np.ones(rows, np.float32), centers, None, None, expected_labels, None, 12, update_centers=False)
                        values.flags.writeable = centers.flags.writeable = norms.flags.writeable = False
                        labels, distances, record = native.lloyd_assignment(values, centers, norms)
                        np.testing.assert_array_equal(distances.view(np.uint32), expected.view(np.uint32))
                        np.testing.assert_array_equal(labels, expected_labels)
                        self.assertEqual(record["execution_device"], "cpu")
                        self.assertEqual(record["command_buffers_per_run"], 0)
                        self.assertEqual(record["gpu_kernels"], [])
                        self.assertIsNone(record["computation_gpu_seconds"])
                        self.assertFalse(record["centroid_updates_replaced"])
            labels, distances, unused = native.lloyd_assignment(np.array([[-0., 0.], [1., -1.]], np.float32), np.zeros((3, 2), np.float32), np.zeros(3, np.float32))
            np.testing.assert_array_equal(labels, [0, 0])
            self.assertTrue(np.isfinite(distances).all())

    def test_small_and_large_normal_operands_unaligned_readonly_and_threads(self):
        with self.context() as native:
            values = np.ndarray((42, 7), np.float32, buffer=bytearray(42 * 7 * 4 + 1), offset=1)
            values[:] = np.random.default_rng(2203).normal(size=values.shape).astype(np.float32)
            centers = values[:3].copy()
            norms = np.einsum("ij,ij->i", centers, centers)
            values.flags.writeable = False
            reference = native.lloyd_assignment(values, centers, norms)[:2]
            with ThreadPoolExecutor(max_workers=2) as executor:
                for labels, distances, unused in executor.map(lambda _: native.lloyd_assignment(values, centers, norms), range(4)):
                    np.testing.assert_array_equal(labels, reference[0])
                    np.testing.assert_array_equal(distances.view(np.uint32), reference[1].view(np.uint32))
            for magnitude in (2. ** -60, 2. ** -33, 2. ** 32):
                x = np.full((7, 17), magnitude, np.float32)
                c = np.full((3, 17), -magnitude, np.float32)
                n = np.einsum("ij,ij->i", c, c)
                expected = sgemm(-2., c.T, x.T, beta=1., c=np.tile(n, (7, 1)).T, trans_a=1).T.copy()
                labels, distances, unused = native.lloyd_assignment(x, c, n)
                np.testing.assert_array_equal(distances.view(np.uint32), expected.view(np.uint32))
                np.testing.assert_array_equal(labels, np.zeros(7, np.int32))

    def test_raw_domains_shapes_abi_and_closed(self):
        x, centers, norms = np.ones((7, 3), np.float32), np.ones((2, 3), np.float32), np.full(2, 3, np.float32)
        with self.context() as native:
            for operand in (x.astype(np.float64), x[:, ::2], x[0]):
                with self.assertRaises(ValueError):
                    native.lloyd_assignment(operand, centers, norms)
            for operand in (np.ones((0, 3), np.float32), np.ones((257, 3), np.float32), np.ones((7, 4), np.float32)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    native.lloyd_assignment(operand, centers, norms)
            for value in (np.nan, np.inf, 2. ** -61, 2. ** 33):
                bad = x.copy(); bad[-1, -1] = value
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    native.lloyd_assignment(bad, centers, norms)
            for value in (-1., np.nan, np.inf, 2. ** -121, 2. ** 75):
                with self.assertRaisesRegex(ValueError, "norm_domain_invalid"):
                    native.lloyd_assignment(x, centers, np.full(2, value, np.float32))
            with patch.object(native, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                native.lloyd_assignment(x, centers, norms)
        with self.assertRaisesRegex(ValueError, "closed"):
            native.lloyd_assignment(x, centers, norms)

    def test_fit_final_pass_attributes_rng_and_composition(self):
        values = np.random.default_rng(2205).normal(size=(42, 17)).astype(np.float32)
        with self.context() as native, self.gpu() as gpu:
            for weighted in (False, True):
                for tolerance, maximum in ((0., 1), (1e6, 3)):
                    weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                    options = dict(n_clusters=3, n_init=3, tol=tolerance, max_iter=maximum, copy_x=False)
                    left_rng, right_rng = np.random.RandomState(17), np.random.RandomState(17)
                    before, after = values.copy(), values.copy()
                    reference = KMeans(**options, random_state=left_rng).fit(before, sample_weight=weights)
                    with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], lloyd_shift_records=[], lloyd_label_records=[], assignment_records=(records := [])):
                        actual = KMeans(**options, random_state=right_rng).fit(after, sample_weight=weights)
                    self.assertEqual(len(records), 3)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), reference.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, reference.labels_)
                    np.testing.assert_array_equal(after.view(np.uint32), before.view(np.uint32))
                    np.testing.assert_array_equal(left_rng.rand(32), right_rng.rand(32))
                    self.assertEqual(actual.inertia_, reference.inertia_)
                    self.assertEqual(actual.n_iter_, reference.n_iter_)
                    self.assertIs(type(actual._tol), type(reference._tol))
                    self.assertEqual(actual._tol, reference._tol)
                    self.assertTrue(all(row["final_assignment_replaced"] and not row["centroid_updates_replaced"] for row in records))

    def test_final_pass_skips_cython_and_leaves_all_centroid_buffers_untouched(self):
        values = np.random.default_rng(2206).normal(size=(42, 7)).astype(np.float32)
        centers, weights = values[:3].copy(), np.ones(42, np.float32)
        expected_labels = np.empty(42, np.int32)
        lloyd_iter_chunked_dense(values, weights, centers, None, None, expected_labels, None, 12, update_centers=False)
        new_centers, cluster_weights, shifts = np.full_like(centers, 91), np.full(3, 92, np.float32), np.full(3, 93, np.float32)
        labels = np.full(42, -1, np.int32)
        with self.context() as native, self.gpu() as gpu:
            scope = LloydNormScope(gpu, [], native, records := [])
            with scope, patch.object(scope, "original_iteration", side_effect=AssertionError("Cython final pass invoked")):
                scope.iteration(values, weights, centers, new_centers, cluster_weights, labels, shifts, 12, update_centers=False)
                np.testing.assert_array_equal(labels, expected_labels)
                np.testing.assert_array_equal(new_centers, np.full_like(centers, 91))
                np.testing.assert_array_equal(cluster_weights, np.full(3, 92, np.float32))
                np.testing.assert_array_equal(shifts, np.full(3, 93, np.float32))
                scope.iteration(values, weights, centers, None, None, labels, None, 12, False)
                np.testing.assert_array_equal(labels, expected_labels)
            self.assertEqual(len(records), 2)
            self.assertIsNone(scope.centers)
            import scipy.linalg.cython_blas as scipy_blas
            wrong = Path(self.directory.name) / "wrong-blas.so"
            wrong.write_bytes(b"unsupported")
            with patch.object(scipy_blas, "__file__", str(wrong)), self.assertRaisesRegex(ValueError, "blas_reference_not_supported"):
                LloydNormScope(gpu, [], native, [])

    def test_bypass_configuration_fault_restore_and_single_chunk_limit(self):
        original_fit, original_norm = KMeans.fit, cython_lloyd.row_norms
        values = np.random.default_rng(2207).normal(size=(42, 7)).astype(np.float32)
        with self.context() as native, self.gpu() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(gpu, [], assignment_records=[]):
                    pass
            with patch.object(native, "library", SimpleNamespace(wxkmeans_weighted_potentials=object())), self.assertRaisesRegex(ValueError, "assignment_abi_unavailable"):
                LloydNormScope(gpu, [], native, [])
            with self.assertRaisesRegex(ValueError, "injected_assignment_failure"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=[]), patch.object(native, "lloyd_assignment", side_effect=ValueError("injected_assignment_failure")):
                    KMeans(n_clusters=3, n_init=1, random_state=17, max_iter=1, tol=0).fit(values)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(cython_lloyd.row_norms, original_norm)
            with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=native, potential_records=[], assignment_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17, max_iter=1).fit, values).result()
                KMeans(n_clusters=3, n_init=1, random_state=17, max_iter=1, algorithm="elkan").fit(values)
                KMeans(n_clusters=3, n_init=1, random_state=17, max_iter=1).fit(sparse.csr_matrix(values))
                self.assertEqual(records, [])
                with self.assertRaisesRegex(ValueError, "single_chunk_required"):
                    KMeans(n_clusters=3, n_init=1, random_state=17, max_iter=1).fit(np.ones((257, 7), np.float32))
            self.assertIs(cython_lloyd.row_norms, original_norm)


if __name__ == "__main__":
    unittest.main()

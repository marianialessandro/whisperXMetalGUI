from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans
import sklearn.metrics.pairwise as pairwise

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library, distance_products
from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline, compile_upcast_distances


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Accelerate required")
class NativeKMeansDistanceProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "potential.dylib"
        cls.identity = build_library(cls.library)
        cls.double = Path(cls.directory.name) / "double.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def test_raw_numpy_dispatch_all_branches_and_maximum(self):
        rng = np.random.default_rng(1021)
        with self.context() as context:
            for rows, columns, features, dispatch in ((1, 1, 1, "ddot"), (1, 1, 1024, "ddot"), (1, 42, 256, "dgemv_colmajor_trans"), (42, 1, 256, "dgemv_colmajor_trans"), (3, 7, 17, "dgemm_rowmajor_nt"), (2, 42, 256, "dgemm_rowmajor_nt"), (1, 7, 1, "scalar_outer"), (7, 1, 1, "scalar_outer"), (7, 3, 1, "scalar_outer"), (256, 256, 1024, "dgemm_rowmajor_nt")):
                left, right = rng.normal(size=(rows, features)), rng.normal(size=(columns, features))
                expected = left @ right.T
                actual, record = context.distance_products(left, right)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(record["numpy_dispatch"], dispatch)
                self.assertEqual(record["execution_device"], "cpu")
                self.assertEqual(record["gpu_kernels"], [])
                self.assertEqual(record["command_buffers_per_run"], 0)
                self.assertIsNone(record["computation_gpu_seconds"])
                self.assertEqual(record["host_input_snapshot_bytes"], left.nbytes + right.nbytes)
                self.assertEqual(record["host_result_bytes"], actual.nbytes)
            for count in (2, 7, 42, 256):
                left = rng.normal(size=(count, 17))
                for right in (left, left.view(), left.copy()):
                    actual, record = context.distance_products(left, right)
                    np.testing.assert_array_equal(actual.view(np.uint64), (left @ right.T).view(np.uint64))
                    self.assertEqual(record["numpy_dispatch"], "dsyrk_rowmajor_upper" if left.ctypes.data == right.ctypes.data else "dgemm_rowmajor_nt")
            for value in (0., -0., 2. ** -200, -(2. ** -200), 2. ** 200, -(2. ** 200)):
                left, right = np.full((3, 1024), value), np.full((7, 1024), value)
                actual, _ = context.distance_products(left, right)
                np.testing.assert_array_equal(actual.view(np.uint64), (left @ right.T).view(np.uint64))
            left = np.array([[1., -1., 2. ** -200, -(2. ** -200)]])
            right = np.array([[1., 1., 1., 1.]])
            actual, _ = context.distance_products(left, right)
            np.testing.assert_array_equal(actual.view(np.uint64), (left @ right.T).view(np.uint64))

    def test_validation_domains_abi_closed_readonly_unaligned_and_thread(self):
        left, right = np.ones((3, 7)), np.ones((5, 7))
        with self.context() as context:
            for a, b in ((left.astype(np.float32), right), (left, right.T), (left, np.asfortranarray(right)), (left[:, ::-1], right[:, ::-1]), (left[0], right)):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    context.distance_products(a, b)
            for rows, columns, features in ((0, 1, 1), (257, 1, 1), (1, 257, 1), (1, 1, 0), (1, 1, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.distance_products(np.zeros((rows, features)), np.zeros((columns, features)))
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                context.distance_products(left, np.zeros((5, 8)))
            for value in (np.nan, np.inf, -np.inf, 2. ** -201, 2. ** 201):
                with self.assertRaisesRegex(ValueError, "value_domain"):
                    context.distance_products(np.full(left.shape, value), right)
                with self.assertRaisesRegex(ValueError, "value_domain"):
                    context.distance_products(left, np.full(right.shape, value))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.distance_products(left, right)
            unaligned = np.ndarray(left.shape, dtype=np.float64, buffer=bytearray(left.nbytes + 1), offset=1)
            unaligned[:] = left; unaligned.flags.writeable = False
            right.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: context.distance_products(unaligned, right)[0], range(4)))
            for actual in outputs:
                np.testing.assert_array_equal(actual, np.full((3, 5), 7.))
            with self.assertRaisesRegex(ValueError, "transposed_right"):
                distance_products(context, [], left, right, True)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.distance_products(left, right)

    def test_compiler_replaces_blas_bridge_and_preserves_guarded_helper_outputs(self):
        rng = np.random.default_rng(1031)
        X, Y = rng.normal(size=(13, 17)).astype(np.float32), rng.normal(size=(19, 17)).astype(np.float32)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            pipeline = ResidentDistancePipeline(gpu, [], [], norm_records=[])
            candidate = compile_upcast_distances(pairwise._euclidean_distances, None, finish=pipeline.finish, clamp=pipeline.clamp, resident=True, resident_norms=True, product=lambda a, b, dense_output: distance_products(cpu, records, a, b, dense_output))
            helper = candidate.__globals__["_euclidean_distances_upcast"]
            for right in (X, X.copy(), Y):
                records = []
                expected = pairwise._euclidean_distances(X, right, squared=True)
                with patch.dict(helper.__globals__, {"safe_sparse_dot": lambda *args, **kwargs: self.fail("original BLAS bridge")}):
                    actual = candidate(X, right, squared=True)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["host_input_staging_copy_bytes"], 0)
            with patch("tools.baseline.native_kmeans_potential.SAFE_SPARSE_DOT_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                compile_upcast_distances(pairwise._euclidean_distances, object(), product=object())
            getsource = inspect.getsource
            changed = getsource(pairwise._euclidean_distances_upcast).replace("dense_output=True", "dense_output=False")
            with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is pairwise._euclidean_distances_upcast else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.UPCAST_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
                compile_upcast_distances(pairwise._euclidean_distances, object(), product=object())

    def test_fits_potential_only_and_all_components_rng_bypass_failure_restore(self):
        values = np.random.default_rng(1033).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        def scope(gpu, cpu, records, composed):
            options = dict(shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True) if composed else {}
            return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], upcast_norm_records=[], product_records=records, **options)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            for composed in (False, True):
                for clusters in (1, 2, 4):
                    weights = np.linspace(.5, 2., len(values), dtype=np.float32)
                    rng1, rng2 = np.random.RandomState(17), np.random.RandomState(17)
                    options = dict(n_clusters=clusters, n_init=2, copy_x=False)
                    expected = KMeans(**options, random_state=rng1).fit(values.copy(), sample_weight=weights)
                    with scope(gpu, cpu, records := [], composed):
                        actual = KMeans(**options, random_state=rng2).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        count = len(records); self.assertEqual(count, 2 * clusters)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options, random_state=17).fit(values.copy())).result()
                        self.assertEqual(len(records), count)
                    np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
            with scope(gpu, cpu, skipped := [], True):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope(gpu, cpu, [], True), patch.object(cpu, "distance_products", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(gpu, [], product_records=[]):
                    pass
        self.assertIs(KMeans.fit, original_fit); self.assertIs(kmeans._kmeans_plusplus, original_plus)

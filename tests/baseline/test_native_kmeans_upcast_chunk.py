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
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library, upcast_chunk
from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline, compile_upcast_distances


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansUpcastChunkTests(unittest.TestCase):
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

    def test_raw_finite_ieee_bits_all_exponents_boundaries_and_maximum(self):
        rng = np.random.default_rng(1039)
        bits = rng.integers(0, 2 ** 32, size=(256, 1024), dtype=np.uint32)
        bits[(bits & 0x7f800000) == 0x7f800000] = 0
        boundaries = [sign | (exponent << 23) | fraction for sign in (0, 0x80000000) for exponent in range(255) for fraction in (0, 1, 0x3fffff, 0x7ffffe, 0x7fffff)]
        bits.reshape(-1)[:len(boundaries)] = boundaries
        values = bits.view(np.float32)
        expected = values.astype(np.float64)
        with self.context() as context:
            for value in (values, values[:1, :1], values[:1], values[:3]):
                actual, record = context.upcast_chunk(value)
                np.testing.assert_array_equal(actual.view(np.uint64), value.astype(np.float64).view(np.uint64))
                self.assertEqual(actual.shape, value.shape)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertEqual(record["execution_device"], "cpu")
                self.assertEqual(record["native_blas_calls"], 0)
                self.assertEqual(record["command_buffers_per_run"], 0)
                self.assertEqual(record["gpu_kernels"], [])
                self.assertIsNone(record["computation_gpu_seconds"])
                self.assertTrue(record["signed_zero_preserved"])
                self.assertTrue(record["binary32_subnormals_preserved"])
                self.assertFalse(record["python_numeric_dtype_conversion"])
            unaligned = np.ndarray(values.shape, dtype=np.float32, buffer=bytearray(values.nbytes + 1), offset=1)
            unaligned[:] = values; unaligned.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: context.upcast_chunk(unaligned)[0], range(4)))
            for actual in outputs:
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    def test_raw_errors_closed_abi_and_role(self):
        with self.context() as context:
            for value in (np.ones((3, 7)), np.ones(7, np.float32), np.ones((3, 7), np.float32)[:, ::-1], np.asfortranarray(np.ones((3, 7), np.float32))):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    context.upcast_chunk(value)
            for rows, features in ((0, 1), (257, 1), (1, 0), (1, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.upcast_chunk(np.zeros((rows, features), np.float32))
            for value in (np.nan, np.inf, -np.inf):
                with self.assertRaisesRegex(ValueError, "finite_input_required"):
                    context.upcast_chunk(np.full((3, 7), value, np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.upcast_chunk(np.zeros((1, 1), np.float32))
            with self.assertRaisesRegex(ValueError, "role_invalid"):
                upcast_chunk(context, [], np.zeros((1, 1), np.float32), "other")
        with self.assertRaisesRegex(ValueError, "closed"):
            context.upcast_chunk(np.zeros((1, 1), np.float32))

    def test_helper_displaces_numpy_upcast_and_source_expression_guard(self):
        rng = np.random.default_rng(1049)
        X, Y = rng.normal(size=(13, 17)).astype(np.float32), rng.normal(size=(19, 17)).astype(np.float32)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            pipeline = ResidentDistancePipeline(gpu, [], [], norm_records=[])
            candidate = compile_upcast_distances(pairwise._euclidean_distances, None, finish=pipeline.finish, clamp=pipeline.clamp, resident=True, resident_norms=True, upcast_chunk=lambda value, target_dtype, role: upcast_chunk(cpu, records, value, role, target_dtype))
            xp = pairwise._euclidean_distances_upcast.__globals__["get_namespace_and_device"](X, Y)[0]
            original_cast = xp.astype
            def reject_upcast(value, dtype, *args, **kwargs):
                if dtype == xp.float64 and value.dtype == np.float32:
                    self.fail("NumPy chunk upcast used")
                return original_cast(value, dtype, *args, **kwargs)
            for right in (X, X.copy(), Y):
                records = []
                expected = pairwise._euclidean_distances(X, right, squared=True)
                with patch.object(xp, "astype", side_effect=reject_upcast):
                    actual = candidate(X, right, squared=True)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual([row["operand_role"] for row in records], ["X_chunk", "Y_chunk"])
            helper = candidate.__globals__["_euclidean_distances_upcast"]
            with patch.dict(helper.__globals__, {"_max_precision_float_dtype": lambda **kwargs: np.float32}), self.assertRaisesRegex(ValueError, "target_dtype_not_supported"):
                candidate(X, Y, squared=True)
            getsource = inspect.getsource
            for expression in ("X[x_slice, :]", "Y[y_slice, :]"):
                changed = getsource(pairwise._euclidean_distances_upcast).replace(expression, expression + ".copy()")
                with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is pairwise._euclidean_distances_upcast else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.UPCAST_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "upcast_chunk_expression"):
                    compile_upcast_distances(pairwise._euclidean_distances, object(), upcast_chunk=object())

    def test_fits_composition_rng_thread_sparse_explicit_failure_restore(self):
        values = np.random.default_rng(1051).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        def scope(gpu, cpu, records, composed=True):
            options = dict(shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True, product_records=[]) if composed else {}
            return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], upcast_norm_records=[], chunk_records=records, **options)
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
                        count = len(records); self.assertEqual(count, 4 * clusters)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options, random_state=17).fit(values.copy())).result()
                        self.assertEqual(len(records), count)
                    np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
            with scope(gpu, cpu, skipped := []):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope(gpu, cpu, []), patch.object(cpu, "upcast_chunk", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(gpu, [], chunk_records=[]):
                    pass
        self.assertIs(KMeans.fit, original_fit); self.assertIs(kmeans._kmeans_plusplus, original_plus)

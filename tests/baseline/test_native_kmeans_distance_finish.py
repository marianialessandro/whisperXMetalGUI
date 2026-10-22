from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans
import sklearn.metrics.pairwise as pairwise

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_upcast_norms import compile_upcast_distances, finish_distances, squared_upcast_norms


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKMeansDistanceFinishTests(unittest.TestCase):
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

    def oracle(self, products, left, right):
        result = -2 * products
        result += left.reshape(-1, 1)
        result += right.reshape(1, -1)
        return result.astype(np.float32)

    def test_rounding_midpoints_subnormals_signed_underflow_and_cancellation(self):
        rng = np.random.default_rng(1235)
        lower = rng.integers(0, 0x7f7ffffe, 4096, dtype=np.uint32).view(np.float32)
        upper = np.nextafter(lower, np.float32(np.inf)).astype(np.float64)
        mid = (lower.astype(np.float64) + upper) / 2
        values = np.concatenate((lower.astype(np.float64), upper, mid, np.nextafter(mid, -np.inf), np.nextafter(mid, np.inf)))
        values = np.concatenate((values, -values, [0., -0., 2. ** -499, -(2. ** -499), 2. ** -150, -(2. ** -150), 2. ** -149, -(2. ** -149)]))
        transition = 2. ** -126 - 2. ** -150
        edges = np.array([transition, np.nextafter(transition, 0.), np.nextafter(transition, np.inf)])
        values = np.concatenate((values, edges, -edges))
        values = np.pad(values, (0, -len(values) % 1024)).reshape(-1, 1024)
        with self.context() as context:
            products = -values / 2
            left, right = np.zeros(len(values)), np.zeros(1024)
            actual, record = context.kmeans_distance_finish(products, left, right)
            np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(products, left, right).view(np.uint32))
            self.assertTrue(record["binary32_subnormals_preserved"])
            self.assertFalse(record["host_numeric_dtype_conversion"])
            for magnitude in (1., 2. ** -400, 2. ** 100):
                left, right = rng.uniform(.5, 1., 33) * magnitude, rng.uniform(.5, 1., 127) * magnitude
                products = (left[:, None] + right[None, :]) / 2
                products[:, ::3] = np.nextafter(products[:, ::3], np.inf)
                actual, _ = context.kmeans_distance_finish(products, left, right)
                np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(products, left, right).view(np.uint32))

    def test_shapes_domains_maximum_budget_layouts_and_closed_abi(self):
        rng = np.random.default_rng(1277)
        with self.context() as context:
            products = rng.normal(size=(1024, 1024))
            products.flags.writeable = False
            left, right = np.ones(1024), np.ones(1024)
            actual, record = context.kmeans_distance_finish(products, left, right)
            np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(products, left, right).view(np.uint32))
            self.assertGreater(record["computation_gpu_seconds"], 0)
            self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            self.assertEqual(record["logical_input_upload_bytes"], products.nbytes + left.nbytes + right.nbytes)
            for shape in ((0, 2), (4097, 2), (2, 4097), (1025, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_distance_finish(np.zeros(shape), np.zeros(shape[0]), np.zeros(shape[1]))
            for value in (np.nan, np.inf, 2. ** -501, 2. ** 501):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_distance_finish(np.full((2, 3), value), np.zeros(2), np.zeros(3))
            with self.assertRaisesRegex(ValueError, "domain_invalid"):
                context.kmeans_distance_finish(np.zeros((2, 3)), -np.ones(2), np.zeros(3))
            with self.assertRaisesRegex(ValueError, "output_overflow"):
                context.kmeans_distance_finish(np.full((2, 3), 2. ** 130), np.zeros(2), np.zeros(3))
            overflow_midpoint = float(np.finfo(np.float32).max) + 2. ** 103
            for sign in (-1., 1.):
                below = sign * np.nextafter(overflow_midpoint, 0.)
                products = np.array([[-below / 2]])
                actual, _ = context.kmeans_distance_finish(products, np.zeros(1), np.zeros(1))
                np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(products, np.zeros(1), np.zeros(1)).view(np.uint32))
                for target in (sign * overflow_midpoint, sign * np.nextafter(overflow_midpoint, np.inf)):
                    with self.assertRaisesRegex(ValueError, "output_overflow"):
                        context.kmeans_distance_finish(np.array([[-target / 2]]), np.zeros(1), np.zeros(1))
            products = rng.normal(size=(3, 7))
            unaligned = np.ndarray(products.shape, dtype=np.float64, buffer=bytearray(products.nbytes + 1), offset=1)
            unaligned[:] = products
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.kmeans_distance_finish(unaligned, np.zeros(3), np.zeros(7))[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result.view(np.uint32), self.oracle(products, np.zeros(3), np.zeros(7)).view(np.uint32))
            strided = np.repeat(products, 2, axis=1)[:, ::2]
            result = finish_distances(context, [], strided, np.zeros((3, 1)), np.zeros((1, 7)))
            np.testing.assert_array_equal(result.view(np.uint32), results[0].view(np.uint32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.kmeans_distance_finish(products, np.zeros(3), np.zeros(7))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_distance_finish(products, np.zeros(3), np.zeros(7))

    def test_full_distance_helpers_chunks_symmetry_and_complete_fits(self):
        rng = np.random.default_rng(1299)
        X, Y = rng.normal(size=(13, 17)).astype(np.float32), rng.normal(size=(19, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as context:
            records = []
            candidate = compile_upcast_distances(pairwise._euclidean_distances, lambda values, squared=False: squared_upcast_norms(context, [], values, squared), finish=lambda dot, left, right: finish_distances(context, records, dot, left, right))
            helper = candidate.__globals__["_euclidean_distances_upcast"]
            for other in (X, Y):
                for squared in (True, False):
                    actual = candidate(X, other, squared=squared)
                    expected = pairwise._euclidean_distances(X, other, squared=squared)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                for batch in (1, 2, 5):
                    np.testing.assert_array_equal(helper(X, Y=other, batch_size=batch).view(np.uint32), pairwise._euclidean_distances_upcast(X, Y=other, batch_size=batch).view(np.uint32))
            XX = np.einsum("ij,ij->i", X.astype(np.float64), X.astype(np.float64))[:, None]
            YY = np.einsum("ij,ij->i", Y.astype(np.float64), Y.astype(np.float64))[None, :]
            np.testing.assert_array_equal(helper(X, XX, Y, YY).view(np.uint32), pairwise._euclidean_distances_upcast(X, XX, Y, YY).view(np.uint32))
            for weighted in (False, True):
                weights = np.linspace(.5, 2., len(Y), dtype=np.float32) if weighted else None
                for clusters in (1, 2, 4):
                    options = dict(n_clusters=clusters, random_state=17, n_init=2, copy_x=False)
                    expected = KMeans(**options).fit(Y.copy(), sample_weight=weights)
                    records = []
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=records):
                        actual = KMeans(**options).fit(Y.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertTrue(records)
                        count = len(records)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(Y.copy(), sample_weight=weights)).result()
                        self.assertEqual(len(records), count)
            skipped = []
            with native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=skipped):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(Y))
                KMeans(n_clusters=2, n_init=1, init=Y[:2]).fit(Y)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected_finish_failure"), native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[]), patch.object(context, "kmeans_distance_finish", side_effect=ValueError("injected_finish_failure")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(Y)
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

    def test_configuration_source_and_capability_guards(self):
        source = inspect.getsource(pairwise._euclidean_distances_upcast)
        changed = source.replace("d += YY_chunk", "d -= YY_chunk")
        getsource = inspect.getsource
        with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is pairwise._euclidean_distances_upcast else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.UPCAST_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "finish_expression_contract_changed"):
            compile_upcast_distances(pairwise._euclidean_distances, object(), finish=object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(context, [], distance_finish_records=[]):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "distance_finish_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[]):
                    pass

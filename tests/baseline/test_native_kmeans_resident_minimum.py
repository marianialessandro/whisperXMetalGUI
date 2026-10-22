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
import torch
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans
import sklearn.metrics.pairwise as pairwise

from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_prefix import compile_native_prefix
from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline, compile_upcast_distances


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKMeansResidentMinimumTests(unittest.TestCase):
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

    def test_raw_bitwise_composition_caps_signed_zero_and_unaligned_readonly(self):
        rng = np.random.default_rng(967)
        with self.context() as context:
            for rows, columns, features in ((1, 7, 1), (3, 42, 17), (256, 256, 1024)):
                left, right = rng.normal(size=(rows, features)), rng.normal(size=(columns, features))
                products, closest = left @ right.T, rng.uniform(size=columns).astype(np.float32)
                closest[0] = -0.
                closest.flags.writeable = False
                for diagonal in ((False, True) if rows == columns else (False,)):
                    distances, _ = context.kmeans_resident_norms(products, left, right, diagonal)
                    expected = np.minimum(closest, distances)
                    actual, record = context.kmeans_resident_minimum(products, left, right, closest, diagonal)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(record["gpu_kernels"], ["double_kmeans_upcast_norms", "double_kmeans_resident_distance_finish", "float_kmeans_candidate_minimum"])
                    self.assertEqual(record["command_buffers_per_run"], 1)
                    self.assertTrue(record["resident_minimum"])
                    self.assertEqual(record["intermediate_candidate_readback_bytes"], 0)
                    self.assertEqual(record["logical_input_upload_bytes"], products.nbytes + left.nbytes + right.nbytes + closest.nbytes)
                    self.assertEqual(record["logical_result_readback_bytes"], actual.nbytes)
            closest = np.ndarray((7,), dtype=np.float32, buffer=bytearray(29), offset=1)
            closest[:] = np.arange(7)
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: context.kmeans_resident_minimum(np.ones((2, 7)), np.ones((2, 3)), np.ones((7, 3)), closest)[0], range(4)))
            for actual in outputs:
                np.testing.assert_array_equal(actual, np.minimum(closest, np.full((2, 7), 4., np.float32)))

    def test_subnormal_full_finite_range_and_signed_zero_bits(self):
        rng = np.random.default_rng(981)
        bits = np.concatenate((np.array([0, 1, 2, 3, 0x007fffff, 0x00800000, 0x00800001, 0x3f800000, 0x7f7fffff], np.uint32), rng.integers(0, 0x7f800000, 247, dtype=np.uint32)))
        values = bits.view(np.float32)
        products, left, right = np.ascontiguousarray((-values.astype(np.float64) * .5)[None, :]), np.zeros((1, 1)), np.zeros((len(values), 1))
        with self.context() as context:
            for closest in (values[::-1].copy(), np.full(len(values), np.nextafter(np.float32(0), np.float32(1))), np.full(len(values), -0., np.float32)):
                actual, _ = context.kmeans_resident_minimum(products, left, right, closest)
                np.testing.assert_array_equal(actual.view(np.uint32), np.minimum(closest, values[None, :]).view(np.uint32))

    def test_raw_errors_and_overflow_cannot_be_hidden_by_minimum(self):
        products, left, right = np.zeros((2, 2)), np.ones((2, 3)), np.ones((2, 3))
        with self.context() as context:
            for closest in (np.ones(2), np.ones((2, 1), np.float32), np.ones((1, 1, 2), np.float32), np.ones(4, np.float32)[::2], np.ones(0, np.float32)):
                with self.assertRaisesRegex(ValueError, "closest_format"):
                    context.kmeans_resident_minimum(products, left, right, closest)
            with self.assertRaisesRegex(ValueError, "closest_shape"):
                context.kmeans_resident_minimum(products, left, right, np.ones(3, np.float32))
            for value in (np.nan, np.inf, -np.inf, -1., -np.nextafter(np.float32(0), np.float32(1))):
                with self.assertRaisesRegex(ValueError, "closest_domain"):
                    context.kmeans_resident_minimum(products, left, right, np.full(2, value, np.float32))
            for diagonal in (False, True):
                with self.assertRaisesRegex(ValueError, "overflow_or_invalid"):
                    context.kmeans_resident_minimum(products, np.full((2, 3), 2. ** 100), right, np.zeros(2, np.float32), diagonal)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "resident_minimum_abi_unavailable"):
                context.kmeans_resident_minimum(products, left, right, np.ones(2, np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_resident_minimum(products, left, right, np.ones(2, np.float32))

    def test_pipeline_identity_reset_source_and_expression_guards(self):
        X, Y = np.random.default_rng(971).normal(size=(2, 7)).astype(np.float32), np.ones((5, 7), np.float32)
        closest = np.linspace(0, 1, 5, dtype=np.float32)[None, :]
        with self.context() as context:
            pipeline = ResidentDistancePipeline(context, finishes := [], [], norm_records=[], minimum_records=(minimum := []))
            reject = lambda *args, **kwargs: self.fail("standalone minimum bridge")
            pipeline.distance = compile_upcast_distances(pairwise._euclidean_distances, reject, finish=pipeline.finish, clamp=pipeline.clamp, resident=True, resident_norms=True)
            expected = np.minimum(closest, pairwise._euclidean_distances(X, Y, squared=True))
            with patch.object(context, "kmeans_candidate_minimum", side_effect=reject):
                actual = pipeline.candidate_distance(closest, X, Y, squared=True)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                with self.assertRaisesRegex(ValueError, "identity"):
                    pipeline.minimum(closest.copy(), actual)
                with self.assertRaisesRegex(ValueError, "state_invalid"):
                    pipeline.candidate_distance(closest, X, Y, squared=True)
                self.assertIs(pipeline.minimum(closest, actual), actual)
                self.assertEqual(minimum[0]["command_buffers_per_run"], 0)
                with self.assertRaisesRegex(ValueError, "identity"):
                    pipeline.minimum(closest, actual)
            with patch.object(context, "kmeans_resident_minimum", side_effect=ValueError("injected")), self.assertRaisesRegex(ValueError, "injected"):
                pipeline.candidate_distance(closest, X, Y, squared=True)
            self.assertIsNone(pipeline.pending_minimum)
            self.assertIsNone(pipeline.closest)
            self.assertIsNone(pipeline.pending)
            actual = pipeline.candidate_distance(closest, X, Y, squared=True)
            pipeline.reset()
            with self.assertRaisesRegex(ValueError, "identity"):
                pipeline.minimum(closest, actual)
        with self.assertRaisesRegex(ValueError, "configuration"):
            compile_native_prefix(kmeans._kmeans_plusplus, None, resident_candidate_distance=object())
        getsource = inspect.getsource
        changed = getsource(kmeans._kmeans_plusplus).replace("X[candidate_ids], X", "X[candidate_ids].copy(), X")
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=lambda function: changed if function is kmeans._kmeans_plusplus else getsource(function)), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "resident_minimum_expression"):
            compile_native_prefix(kmeans._kmeans_plusplus, None, minimum=object(), upcast_distance=object(), resident_candidate_distance=object())

    def test_complete_fits_composition_bypass_failure_restore_and_configuration(self):
        values = np.random.default_rng(977).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        def scope(context, finishes, minimum):
            return native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=minimum, probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=finishes, distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True)
        with self.context() as context:
            for weighted in (False, True):
                weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                for clusters in (1, 2, 4):
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    with scope(context, finishes := [], minimum := []), patch.object(context, "kmeans_candidate_minimum", side_effect=AssertionError("standalone minimum")):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertEqual(len(minimum), 2 * (clusters - 1))
                        self.assertTrue(all(row["resident_minimum_reuse"] and row["command_buffers_per_run"] == 0 for row in minimum))
                        count = len(finishes)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(values.copy())).result()
                        self.assertEqual(len(finishes), count)
            with scope(context, skipped := [], []):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope(context, [], []), patch.object(context, "kmeans_resident_minimum", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(context, [], resident_minimum=True):
                    pass
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

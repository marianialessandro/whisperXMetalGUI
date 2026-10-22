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
from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline, compile_upcast_distances


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKMeansResidentNormTests(unittest.TestCase):
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

    def test_raw_matches_separate_norms_and_finish_at_boundaries_and_maximum_budget(self):
        rng = np.random.default_rng(941)
        with self.context() as context:
            for rows, columns, features in ((1, 7, 1), (3, 7, 7), (42, 42, 256), (256, 256, 1024)):
                left, right = rng.normal(size=(rows, features)), rng.normal(size=(columns, features))
                products = left @ right.T
                a, _ = context.kmeans_upcast_norms(left)
                b, _ = context.kmeans_upcast_norms(right)
                for diagonal in ((False, True) if rows == columns else (False,)):
                    expected, _ = context.kmeans_resident_distances(products, a, b, diagonal)
                    actual, record = context.kmeans_resident_norms(products, left, right, diagonal)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(record["gpu_kernels"], ["double_kmeans_upcast_norms", "double_kmeans_resident_distance_finish"])
                    self.assertEqual(record["command_buffers_per_run"], 1)
                    self.assertEqual(record["intermediate_norm_readback_bytes"], 0)
                    self.assertEqual(record["logical_input_upload_bytes"], products.nbytes + left.nbytes + right.nbytes)
                    self.assertEqual(record["logical_result_readback_bytes"], actual.nbytes)
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for value in (0., -0., 2. ** -200):
                actual, _ = context.kmeans_resident_norms(np.zeros((2, 2)), np.full((2, 3), value), np.full((2, 3), value))
                np.testing.assert_array_equal(actual.view(np.uint32), np.zeros((2, 2), np.uint32))

    def test_raw_domains_shapes_readonly_unaligned_serialization_and_closed_abi(self):
        with self.context() as context:
            products, left, right = np.ones((3, 7)), np.ones((3, 8)), np.ones((7, 8))
            unaligned = np.ndarray(left.shape, dtype=np.float64, buffer=bytearray(left.nbytes + 1), offset=1)
            unaligned[:] = left; unaligned.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.kmeans_resident_norms(products, unaligned, right)[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result, np.full((3, 7), 14., np.float32))
            for rows, columns, features, policy in ((0, 1, 1, False), (257, 1, 1, False), (1, 257, 1, False), (2, 2, 0, False), (2, 2, 1025, False), (2, 3, 7, True), (2, 2, 7, 1)):
                with self.assertRaisesRegex(ValueError, "shape_or_policy_invalid"):
                    context.kmeans_resident_norms(np.zeros((rows, columns)), np.zeros((rows, features)), np.zeros((columns, features)), policy)
            for value in (np.inf, np.nan, 2. ** -201, 2. ** 201):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_resident_norms(products, np.full(left.shape, value), right)
            for value in (np.inf, 2. ** -501, 2. ** 501):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_resident_norms(np.full(products.shape, value), left, right)
            with self.assertRaisesRegex(ValueError, "output_overflow"):
                context.kmeans_resident_norms(np.zeros((2, 2)), np.full((2, 3), 2. ** 100), np.zeros((2, 3)))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "resident_norms_abi_unavailable"):
                context.kmeans_resident_norms(products, left, right)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_resident_norms(products, left, right)

    def test_whole_helpers_displace_host_norm_bridges_and_reject_provided_binary64_norms(self):
        rng = np.random.default_rng(947)
        X, Y = rng.normal(size=(13, 17)).astype(np.float32), rng.normal(size=(19, 17)).astype(np.float32)
        with self.context() as context:
            pipeline = ResidentDistancePipeline(context, finishes := [], clamps := [], norm_records=(norms := []))
            def reject_bridge(*args, **kwargs):
                raise AssertionError("host norm bridge or CPU postprocessing")
            candidate = compile_upcast_distances(pairwise._euclidean_distances, reject_bridge, finish=pipeline.finish, clamp=pipeline.clamp, resident=True, resident_norms=True)
            for other in (X, X.copy(), Y):
                expected = pairwise._euclidean_distances(X, other, squared=True)
                with patch.object(context, "kmeans_upcast_norms", side_effect=reject_bridge), patch.object(context, "kmeans_resident_distances", side_effect=reject_bridge), patch.dict(candidate.__globals__, {"_modify_in_place_if_numpy": reject_bridge, "_fill_diagonal": reject_bridge}):
                    actual = candidate(X, other, squared=True)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(len(norms), 2 * len(finishes))
                self.assertTrue(all(row["resident_norm_reuse"] and row["command_buffers_per_run"] == 0 for row in norms))
                self.assertIsNone(pipeline.pending)
            with self.assertRaisesRegex(ValueError, "provided_binary64_norms"):
                candidate(X, Y, X_norm_squared=np.ones(len(X)), squared=True)
            helper = candidate.__globals__["_euclidean_distances_upcast"]
            with self.assertRaisesRegex(ValueError, "provided_chunk_norms"):
                helper(X, XX=np.ones((len(X), 1)), Y=Y)
            expected = pairwise._euclidean_distances(X, Y, Y_norm_squared=np.ones(len(Y), np.float32), squared=True)
            actual = candidate(X, Y, Y_norm_squared=np.ones(len(Y), np.float32), squared=True)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            source = inspect.getsource(pairwise._euclidean_distances_upcast)
            changed = source.replace("XX_chunk = XX[x_slice]", "XX_chunk = XX.copy()[x_slice]")
            getsource = inspect.getsource
            with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is pairwise._euclidean_distances_upcast else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.UPCAST_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "resident_norms_expression_contract_changed"):
                compile_upcast_distances(pairwise._euclidean_distances, object(), finish=object(), clamp=object(), resident=True, resident_norms=True)

    def test_complete_fits_scope_composition_sparse_owner_explicit_failure_and_configuration(self):
        values = np.random.default_rng(953).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as context:
            for weighted in (False, True):
                weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                for clusters in (1, 2, 4):
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    finishes, norms = [], []
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=norms, distance_finish_records=finishes, distance_clamp_records=[], resident_distances=True, resident_norms=True):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        count = len(finishes)
                        self.assertTrue(count)
                        self.assertEqual(len(norms), count * 2)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(values.copy(), sample_weight=weights)).result()
                        self.assertEqual(len(finishes), count)
            skipped = []
            with native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=skipped, distance_clamp_records=[], resident_distances=True, resident_norms=True):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected_resident_norm_failure"), native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True), patch.object(context, "kmeans_resident_norms", side_effect=ValueError("injected_resident_norm_failure")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(context, [], resident_norms=True):
                    pass
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

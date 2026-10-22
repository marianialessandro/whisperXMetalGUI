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
from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline, compile_upcast_distances, squared_upcast_norms


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKMeansResidentDistanceTests(unittest.TestCase):
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

    def test_fused_raw_exact_rounding_clamp_diagonal_layout_and_failure_guards(self):
        rng = np.random.default_rng(817)
        with self.context() as context:
            for rows, columns in ((1, 7), (3, 7), (42, 42), (256, 256)):
                products, left, right = rng.normal(size=(rows, columns)), rng.uniform(0., 1., rows), rng.uniform(0., 1., columns)
                for diagonal in ((False, True) if rows == columns else (False,)):
                    finished, _ = context.kmeans_distance_finish(products, left, right)
                    expected, _ = context.kmeans_distance_clamp(finished, diagonal)
                    actual, record = context.kmeans_resident_distances(products, left, right, diagonal)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(record["gpu_kernels"], ["double_kmeans_resident_distance_finish"])
                    self.assertEqual(record["command_buffers_per_run"], 1)
                    self.assertEqual(record["intermediate_distance_readback_bytes"], 0)
                    self.assertTrue(record["distance_clamping_replaced"])
                    self.assertGreater(record["computation_gpu_seconds"], 0)
            targets = np.array([[0., -0., 2. ** -149, -(2. ** -149), 1. + 2. ** -24, 2. ** -126 - 2. ** -150]])
            actual, _ = context.kmeans_resident_distances(-targets / 2, np.zeros(1), np.zeros(6))
            expected = -2 * (-targets / 2); expected += 0.; expected += 0.
            np.testing.assert_array_equal(actual.view(np.uint32), np.maximum(expected.astype(np.float32), np.float32(0)).view(np.uint32))
            for diagonal in (False, True):
                for product in (2. ** 130, -(2. ** 130)):
                    with self.assertRaisesRegex(ValueError, "output_overflow"):
                        context.kmeans_resident_distances(np.full((2, 2), product), np.zeros(2), np.zeros(2), diagonal)
            for rows, columns, policy in ((257, 1, False), (1, 257, False), (2, 3, True), (2, 2, 1)):
                with self.assertRaisesRegex(ValueError, "shape_or_policy_invalid"):
                    context.kmeans_resident_distances(np.ones((rows, columns)), np.ones(rows), np.ones(columns), policy)
            products = np.ones((3, 7))
            unaligned = np.ndarray(products.shape, dtype=np.float64, buffer=bytearray(products.nbytes + 1), offset=1)
            unaligned[:] = products
            unaligned.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.kmeans_resident_distances(unaligned, np.zeros(3), np.zeros(7))[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result, np.zeros((3, 7), np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "resident_distances_abi_unavailable"):
                context.kmeans_resident_distances(products, np.zeros(3), np.zeros(7))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_resident_distances(products, np.zeros(3), np.zeros(7))

    def test_whole_helpers_exact_identity_reuse_provided_norms_and_single_chunk_bounds(self):
        rng = np.random.default_rng(821)
        with self.context() as context:
            pipeline = ResidentDistancePipeline(context, finishes := [], clamps := [])
            norms = lambda values, squared=False: squared_upcast_norms(context, [], values, squared)
            candidate = compile_upcast_distances(pairwise._euclidean_distances, norms, finish=pipeline.finish, clamp=pipeline.clamp, resident=True)
            for rows, columns, features in ((1, 7, 1), (13, 19, 17), (42, 42, 256), (256, 256, 1024)):
                X, Y = rng.normal(size=(rows, features)).astype(np.float32), rng.normal(size=(columns, features)).astype(np.float32)
                for other in (Y, X, X.copy()):
                    expected = pairwise._euclidean_distances(X, other, squared=True)
                    def reject_cpu_postprocess(*args, **kwargs):
                        raise AssertionError("CPU maximum or diagonal")
                    with patch.object(context, "kmeans_distance_finish", side_effect=AssertionError("standalone finish bridge")), patch.object(context, "kmeans_distance_clamp", side_effect=AssertionError("standalone clamp bridge")), patch.dict(candidate.__globals__, {"_modify_in_place_if_numpy": reject_cpu_postprocess, "_fill_diagonal": reject_cpu_postprocess}):
                        actual = candidate(X, other, squared=True)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(clamps[-1]["zero_diagonal"], X is other)
                    self.assertEqual(clamps[-1]["command_buffers_per_run"], 0)
                    self.assertEqual(clamps[-1]["shared_finish_dispatch_index"], len(finishes) - 1)
                    self.assertIsNone(pipeline.pending)
            helper = candidate.__globals__["_euclidean_distances_upcast"]
            X = rng.normal(size=(3, 7)).astype(np.float32)
            XX = np.einsum("ij,ij->i", X.astype(np.float64), X.astype(np.float64))[:, None]
            actual = helper(X, XX, X, XX.T, batch_size=3)
            expected = pairwise._euclidean_distances_upcast(X, XX, X, XX.T, batch_size=3)
            expected = np.maximum(expected, np.float32(0)); np.fill_diagonal(expected, 0)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            pipeline.clamp(actual, True)
            with self.assertRaisesRegex(ValueError, "single_chunk_batch_required"):
                helper(X, Y=X, batch_size=1)
            for bad in (np.ones((257, 7), np.float32), np.ones((3, 1025), np.float32), np.ones((0, 7), np.float32)):
                with self.assertRaisesRegex(ValueError, "single_chunk_shape_required"):
                    candidate(bad, bad, squared=True)
            with self.assertRaisesRegex(ValueError, "squared_float32_policy_required"):
                candidate(X, X, squared=False)

    def test_result_cache_checks_configuration_source_and_reset(self):
        with self.context() as context:
            pipeline = ResidentDistancePipeline(context, [], [])
            products, left, right = np.ones((2, 2)), np.ones((2, 1)), np.ones((1, 2))
            result = pipeline.finish(products, left, right, False)
            with self.assertRaisesRegex(ValueError, "previous_result_not_consumed"):
                pipeline.finish(products, left, right, False)
            with self.assertRaisesRegex(ValueError, "reuse_mismatch"):
                pipeline.clamp(result.copy(), False)
            result = pipeline.finish(products, left, right, False)
            pipeline.reset()
            with self.assertRaisesRegex(ValueError, "reuse_mismatch"):
                pipeline.clamp(result, False)
            for kwargs in ({"resident": True}, {"resident": 1}):
                with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                    compile_upcast_distances(pairwise._euclidean_distances, object(), **kwargs)
            source = inspect.getsource(pairwise._euclidean_distances_upcast)
            changed = source.replace("copy=False", "copy=True")
            getsource = inspect.getsource
            with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is pairwise._euclidean_distances_upcast else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.UPCAST_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "resident_distances_expression_contract_changed"):
                compile_upcast_distances(pairwise._euclidean_distances, object(), finish=object(), clamp=object(), resident=True)

    def test_full_fits_composition_sparse_owner_explicit_and_fault_restore(self):
        rng = np.random.default_rng(827)
        values = rng.normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as context:
            for weighted in (False, True):
                weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                for clusters in (1, 2, 4):
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    finishes, clamps = [], []
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=finishes, distance_clamp_records=clamps, resident_distances=True):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertTrue(finishes)
                        self.assertTrue(all(row["resident_clamp_reuse"] for row in clamps))
                        count = len(finishes)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(values.copy(), sample_weight=weights)).result()
                        self.assertEqual(len(finishes), count)
            skipped = []
            with native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=skipped, distance_clamp_records=[], resident_distances=True):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected_resident_failure"), native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True), patch.object(context, "kmeans_resident_distances", side_effect=ValueError("injected_resident_failure")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(context, [], resident_distances=True):
                    pass
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

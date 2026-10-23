from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans
import sklearn.metrics.pairwise as pairwise

from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library as build_cpu
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_upcast_norms import ResidentDistancePipeline, compile_upcast_distances


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Metal required")
class NativeKMeansResident512Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.gpu = Path(cls.directory.name) / "gpu.dylib"
        cls.identity = build_library(cls.gpu)
        cls.cpu = Path(cls.directory.name) / "cpu.dylib"
        cls.cpu_identity = build_cpu(cls.cpu, with_two_chunks=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeDoubleNormalization(self.gpu, self.identity["library_sha256"])

    def test_raw_boundaries_norm_order_clamp_diagonal_minimum_and_metrics(self):
        rng = np.random.default_rng(512101)
        with self.context() as gpu:
            for rows, columns in ((1, 257), (3, 512), (257, 1), (512, 3), (257, 257), (511, 512), (512, 512)):
                for features in (1, 7, 17, 63, 64, 65, 256, 1024):
                    left, right = rng.normal(size=(rows, features)), rng.normal(size=(columns, features))
                    products = left @ right.T
                    a, b = pairwise.row_norms(left, squared=True), pairwise.row_norms(right, squared=True)
                    for diagonal in (False, True) if rows == columns else (False,):
                        expected = -2 * products
                        expected += a[:, None]
                        expected += b[None, :]
                        expected = np.maximum(expected.astype(np.float32), np.float32(0))
                        if diagonal:
                            np.fill_diagonal(expected, 0)
                        first, first_record = gpu.kmeans_resident_distances(products, a, b, diagonal, extended_shape=True)
                        actual, record = gpu.kmeans_resident_norms(products, left, right, diagonal, extended_shape=True)
                        closest = rng.uniform(0, 2, columns).astype(np.float32)
                        minimum, minimum_record = gpu.kmeans_resident_minimum(products, left, right, closest, diagonal, extended_shape=True)
                        for value in (first, actual):
                            np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
                        np.testing.assert_array_equal(minimum.view(np.uint32), np.minimum(closest, expected).view(np.uint32))
                        for item, kernels in ((first_record, 1), (record, 2), (minimum_record, 3)):
                            self.assertTrue(item["extended_shape_enabled"])
                            self.assertEqual(item["maximum_extent"], 512)
                            self.assertEqual(item["command_buffers_per_run"], 1)
                            self.assertEqual(len(item["gpu_kernels"]), kernels)
                            self.assertEqual(item["intermediate_distance_readback_bytes"], 0)
                            self.assertGreater(item["computation_gpu_seconds"], 0)
                            self.assertLess(item["allocated_shared_buffer_bytes"], 48 * 1024 * 1024)
                        self.assertEqual(record["intermediate_norm_readback_bytes"], 0)
                        self.assertEqual(minimum_record["intermediate_candidate_readback_bytes"], 0)
                        self.assertEqual(minimum_record["logical_result_readback_bytes"], minimum.nbytes)

    def test_raw_limits_legacy_missing_abi_domain_concurrency_and_closed(self):
        left, right = np.ones((3, 17)), np.ones((512, 17))
        products = left @ right.T
        a, b = np.full(3, 17.), np.full(512, 17.)
        with self.context() as gpu:
            calls = (lambda **kwargs: gpu.kmeans_resident_distances(products, a, b, **kwargs), lambda **kwargs: gpu.kmeans_resident_norms(products, left, right, **kwargs), lambda **kwargs: gpu.kmeans_resident_minimum(products, left, right, np.ones(512, np.float32), **kwargs))
            for call in calls:
                with self.assertRaisesRegex(ValueError, "shape_or_policy"):
                    call()
                for flag in (1, None, "yes"):
                    with self.assertRaisesRegex(ValueError, "configuration"):
                        call(extended_shape=flag)
                with patch.object(gpu, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "resident_512_abi_unavailable"):
                    call(extended_shape=True)
            unaligned = np.ndarray(right.shape, dtype=np.float64, buffer=bytearray(right.nbytes + 1), offset=1)
            unaligned[:] = right
            unaligned.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: gpu.kmeans_resident_norms(products, left, unaligned, extended_shape=True)[0], range(4)))
            for output in outputs:
                np.testing.assert_array_equal(output.view(np.uint32), np.zeros(products.shape, np.uint32))
            with self.assertRaisesRegex(ValueError, "shape_or_policy"):
                gpu.kmeans_resident_norms(np.ones((3, 513)), left, np.ones((513, 17)), extended_shape=True)
            right[-1, -1] = np.inf
            with self.assertRaisesRegex(ValueError, "value_domain"):
                gpu.kmeans_resident_norms(products, left, right, extended_shape=True)
        with self.assertRaisesRegex(ValueError, "closed"):
            gpu.kmeans_resident_norms(products, left, right, extended_shape=True)

    def test_helper_and_composed_fits_preserve_original_batching_and_rng(self):
        with self.context() as gpu, NativeKMeansPotentials(self.cpu, self.cpu_identity["library_sha256"]) as cpu:
            for rows in (257, 512):
                x = np.random.default_rng(rows).normal(size=(rows, 17)).astype(np.float32)
                for mode in (0, 1, 2):
                    finishes, clamps, norms, minima = [], [], [], []
                    pipeline = ResidentDistancePipeline(gpu, finishes, clamps, norm_records=norms if mode else None, extended_shape=True)
                    candidate = compile_upcast_distances(pairwise._euclidean_distances, pairwise.row_norms, finish=pipeline.finish, clamp=pipeline.clamp, resident=True, resident_norms=bool(mode), resident_512=True)
                    for right in (x, x.copy(), x[:3].copy()):
                        expected = pairwise._euclidean_distances(x, right, squared=True)
                        actual = candidate(x, right, squared=True)
                        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    rng1, rng2 = np.random.RandomState(17), np.random.RandomState(17)
                    left, right = x.copy(), x.copy()
                    weights = np.linspace(.125, 2, rows, dtype=np.float32)
                    expected = KMeans(n_clusters=4, n_init=2, copy_x=False, random_state=rng1).fit(left, sample_weight=weights)
                    finishes.clear()
                    with native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], upcast_norm_records=norms, chunk_records=[], upcast_512=True, product_records=[], products_512=True, distance_finish_records=finishes, distance_clamp_records=clamps, resident_distances=True, resident_norms=bool(mode), minimum_records=minima if mode == 2 else None, resident_minimum=mode == 2, resident_512=True):
                        actual = KMeans(n_clusters=4, n_init=2, copy_x=False, random_state=rng2).fit(right, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
                    np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._n_threads), (expected.inertia_, expected.n_iter_, expected._n_threads))
                    self.assertTrue(any(max(r["rows"], r["columns"]) > 256 for r in finishes))
                    self.assertTrue(all(r["command_buffers_per_run"] == 1 and len(r["gpu_kernels"]) in ((2, 3) if mode == 2 else (mode + 1,)) for r in finishes))
                    if mode == 2:
                        self.assertTrue(any(len(r["gpu_kernels"]) == 3 for r in finishes))
                        self.assertTrue(minima)
                        self.assertTrue(all(r["resident_minimum_reuse"] and r["command_buffers_per_run"] == 0 for r in minima))

    def test_configuration_missing_selected_abi_and_exception_restore(self):
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], resident_512=True):
                pass
            options = dict(upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_512=True)
            for mode, missing in ((0, "distances"), (1, "norms"), (2, "minimum")):
                library = SimpleNamespace(**{name: getattr(gpu.library, name) for name in ("wxdouble_vbx_kmeans_centroids", "wxdouble_kmeans_distance_finish", "wxdouble_kmeans_distance_clamp", "wxdouble_kmeans_resident_distances", "wxdouble_kmeans_resident_norms", "wxdouble_kmeans_resident_minimum", "wxdouble_kmeans_resident_distances_512", "wxdouble_kmeans_resident_norms_512", "wxdouble_kmeans_resident_minimum_512") if name != "wxdouble_kmeans_resident_" + missing + "_512"})
                with patch.object(gpu, "library", library), self.assertRaisesRegex(ValueError, "resident_512_abi_unavailable"), native_kmeans_feature_mean(gpu, [], resident_norms=mode > 0, resident_minimum=mode == 2, minimum_records=[] if mode == 2 else None, **options):
                    pass
            with self.assertRaisesRegex(ValueError, "injected"), native_kmeans_feature_mean(gpu, [], **options), patch.object(gpu, "kmeans_resident_distances", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(np.ones((257, 3), np.float32))
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

    def test_maximum_original_batch_and_explicit_rechunk_rejection(self):
        x = np.random.default_rng(512103).normal(size=(512, 1024)).astype(np.float32)
        with self.context() as gpu:
            finishes = []
            pipeline = ResidentDistancePipeline(gpu, finishes, [], norm_records=[], extended_shape=True)
            candidate = compile_upcast_distances(pairwise._euclidean_distances, None, finish=pipeline.finish, clamp=pipeline.clamp, resident=True, resident_norms=True, resident_512=True)
            expected = pairwise._euclidean_distances(x, x, squared=True)
            actual = candidate(x, x, squared=True)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(len(finishes), 1)
            self.assertEqual((finishes[0]["rows"], finishes[0]["columns"]), (512, 512))
            helper = candidate.__globals__["_euclidean_distances_upcast"]
            with self.assertRaisesRegex(ValueError, "single_chunk_batch_required"):
                helper(x, Y=x, batch_size=256)
            self.assertEqual(len(finishes), 1)

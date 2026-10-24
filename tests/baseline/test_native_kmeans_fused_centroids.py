from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansFusedCentroidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.cpu = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.cpu, with_two_chunks=True)
        cls.gpu = Path(cls.directory.name) / "gpu.dylib"
        cls.gpu_identity = build_double(cls.gpu)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.cpu, self.identity["library_sha256"])

    def test_raw_matches_separate_operations_at_boundaries_and_empty_clusters(self):
        rng = np.random.default_rng(56101)
        with self.context() as cpu:
            for rows in (1, 42, 256, 257, 511, 512):
                clusters = min(rows, 4)
                for features in (1, 3, 17, 63, 64, 65, 256, 1024):
                    x = rng.normal(size=(rows, features)).astype(np.float32)
                    weights = np.linspace(.125, 2, rows, dtype=np.float32)
                    for empty in (False, True):
                        labels = np.zeros(rows, np.int32) if empty else np.arange(rows, dtype=np.int32) % clusters
                        previous = x[:clusters].copy()
                        for threads in (1, 2, 12):
                            expected = cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=threads)
                            expected_shifts = cpu.lloyd_center_shifts(expected[0], previous)[0]
                            actual = cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=threads, with_shifts=True)
                            for left, right in zip(actual[:4], expected[:4]):
                                np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
                            np.testing.assert_array_equal(actual[4].view(np.uint32), expected_shifts.view(np.uint32))
                            record = actual[5]
                            self.assertTrue(record["center_shift_owned_buffer_reuse"])
                            self.assertEqual(record["center_shift_snapshot_bytes_avoided"], previous.nbytes * 2)
                            self.assertEqual(record["center_shift_input_snapshot_bytes"], 0)
                            self.assertTrue(record["fused_center_shifts"])
                            self.assertFalse(record["python_center_shift_bridge"])
                            self.assertEqual(record["intermediate_centroid_transfer_for_shift_bytes"], 0)
                            self.assertEqual(record["relocated_clusters"], expected[4]["relocated_clusters"])
                            self.assertEqual(record["actual_threads"], min(threads, (rows + 255) // 256))

    def test_raw_validation_legacy_missing_abi_readonly_unaligned_and_closed(self):
        x = np.random.default_rng(56103).normal(size=(512, 17)).astype(np.float32)
        unaligned = np.ndarray(x.shape, dtype=np.float32, buffer=bytearray(x.nbytes + 1), offset=1)
        unaligned[:] = x
        unaligned.flags.writeable = False
        weights, labels, previous = np.ones(512, np.float32), np.arange(512, dtype=np.int32) % 4, x[:4].copy()
        with self.context() as cpu:
            for flag in (1, None, "yes"):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12, with_shifts=flag)
            with self.assertRaisesRegex(ValueError, "configuration"):
                cpu.lloyd_relocated_centroids(x, weights, labels, previous, with_shifts=True)
            expected = cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12, with_shifts=True)
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: cpu.lloyd_relocated_centroids(unaligned, weights, labels, previous, n_threads=12, with_shifts=True), range(4)))
            for output in outputs:
                for left, right in zip(output[:5], expected[:5]):
                    np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
            weights[-1] = -1
            with self.assertRaisesRegex(ValueError, "weight_domain"):
                cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12, with_shifts=True)
            with patch.object(cpu, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "fused_centroids_abi_unavailable"):
                cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12, with_shifts=True)
        with self.assertRaisesRegex(ValueError, "closed"):
            cpu.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12, with_shifts=True)

    def test_complete_fits_shared_records_and_failure_restore(self):
        original = KMeans.fit
        with self.context() as cpu, NativeDoubleNormalization(self.gpu, self.gpu_identity["library_sha256"]) as gpu:
            def scope(records, shifts):
                return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], norm_records=[], lloyd_norm_records=[], assignment_records=[], centroid_records=records, center_shift_records=shifts, inertia_records=[], relocation_records=[], two_chunks=True, fused_centroids=True)
            for rows in (42, 257, 512):
                x = np.random.default_rng(rows).normal(size=(rows, 17)).astype(np.float32)
                for duplicate in (False, True):
                    for maximum in (1, 300):
                        a, b = np.random.RandomState(17), np.random.RandomState(17)
                        left, right = x.copy(), x.copy()
                        weights = np.linspace(.125, 2, rows, dtype=np.float32)
                        options = dict(n_clusters=4, n_init=1 if duplicate else 3, init=np.repeat(x[:1], 4, axis=0) if duplicate else "k-means++", copy_x=False, max_iter=maximum)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            expected = KMeans(**options, random_state=a).fit(left, sample_weight=weights)
                            with scope(records := [], shifts := []):
                                actual = KMeans(**options, random_state=b).fit(right, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
                        np.testing.assert_array_equal(a.uniform(size=32), b.uniform(size=32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._n_threads, actual._tol, type(actual._tol)), (expected.inertia_, expected.n_iter_, expected._n_threads, expected._tol, type(expected._tol)))
                        self.assertTrue(records)
                        self.assertEqual(len(records), len(shifts))
                        self.assertTrue(all(r["fused_center_shifts"] for r in records))
                        self.assertTrue(all(r["shared_operation_index"] == i and r["native_wall_ns"] == r["host_input_snapshot_bytes"] == r["host_result_bytes"] == 0 for i, r in enumerate(shifts)))
            with patch.object(cpu.library, "wxkmeans_lloyd_centroids_and_shifts", None), self.assertRaisesRegex(ValueError, "fused_centroids_abi_unavailable"), scope([], []):
                pass
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], fused_centroids=True):
                pass
            with scope([], []), patch.object(cpu, "lloyd_center_shifts", side_effect=AssertionError("separate shift bridge used")):
                KMeans(n_clusters=4, n_init=1, random_state=17).fit(x)
            with self.assertRaisesRegex(ValueError, "injected"), scope([], []), patch.object(cpu, "lloyd_relocated_centroids", side_effect=ValueError("injected")):
                KMeans(n_clusters=4, n_init=1, random_state=17).fit(x)
        self.assertIs(KMeans.fit, original)

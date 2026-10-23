from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
import sklearn.cluster._k_means_lloyd as cython_lloyd

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Pinned Apple native core required")
class NativeKMeansRelocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.library, with_inertia=True)
        cls.double = Path(cls.directory.name) / "gpu.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def gpu(self):
        return NativeDoubleNormalization(self.double, self.double_identity["library_sha256"])

    def test_duplicate_and_distinct_centers_match_cython_weighted_outputs(self):
        rng = np.random.default_rng(2805)
        with self.context() as core:
            for rows in (1, 7, 42, 256):
                for features in (1, 2, 4, 7, 16, 17, 129, 256, 1024):
                    x = rng.normal(size=(rows, features)).astype(np.float32)
                    for clusters in (1, 2, 4, 8):
                        if clusters > rows:
                            continue
                        for duplicate in (False, True):
                            for weighted in (False, True):
                                previous = np.repeat(x[:1], clusters, axis=0) if duplicate else x[:clusters].copy()
                                weights = np.linspace(.5, 2., rows, dtype=np.float32) if weighted else np.ones(rows, np.float32)
                                centers, totals, labels, shifts = np.empty_like(previous), np.empty(clusters, np.float32), np.empty(rows, np.int32), np.empty(clusters, np.float32)
                                lloyd_iter_chunked_dense(x, weights, previous, centers, totals, labels, shifts, 12)
                                saved = labels.copy()
                                actual, actual_totals, distances, selected, record = core.lloyd_relocated_centroids(x, weights, labels, previous)
                                np.testing.assert_array_equal(actual.view(np.uint32), centers.view(np.uint32))
                                np.testing.assert_array_equal(actual_totals.view(np.uint32), totals.view(np.uint32))
                                np.testing.assert_array_equal(labels, saved)
                                count = sum(not np.any(weights[labels == cluster] > 0) for cluster in range(clusters))
                                self.assertEqual(record["empty_clusters"], count)
                                if count:
                                    expected_distances = ((x - previous[labels]) ** 2).sum(axis=1)
                                    np.testing.assert_array_equal(distances.view(np.uint32), expected_distances.view(np.uint32))
                                    np.testing.assert_array_equal(selected, np.argpartition(expected_distances, -count)[:-count-1:-1].astype(np.int32))
                                self.assertFalse(record["python_argpartition"])
                                self.assertEqual(record["execution_device"], "cpu")
                                self.assertIsNone(record["computation_gpu_seconds"])

    def test_every_selection_size_ties_zero_and_adversarial_roll_order(self):
        rng = np.random.default_rng(2807)
        with self.context() as core:
            for rows in range(2, 257):
                weights, labels = np.ones(rows, np.float32), np.zeros(rows, np.int32)
                for values in (rng.integers(-4, 5, rows).astype(np.float32), np.zeros(rows, np.float32), np.roll(np.arange(rows, dtype=np.float32), rows // 2)):
                    x = values[:, None].copy()
                    distances = values ** 2
                    for clusters in (2, min(3, rows), min(8, rows), rows):
                        previous = np.zeros((clusters, 1), np.float32)
                        actual, totals, actual_distances, selected, record = core.lloyd_relocated_centroids(x, weights, labels, previous)
                        count = clusters - 1
                        np.testing.assert_array_equal(selected, np.argpartition(distances, -count)[:-count-1:-1].astype(np.int32))
                        np.testing.assert_array_equal(actual_distances.view(np.uint32), distances.view(np.uint32))
                        self.assertEqual(record["relocated_clusters"], count if np.max(distances) else 0)
            x = np.ones((6, 3), np.float32)
            previous = np.array([[2., 2., 2.], [1., 1., 1.], [3., 3., 3.]], np.float32)
            centers, totals, labels, shifts = np.empty_like(previous), np.empty(3, np.float32), np.empty(6, np.int32), np.empty(3, np.float32)
            lloyd_iter_chunked_dense(x, np.ones(6, np.float32), previous, centers, totals, labels, shifts, 12)
            actual, actual_totals, distances, selected, record = core.lloyd_relocated_centroids(x, np.ones(6, np.float32), labels, previous)
            np.testing.assert_array_equal(actual, centers)
            np.testing.assert_array_equal(actual_totals, totals)
            self.assertEqual(record["relocated_clusters"], 0)
            self.assertEqual(record["selection_calls"], 1)
            np.testing.assert_array_equal(labels, np.ones(6, np.int32))

    def test_unaligned_readonly_threads_and_domain_rejections(self):
        x = np.ndarray((42, 17), np.float32, buffer=bytearray(42 * 17 * 4 + 1), offset=1)
        x[:] = np.random.default_rng(2809).normal(size=x.shape)
        weights, labels, previous = np.ones(42, np.float32), np.zeros(42, np.int32), np.repeat(x[:1], 4, axis=0)
        with self.context() as core:
            expected = core.lloyd_relocated_centroids(x, weights, labels, previous)
            x.flags.writeable = weights.flags.writeable = labels.flags.writeable = previous.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in executor.map(lambda _: core.lloyd_relocated_centroids(x, weights, labels, previous), range(4)):
                    for actual, reference in zip(result[:4], expected[:4]):
                        np.testing.assert_array_equal(actual, reference)
            for value in (np.nan, np.inf, 2. ** 33):
                bad = x.copy(); bad[-1, -1] = value
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    core.lloyd_relocated_centroids(bad, weights, labels, previous)
            bad = weights.copy(); bad[-1] = -1
            with self.assertRaisesRegex(ValueError, "weight_domain_invalid"):
                core.lloyd_relocated_centroids(x, bad, labels, previous)
            bad = labels.copy(); bad[-1] = 4
            with self.assertRaisesRegex(ValueError, "label_domain_invalid"):
                core.lloyd_relocated_centroids(x, weights, bad, previous)
            with patch.object(core, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                core.lloyd_relocated_centroids(x, weights, labels, previous)
        with self.assertRaisesRegex(ValueError, "closed"):
            core.lloyd_relocated_centroids(x, weights, labels, previous)

    def test_composed_duplicate_initialization_fits_and_zero_distance_branch(self):
        x = np.random.default_rng(2811).normal(size=(42, 17)).astype(np.float32)
        with self.context() as core, self.gpu() as gpu:
            for weighted in (False, True):
                for clusters in (2, 4, 8):
                    for maximum in (1, 300):
                        weights = np.linspace(.5, 2., len(x), dtype=np.float32) if weighted else None
                        options = dict(n_clusters=clusters, n_init=1, init=np.repeat(x[:1], clusters, axis=0), random_state=17, max_iter=maximum, copy_x=False)
                        before, after = x.copy(), x.copy()
                        reference = KMeans(**options).fit(before, sample_weight=weights)
                        with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=(centroid_records := []), center_shift_records=[], inertia_records=[], relocation_records=(records := [])):
                            actual = KMeans(**options).fit(after, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), reference.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, reference.labels_)
                        np.testing.assert_array_equal(before.view(np.uint32), after.view(np.uint32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (reference.inertia_, reference.n_iter_, reference._tol))
                        self.assertTrue(any(row["relocated_clusters"] > 0 for row in records))
                        self.assertEqual(len(records), len(centroid_records))
                        self.assertTrue(all(row["shared_centroid_update"] and row["native_wall_ns"] == 0 for row in records))
            values = np.ones((6, 3), np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reference = KMeans(n_clusters=3, n_init=1, random_state=17).fit(values)
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], center_shift_records=[], relocation_records=(records := [])):
                    actual = KMeans(n_clusters=3, n_init=1, random_state=17).fit(values)
            np.testing.assert_array_equal(actual.cluster_centers_, reference.cluster_centers_)
            self.assertTrue(any(row["empty_clusters"] for row in records))
            self.assertTrue(all(row["relocated_clusters"] == 0 for row in records))

    def test_native_abi_under_address_and_undefined_sanitizers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "relocation-sanitizers"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-ffp-contract=off", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(root / "tools/baseline/native_kmeans_relocation.cpp"), str(root / "tests/fixtures/baseline/native_kmeans_relocation_abi.cpp"), "-o", str(binary)], check=True, capture_output=True, timeout=60)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("runtime error:", result.stderr)
            self.assertNotIn("ERROR: AddressSanitizer", result.stderr)

    def test_configuration_fault_restore_and_legacy_empty_rejection(self):
        original_fit, original_norms = KMeans.fit, cython_lloyd.row_norms
        x = np.random.default_rng(2813).normal(size=(42, 7)).astype(np.float32)
        with self.context() as core, self.gpu() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(gpu, [], relocation_records=[]):
                    pass
            with self.assertRaisesRegex(ValueError, "injected_relocation_failure"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], relocation_records=[]), patch.object(core, "lloyd_relocated_centroids", side_effect=ValueError("injected_relocation_failure")):
                    KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
            with patch("tools.baseline.native_kmeans_lloyd_norms.ARGPARTITION_SHA256", "changed"), self.assertRaisesRegex(ValueError, "numpy_reference_not_supported"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], relocation_records=[]):
                    pass
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(cython_lloyd.row_norms, original_norms)
            with self.assertRaisesRegex(ValueError, "empty_cluster_not_supported"):
                core.lloyd_centroids(x, np.ones(42, np.float32), np.zeros(42, np.int32), 3)


if __name__ == "__main__":
    unittest.main()

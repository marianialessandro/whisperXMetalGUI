from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import os
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
from sklearn.cluster._k_means_common import _inertia_dense
import sklearn.cluster._k_means_lloyd as cython_lloyd

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd_norms import LloydNormScope


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Pinned Apple OpenMP core required")
class NativeKMeansTwoChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.library, with_two_chunks=True)
        cls.double = Path(cls.directory.name) / "gpu.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def gpu(self):
        return NativeDoubleNormalization(self.double, self.double_identity["library_sha256"])

    def test_raw_boundaries_worker_grouping_labels_centroids_shifts_and_inertia(self):
        rng = np.random.default_rng(3003)
        with self.context() as core:
            for rows in (1, 256, 257, 511, 512):
                for features in (1, 3, 17, 63, 64, 65, 256, 1024):
                    x = rng.normal(size=(rows, features)).astype(np.float32)
                    for clusters in (1, 4):
                        if clusters > rows:
                            continue
                        for duplicate in (False, True):
                            previous = np.repeat(x[:1], clusters, axis=0) if duplicate else x[:clusters].copy()
                            norms, weights = np.einsum("ij,ij->i", previous, previous), np.linspace(.5, 2., rows, dtype=np.float32)
                            for threads in (1, 2, 12):
                                expected = (np.empty_like(previous), np.empty(clusters, np.float32), np.empty(rows, np.int32), np.empty(clusters, np.float32))
                                lloyd_iter_chunked_dense(x, weights, previous, *expected, threads)
                                labels, scores, assignment_record = core.lloyd_assignment(x, previous, norms, n_threads=threads)
                                centers, totals, distances, selected, record = core.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=threads)
                                shifts, unused = core.lloyd_center_shifts(centers, previous)
                                for actual, reference in zip((centers, totals, labels, shifts), expected):
                                    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
                                value, inertia_record = core.inertia_dense(x, weights, centers, labels, threads, two_chunks=True)
                                self.assertEqual(np.float32(value).view(np.uint32), np.float32(_inertia_dense(x, weights, centers, labels, threads)).view(np.uint32))
                                chunks = (rows + 255) // 256
                                self.assertEqual(assignment_record["native_blas_calls"], chunks)
                                self.assertEqual(record["actual_threads"], min(threads, chunks))
                                self.assertEqual(inertia_record["actual_threads"], threads)
                                self.assertFalse(record["thread_count_changed"])
            x = rng.normal(size=(512, 17)).astype(np.float32)
            labels, weights, previous = np.arange(512, dtype=np.int32) % 4, rng.uniform(.5, 2., 512).astype(np.float32), x[:4].copy()
            one = core.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=1)[0]
            two = core.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12)[0]
            self.assertNotEqual(one.tobytes(), two.tobytes())

    def test_extended_selection_ties_zero_and_rotated_tail(self):
        rng = np.random.default_rng(3005)
        with self.context() as core:
            for rows in (257, 300, 384, 511, 512):
                for values in (rng.integers(-4, 5, rows).astype(np.float32), np.zeros(rows, np.float32), np.roll(np.arange(rows, dtype=np.float32), rows // 2)):
                    for clusters in (2, 3, 8, 128, 256):
                        x, weights, labels, previous = values[:, None].copy(), np.ones(rows, np.float32), np.zeros(rows, np.int32), np.zeros((clusters, 1), np.float32)
                        actual, totals, distances, selected, record = core.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12)
                        expected = values ** 2
                        np.testing.assert_array_equal(distances.view(np.uint32), expected.view(np.uint32))
                        np.testing.assert_array_equal(selected, np.argpartition(expected, -(clusters - 1))[:-clusters:-1].astype(np.int32))
                        self.assertEqual(record["empty_clusters"], clusters - 1)

    def test_legacy_limits_new_limits_abi_closed_unaligned_and_concurrent(self):
        x = np.ndarray((512, 17), np.float32, buffer=bytearray(512 * 17 * 4 + 1), offset=1)
        x[:] = np.random.default_rng(3007).normal(size=x.shape)
        previous, weights, labels = x[:4].copy(), np.ones(512, np.float32), np.arange(512, dtype=np.int32) % 4
        norms = np.einsum("ij,ij->i", previous, previous)
        with self.context() as core:
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                core.lloyd_assignment(x, previous, norms)
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                core.lloyd_relocated_centroids(x, weights, labels, previous)
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                core.inertia_dense(x, weights, previous, labels, 12)
            for threads in (0, 3, 4, 7, True):
                with self.assertRaisesRegex(ValueError, "thread_count_not_supported"):
                    core.lloyd_assignment(x, previous, norms, n_threads=threads)
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                core.lloyd_assignment(np.ones((513, 17), np.float32), previous, norms, n_threads=12)
            with patch.object(core, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "two_chunks_assignment_abi_unavailable"):
                core.lloyd_assignment(x, previous, norms, n_threads=12)
            expected = core.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12)
            for value in (x, previous, weights, labels):
                value.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                for actual in executor.map(lambda _: core.lloyd_relocated_centroids(x, weights, labels, previous, n_threads=12), range(4)):
                    for left, right in zip(actual[:4], expected[:4]):
                        np.testing.assert_array_equal(left, right)
        with self.assertRaisesRegex(ValueError, "closed"):
            core.lloyd_assignment(x, previous, norms, n_threads=12)

    def test_composed_fits_large_inputs_original_initialization_and_duplicate_centers(self):
        with self.context() as core, self.gpu() as gpu:
            for rows in (257, 512):
                x = np.random.default_rng(rows + 3009).normal(size=(rows, 17)).astype(np.float32)
                for duplicate in (False, True):
                    for weighted in (False, True):
                        for maximum in (1, 300):
                            weights = np.linspace(.5, 2., rows, dtype=np.float32) if weighted else None
                            options = dict(n_clusters=4, n_init=1 if duplicate else 2, init=np.repeat(x[:1], 4, axis=0) if duplicate else "k-means++", max_iter=maximum, copy_x=False)
                            left_rng, right_rng = np.random.RandomState(17), np.random.RandomState(17)
                            before, after = x.copy(), x.copy()
                            expected = KMeans(**options, random_state=left_rng).fit(before, sample_weight=weights)
                            with native_kmeans_feature_mean(gpu, [], norm_records=[], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=(records := []), center_shift_records=[], inertia_records=(inertias := []), relocation_records=[], two_chunks=True):
                                actual = KMeans(**options, random_state=right_rng).fit(after, sample_weight=weights)
                            np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                            np.testing.assert_array_equal(actual.labels_, expected.labels_)
                            np.testing.assert_array_equal(after.view(np.uint32), before.view(np.uint32))
                            np.testing.assert_array_equal(left_rng.rand(32), right_rng.rand(32))
                            self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol, actual._n_threads), (expected.inertia_, expected.n_iter_, expected._tol, expected._n_threads))
                            self.assertTrue(all(row["chunk_count"] == 2 and row["actual_threads"] == 2 and row["native_blas_calls"] == 2 for row in records))
                            self.assertTrue(all(row["actual_threads"] == actual._n_threads for row in inertias))

    def test_abi_under_sanitizers_and_limited_team_preserves_outputs(self):
        from tools.baseline.native_kmeans_inertia import runtime_path
        root, runtime = Path(__file__).resolve().parents[2], runtime_path()
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "two-chunks-sanitizers"
            command = ["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(root / "tools/baseline/native_kmeans_two_chunks.cpp"), str(root / "tools/baseline/native_kmeans_assignment.cpp"), str(root / "tools/baseline/native_kmeans_inertia.cpp"), str(root / "tests/fixtures/baseline/native_kmeans_two_chunks_abi.cpp"), str(runtime), "-Wl,-rpath," + str(runtime.parent), "-framework", "Accelerate", "-o", str(binary)]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            install_id = subprocess.run(["otool", "-D", str(runtime)], check=True, capture_output=True, text=True).stdout.splitlines()[1].strip()
            subprocess.run(["install_name_tool", "-change", install_id, "@rpath/libomp.dylib", str(binary)], check=True, capture_output=True)
            subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(binary)], check=True, capture_output=True)
            for args, environment in (([], os.environ.copy()), (["limited"], dict(os.environ, OMP_THREAD_LIMIT="1"))):
                result = subprocess.run([str(binary), *args], env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("runtime error:", result.stderr)
                self.assertNotIn("ERROR: AddressSanitizer", result.stderr)

    def test_configuration_fault_and_owner_thread_restoration(self):
        original_fit, original_norms = KMeans.fit, cython_lloyd.row_norms
        x = np.random.default_rng(3011).normal(size=(512, 17)).astype(np.float32)
        with self.context() as core, self.gpu() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(gpu, [], two_chunks=True):
                    pass
            with patch.object(core, "library", SimpleNamespace(wxkmeans_lloyd_assignment=object(), wxkmeans_lloyd_centroids=object(), wxkmeans_lloyd_center_shifts=object(), wxkmeans_lloyd_relocated_centroids=object())), self.assertRaisesRegex(ValueError, "two_chunks_abi_unavailable"):
                LloydNormScope(gpu, [], core, [], [], [], [], two_chunks=True)
            with self.assertRaisesRegex(ValueError, "injected_two_chunks_failure"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], center_shift_records=[], inertia_records=[], relocation_records=[], two_chunks=True), patch.object(core, "lloyd_relocated_centroids", side_effect=ValueError("injected_two_chunks_failure")):
                    KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(cython_lloyd.row_norms, original_norms)
            with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=(records := []), center_shift_records=[], inertia_records=[], relocation_records=[], two_chunks=True):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17).fit, x).result()
                KMeans(n_clusters=3, n_init=1, random_state=17, algorithm="elkan").fit(x)
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                self.assertEqual(records, [])
                with self.assertRaisesRegex(ValueError, "two_chunks_sample_limit"):
                    KMeans(n_clusters=3, n_init=1, random_state=17).fit(np.ones((513, 17), np.float32))
        self.assertIs(cython_lloyd.row_norms, original_norms)


if __name__ == "__main__":
    unittest.main()

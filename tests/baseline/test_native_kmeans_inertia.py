from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import ctypes as ct
import hashlib
import inspect
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_common import _inertia_dense
import sklearn.cluster._kmeans as kmeans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd import compile_native_lloyd
from tools.baseline.native_kmeans_inertia import inertia_value, runtime_path, validate_reference


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Pinned Apple OpenMP core required")
class NativeKMeansInertiaTests(unittest.TestCase):
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

    def test_supported_thread_counts_static_reduction_and_single_label_bits(self):
        rng = np.random.default_rng(2703)
        with self.context() as core:
            for rows in (1, 7, 42, 129, 256):
                for features in (1, 3, 4, 7, 16, 17, 63, 64, 65, 256, 1024):
                    x, centers = rng.normal(size=(rows, features)).astype(np.float32), rng.normal(size=(4, features)).astype(np.float32)
                    weights, labels = rng.uniform(.5, 2., rows).astype(np.float32), rng.integers(0, 4, rows, dtype=np.int32)
                    for threads in (1, 2, 12):
                        for single in (-1, 0, 3):
                            expected = np.float32(_inertia_dense(x, weights, centers, labels, threads, single))
                            for unused in range(3):
                                actual, record = core.inertia_dense(x, weights, centers, labels, threads, single)
                                self.assertIs(type(actual), float)
                                self.assertEqual(np.float32(actual).view(np.uint32), expected.view(np.uint32))
                            self.assertEqual(record["actual_threads"], threads)
                            self.assertFalse(record["thread_count_changed"])
                            self.assertEqual(record["execution_device"], "cpu")
                            self.assertEqual(record["command_buffers_per_run"], 0)
                            self.assertIsNone(record["computation_gpu_seconds"])

    def test_normal_bounds_subnormals_zero_unaligned_readonly_and_concurrent_calls(self):
        with self.context() as core:
            for magnitude in (0., np.nextafter(np.float32(0), np.float32(1)), 2. ** -60, 2. ** 32):
                x, centers, weights, labels = np.full((42, 17), magnitude, np.float32), np.full((2, 17), -magnitude, np.float32), np.full(42, 2. ** 32, np.float32), np.zeros(42, np.int32)
                expected = np.float32(_inertia_dense(x, weights, centers, labels, 12))
                actual, unused = core.inertia_dense(x, weights, centers, labels, 12)
                self.assertEqual(np.float32(actual).view(np.uint32), expected.view(np.uint32))
            x = np.ndarray((42, 17), np.float32, buffer=bytearray(42 * 17 * 4 + 1), offset=1)
            x[:] = np.random.default_rng(2705).normal(size=x.shape)
            centers, weights, labels = x[:2].copy(), np.ones(42, np.float32), np.arange(42, dtype=np.int32) % 2
            expected = np.float32(_inertia_dense(x, weights, centers, labels, 12))
            for value in (x, centers, weights, labels):
                value.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                for actual, unused in executor.map(lambda _: core.inertia_dense(x, weights, centers, labels, 12), range(6)):
                    self.assertEqual(np.float32(actual).view(np.uint32), expected.view(np.uint32))

    def test_domain_shape_thread_runtime_policy_closed_and_abi_guards(self):
        x, weights, centers, labels = np.ones((7, 3), np.float32), np.ones(7, np.float32), np.ones((2, 3), np.float32), np.zeros(7, np.int32)
        with self.context() as core:
            for threads in (0, 3, 4, 7, 64, True, np.int64(12)):
                with self.assertRaisesRegex(ValueError, "thread_count_not_supported"):
                    core.inertia_dense(x, weights, centers, labels, threads)
            for single in (-2, 2, True):
                with self.assertRaisesRegex(ValueError, "single_label_invalid"):
                    core.inertia_dense(x, weights, centers, labels, 12, single)
            for name in ("KMP_FORCE_REDUCTION", "KMP_DETERMINISTIC_REDUCTION"):
                with patch.dict(os.environ, {name: "true"}), self.assertRaisesRegex(ValueError, "runtime_policy_not_supported"):
                    core.inertia_dense(x, weights, centers, labels, 12)
            for value in (np.nan, np.inf, 2. ** 33):
                bad = x.copy(); bad[-1, -1] = value
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    core.inertia_dense(bad, weights, centers, labels, 12)
            for value in (-1., 2. ** -33, np.nan):
                bad = weights.copy(); bad[-1] = value
                with self.assertRaisesRegex(ValueError, "weight_domain_invalid"):
                    core.inertia_dense(x, bad, centers, labels, 12)
            bad = labels.copy(); bad[-1] = 2
            with self.assertRaisesRegex(ValueError, "label_domain_invalid"):
                core.inertia_dense(x, weights, centers, bad, 12)
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                core.inertia_dense(x.astype(np.float64), weights, centers, labels, 12)
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                core.inertia_dense(np.ones((257, 3), np.float32), np.ones(257, np.float32), centers, np.zeros(257, np.int32), 12)
            with patch.object(core, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                core.inertia_dense(x, weights, centers, labels, 12)
        with self.assertRaisesRegex(ValueError, "closed"):
            core.inertia_dense(x, weights, centers, labels, 12)

    def test_private_helper_replaces_only_dense_inertia_and_preserves_type(self):
        x = np.random.default_rng(2707).normal(size=(42, 17)).astype(np.float32)
        weights, centers = np.ones(42, np.float32), x[:3].copy()
        original = kmeans._inertia_dense
        expected = kmeans._kmeans_single_lloyd(x, weights, centers.copy(), max_iter=1, n_threads=12)
        with self.context() as core:
            callback = lambda X, weights, centers, labels, n_threads, single_label=-1: inertia_value(core, records, X, weights, centers, labels, n_threads, single_label)
            records = []
            candidate = compile_native_lloyd(kmeans._kmeans_single_lloyd, inertia=callback)
            self.assertIs(inspect.unwrap(candidate).__globals__["_inertia_dense"], callback)
            with patch.object(kmeans, "_inertia_dense", side_effect=AssertionError("original inertia invoked")):
                actual = candidate(x, weights, centers.copy(), max_iter=1, n_threads=12)
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)
            self.assertIs(type(actual[1]), type(expected[1]))
            self.assertEqual(len(records), 1)
            expected_log, actual_log = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(expected_log):
                expected_verbose = kmeans._kmeans_single_lloyd(x, weights, centers.copy(), max_iter=1, n_threads=12, verbose=True)
            with contextlib.redirect_stdout(actual_log):
                actual_verbose = candidate(x, weights, centers.copy(), max_iter=1, n_threads=12, verbose=True)
            self.assertEqual(actual_log.getvalue(), expected_log.getvalue())
            for left, right in zip(actual_verbose, expected_verbose):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(len(records), 3)
        self.assertIs(kmeans._inertia_dense, original)
        with patch("tools.baseline.native_kmeans_inertia.COMMON_BINARY_SHA256", "changed"), self.assertRaisesRegex(ValueError, "reference_not_supported"):
            validate_reference(original)
        with patch("tools.baseline.native_kmeans_inertia.OPENMP_SHA256", "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
            runtime_path()

    def test_composed_fits_preserve_attributes_input_and_rng_with_original_threads(self):
        x = np.random.default_rng(2709).normal(size=(42, 17)).astype(np.float32)
        with self.context() as core, self.gpu() as gpu:
            for weighted in (False, True):
                for clusters in (1, 2, 4, 8):
                    for maximum in (1, 300):
                        weights = np.linspace(.5, 2., len(x), dtype=np.float32) if weighted else None
                        left_rng, right_rng = np.random.RandomState(17), np.random.RandomState(17)
                        options = dict(n_clusters=clusters, n_init=3, max_iter=maximum, copy_x=False, tol=0. if maximum == 1 else 1e-4)
                        before, after = x.copy(), x.copy()
                        expected = KMeans(**options, random_state=left_rng).fit(before, sample_weight=weights)
                        with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], center_shift_records=[], inertia_records=(records := [])):
                            actual = KMeans(**options, random_state=right_rng).fit(after, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(before.view(np.uint32), after.view(np.uint32))
                        np.testing.assert_array_equal(left_rng.rand(32), right_rng.rand(32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol, actual._n_threads), (expected.inertia_, expected.n_iter_, expected._tol, expected._n_threads))
                        self.assertEqual(len(records), 3)
                        self.assertTrue(all(row["requested_threads"] == actual._n_threads for row in records))

    def test_native_abi_sanitizers_and_actual_team_guard(self):
        root = Path(__file__).resolve().parents[2]
        runtime = runtime_path()
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "inertia-sanitizers"
            command = ["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(root / "tools/baseline/native_kmeans_inertia.cpp"), str(root / "tests/fixtures/baseline/native_kmeans_inertia_abi.cpp"), str(runtime), "-Wl,-rpath," + str(runtime.parent), "-o", str(binary)]
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            install_id = subprocess.run(["otool", "-D", str(runtime)], check=True, capture_output=True, text=True).stdout.splitlines()[1].strip()
            subprocess.run(["install_name_tool", "-change", install_id, "@rpath/libomp.dylib", str(binary)], check=True, capture_output=True)
            subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(binary)], check=True, capture_output=True)
            for arguments, environment in (([], os.environ.copy()), (["limited"], dict(os.environ, OMP_THREAD_LIMIT="1"))):
                result = subprocess.run([str(binary), *arguments], env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("runtime error:", result.stderr)
                self.assertNotIn("ERROR: AddressSanitizer", result.stderr)

    def test_configuration_bypass_fault_restore_and_opt_in_build(self):
        x = np.random.default_rng(2711).normal(size=(42, 7)).astype(np.float32)
        original_fit = KMeans.fit
        with self.context() as core, self.gpu() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(gpu, [], inertia_records=[]):
                    pass
            with self.assertRaisesRegex(ValueError, "injected_inertia_failure"):
                with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], center_shift_records=[], inertia_records=[]), patch.object(core, "inertia_dense", side_effect=ValueError("injected_inertia_failure")):
                    KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
            self.assertIs(KMeans.fit, original_fit)
            with native_kmeans_feature_mean(gpu, [], lloyd_norm_records=[], potential_context=core, potential_records=[], assignment_records=[], centroid_records=[], center_shift_records=[], inertia_records=(records := [])):
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                KMeans(n_clusters=3, n_init=1, random_state=17, algorithm="elkan").fit(x)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17).fit, x).result()
                self.assertEqual(records, [])
        plain = Path(self.directory.name) / "plain.dylib"
        identity = build_library(plain)
        self.assertNotIn("openmp_runtime_sha256", identity)
        with NativeKMeansPotentials(plain, identity["library_sha256"]) as core:
            self.assertIsNone(getattr(core.library, "wxkmeans_inertia_dense", None))
        self.assertIn("openmp_runtime_sha256", self.identity)


if __name__ == "__main__":
    unittest.main()

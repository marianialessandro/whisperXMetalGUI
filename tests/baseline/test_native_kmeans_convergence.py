from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import inspect
import io
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

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd import compile_native_lloyd


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.cpu = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.cpu)
        cls.gpu = Path(cls.directory.name) / "gpu.dylib"
        cls.gpu_identity = build_double(cls.gpu)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.cpu, self.identity["library_sha256"])

    def test_raw_numpy_pairwise_and_scalar_promotion_at_thresholds(self):
        rng = np.random.default_rng(54101)
        with self.context() as cpu:
            for count in (1, 2, 7, 8, 63, 64, 65, 127, 128, 129, 256, 512, 1024):
                for values in (np.zeros(count, np.float32), np.full(count, 2. ** -60, np.float32), rng.uniform(0, 2, count).astype(np.float32)):
                    expected = (values ** 2).sum()
                    middle = float(expected)
                    for tolerance in (0., float(np.nextafter(np.float64(middle), -np.inf)) if middle > 0 else 0., middle, float(np.nextafter(np.float64(middle), np.inf))):
                        for kind in (float, np.float32, np.float64):
                            tol = kind(tolerance)
                            actual, converged, record = cpu.lloyd_convergence(values, tol)
                            self.assertEqual(actual.view(np.uint32), expected.view(np.uint32))
                            self.assertEqual(converged, bool(expected <= tol))
                            self.assertIs(type(actual), np.float32)
                            self.assertIs(type(converged), bool)
                            self.assertEqual(record["comparison_precision"], "binary64" if kind is np.float64 else "binary32")
            values = np.ones(1, np.float32)
            threshold = np.nextafter(np.float64(1), np.float64(0))
            self.assertTrue(cpu.lloyd_convergence(values, float(threshold))[1])
            self.assertFalse(cpu.lloyd_convergence(values, threshold)[1])
            self.assertTrue(cpu.lloyd_convergence(values, np.float32(threshold))[1])

    def test_raw_errors_unaligned_readonly_threads_and_closed(self):
        values = np.ones(17, np.float32)
        unaligned = np.ndarray(values.shape, dtype=np.float32, buffer=bytearray(values.nbytes + 1), offset=1)
        unaligned[:] = values
        unaligned.flags.writeable = False
        with self.context() as cpu:
            with ThreadPoolExecutor(max_workers=2) as executor:
                result = list(executor.map(lambda _: cpu.lloyd_convergence(unaligned, 17.)[:2], range(4)))
            self.assertEqual(result, [(np.float32(17), True)] * 4)
            for value in (True, 0, None, -1., np.nan, np.inf, 2. ** 33, np.float16(1)):
                with self.assertRaisesRegex(ValueError, "tolerance_invalid"):
                    cpu.lloyd_convergence(values, value)
            for value in (np.ones(17), values[::-1], values.reshape(1, -1)):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    cpu.lloyd_convergence(value, 1.)
            for count in (0, 1025):
                with self.assertRaisesRegex(ValueError, "count_invalid"):
                    cpu.lloyd_convergence(np.ones(count, np.float32), 1.)
            values[-1] = np.nan
            with self.assertRaisesRegex(ValueError, "value_domain"):
                cpu.lloyd_convergence(values, 1.)
            with patch.object(cpu, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                cpu.lloyd_convergence(np.ones(1, np.float32), 1.)
        with self.assertRaisesRegex(ValueError, "closed"):
            cpu.lloyd_convergence(np.ones(1, np.float32), 1.)

    def test_original_helper_strict_tolerance_maxiter_verbose_and_final_assignment(self):
        x = np.random.default_rng(54103).normal(size=(42, 17)).astype(np.float32)
        weights = np.linspace(.125, 2, 42, dtype=np.float32)
        with self.context() as cpu:
            calls = []
            def convergence(values, tolerance):
                total, within, record = cpu.lloyd_convergence(values, tolerance)
                calls.append(record)
                return total, within
            candidate = compile_native_lloyd(kmeans._kmeans_single_lloyd, convergence=convergence)
            for maximum in (1, 300):
                for tolerance in (0., np.float32(1.e-4), np.float64(1.e4)):
                    for verbose in (False, True):
                        first, second = io.StringIO(), io.StringIO()
                        with contextlib.redirect_stdout(first):
                            expected = kmeans._kmeans_single_lloyd(x, weights, x[:4].copy(), max_iter=maximum, tol=tolerance, verbose=verbose, n_threads=1)
                        with contextlib.redirect_stdout(second):
                            actual = candidate(x, weights, x[:4].copy(), max_iter=maximum, tol=tolerance, verbose=verbose, n_threads=1)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_array_equal(actual[2].view(np.uint32), expected[2].view(np.uint32))
                        self.assertEqual((actual[1], actual[3]), (expected[1], expected[3]))
                        self.assertEqual(second.getvalue(), first.getvalue())
            self.assertTrue(calls)

    def test_fit_composition_shared_reduction_sparse_elkan_thread_and_restore(self):
        original = KMeans.fit
        x = np.random.default_rng(54107).normal(size=(42, 17)).astype(np.float32)
        with self.context() as cpu, NativeDoubleNormalization(self.gpu, self.gpu_identity["library_sha256"]) as gpu:
            def scope(convergence, shifts):
                return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], lloyd_shift_records=shifts, convergence_records=convergence)
            expected = KMeans(n_clusters=4, n_init=3, random_state=17).fit(x.copy())
            with scope(records := [], shifts := []):
                actual = KMeans(n_clusters=4, n_init=3, random_state=17).fit(x.copy())
            np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
            np.testing.assert_array_equal(actual.labels_, expected.labels_)
            self.assertEqual((actual.inertia_, actual.n_iter_), (expected.inertia_, expected.n_iter_))
            self.assertTrue(records)
            self.assertEqual(len(shifts), len(records))
            self.assertTrue(all(r["native_wall_ns"] == r["host_result_bytes"] == 0 and r["shared_convergence_index"] == i for i, r in enumerate(shifts)))
            with scope(skipped := [], []):
                KMeans(n_clusters=4, n_init=1, random_state=17, algorithm="elkan").fit(x)
                KMeans(n_clusters=4, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=4, n_init=1, random_state=17).fit, x).result()
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope([], []), patch.object(cpu, "lloyd_convergence", side_effect=ValueError("injected")):
                KMeans(n_clusters=4, n_init=1, random_state=17).fit(x)
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], convergence_records=[]):
                pass
        self.assertIs(KMeans.fit, original)

    def test_source_comparison_drift_is_rejected(self):
        original = kmeans._kmeans_single_lloyd
        inner = inspect.unwrap(original)
        getsource = inspect.getsource
        source = getsource(inner).replace("center_shift_tot <= tol", "center_shift_tot < tol")
        with patch("tools.baseline.native_kmeans_lloyd.inspect.getsource", side_effect=lambda fn: source if fn is inner else getsource(fn)), patch("tools.baseline.native_kmeans_lloyd.LLOYD_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract"):
            compile_native_lloyd(original, convergence=object())

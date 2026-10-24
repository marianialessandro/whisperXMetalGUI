from concurrent.futures import ThreadPoolExecutor
import contextlib
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
from tools.baseline.native_kmeans_lloyd import compile_native_lloyd
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansDecisionTests(unittest.TestCase):
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

    def test_raw_strict_short_circuits_unusable_shifts_and_tolerance(self):
        with self.context() as cpu:
            for rows in (1, 42, 256, 257, 512, 4096):
                labels = np.arange(rows, dtype=np.int32)
                for shifts, tolerance in ((None, None), (np.array([np.nan], np.float32), -1.), (np.ones(1025), object())):
                    strict, total, within, record = cpu.lloyd_decision(labels, labels.copy(), shifts, tolerance)
                    self.assertTrue(strict)
                    self.assertEqual(total.view(np.uint32), 0)
                    self.assertFalse(within)
                    self.assertFalse(record["tolerance_evaluated"])
                    self.assertIsNone(record["comparison_precision"])
                    self.assertEqual(record["host_input_snapshot_bytes"], labels.nbytes * 2)
            labels = np.array([0, 1], np.int32)
            for shifts, tolerance in ((None, None), (np.array([np.nan], np.float32), 1.)):
                with self.assertRaises(ValueError):
                    cpu.lloyd_decision(labels, labels[::-1].copy(), shifts, tolerance)

    def test_raw_non_strict_matches_numpy_and_promotion_validation(self):
        rng = np.random.default_rng(55101)
        with self.context() as cpu:
            for rows in (1, 42, 257, 512, 4096):
                labels = rng.integers(0, 8, rows, dtype=np.int32)
                previous = labels.copy()
                previous[-1] = -1
                for clusters in (1, 7, 64, 65, 128, 129, 256, 1024):
                    shifts = rng.uniform(0, 2, clusters).astype(np.float32)
                    total = (shifts ** 2).sum()
                    for kind in (float, np.float32, np.float64):
                        for tolerance in (kind(0), kind(float(total)), kind(np.nextafter(np.float64(total), np.float64(0)))):
                            strict, actual, within, record = cpu.lloyd_decision(labels, previous, shifts, tolerance)
                            self.assertFalse(strict)
                            self.assertEqual(actual.view(np.uint32), total.view(np.uint32))
                            self.assertEqual(within, bool(total <= tolerance))
                            self.assertTrue(record["tolerance_evaluated"])
            labels = np.array([0, 1], np.int32)
            unaligned = np.ndarray(labels.shape, dtype=np.int32, buffer=bytearray(labels.nbytes + 1), offset=1)
            unaligned[:] = labels
            unaligned.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: cpu.lloyd_decision(unaligned, labels, None, None)[0], range(4)))
            self.assertEqual(results, [True] * 4)
            for values in (labels.astype(np.int64), labels.reshape(1, 2), labels[::-1]):
                with self.assertRaisesRegex(ValueError, "label_format"):
                    cpu.lloyd_decision(values, labels)
            for values, previous in ((np.zeros(0, np.int32), np.zeros(0, np.int32)), (np.zeros(4097, np.int32), np.zeros(4097, np.int32)), (labels, labels[:1])):
                with self.assertRaisesRegex(ValueError, "label_count"):
                    cpu.lloyd_decision(values, previous)
            with self.assertRaisesRegex(ValueError, "value_domain"):
                cpu.lloyd_decision(np.array([4096], np.int32), np.array([4096], np.int32))
            with patch.object(cpu, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                cpu.lloyd_decision(labels, labels)
        with self.assertRaisesRegex(ValueError, "closed"):
            cpu.lloyd_decision(labels, labels)

    def test_original_loop_exact_verbose_branches_and_final_assignment(self):
        x = np.random.default_rng(55103).normal(size=(42, 17)).astype(np.float32)
        weights = np.linspace(.125, 2, 42, dtype=np.float32)
        with self.context() as cpu:
            calls = []
            def decide(labels, previous, shifts, tolerance):
                strict, total, within, record = cpu.lloyd_decision(labels, previous, shifts, tolerance)
                calls.append(record)
                return strict, total, within
            candidate = compile_native_lloyd(kmeans._kmeans_single_lloyd, decision=decide)
            for maximum in (1, 300):
                for tolerance in (0., np.float32(1.e-4), np.float64(1.e4)):
                    first, second = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(first):
                        expected = kmeans._kmeans_single_lloyd(x, weights, x[:4].copy(), max_iter=maximum, tol=tolerance, verbose=True, n_threads=1)
                    with contextlib.redirect_stdout(second):
                        actual = candidate(x, weights, x[:4].copy(), max_iter=maximum, tol=tolerance, verbose=True, n_threads=1)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[2].view(np.uint32), expected[2].view(np.uint32))
                    self.assertEqual((actual[1], actual[3]), (expected[1], expected[3]))
                    self.assertEqual(second.getvalue(), first.getvalue())
            self.assertTrue(any(r["strict_convergence"] for r in calls))
            self.assertTrue(any(r["converged_by_tolerance"] for r in calls))
            self.assertTrue(any(not r["strict_convergence"] and not r["converged_by_tolerance"] for r in calls))

    def test_fit_shared_records_skip_routes_failure_and_configuration(self):
        original = KMeans.fit
        x = np.random.default_rng(55107).normal(size=(42, 17)).astype(np.float32)
        with self.context() as cpu, NativeDoubleNormalization(self.gpu, self.gpu_identity["library_sha256"]) as gpu:
            def scope(decisions, labels, convergence, shifts):
                return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], lloyd_shift_records=shifts, lloyd_label_records=labels, convergence_records=convergence, decision_records=decisions)
            expected = KMeans(n_clusters=4, n_init=3, random_state=17).fit(x.copy())
            with scope(decisions := [], labels := [], convergence := [], shifts := []):
                actual = KMeans(n_clusters=4, n_init=3, random_state=17).fit(x.copy())
            np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
            np.testing.assert_array_equal(actual.labels_, expected.labels_)
            self.assertEqual((actual.inertia_, actual.n_iter_), (expected.inertia_, expected.n_iter_))
            self.assertEqual(len(labels), len(decisions))
            self.assertEqual(len(convergence), sum(not r["strict_convergence"] for r in decisions))
            self.assertEqual(len(shifts), len(convergence))
            self.assertTrue(all(r["native_wall_ns"] == r["host_input_snapshot_bytes"] == r["host_result_bytes"] == 0 for r in labels + convergence + shifts))
            with scope(skipped := [], [], [], []):
                KMeans(n_clusters=4, n_init=1, random_state=17, algorithm="elkan").fit(x)
                KMeans(n_clusters=4, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=4, n_init=1, random_state=17).fit, x).result()
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope([], [], [], []), patch.object(cpu, "lloyd_decision", side_effect=ValueError("injected")):
                KMeans(n_clusters=4, n_init=1, random_state=17).fit(x)
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], decision_records=[]):
                pass
        self.assertIs(KMeans.fit, original)

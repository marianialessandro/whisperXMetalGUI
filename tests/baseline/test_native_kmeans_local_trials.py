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
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_prefix import compile_native_prefix


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansLocalTrialTests(unittest.TestCase):
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

    def test_exhaustive_original_numpy_default_domain(self):
        with self.context() as cpu:
            for clusters in range(1, 4097):
                actual, record = cpu.local_trials(clusters)
                self.assertEqual(actual, 2 + int(np.log(clusters)))
                self.assertIs(type(actual), int)
                self.assertEqual(record["local_trials"], actual)
                self.assertEqual(record["execution_device"], "cpu")
                self.assertEqual(record["gpu_kernels"], [])
                self.assertEqual(record["native_blas_calls"], 0)
                self.assertFalse(record["python_numeric_logarithm"])
            for value in (0, 4097, -1, 2**32, True, 1., np.int64(2), None):
                with self.assertRaisesRegex(ValueError, "cluster_count_invalid"):
                    cpu.local_trials(value)
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(list(executor.map(lambda _: cpu.local_trials(4096)[0], range(4))), [10] * 4)
            with patch.object(cpu, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                cpu.local_trials(1)
        with self.assertRaisesRegex(ValueError, "closed"):
            cpu.local_trials(1)

    def test_original_helper_explicit_trials_rng_and_log_displacement(self):
        x = np.random.default_rng(52101).normal(size=(42, 17)).astype(np.float32)
        norms = np.einsum("ij,ij->i", x, x)
        weights = np.linspace(.125, 2, 42, dtype=np.float32)
        with self.context() as cpu:
            records = []
            def trials(clusters):
                output, record = cpu.local_trials(clusters)
                records.append(record)
                return output
            candidate = compile_native_prefix(kmeans._kmeans_plusplus, None, local_trials=trials)
            for clusters in (1, 2, 3, 8, 21):
                for explicit in (None, 1, 3, 11):
                    rng1, rng2 = np.random.RandomState(17), np.random.RandomState(17)
                    expected = kmeans._kmeans_plusplus(x, clusters, norms, weights, rng1, n_local_trials=explicit)
                    before = len(records)
                    with patch.object(np, "log", side_effect=AssertionError("Python logarithm used")):
                        actual = candidate(x, clusters, norms, weights, rng2, n_local_trials=explicit)
                    for left, right in zip(actual, expected):
                        np.testing.assert_array_equal(left, right)
                    np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
                    self.assertEqual(len(records) - before, int(explicit is None))
            for expression, error in (("2 + int(np.log(n_clusters))", "expression_contract"), ("n_local_trials is None", "default_guard")):
                getsource = inspect.getsource
                source = getsource(kmeans._kmeans_plusplus).replace(expression, "1" if expression.startswith("2") else "True")
                with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=lambda fn: source if fn is kmeans._kmeans_plusplus else getsource(fn)), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, error):
                    compile_native_prefix(kmeans._kmeans_plusplus, None, local_trials=object())

    def test_fit_composition_sparse_thread_fault_and_configuration(self):
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as cpu, NativeDoubleNormalization(self.gpu, self.gpu_identity["library_sha256"]) as gpu:
            def scope(records):
                return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], local_trial_records=records)
            for rows in (42, 257, 512):
                x = np.random.default_rng(rows).normal(size=(rows, 17)).astype(np.float32)
                for clusters in (1, 3, 8):
                    rng1, rng2 = np.random.RandomState(17), np.random.RandomState(17)
                    left, right = x.copy(), x.copy()
                    weights = np.linspace(.125, 2, rows, dtype=np.float32)
                    expected = KMeans(n_clusters=clusters, n_init=3, random_state=rng1, copy_x=False).fit(left, sample_weight=weights)
                    records = []
                    with scope(records):
                        actual = KMeans(n_clusters=clusters, n_init=3, random_state=rng2, copy_x=False).fit(right, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
                    np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._n_threads), (expected.inertia_, expected.n_iter_, expected._n_threads))
                    self.assertEqual(len(records), 3)
            with scope(skipped := []):
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                KMeans(n_clusters=3, n_init=1, init=x[:3]).fit(x)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17).fit, x).result()
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope([]), patch.object(cpu, "local_trials", side_effect=ValueError("injected")):
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], local_trial_records=[]):
                pass
            with patch.object(cpu, "library", SimpleNamespace(wxkmeans_weighted_potentials=object())), self.assertRaisesRegex(ValueError, "local_trials_abi_unavailable"), scope([]):
                pass
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

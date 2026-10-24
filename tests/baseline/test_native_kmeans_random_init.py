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

from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_random_init import compile_random_initializer, random_probabilities


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Metal required")
class NativeKMeansRandomInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "gpu.dylib"
        cls.identity = build_library(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeDoubleNormalization(self.library, self.identity["library_sha256"])

    def test_original_helper_probabilities_seeds_rng_subsampling_and_overrides(self):
        original = KMeans._init_centroids
        with self.context() as gpu:
            records = []
            candidate = compile_random_initializer(original, lambda weights: random_probabilities(gpu, records, weights))
            for rows in (42, 257, 512):
                x = np.random.default_rng(rows).normal(size=(rows, 17)).astype(np.float32)
                norms = np.einsum("ij,ij->i", x, x)
                for pattern in ("uniform", "linear", "zero"):
                    weights = np.ones(rows, np.float32) if pattern == "uniform" else np.linspace(.125, 2, rows, dtype=np.float32)
                    if pattern == "zero":
                        weights[::3] = 0
                    for clusters in (1, 2, 8):
                        for init_size in (None, 32):
                            a, b = np.random.RandomState(17), np.random.RandomState(17)
                            instance = KMeans(n_clusters=clusters, init="random")
                            expected = original(instance, x, norms, "random", a, weights, init_size=init_size)
                            actual = candidate(instance, x, norms, "random", b, weights, init_size=init_size)
                            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                            np.testing.assert_array_equal(a.uniform(size=32), b.uniform(size=32))
                            self.assertEqual(records[-1]["count"], rows if init_size is None else 32)
                a, b = np.random.RandomState(17), np.random.RandomState(17)
                instance = KMeans(n_clusters=8, init="random")
                expected = original(instance, x, norms, "random", a, weights, n_centroids=3)
                actual = candidate(instance, x, norms, "random", b, weights, n_centroids=3)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(a.uniform(size=32), b.uniform(size=32))
            self.assertTrue(all(not r["rng_replaced"] and not r["choice_replace"] and r["computation_gpu_seconds"] > 0 for r in records))

    def test_dense_random_fits_exact_with_weights_copy_modes_and_algorithms(self):
        with self.context() as gpu:
            for rows in (42, 257, 512):
                for algorithm in ("lloyd", "elkan"):
                    for copy_x in (False, True):
                        x = np.random.default_rng(rows).normal(size=(rows, 17)).astype(np.float32)
                        weights = np.linspace(.125, 2, rows, dtype=np.float32)
                        weights[::3] = 0
                        a, b = np.random.RandomState(17), np.random.RandomState(17)
                        left, right = x.copy(), x.copy()
                        options = dict(n_clusters=4, n_init=3, init="random", algorithm=algorithm, copy_x=copy_x)
                        expected = KMeans(**options, random_state=a).fit(left, sample_weight=weights)
                        with native_kmeans_feature_mean(gpu, [], random_probability_records=(records := [])):
                            actual = KMeans(**options, random_state=b).fit(right, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
                        np.testing.assert_array_equal(a.uniform(size=32), b.uniform(size=32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._n_threads), (expected.inertia_, expected.n_iter_, expected._n_threads))
                        self.assertEqual(len(records), 3)

    def test_skip_nonrandom_sparse_explicit_callable_and_nonowner(self):
        x = np.random.default_rng(53101).normal(size=(42, 17)).astype(np.float32)
        original = KMeans._init_centroids
        class CustomKMeans(KMeans):
            def _init_centroids(self, X, *args, **kwargs):
                return X[:self.n_clusters].copy()
        with self.context() as gpu, native_kmeans_feature_mean(gpu, [], random_probability_records=(records := [])):
            for init in ("k-means++", x[:3].copy(), lambda X, clusters, random_state: X[:clusters].copy()):
                KMeans(n_clusters=3, n_init=1, random_state=17, init=init).fit(x)
            KMeans(n_clusters=3, n_init=1, random_state=17, init="random").fit(sparse.csr_matrix(x))
            CustomKMeans(n_clusters=3, n_init=1, random_state=17, init="random").fit(x)
            instance = KMeans(n_clusters=3, n_init=1, random_state=17, init="random")
            instance._init_centroids = lambda X, **kwargs: X[:3].copy()
            instance.fit(x)
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17, init="random").fit, x).result()
            self.assertEqual(records, [])
            self.assertIs(KMeans._init_centroids, original)

    def test_source_expression_choice_drift_config_fault_restoration(self):
        original_fit, original_init = KMeans.fit, KMeans._init_centroids
        getsource = inspect.getsource
        for before, after, error in (("sample_weight / sample_weight.sum()", "sample_weight.copy() / sample_weight.sum()", "expression_changed"), ("replace=False", "replace=True", "choice_contract")):
            source = getsource(original_init).replace(before, after)
            with patch("tools.baseline.native_kmeans_random_init.inspect.getsource", side_effect=lambda fn: source if fn is original_init else getsource(fn)), patch("tools.baseline.native_kmeans_random_init.INIT_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, error):
                compile_random_initializer(original_init, object())
        with self.context() as gpu:
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], random_probability_records=()):
                pass
            with patch.object(gpu, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "probability_abi_unavailable"), native_kmeans_feature_mean(gpu, [], random_probability_records=[]):
                pass
            with self.assertRaisesRegex(ValueError, "injected"), native_kmeans_feature_mean(gpu, [], random_probability_records=[]), patch.object(gpu, "kmeans_seed_probabilities", side_effect=ValueError("injected")):
                KMeans(n_clusters=3, n_init=1, random_state=17, init="random").fit(np.ones((42, 17), np.float32))
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(KMeans._init_centroids, original_init)

    def test_default_plusplus_keeps_existing_scoped_numeric_hooks(self):
        from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library as build_cpu
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "cpu.dylib"
            identity = build_cpu(library)
            with self.context() as gpu, NativeKMeansPotentials(library, identity["library_sha256"]) as cpu:
                x = np.random.default_rng(53103).normal(size=(42, 17)).astype(np.float32)
                expected = KMeans(n_clusters=4, n_init=3, random_state=17).fit(x.copy())
                with native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=(potentials := []), local_trial_records=(trials := []), random_probability_records=(random_records := [])):
                    actual = KMeans(n_clusters=4, n_init=3, random_state=17).fit(x.copy())
                self.assertEqual(random_records, [])
                self.assertEqual(len(trials), 3)
                self.assertTrue(potentials)
                np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                np.testing.assert_array_equal(actual.labels_, expected.labels_)

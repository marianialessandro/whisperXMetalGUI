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
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean, compile_native_mean
from tools.baseline.native_kmeans_equivalence import validate_reference


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.library)
        cls.double = Path(cls.directory.name) / "double.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def test_raw_matches_original_directed_mapping_and_permutations(self):
        rng = np.random.default_rng(1103)
        with self.context() as context:
            for count in (1, 7, 42, 4096):
                for clusters in (1, 2, 7, 4096):
                    labels = rng.integers(0, clusters, count, dtype=np.int32)
                    for previous in (labels, labels.copy(), rng.permutation(clusters).astype(np.int32)[labels], rng.integers(0, clusters, count, dtype=np.int32)):
                        expected = kmeans._is_same_clustering(labels, previous, clusters)
                        actual, record = context.same_clustering(labels, previous, clusters)
                        self.assertIs(type(actual), bool); self.assertEqual(actual, expected)
                        self.assertEqual(record["execution_device"], "cpu")
                        self.assertEqual(record["command_buffers_per_run"], 0)
                        self.assertEqual(record["gpu_kernels"], [])
                        self.assertIsNone(record["computation_gpu_seconds"])
            a, b = np.array([0, 1], np.int32), np.array([0, 0], np.int32)
            self.assertTrue(context.same_clustering(a, b, 2)[0])
            self.assertFalse(context.same_clustering(b, a, 2)[0])
            labels = np.ndarray((42,), dtype=np.int32, buffer=bytearray(169), offset=1)
            labels[:] = rng.integers(0, 7, 42); labels.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: context.same_clustering(labels, labels, 7)[0], range(4)))
            self.assertEqual(outputs, [True] * 4)

    def test_raw_domains_abi_closed_and_binary_guard(self):
        labels = np.arange(7, dtype=np.int32)
        with self.context() as context:
            for previous in (labels.astype(np.int64), labels[None, :], labels[::-1]):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    context.same_clustering(labels, previous, 7)
            for clusters in (0, 4097, True, np.int64(7)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.same_clustering(labels, labels, clusters)
            for count in (0, 4097):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.same_clustering(np.zeros(count, np.int32), np.zeros(count, np.int32), 7)
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                context.same_clustering(labels, labels[:3], 7)
            for value in (-1, 7, np.iinfo(np.int32).max):
                previous = labels.copy(); previous[-1] = value
                with self.assertRaisesRegex(ValueError, "label_domain"):
                    context.same_clustering(labels, previous, 7)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.same_clustering(labels, labels, 7)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.same_clustering(labels, labels, 7)
        with patch("tools.baseline.native_kmeans_equivalence.COMMON_BINARY_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
            validate_reference(kmeans._is_same_clustering)

    def test_compiler_expression_guard(self):
        original = KMeans.fit; inner = inspect.unwrap(original); getsource = inspect.getsource
        changed = getsource(inner).replace("_is_same_clustering(labels, best_labels, self.n_clusters)", "_is_same_clustering(labels.copy(), best_labels, self.n_clusters)")
        with patch("tools.baseline.native_kmeans_mean.inspect.getsource", side_effect=lambda function: changed if function is inner else getsource(function)), patch("tools.baseline.native_kmeans_mean.FIT_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "same_clustering_expression"):
            compile_native_mean(original, object(), equivalence=object())

    def test_fits_composition_rng_sparse_thread_explicit_restore_and_failure(self):
        values = np.random.default_rng(1109).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_mapping = KMeans.fit, kmeans._is_same_clustering
        def scope(gpu, cpu, records, composed):
            options = dict(shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True, product_records=[], chunk_records=[], lloyd_shift_records=[], lloyd_label_records=[]) if composed else {}
            return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], equivalence_records=records, **options)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            total = 0
            trigger = None
            for composed in (False, True):
                for clusters in (1, np.int64(2), 4):
                    rng1, rng2 = np.random.RandomState(17), np.random.RandomState(17)
                    weights = np.linspace(.5, 2., len(values), dtype=np.float32)
                    options = dict(n_clusters=clusters, n_init=6, copy_x=False)
                    expected = KMeans(**options, random_state=rng1).fit(values.copy(), sample_weight=weights)
                    with scope(gpu, cpu, records := [], composed):
                        actual = KMeans(**options, random_state=rng2).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        count = len(records); total += count
                        if count and not composed:
                            trigger = clusters
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options, random_state=17).fit(values.copy())).result()
                        self.assertEqual(len(records), count)
                        self.assertIs(kmeans._is_same_clustering, original_mapping)
                    np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
            self.assertGreater(total, 0)
            with scope(gpu, cpu, skipped := [], False):
                KMeans(n_clusters=2, n_init=6, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            self.assertIsNotNone(trigger)
            with self.assertRaisesRegex(ValueError, "injected"), scope(gpu, cpu, [], False), patch.object(cpu, "same_clustering", side_effect=ValueError("injected")):
                KMeans(n_clusters=trigger, n_init=6, random_state=17, copy_x=False).fit(values.copy(), sample_weight=weights)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(gpu, [], equivalence_records=[]):
                    pass
        self.assertIs(KMeans.fit, original_fit); self.assertIs(kmeans._is_same_clustering, original_mapping)

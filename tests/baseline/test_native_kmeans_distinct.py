from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
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
import sklearn.cluster._kmeans as kmeans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean, compile_native_mean


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansDistinctTests(unittest.TestCase):
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

    def gpu_context(self):
        return NativeDoubleNormalization(self.gpu, self.gpu_identity["library_sha256"])

    def test_raw_every_label_bitset_word_boundaries_and_order(self):
        rng = np.random.default_rng(51701)
        with self.context() as cpu:
            for count in (1, 2, 63, 64, 65, 255, 256, 257, 511, 512, 4095, 4096):
                for labels in (np.arange(count, dtype=np.int32), np.zeros(count, np.int32), np.full(count, 4095, np.int32), rng.integers(0, 4096, count, dtype=np.int32)):
                    for values in (labels, labels[::-1].copy()):
                        actual, record = cpu.distinct_labels(values)
                        self.assertEqual(actual, len(set(values)))
                        self.assertIs(type(actual), int)
                        self.assertEqual(record["distinct_labels"], actual)
                        self.assertEqual(record["execution_device"], "cpu")
                        self.assertEqual(record["worker_bitset_bytes"], 512)
                        self.assertEqual(record["gpu_kernels"], [])
                        self.assertEqual(record["native_blas_calls"], 0)
                        self.assertFalse(record["python_set_cardinality"])

    def test_raw_validation_unaligned_readonly_concurrent_and_closed(self):
        labels = np.arange(4096, dtype=np.int32)
        unaligned = np.ndarray(labels.shape, dtype=np.int32, buffer=bytearray(labels.nbytes + 1), offset=1)
        unaligned[:] = labels
        unaligned.flags.writeable = False
        with self.context() as cpu:
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(list(executor.map(lambda _: cpu.distinct_labels(unaligned)[0], range(4))), [4096] * 4)
            for value in (labels.astype(np.int64), labels.reshape(2, -1), labels[::-1], [0, 1]):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    cpu.distinct_labels(value)
            for count in (0, 4097):
                with self.assertRaisesRegex(ValueError, "count_invalid"):
                    cpu.distinct_labels(np.zeros(count, np.int32))
            for value in (-1, 4096, np.iinfo(np.int32).max):
                labels[-1] = value
                with self.assertRaisesRegex(ValueError, "label_domain"):
                    cpu.distinct_labels(labels)
            with patch.object(cpu, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                cpu.distinct_labels(np.zeros(1, np.int32))
        with self.assertRaisesRegex(ValueError, "closed"):
            cpu.distinct_labels(np.zeros(1, np.int32))

    def test_dense_fits_attributes_rng_input_and_duplicate_warnings(self):
        with self.context() as cpu, self.gpu_context() as gpu:
            for rows in (42, 257, 512):
                for algorithm in ("lloyd", "elkan"):
                    for duplicate in (False, True):
                        x = np.ones((rows, 17), np.float32) if duplicate else np.random.default_rng(rows).normal(size=(rows, 17)).astype(np.float32)
                        weights = np.linspace(.125, 2, rows, dtype=np.float32)
                        rng1, rng2 = np.random.RandomState(17), np.random.RandomState(17)
                        left, right = x.copy(), x.copy()
                        options = dict(n_clusters=4, n_init=3, copy_x=False, algorithm=algorithm)
                        with warnings.catch_warnings(record=True) as expected_warnings:
                            warnings.simplefilter("always")
                            expected = KMeans(**options, random_state=rng1).fit(left, sample_weight=weights)
                        records = []
                        with warnings.catch_warnings(record=True) as actual_warnings, native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], distinct_records=records):
                            warnings.simplefilter("always")
                            actual = KMeans(**options, random_state=rng2).fit(right, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
                        np.testing.assert_array_equal(rng1.uniform(size=32), rng2.uniform(size=32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._n_threads), (expected.inertia_, expected.n_iter_, expected._n_threads))
                        expected_convergence = [(str(w.message), w.category) for w in expected_warnings if issubclass(w.category, kmeans.ConvergenceWarning)]
                        actual_convergence = [(str(w.message), w.category) for w in actual_warnings if issubclass(w.category, kmeans.ConvergenceWarning)]
                        self.assertEqual(actual_convergence, expected_convergence)
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0]["distinct_labels"], 1 if duplicate else 4)

    def test_source_contract_and_python_set_displaced(self):
        original = KMeans.fit
        with self.context() as cpu:
            candidate = compile_native_mean(original, lambda x: x.mean(axis=0), distinct=lambda X, labels: cpu.distinct_labels(labels)[0])
            def reject(*args):
                self.fail("Python set cardinality used")
            inspect.unwrap(candidate).__globals__["set"] = reject
            candidate(KMeans(n_clusters=4, n_init=2, random_state=17), np.random.default_rng(51703).normal(size=(42, 17)).astype(np.float32))
            inner = inspect.unwrap(original)
            with patch.dict(inner.__globals__, {"set": reject}), self.assertRaisesRegex(ValueError, "builtins_not_supported"):
                compile_native_mean(original, object(), distinct=object())
            source = inspect.getsource(inner).replace("len(set(best_labels))", "len(set(best_labels.copy()))")
            getsource = inspect.getsource
            with patch("tools.baseline.native_kmeans_mean.inspect.getsource", side_effect=lambda fn: source if fn is inner else getsource(fn)), patch("tools.baseline.native_kmeans_mean.FIT_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "distinct_expression"):
                compile_native_mean(original, object(), distinct=object())

    def test_scope_sparse_owner_thread_and_failure_restore(self):
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        x = np.random.default_rng(51707).normal(size=(42, 17)).astype(np.float32)
        with self.context() as cpu, self.gpu_context() as gpu:
            def scope(records):
                return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], distinct_records=records)
            records = []
            with scope(records):
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(sparse.csr_matrix(x))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=17).fit, x).result()
                self.assertEqual(records, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope([]), patch.object(cpu, "distinct_labels", side_effect=ValueError("injected")):
                KMeans(n_clusters=3, n_init=1, random_state=17).fit(x)
            with self.assertRaisesRegex(ValueError, "configuration"), native_kmeans_feature_mean(gpu, [], distinct_records=[]):
                pass
            with patch.object(cpu, "library", SimpleNamespace(wxkmeans_weighted_potentials=object())), self.assertRaisesRegex(ValueError, "distinct_abi_unavailable"), scope([]):
                pass
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

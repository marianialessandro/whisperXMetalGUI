from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans_module
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_prefix import compile_native_prefix, seed_probabilities


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "double.dylib"
        cls.identity = build_library(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeDoubleNormalization(self.library, self.identity["library_sha256"])

    def test_pairwise_sum_division_bits_recursion_boundaries_zeros_extremes_and_budget(self):
        rng = np.random.default_rng(2317)
        with self.context() as context:
            for count in (1, 2, 3, 7, 8, 9, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4095, 4096):
                for weights in (rng.uniform(.01, 3, count).astype(np.float32), np.full(count, 2. ** -32, np.float32), np.full(count, 2. ** 32, np.float32)):
                    weights.flags.writeable = False
                    expected = weights / weights.sum()
                    actual, record = context.kmeans_seed_probabilities(weights)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(len(record["gpu_kernels"]), 2)
                    self.assertEqual(record["logical_result_readback_bytes"], weights.nbytes)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            weights = np.array([-0., 0., 2. ** -32, 2. ** 32], np.float32)
            expected = weights / weights.sum()
            with patch("numpy._core._methods._sum", side_effect=AssertionError("CPU sum executed")):
                actual = seed_probabilities(context, records := [], weights)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(len(records), 1)

    def test_invalid_inputs_zero_total_and_closed_context_fail(self):
        with self.context() as context:
            for weights in (np.ones(3, np.float64), np.ones((1, 3), np.float32), np.ones(6, np.float32)[::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_seed_probabilities(weights)
            for count in (0, 4097):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_seed_probabilities(np.ones(count, np.float32))
            for value in (-1., np.nan, np.inf, 2. ** -33, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_seed_probabilities(np.full(3, value, np.float32))
            with self.assertRaisesRegex(ValueError, "result_invalid"):
                context.kmeans_seed_probabilities(np.zeros(3, np.float32))
            actual, _ = context.kmeans_seed_probabilities(np.ones(2, np.float32))
            np.testing.assert_array_equal(actual, np.array([.5, .5], np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_seed_probabilities(np.ones(2, np.float32))

    def test_seed_indices_centers_rng_state_and_complete_fit_attributes(self):
        values = np.random.default_rng(2319).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        norms = kmeans_module.row_norms(values, squared=True)
        original = kmeans_module._kmeans_plusplus
        with self.context() as context:
            for combined in (False, True):
                for clusters in (1, 2, 4, 8):
                    reference_rng, actual_rng = np.random.RandomState(42), np.random.RandomState(42)
                    expected = original(values, clusters, norms, weights, reference_rng)
                    with native_kmeans_feature_mean(context, [], prefix_records=[] if combined else None, minimum_records=[] if combined else None, probability_records=(records := [])):
                        actual = kmeans_module._kmeans_plusplus(values, clusters, norms, weights, actual_rng)
                    np.testing.assert_array_equal(actual[0].view(np.uint32), expected[0].view(np.uint32))
                    np.testing.assert_array_equal(actual[1], expected[1])
                    np.testing.assert_array_equal(actual_rng.random_sample(32), reference_rng.random_sample(32))
                    self.assertEqual(len(records), 1)
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    reference_input, actual_input = values.copy(), values.copy()
                    options = dict(n_clusters=3, n_init=3, random_state=42, algorithm=algorithm, copy_x=copy_x)
                    expected = KMeans(**options).fit(reference_input, sample_weight=weights)
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=(records := [])):
                        actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                    self.assertEqual(len(records), 3)
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_sparse_explicit_random_initialization_and_other_threads_bypass(self):
        values = np.random.default_rng(2321).normal(size=(129, 17)).astype(np.float32)
        with self.context() as context, native_kmeans_feature_mean(context, [], probability_records=(records := [])):
            KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
            for init in (values[:3].copy(), "random"):
                KMeans(n_clusters=3, n_init=1, init=init, random_state=42).fit(values.copy())
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
            self.assertEqual(records, [])

    def test_caps_nested_failure_source_expression_and_sum_drift_restore(self):
        original_fit, original = KMeans.fit, kmeans_module._kmeans_plusplus
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "probabilities_records_required"):
                with native_kmeans_feature_mean(context, [], probability_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "probabilities_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], probability_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], probability_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], probability_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_probability_failure"):
                with native_kmeans_feature_mean(context, [], probability_records=(records := [])), patch.object(context, "kmeans_seed_probabilities", side_effect=ValueError("injected_probability_failure")):
                    KMeans(n_clusters=2, random_state=42).fit(np.ones((3, 2), np.float32))
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._kmeans_plusplus, original)
        getsource = inspect.getsource
        source = getsource(original).replace("sample_weight / sample_weight.sum()", "sample_weight / sample_weight.mean()")
        def changed(function):
            return source if function is original else getsource(function)
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "probabilities_expression_contract_changed"):
            compile_native_prefix(original, None, probabilities=object())
        with patch("tools.baseline.native_kmeans_prefix.SUM_SHA256", "changed"), self.assertRaisesRegex(ValueError, "probabilities_runtime_not_supported"):
            compile_native_prefix(original, None, probabilities=object())


if __name__ == "__main__":
    unittest.main()

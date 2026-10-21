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
from tools.baseline.native_kmeans_prefix import compile_native_prefix, weighted_prefix


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansPrefixTests(unittest.TestCase):
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

    def test_numpy_multiply_cumsum_bits_shapes_zeros_extremes_and_budget(self):
        rng = np.random.default_rng(2111)
        with self.context() as context:
            for count in (1, 2, 3, 15, 16, 17, 42, 127, 128, 129, 1024, 4096):
                for shape in ((count,), (1, count)):
                    weights = rng.uniform(.01, 2, count).astype(np.float32)
                    distances = rng.uniform(.01, 3, count).astype(np.float32).reshape(shape)
                    distances.flags.writeable = weights.flags.writeable = False
                    expected = np.cumsum(weights * distances)
                    actual, record = context.kmeans_weighted_prefix(weights, distances)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(len(record["gpu_kernels"]), 2)
                    self.assertEqual(record["logical_result_readback_bytes"], count * 4)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            actual, _ = context.kmeans_weighted_prefix(np.full(4096, 2. ** -32, np.float32), np.full(4096, 2. ** -64, np.float32))
            expected = np.cumsum(np.full(4096, 2. ** -96, np.float32))
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            for value in (0., -0., 2. ** -32, 2. ** 32):
                values = np.full(4096, value, np.float32)
                expected = np.cumsum(values * values)
                actual, _ = context.kmeans_weighted_prefix(values, values)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_invalid_operands_and_domains_do_not_return_cpu_results(self):
        values = np.ones(3, np.float32)
        with self.context() as context:
            for bad in (values.astype(np.float64), values[::2], values.reshape(3, 1)):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_weighted_prefix(values, bad)
            for count in (0, 4097):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_weighted_prefix(np.ones(count, np.float32), np.ones(count, np.float32))
            for value in (-1., np.nan, np.inf, 2. ** -65, 2. ** 33):
                bad = np.full(3, value, np.float32)
                for weights, distances in ((bad, values), (values, bad)):
                    with self.assertRaisesRegex(ValueError, "domain_invalid"):
                        context.kmeans_weighted_prefix(weights, distances)
            with self.assertRaisesRegex(ValueError, "weight_domain_invalid"):
                context.kmeans_weighted_prefix(np.full(3, 2. ** -33, np.float32), values)
            with patch("numpy.cumsum", side_effect=AssertionError("CPU cumsum executed")):
                actual = weighted_prefix(context, records := [], values, values)
            np.testing.assert_array_equal(actual, np.array([1, 2, 3], np.float32))
            self.assertEqual(len(records), 1)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_weighted_prefix(values, values)

    def test_seed_indices_centers_rng_state_sparse_and_other_threads_preserved(self):
        values = np.random.default_rng(2117).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        norms = kmeans_module.row_norms(values, squared=True)
        original = kmeans_module._kmeans_plusplus
        with self.context() as context:
            for clusters in (1, 2, 4, 8):
                for trials in (None, 1, 5):
                    reference_rng, actual_rng = np.random.RandomState(42), np.random.RandomState(42)
                    expected = original(values, clusters, norms, weights, reference_rng, trials)
                    with native_kmeans_feature_mean(context, [], prefix_records=(records := [])):
                        actual = kmeans_module._kmeans_plusplus(values, clusters, norms, weights, actual_rng, trials)
                    np.testing.assert_array_equal(actual[0].view(np.uint32), expected[0].view(np.uint32))
                    np.testing.assert_array_equal(actual[1], expected[1])
                    np.testing.assert_array_equal(actual_rng.random_sample(32), reference_rng.random_sample(32))
                    self.assertEqual(len(records), clusters - 1)
            with native_kmeans_feature_mean(context, means := [], prefix_records=(records := [])):
                KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
                self.assertEqual(means, [])
                self.assertEqual(records, [])
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_complete_fit_and_explicit_init_attributes_with_all_native_stages(self):
        values = np.random.default_rng(2119).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        with self.context() as context:
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    for explicit in (False, True):
                        options = dict(n_clusters=3, n_init=1 if explicit else 3, random_state=42, algorithm=algorithm, copy_x=copy_x, init=values[:3].copy() if explicit else "k-means++")
                        reference_input, actual_input = values.copy(), values.copy()
                        expected = KMeans(**options).fit(reference_input, sample_weight=weights)
                        with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=(records := [])):
                            actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertEqual(len(records), 0 if explicit else 6)

    def test_missing_abi_nested_and_native_fault_restore_all_bindings(self):
        original_fit, original = KMeans.fit, kmeans_module._kmeans_plusplus
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "prefix_records_required"):
                with native_kmeans_feature_mean(context, [], prefix_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "prefix_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], prefix_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], prefix_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], prefix_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_prefix_failure"):
                with native_kmeans_feature_mean(context, [], tolerance_records=[], prefix_records=(records := [])), patch.object(context, "kmeans_weighted_prefix", side_effect=ValueError("injected_prefix_failure")):
                    KMeans(n_clusters=2, random_state=42).fit(np.ones((3, 2), np.float32))
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_source_expression_cumsum_and_binary_drift_are_rejected(self):
        original, getsource = kmeans_module._kmeans_plusplus, inspect.getsource
        source = getsource(original).replace("sample_weight * closest_dist_sq", "sample_weight + closest_dist_sq")
        def changed(function):
            return source if function is original else getsource(function)
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_prefix(original, object())
        for name in ("CUMSUM_SHA256", "NUMPY_CORE_SHA256"):
            with patch("tools.baseline.native_kmeans_prefix." + name, "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                compile_native_prefix(original, object())


if __name__ == "__main__":
    unittest.main()

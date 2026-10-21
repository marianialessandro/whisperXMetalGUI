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
from tools.baseline.native_kmeans_prefix import compile_native_prefix, scale_queries


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansQueryTests(unittest.TestCase):
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

    def test_original_broadcast_binary64_multiplication_bits_and_reused_one_term_kernel(self):
        rng = np.random.RandomState(17)
        with self.context() as context:
            for count in (1, 2, 3, 17, 129, 1024):
                for scale in (0., 2. ** -100, 2. ** -96, .7, 1., 2. ** 76):
                    random_values = rng.uniform(size=count)
                    random_values[0] = 0.
                    if count > 1:
                        random_values[1] = 2. ** -53
                    if count > 2:
                        random_values[2] = np.nextafter(1., 0.)
                    random_values.flags.writeable = False
                    potential = np.array([scale], np.float32)
                    expected = random_values * potential
                    actual = scale_queries(context, records := [], random_values, potential)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(actual.dtype, np.float64)
                    self.assertTrue(actual.flags.c_contiguous)
                    record = records[0]
                    self.assertEqual(record["gpu_kernels"], ["double_matmul"])
                    self.assertEqual(record["inner"], 1)
                    self.assertEqual(record["columns"], 1)
                    self.assertEqual(record["host_exact_scalar_promotion_bytes"], 8)
                    self.assertEqual(record["host_random_view_copy_bytes"], 0)
                    self.assertEqual(record["logical_result_readback_bytes"], count * 8)
                    self.assertFalse(record["rng_calls_replaced"])
                    self.assertGreater(record["computation_gpu_seconds"], 0)

    def test_unsupported_inputs_fail_without_changing_rng_or_cpu_multiplication(self):
        values, potential = np.array([0., .5]), np.array([1.], np.float32)
        with self.context() as context:
            for left, right in ((values.astype(np.float32), potential), (values.reshape(1, 2), potential), (np.ones(4)[::2], potential), (values, potential.astype(np.float64)), (values, potential.reshape(()))):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    scale_queries(context, [], left, right)
            for count in (0, 1025):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    scale_queries(context, [], np.zeros(count), potential)
            for value in (-0., -1., 1., 2. ** -54, np.nan, np.inf):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    scale_queries(context, [], np.array([value]), potential)
            for value in (-0., -1., 2. ** -101, 2. ** 77, np.nan, np.inf):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    scale_queries(context, [], values, np.array([value], np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            scale_queries(context, [], values, potential)

    def test_seed_indices_rng_sequence_and_complete_fit_attributes_preserved(self):
        values = np.random.default_rng(2619).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        norms = kmeans_module.row_norms(values, squared=True)
        original = kmeans_module._kmeans_plusplus
        with self.context() as context:
            for combined in (False, True):
                for clusters in (1, 2, 4, 8):
                    reference_rng, actual_rng = np.random.RandomState(42), np.random.RandomState(42)
                    expected = original(values, clusters, norms, weights, reference_rng)
                    with native_kmeans_feature_mean(context, [], prefix_records=[] if combined else None, minimum_records=[] if combined else None, probability_records=[] if combined else None, search_records=[] if combined else None, best_records=[] if combined else None, query_records=(records := [])):
                        actual = kmeans_module._kmeans_plusplus(values, clusters, norms, weights, actual_rng)
                    np.testing.assert_array_equal(actual[0].view(np.uint32), expected[0].view(np.uint32))
                    np.testing.assert_array_equal(actual[1], expected[1])
                    np.testing.assert_array_equal(actual_rng.random_sample(32), reference_rng.random_sample(32))
                    self.assertEqual(len(records), clusters - 1)
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    reference_input, actual_input = values.copy(), values.copy()
                    options = dict(n_clusters=3, n_init=3, random_state=42, algorithm=algorithm, copy_x=copy_x)
                    expected = KMeans(**options).fit(reference_input, sample_weight=weights)
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=(records := [])):
                        actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                    self.assertEqual(len(records), 6)
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_sparse_explicit_random_initialization_and_other_threads_bypass(self):
        values = np.random.default_rng(2621).normal(size=(129, 17)).astype(np.float32)
        with self.context() as context, native_kmeans_feature_mean(context, [], query_records=(records := [])):
            KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
            for init in (values[:3].copy(), "random"):
                KMeans(n_clusters=3, n_init=1, init=init, random_state=42).fit(values.copy())
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
            self.assertEqual(records, [])

    def test_caps_fault_nesting_and_expression_drift_restore_original_bindings(self):
        original_fit, original = KMeans.fit, kmeans_module._kmeans_plusplus
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "query_scaling_records_required"):
                with native_kmeans_feature_mean(context, [], query_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "query_scaling_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], query_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], query_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], query_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_query_failure"):
                with native_kmeans_feature_mean(context, [], query_records=(records := [])), patch.object(context, "matmul", side_effect=ValueError("injected_query_failure")):
                    KMeans(n_clusters=2, random_state=42).fit(np.ones((3, 2), np.float32))
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._kmeans_plusplus, original)
        getsource = inspect.getsource
        source = getsource(original).replace("random_state.uniform(size=n_local_trials) * current_pot", "random_state.uniform(size=n_local_trials) + current_pot")
        def changed(function):
            return source if function is original else getsource(function)
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "query_scaling_expression_contract_changed"):
            compile_native_prefix(original, None, query_scale=object())


if __name__ == "__main__":
    unittest.main()

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
from tools.baseline.native_kmeans_prefix import candidate_search, compile_native_prefix


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansSearchTests(unittest.TestCase):
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

    def test_exact_search_and_clip_at_equal_neighbors_repeats_zeros_and_maximum_budget(self):
        rng = np.random.default_rng(2417)
        with self.context() as context:
            for count in (1, 2, 3, 7, 17, 42, 127, 128, 129, 1024, 4096):
                cdf = np.cumsum(rng.integers(0, 4, count).astype(np.float32))
                queries = rng.uniform(0, max(1., float(cdf[-1])) * 2, 1024).astype(np.float64)
                queries[:4] = (0., -0., np.nextafter(0., 1.), 2. ** 80)
                for i, value in enumerate(cdf[:200].astype(np.float64)):
                    queries[4 + i * 3:7 + i * 3] = (max(0., np.nextafter(value, -np.inf)), value, np.nextafter(value, np.inf))
                expected = np.clip(np.searchsorted(cdf, queries), None, count - 1)
                cdf.flags.writeable = queries.flags.writeable = False
                actual, record = context.kmeans_candidate_search(cdf, queries)
                self.assertEqual(actual.dtype, np.int64)
                np.testing.assert_array_equal(actual, expected)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertEqual(record["logical_result_readback_bytes"], len(queries) * 8)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            cdf = np.array([-0., 0., 2. ** -100, 2. ** 76], np.float32)
            queries = np.array([-0., 0., 2. ** -101, 2. ** -100, 2. ** 76, 2. ** 80], np.float64)
            expected = np.clip(np.searchsorted(cdf, queries), None, 3)
            with patch("numpy.searchsorted", side_effect=AssertionError("CPU search executed")), patch("numpy.clip", side_effect=AssertionError("CPU clip executed")):
                actual = candidate_search(context, records := [], cdf, queries)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(records), 1)

    def test_invalid_format_shapes_cdf_order_query_domains_and_closed_context(self):
        cdf, queries = np.array([0., 1., 2.], np.float32), np.array([0., 1.], np.float64)
        with self.context() as context:
            for a, b in ((cdf.astype(np.float64), queries), (cdf[::2], queries), (cdf, queries.astype(np.float32)), (cdf, queries.reshape(1, 2))):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_candidate_search(a, b)
            for a, b in ((np.zeros(0, np.float32), queries), (np.zeros(4097, np.float32), queries), (cdf, np.zeros(0)), (cdf, np.zeros(1025))):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_candidate_search(a, b)
            for value in (-1., np.nan, np.inf, np.nextafter(np.float32(0), np.float32(1)), 2. ** -101, 2. ** 77):
                with self.assertRaisesRegex(ValueError, "cdf_domain_invalid"):
                    context.kmeans_candidate_search(np.full(3, value, np.float32), queries)
            with self.assertRaisesRegex(ValueError, "cdf_domain_invalid"):
                context.kmeans_candidate_search(cdf[::-1].copy(), queries)
            for value in (-1., -np.nextafter(0., 1.), np.nan, np.inf, 2. ** 81):
                with self.assertRaisesRegex(ValueError, "query_domain_invalid"):
                    context.kmeans_candidate_search(cdf, np.full(2, value))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_candidate_search(cdf, queries)

    def test_initial_centers_indices_rng_and_complete_fit_with_native_preparation(self):
        values = np.random.default_rng(2419).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        norms = kmeans_module.row_norms(values, squared=True)
        original = kmeans_module._kmeans_plusplus
        with self.context() as context:
            for combined in (False, True):
                for clusters in (1, 2, 4, 8):
                    reference_rng, actual_rng = np.random.RandomState(42), np.random.RandomState(42)
                    expected = original(values, clusters, norms, weights, reference_rng)
                    with native_kmeans_feature_mean(context, [], prefix_records=[] if combined else None, minimum_records=[] if combined else None, probability_records=[] if combined else None, search_records=(records := [])):
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
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=(records := [])):
                        actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                    self.assertEqual(len(records), 6)
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_sparse_explicit_random_initialization_and_other_threads_bypass(self):
        values = np.random.default_rng(2421).normal(size=(129, 17)).astype(np.float32)
        with self.context() as context, native_kmeans_feature_mean(context, [], search_records=(records := [])):
            KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
            for init in (values[:3].copy(), "random"):
                KMeans(n_clusters=3, n_init=1, init=init, random_state=42).fit(values.copy())
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
            self.assertEqual(records, [])

    def test_caps_fault_nesting_source_expression_and_wrapper_drift_restore(self):
        original_fit, original = KMeans.fit, kmeans_module._kmeans_plusplus
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "search_records_required"):
                with native_kmeans_feature_mean(context, [], search_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "search_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], search_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], search_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], search_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_search_failure"):
                with native_kmeans_feature_mean(context, [], prefix_records=[], search_records=(records := [])), patch.object(context, "kmeans_candidate_search", side_effect=ValueError("injected_search_failure")):
                    KMeans(n_clusters=2, random_state=42).fit(np.ones((3, 2), np.float32))
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._kmeans_plusplus, original)
        getsource = inspect.getsource
        for old, new in (("np.searchsorted(", "np.searchsorted_changed("), ("out=candidate_ids", "out=None")):
            source = getsource(original).replace(old, new)
            def changed(function):
                return source if function is original else getsource(function)
            with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "search_expression_contract_changed"):
                compile_native_prefix(original, None, search=object())
        for name in ("SEARCHSORTED_SHA256", "CLIP_SHA256"):
            with patch("tools.baseline.native_kmeans_prefix." + name, "changed"), self.assertRaisesRegex(ValueError, "search_runtime_not_supported"):
                compile_native_prefix(original, None, search=object())


if __name__ == "__main__":
    unittest.main()

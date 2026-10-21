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
from tools.baseline.native_kmeans_minimum import candidate_minimum
from tools.baseline.native_kmeans_prefix import compile_native_prefix


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansMinimumTests(unittest.TestCase):
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

    def test_numpy_broadcast_bits_shapes_bounds_signed_zeros_and_inplace_identity(self):
        rng = np.random.default_rng(2211)
        with self.context() as context:
            for rows, columns in ((1, 1), (3, 2), (3, 3), (7, 17), (5, 42), (13, 129), (1024, 4096)):
                for row_shape in ((columns,), (1, columns)):
                    closest = rng.uniform(.01, 3, columns).astype(np.float32).reshape(row_shape)
                    candidates = rng.uniform(.01, 3, (rows, columns)).astype(np.float32)
                    expected = np.minimum(closest, candidates)
                    actual, record = context.kmeans_candidate_minimum(closest, candidates)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(record["logical_result_readback_bytes"], candidates.nbytes)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            closest = np.array([0., -0., 2. ** -100, 2. ** 64], np.float32)
            candidates = np.array([[-0., 0., 2. ** 64, 2. ** -100]], np.float32)
            expected = np.minimum(closest, candidates)
            with patch("numpy.minimum", side_effect=AssertionError("CPU minimum executed")):
                actual = candidate_minimum(context, records := [], closest, candidates)
            self.assertIs(actual, candidates)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(records[0]["host_inplace_copy_bytes"], candidates.nbytes)

    def test_invalid_domains_formats_readonly_and_closed_context(self):
        closest, candidates = np.ones(3, np.float32), np.ones((2, 3), np.float32)
        with self.context() as context:
            for bad in (candidates.astype(np.float64), candidates[:, ::2], candidates[0]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_candidate_minimum(closest, bad)
            for shape in ((0, 3), (1025, 3), (2, 0), (2, 4097)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_candidate_minimum(np.ones(shape[1], np.float32), np.ones(shape, np.float32))
            for value in (-1., np.nan, np.inf, 2. ** -101, 2. ** 65):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_candidate_minimum(np.full(3, value, np.float32), candidates)
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_candidate_minimum(closest, np.full((2, 3), value, np.float32))
            candidates.flags.writeable = False
            with self.assertRaisesRegex(ValueError, "writable_array_required"):
                candidate_minimum(context, [], closest, candidates)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_candidate_minimum(closest, candidates)

    def test_seed_indices_rng_and_full_fit_attributes_with_and_without_prefix(self):
        values = np.random.default_rng(2217).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        norms = kmeans_module.row_norms(values, squared=True)
        original = kmeans_module._kmeans_plusplus
        with self.context() as context:
            for prefix_enabled in (False, True):
                for clusters in (1, 2, 4, 8):
                    reference_rng, actual_rng = np.random.RandomState(42), np.random.RandomState(42)
                    expected = original(values, clusters, norms, weights, reference_rng)
                    with native_kmeans_feature_mean(context, [], prefix_records=[] if prefix_enabled else None, minimum_records=(records := [])):
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
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=(records := [])):
                        actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                    self.assertEqual(len(records), 6)
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_sparse_explicit_initialization_and_other_threads_bypass(self):
        values = np.random.default_rng(2221).normal(size=(129, 17)).astype(np.float32)
        with self.context() as context, native_kmeans_feature_mean(context, [], minimum_records=(records := [])):
            KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
            KMeans(n_clusters=3, n_init=1, init=values[:3].copy()).fit(values.copy())
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
            self.assertEqual(records, [])

    def test_capability_fault_nesting_and_source_expression_restore(self):
        original_fit, original = KMeans.fit, kmeans_module._kmeans_plusplus
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "minimum_records_required"):
                with native_kmeans_feature_mean(context, [], minimum_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "minimum_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], minimum_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], minimum_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], minimum_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_minimum_failure"):
                with native_kmeans_feature_mean(context, [], prefix_records=[], minimum_records=(records := [])), patch.object(context, "kmeans_candidate_minimum", side_effect=ValueError("injected_minimum_failure")):
                    KMeans(n_clusters=2, random_state=42).fit(np.ones((3, 2), np.float32))
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._kmeans_plusplus, original)
        getsource = inspect.getsource
        source = getsource(original).replace("out=distance_to_candidates", "out=None")
        def changed(function):
            return source if function is original else getsource(function)
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "minimum_expression_contract_changed"):
            compile_native_prefix(original, None, minimum=object())


if __name__ == "__main__":
    unittest.main()

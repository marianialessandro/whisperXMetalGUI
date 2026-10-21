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
from tools.baseline.native_kmeans_prefix import best_candidate, compile_native_prefix


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansBestTests(unittest.TestCase):
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

    def test_first_tie_argmin_scalar_type_shapes_extremes_signed_zero_and_no_cpu_argmin(self):
        rng = np.random.default_rng(2517)
        with self.context() as context:
            for count in (1, 2, 3, 7, 17, 42, 129, 1024):
                for shape in ((count,), (count, 1)):
                    potentials = rng.uniform(1, 3, count).astype(np.float32).reshape(shape)
                    potentials.reshape(-1)[0] = 2. ** -100
                    if count > 1:
                        potentials.reshape(-1)[-1] = 2. ** -100
                    potentials.flags.writeable = False
                    expected = np.argmin(potentials)
                    actual, record = context.kmeans_best_candidate(potentials)
                    self.assertIs(type(actual), type(expected))
                    self.assertEqual(actual, expected)
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertEqual(record["logical_result_readback_bytes"], 8)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for potentials in (np.array([0., -0., 0.], np.float32), np.array([-0., 0., -0.], np.float32), np.array([2. ** 76, 2. ** -100, 2. ** -100], np.float32)):
                expected = np.argmin(potentials)
                with patch("numpy.argmin", side_effect=AssertionError("CPU argmin executed")):
                    actual = best_candidate(context, records := [], potentials)
                self.assertEqual(actual, expected)
                self.assertEqual(len(records), 1)

    def test_invalid_format_shapes_domains_and_closed_context(self):
        with self.context() as context:
            for bad in (np.ones(3, np.float64), np.ones((1, 3), np.float32), np.ones(6, np.float32)[::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_best_candidate(bad)
            for count in (0, 1025):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_best_candidate(np.ones(count, np.float32))
            for value in (-1., np.nan, np.inf, np.nextafter(np.float32(0), np.float32(1)), 2. ** -101, 2. ** 77):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_best_candidate(np.full(3, value, np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_best_candidate(np.ones(2, np.float32))

    def test_seed_indices_rng_and_complete_fit_preserved_using_original_potentials(self):
        values = np.random.default_rng(2519).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        norms = kmeans_module.row_norms(values, squared=True)
        original = kmeans_module._kmeans_plusplus
        with self.context() as context:
            for combined in (False, True):
                for clusters in (1, 2, 4, 8):
                    reference_rng, actual_rng = np.random.RandomState(42), np.random.RandomState(42)
                    expected = original(values, clusters, norms, weights, reference_rng)
                    with native_kmeans_feature_mean(context, [], prefix_records=[] if combined else None, minimum_records=[] if combined else None, probability_records=[] if combined else None, search_records=[] if combined else None, best_records=(records := [])):
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
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=(records := [])):
                        actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, expected.labels_)
                    np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                    self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                    self.assertEqual(len(records), 6)
        self.assertIs(kmeans_module._kmeans_plusplus, original)

    def test_sparse_explicit_random_initialization_and_other_threads_bypass(self):
        values = np.random.default_rng(2521).normal(size=(129, 17)).astype(np.float32)
        with self.context() as context, native_kmeans_feature_mean(context, [], best_records=(records := [])):
            KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
            for init in (values[:3].copy(), "random"):
                KMeans(n_clusters=3, n_init=1, init=init, random_state=42).fit(values.copy())
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
            self.assertEqual(records, [])

    def test_caps_fault_nesting_source_expression_and_wrapper_drift_restore(self):
        original_fit, original = KMeans.fit, kmeans_module._kmeans_plusplus
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "best_candidate_records_required"):
                with native_kmeans_feature_mean(context, [], best_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "best_candidate_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], best_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], best_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], best_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_best_failure"):
                with native_kmeans_feature_mean(context, [], best_records=(records := [])), patch.object(context, "kmeans_best_candidate", side_effect=ValueError("injected_best_failure")):
                    KMeans(n_clusters=2, random_state=42).fit(np.ones((3, 2), np.float32))
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._kmeans_plusplus, original)
        getsource = inspect.getsource
        source = getsource(original).replace("np.argmin(candidates_pot)", "np.argmin(candidates_pot, axis=0)")
        def changed(function):
            return source if function is original else getsource(function)
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "best_candidate_expression_contract_changed"):
            compile_native_prefix(original, None, best=object())
        with patch("tools.baseline.native_kmeans_prefix.ARGMIN_SHA256", "changed"), self.assertRaisesRegex(ValueError, "best_candidate_runtime_not_supported"):
            compile_native_prefix(original, None, best=object())


if __name__ == "__main__":
    unittest.main()

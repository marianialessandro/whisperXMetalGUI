from concurrent.futures import ThreadPoolExecutor
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
import sklearn.cluster._k_means_lloyd as cython_lloyd

from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd import compile_native_lloyd
from tools.baseline.native_kmeans_lloyd_norms import LloydNormScope


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansLloydNormTests(unittest.TestCase):
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

    def test_fit_attributes_inputs_and_random_state_match_original(self):
        values = np.random.default_rng(2101).normal(size=(129, 17)).astype(np.float32)
        original_norms, original_lloyd = cython_lloyd.row_norms, kmeans._kmeans_single_lloyd
        with self.context() as context:
            for weighted in (False, True):
                for clusters in (1, 2, 4, 8):
                    options = dict(n_clusters=clusters, n_init=3, copy_x=False)
                    left_rng, right_rng = np.random.RandomState(17), np.random.RandomState(17)
                    weights = np.linspace(.5, 1.5, len(values), dtype=np.float32) if weighted else None
                    reference_input, candidate_input = values.copy(), values.copy()
                    reference = KMeans(**options, random_state=left_rng).fit(reference_input, sample_weight=weights)
                    with native_kmeans_feature_mean(context, [], lloyd_norm_records=(records := [])):
                        actual = KMeans(**options, random_state=right_rng).fit(candidate_input, sample_weight=weights)
                        self.assertIs(kmeans._kmeans_single_lloyd, original_lloyd)
                    self.assertTrue(records)
                    np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), reference.cluster_centers_.view(np.uint32))
                    np.testing.assert_array_equal(actual.labels_, reference.labels_)
                    np.testing.assert_array_equal(candidate_input.view(np.uint32), reference_input.view(np.uint32))
                    np.testing.assert_array_equal(left_rng.rand(32), right_rng.rand(32))
                    self.assertEqual(actual.inertia_, reference.inertia_)
                    self.assertEqual(actual.n_iter_, reference.n_iter_)
                    self.assertEqual(actual._tol.view(np.uint32), reference._tol.view(np.uint32))
                    for record in records:
                        self.assertEqual(record["adapter_scope"], "sklearn_dense_float32_lloyd_centroid_norms")
                        self.assertEqual(record["host_operand_view_copy_bytes"], 0)
                        self.assertFalse(record["python_numeric_norm_computation"])
                        self.assertFalse(record["lloyd_iteration_replaced"])
                        self.assertGreater(record["computation_gpu_seconds"], 0)
        self.assertIs(cython_lloyd.row_norms, original_norms)

    def test_only_active_centroid_storage_uses_gpu_and_other_threads_bypass(self):
        values = np.random.default_rng(2103).normal(size=(42, 7)).astype(np.float32)
        original = cython_lloyd.row_norms
        with self.context() as context, native_kmeans_feature_mean(context, [], lloyd_norm_records=(records := [])):
            with patch.object(context, "kmeans_squared_norms", side_effect=AssertionError("unexpected GPU norm")):
                np.testing.assert_array_equal(cython_lloyd.row_norms(values, squared=True), original(values, squared=True))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=2, n_init=1, random_state=17).fit, values).result()
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, random_state=17, algorithm="elkan").fit(values.copy())
            self.assertEqual(records, [])
        self.assertIs(cython_lloyd.row_norms, original)

    def test_private_lloyd_helper_uses_gpu_without_original_row_norms(self):
        values = np.random.default_rng(2105).normal(size=(42, 7)).astype(np.float32)
        weights = np.ones(42, np.float32)
        centers = values[:2].copy()
        expected = kmeans._kmeans_single_lloyd(values, weights, centers.copy(), tol=0, max_iter=2)
        with self.context() as context:
            scope = LloydNormScope(context, records := [])
            helper = compile_native_lloyd(kmeans._kmeans_single_lloyd, dense_iteration=scope.iteration)
            with scope, patch.object(scope, "original_norms", side_effect=AssertionError("CPU row norm executed")):
                actual = helper(values, weights, centers.copy(), tol=0, max_iter=2)
            self.assertTrue(records)
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)
            self.assertIsNone(scope.centers)

    def test_explicit_initialization_and_assignment_only_pass_preserve_results(self):
        values = np.random.default_rng(2106).normal(size=(42, 7)).astype(np.float32)
        with self.context() as context:
            for tolerance, maximum in ((0., 1), (1e6, 3)):
                options = dict(n_clusters=3, init=values[:3].copy(), n_init=1, random_state=17, tol=tolerance, max_iter=maximum)
                expected = KMeans(**options).fit(values.copy())
                with native_kmeans_feature_mean(context, [], lloyd_norm_records=(records := [])):
                    actual = KMeans(**options).fit(values.copy())
                self.assertEqual(len(records), actual.n_iter_ + 1)
                np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                np.testing.assert_array_equal(actual.labels_, expected.labels_)
                self.assertEqual(actual.inertia_, expected.inertia_)
                self.assertEqual(actual.n_iter_, expected.n_iter_)

    def test_fault_domain_and_binding_errors_restore_all_hooks(self):
        values = np.random.default_rng(2107).normal(size=(42, 7)).astype(np.float32)
        original_fit, original_norms = KMeans.fit, cython_lloyd.row_norms
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "injected_lloyd_norm_failure"):
                with native_kmeans_feature_mean(context, [], lloyd_norm_records=(records := [])), patch.object(context, "kmeans_squared_norms", side_effect=ValueError("injected_lloyd_norm_failure")):
                    KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            self.assertEqual(records, [])
            self.assertIs(KMeans.fit, original_fit)
            self.assertIs(cython_lloyd.row_norms, original_norms)
            scope = LloydNormScope(context, [])
            with self.assertRaisesRegex(ValueError, "scope_not_ready"):
                scope.iteration(values, None, values[:2])
            with scope:
                scope.centers = values[:2]
                for operand, squared in ((values[1:3], True), (values[:2].copy(), True), (values[:2], False)):
                    with self.assertRaisesRegex(ValueError, "operand_binding_changed"):
                        scope.row_norms(operand, squared)
                scope.centers = np.full((1, 7), np.float32(2. ** -61))
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    scope.row_norms(scope.centers, True)
            self.assertIsNone(scope.centers)
            with native_kmeans_feature_mean(context, [], lloyd_norm_records=[]):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(cython_lloyd.row_norms, original_norms)

    def test_configuration_binary_source_and_iteration_binding_guards(self):
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                LloydNormScope(context, ())
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                LloydNormScope(context, [])
            with patch("tools.baseline.native_kmeans_lloyd_norms.LLOYD_BINARY_SHA256", "changed"), self.assertRaisesRegex(ValueError, "binary_not_supported"):
                LloydNormScope(context, [])
            with patch.object(cython_lloyd, "row_norms", lambda values, squared=False: None), self.assertRaisesRegex(ValueError, "source_not_supported"):
                LloydNormScope(context, [])
            scope = LloydNormScope(context, [])
            with patch.object(cython_lloyd, "row_norms", object()), self.assertRaisesRegex(ValueError, "binding_changed"):
                scope.__enter__()
            inner = inspect.unwrap(kmeans._kmeans_single_lloyd)
            with patch.dict(inner.__globals__, {"lloyd_iter_chunked_dense": object()}), self.assertRaisesRegex(ValueError, "iteration_binding_changed"):
                compile_native_lloyd(kmeans._kmeans_single_lloyd, dense_iteration=scope.iteration)


if __name__ == "__main__":
    unittest.main()

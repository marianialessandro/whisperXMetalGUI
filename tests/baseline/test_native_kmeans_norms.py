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
from tools.baseline.native_kmeans_mean import compile_native_mean, native_kmeans_feature_mean
from tools.baseline.native_kmeans_norms import squared_norms, validate_reference


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansNormsTests(unittest.TestCase):
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

    def test_neon_einsum_bitwise_boundaries_tails_signed_zeros_and_maximum_budget(self):
        rng = np.random.default_rng(2011)
        with self.context() as context:
            for columns in (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 501, 1023, 1024):
                values = rng.normal(size=(37, columns)).astype(np.float32)
                values[0] = -0.
                values[1] = 2. ** -32
                values[2] = 2. ** 32
                values.flags.writeable = False
                expected = np.einsum("ij,ij->i", values, values)
                actual, record = context.kmeans_squared_norms(values)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(record["logical_result_readback_bytes"], len(values) * 4)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertEqual(record["gpu_kernels"], ["float_kmeans_squared_norms"])
                self.assertFalse(record["numpy_bitwise_certified"])
            values = rng.normal(size=(4096, 1024)).astype(np.float32)
            expected = np.einsum("ij,ij->i", values, values)
            actual, record = context.kmeans_squared_norms(values)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_domains_and_closed_context_fail_without_replacement(self):
        with self.context() as context:
            for values in (np.ones((3, 2), np.float64), np.ones((3, 4), np.float32)[:, ::2], np.ones(2, np.float32)):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_squared_norms(values)
            for shape in ((0, 2), (4097, 2), (2, 0), (2, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_squared_norms(np.ones(shape, np.float32))
            for value in (np.nan, np.inf, 2. ** -33, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.kmeans_squared_norms(np.full((3, 2), value, np.float32))
            result, _ = context.kmeans_squared_norms(np.ones((3, 2), np.float32))
            np.testing.assert_array_equal(result, np.full(3, 2, np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_squared_norms(np.ones((3, 2), np.float32))

    def test_adapter_avoids_cpu_einsum_and_mean_while_scoped_fit_keeps_sparse_and_threads(self):
        values = np.random.default_rng(2017).normal(size=(129, 7)).astype(np.float32)
        expected = np.einsum("ij,ij->i", values, values)
        original_fit, original_norms = KMeans.fit, kmeans_module.row_norms
        with self.context() as context:
            with patch("numpy.einsum", side_effect=AssertionError("CPU norm executed")):
                actual = squared_norms(context, records := [], values)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(len(records), 1)
            sparse_input = sparse.csr_matrix(values)
            expected_sparse = KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse_input)
            with native_kmeans_feature_mean(context, means := [], norm_records=(records := [])):
                actual_sparse = KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse_input)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, values).result()
                self.assertEqual(records, [])
                self.assertEqual(means, [])
            np.testing.assert_array_equal(actual_sparse.cluster_centers_, expected_sparse.cluster_centers_)
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module.row_norms, original_norms)

    def test_complete_fit_preserves_all_attributes_with_centering_and_tolerance(self):
        values = np.random.default_rng(2019).normal(size=(129, 17)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        with self.context() as context:
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    for explicit in (False, True):
                        options = dict(n_clusters=3, n_init=1 if explicit else 3, random_state=42, algorithm=algorithm, copy_x=copy_x, init=values[:3].copy() if explicit else "k-means++")
                        reference_input, actual_input = values.copy(), values.copy()
                        expected = KMeans(**options).fit(reference_input, sample_weight=weights)
                        with native_kmeans_feature_mean(context, means := [], shift_records=[], tolerance_records=[], norm_records=(records := [])):
                            actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                        self.assertEqual(actual.inertia_, expected.inertia_)
                        self.assertEqual(actual.n_iter_, expected.n_iter_)
                        self.assertIs(type(actual._tol), type(expected._tol))
                        self.assertEqual(actual._tol.view(np.uint32), expected._tol.view(np.uint32))
                        self.assertEqual(len(records), 1)
                        self.assertNotIn("squared_norms", means[0]["cpu_residuals"])

    def test_capabilities_faults_nesting_restore_original(self):
        original_fit, original_norms = KMeans.fit, kmeans_module.row_norms
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "norms_records_required"):
                with native_kmeans_feature_mean(context, [], norm_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "norms_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], norm_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], norm_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], norm_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_norm_failure"):
                with native_kmeans_feature_mean(context, means := [], norm_records=(records := [])), patch.object(context, "kmeans_squared_norms", side_effect=ValueError("injected_norm_failure")):
                    KMeans(n_clusters=2).fit(np.ones((3, 2), np.float32))
            self.assertEqual(len(means), 1)
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module.row_norms, original_norms)

    def test_source_expression_runtime_and_binary_drift_are_rejected(self):
        original = KMeans.fit
        getsource = inspect.getsource
        source = getsource(inspect.unwrap(original)).replace("row_norms(X, squared=True)", "row_norms(X, squared=False)")
        def changed(function):
            return source if function is inspect.unwrap(original) else getsource(function)
        with patch("tools.baseline.native_kmeans_mean.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_mean.FIT_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_mean(original, object(), norms=object())
        for name in ("EINSUM_SHA256", "NUMPY_CORE_SHA256"):
            with patch("tools.baseline.native_kmeans_norms." + name, "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                validate_reference(kmeans_module.row_norms)
        with patch("tools.baseline.native_kmeans_norms.ROW_NORMS_SHA256", "changed"), self.assertRaisesRegex(ValueError, "source_not_supported"):
            validate_reference(kmeans_module.row_norms)


if __name__ == "__main__":
    unittest.main()

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
from tools.baseline.native_kmeans_tolerance import compile_native_tolerance


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansToleranceTests(unittest.TestCase):
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

    def test_numpy_variance_and_pairwise_tolerance_bits_shapes_and_parameters(self):
        rng = np.random.default_rng(1913)
        with self.context() as context:
            for rows, columns in ((1, 2), (19, 3), (127, 7), (128, 8), (129, 9), (501, 127), (129, 128), (17, 129), (42, 256), (4096, 1024)):
                values = rng.normal(size=(rows, columns)).astype(np.float32)
                values.flags.writeable = False
                for tolerance in (1e-4, .5, 2. ** -32, 2. ** 16, np.float32(.25), 1):
                    expected = np.mean(np.var(values, axis=0)) * tolerance
                    actual, record = context.kmeans_tolerance(values, tolerance)
                    self.assertIs(type(actual), np.float32)
                    self.assertEqual(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(len(record["gpu_kernels"]), 4)
                    self.assertEqual(record["logical_result_readback_bytes"], 4)
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            result, _ = context.kmeans_tolerance(np.ones((129, 7), np.float32), .5)
            self.assertEqual(result.view(np.uint32), np.float32(0).view(np.uint32))

    def test_invalid_domains_and_intermediate_underflow_fail_without_cpu_replacement(self):
        with self.context() as context:
            values = np.ones((3, 2), np.float32)
            for tolerance in (0, -1, np.nan, np.inf, 2. ** -33, 2. ** 17):
                with self.assertRaisesRegex(ValueError, "shape_or_scalar_invalid"):
                    context.kmeans_tolerance(values, tolerance)
            for value in (np.nan, np.inf, 2. ** -33, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.kmeans_tolerance(np.full((3, 2), value, np.float32), .5)
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.kmeans_tolerance(values, np.float64(.5))
            with self.assertRaisesRegex(ValueError, "shape_or_scalar_invalid"):
                context.kmeans_tolerance(np.ones((3, 1), np.float32), .5)
            underflow = np.zeros((4096, 2), np.float32)
            underflow[0] = 2. ** -32
            underflow[1] = -np.nextafter(np.float32(2. ** -32), np.float32(1.))
            with self.assertRaisesRegex(ValueError, "intermediate_not_supported"):
                context.kmeans_tolerance(underflow, .5)
            result, _ = context.kmeans_tolerance(values, .5)
            self.assertEqual(result, 0)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_tolerance(values, .5)

    def test_pinned_helper_avoids_cpu_variance_and_preserves_zero_sparse_and_threads(self):
        original = kmeans_module._tolerance
        values = np.random.default_rng(1919).normal(size=(129, 7)).astype(np.float32)
        expected = original(values, 1e-4)
        sparse_input = sparse.csr_matrix(values)
        sparse_expected = original(sparse_input, 1e-4)
        with self.context() as context, native_kmeans_feature_mean(context, [], tolerance_records=(records := [])):
            with patch("numpy.var", side_effect=AssertionError("CPU variance executed")), patch("numpy.mean", side_effect=AssertionError("CPU variance mean executed")):
                actual = kmeans_module._tolerance(values, 1e-4)
            self.assertEqual(actual.view(np.uint32), expected.view(np.uint32))
            for zero in (0, 0., -0., np.float64(0)):
                result = kmeans_module._tolerance(None, zero)
                self.assertIs(type(result), int)
                self.assertEqual(result, 0)
            self.assertEqual(kmeans_module._tolerance(sparse_input, 1e-4), sparse_expected)
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(executor.submit(kmeans_module._tolerance, values, 1e-4).result(), expected)
            self.assertEqual(len(records), 1)
        self.assertIs(kmeans_module._tolerance, original)

    def test_complete_fit_preserves_tolerance_and_all_attributes_with_other_native_stages(self):
        values = np.random.default_rng(1921).normal(size=(129, 7)).astype(np.float32)
        weights = np.linspace(.5, 2., len(values), dtype=np.float32)
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with self.context() as context:
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    for tolerance in (0., 1e-4):
                        options = dict(n_clusters=3, n_init=3, random_state=42, algorithm=algorithm, copy_x=copy_x, tol=tolerance)
                        reference_input, actual_input = values.copy(), values.copy()
                        expected = KMeans(**options).fit(reference_input, sample_weight=weights)
                        with native_kmeans_feature_mean(context, means := [], shift_records=[], tolerance_records=(records := [])):
                            actual = KMeans(**options).fit(actual_input, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                        self.assertEqual(actual.inertia_, expected.inertia_)
                        self.assertEqual(actual.n_iter_, expected.n_iter_)
                        self.assertIs(type(actual._tol), type(expected._tol))
                        self.assertEqual(actual._tol, expected._tol)
                        self.assertEqual(len(records), int(tolerance != 0))
                        self.assertNotIn("tolerance_variance", means[0]["cpu_residuals"])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._tolerance, original_tolerance)

    def test_missing_abi_nested_and_native_failure_restore_both_bindings(self):
        original_fit, original_tolerance = KMeans.fit, kmeans_module._tolerance
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "tolerance_records_required"):
                with native_kmeans_feature_mean(context, [], tolerance_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "tolerance_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], tolerance_records=[]):
                    pass
            with native_kmeans_feature_mean(context, [], tolerance_records=[]):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, [], tolerance_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_tolerance_failure"):
                with native_kmeans_feature_mean(context, means := [], tolerance_records=(records := [])), patch.object(context, "kmeans_tolerance", side_effect=ValueError("injected_tolerance_failure")):
                    KMeans(n_clusters=2).fit(np.ones((3, 2), np.float32))
            self.assertEqual(means, [])
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans_module._tolerance, original_tolerance)

    def test_source_expression_and_numpy_reference_drift_are_rejected(self):
        original = kmeans_module._tolerance
        getsource = inspect.getsource
        source = getsource(original).replace("axis=0", "axis=1")
        def changed(function):
            return source if function is original else getsource(function)
        with patch("tools.baseline.native_kmeans_tolerance.inspect.getsource", side_effect=changed), patch("tools.baseline.native_kmeans_tolerance.TOLERANCE_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_tolerance(original, object())
        with patch("tools.baseline.native_kmeans_tolerance.NUMPY_VAR_SHA256", "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
            compile_native_tolerance(original, object())


if __name__ == "__main__":
    unittest.main()

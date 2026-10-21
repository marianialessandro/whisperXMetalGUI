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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import compile_native_mean, native_kmeans_feature_mean, shift_inplace


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeFloatShiftTests(unittest.TestCase):
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

    def test_primitive_add_subtract_bitwise_signed_zeros_neighbors_and_maximum(self):
        rng = np.random.default_rng(1889)
        with self.context() as context:
            for rows, columns in ((1, 1), (17, 3), (129, 256), (4096, 1024)):
                values = rng.normal(size=(rows, columns)).astype(np.float32)
                mean = rng.normal(size=columns).astype(np.float32)
                values.flags.writeable = mean.flags.writeable = False
                for add in (False, True):
                    actual, record = context.float_shift_vbx(values, mean, add)
                    expected = values + mean if add else values - mean
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for value in (0., -0., 2. ** -100, np.nextafter(np.float32(2. ** -100), np.float32(1)), 2. ** 50, -2. ** 50):
                for offset in (0., -0., 2. ** -100, 2. ** 50):
                    values, mean = np.full((1, 2), value, np.float32), np.full(2, offset, np.float32)
                    for add in (False, True):
                        actual, _ = context.float_shift_vbx(values, mean, add)
                        expected = values + mean if add else values - mean
                        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_domains_shapes_capability_and_inplace_copy(self):
        values, mean = np.ones((3, 2), np.float32), np.ones(2, np.float32)
        with self.context() as context:
            for invalid in (np.nan, np.inf, 2. ** -101, 2. ** 51):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.float_shift_vbx(np.full((3, 2), invalid, np.float32), mean)
                with self.assertRaisesRegex(ValueError, "mean_domain_invalid"):
                    context.float_shift_vbx(values, np.full(2, invalid, np.float32))
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                context.float_shift_vbx(values, np.ones(3, np.float32))
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.float_shift_vbx(values, mean, 1)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.float_shift_vbx(values, mean)
            original = values.copy()
            self.assertIs(shift_inplace(context, records := [], values, mean, False, "X"), values)
            np.testing.assert_array_equal(values, original - mean)
            self.assertEqual(records[0]["host_inplace_copy_bytes"], values.nbytes)
            values.flags.writeable = False
            with self.assertRaisesRegex(ValueError, "writable_float32_array_required"):
                shift_inplace(context, [], values, mean, False, "X")
            with self.assertRaisesRegex(ValueError, "role_not_supported"):
                shift_inplace(context, [], original, mean, True, "private-role")
        with self.assertRaisesRegex(ValueError, "closed"):
            context.float_shift_vbx(values, mean)

    def test_complete_fit_both_algorithms_copy_modes_and_explicit_initializers(self):
        features = np.random.default_rng(1891).normal(size=(129, 7)).astype(np.float32)
        weights = np.linspace(.5, 2., len(features), dtype=np.float32)
        original = KMeans.fit
        with self.context() as context:
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    for explicit in (False, True):
                        options = dict(n_clusters=3, n_init=1, random_state=42, algorithm=algorithm, copy_x=copy_x, init=features[:3].copy() if explicit else "k-means++")
                        baseline_input, candidate_input = features.copy(), features.copy()
                        expected = KMeans(**options).fit(baseline_input, sample_weight=weights)
                        with native_kmeans_feature_mean(context, means := [], shift_records=(records := [])):
                            actual = KMeans(**options).fit(candidate_input, sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(candidate_input.view(np.uint32), baseline_input.view(np.uint32))
                        self.assertEqual(actual.inertia_, expected.inertia_)
                        self.assertEqual(actual.n_iter_, expected.n_iter_)
                        self.assertEqual(len(means), 1)
                        self.assertEqual(len(records), 2 + int(not copy_x) + int(explicit))
                        self.assertNotIn("feature_subtraction_and_restoration", means[0]["cpu_residuals"])
                        self.assertIn("tolerance_variance", means[0]["cpu_residuals"])
                        self.assertEqual(records[0]["operand_role"], "X")
                        self.assertEqual(records[-1]["operand_role"], "best_centers")
        self.assertIs(KMeans.fit, original)

    def test_sparse_background_and_failure_preserve_original_fit(self):
        values = np.random.default_rng(1897).normal(size=(21, 3)).astype(np.float32)
        original = KMeans.fit
        with self.context() as context, native_kmeans_feature_mean(context, means := [], shift_records=(records := [])):
            KMeans(n_clusters=2, n_init=1, random_state=42).fit(sparse.csr_matrix(values))
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=2, n_init=1, random_state=42).fit, values).result()
            self.assertEqual(means, [])
            self.assertEqual(records, [])
        with self.context() as context, self.assertRaisesRegex(ValueError, "injected_shift_failure"):
            with native_kmeans_feature_mean(context, [], shift_records=(records := [])), patch.object(context, "float_shift_vbx", side_effect=ValueError("injected_shift_failure")):
                KMeans(n_clusters=2).fit(values)
        self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original)

    def test_missing_shift_abi_and_augmented_assignment_source_drift(self):
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "shift_records_required"):
                with native_kmeans_feature_mean(context, [], shift_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())), self.assertRaisesRegex(ValueError, "shift_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], shift_records=[]):
                    pass
        source = inspect.getsource(inspect.unwrap(KMeans.fit)).replace("X += X_mean", "X -= X_mean")
        with patch("tools.baseline.native_kmeans_mean.inspect.getsource", return_value=source), patch("tools.baseline.native_kmeans_mean.FIT_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "shift_expression_contract_changed"):
            compile_native_mean(KMeans.fit, object(), shift=object())


if __name__ == "__main__":
    unittest.main()

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import platform
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.distance import cdist
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization, cosine_distances
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeCosineTests(unittest.TestCase):
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

    def reference(self, embeddings, centroids):
        def norm(values):
            total = np.float64(0)
            for value in values.astype(np.float64):
                total = total + value * value
            return np.sqrt(total)
        a, b = [norm(row) for row in embeddings], [norm(row) for row in centroids]
        result = np.empty((len(a), len(b)))
        for row, left in enumerate(embeddings.astype(np.float64)):
            for column, right in enumerate(centroids.astype(np.float64)):
                if not a[row] or not b[column] or np.isnan(a[row]) or np.isnan(b[column]):
                    result[row, column] = np.nan
                    continue
                dot = np.float64(0)
                for x, y in zip(left, right):
                    dot = dot + x * y
                result[row, column] = 1. - np.clip(dot / (a[row] * b[column]), -1., 1.)
        return result

    def test_cosine_ordered_bits_scipy_guard_layouts_dtypes_and_limits(self):
        rng = np.random.default_rng(1507)
        with self.context() as context:
            for rows, count, features in ((1, 1, 1), (19, 3, 8), (3, 7, 127), (3, 7, 128), (3, 7, 129), (4096, 1, 1), (1, 1024, 1), (2, 1, 1024)):
                for order in ("C", "F"):
                    for dtype in (np.float32, np.float64):
                        a = np.array(rng.normal(size=(rows, features)), dtype=dtype, order=order)
                        b = np.array(rng.normal(size=(count, features)), order=order)
                        a.flags.writeable = b.flags.writeable = False
                        records = []
                        actual = cosine_distances(context, records, a, b)
                        np.testing.assert_array_equal(actual.view(np.uint64), self.reference(a, b).view(np.uint64))
                        np.testing.assert_allclose(actual, cdist(a, b, metric="cosine"), rtol=3e-14, atol=3e-15)
                        self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                        self.assertEqual(len(records[0]["gpu_kernels"]), 2)
                        self.assertFalse(records[0]["scipy_bitwise_certified"])
                        self.assertLessEqual(records[0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_undefined_nan_zero_pairs_signed_zero_antipodes_and_extremes(self):
        a = np.array([[0., -0.], [np.nan, 1.], [1., 0.], [-1., -0.], [2. ** -100, 2. ** 100]])
        b = np.array([[1., 0.], [-1., -0.], [np.nan, np.nan], [0., 0.], [2. ** -100, 2. ** 100]])
        with self.context() as context:
            actual, record = context.cosine_vbx(a, b)
            expected = cdist(a, b, metric="cosine")
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
            finite = np.isfinite(actual)
            ordered = self.reference(a, b)
            np.testing.assert_array_equal(actual[finite].view(np.uint64), ordered[finite].view(np.uint64))
            self.assertEqual(record["undefined_pairs"], int(np.isnan(expected).sum()))
            self.assertEqual(actual[2, 0], 0.)
            self.assertEqual(actual[2, 1], 2.)

    def test_validation_precedes_staging_and_rejects_other_metrics_domains(self):
        with self.context() as context:
            for a, b in ((np.ones((4097, 1)), np.ones((1, 1))), (np.ones((1, 1)), np.ones((1025, 1))), (np.ones((1, 1025)), np.ones((1, 1025))), (np.ones((4096, 1024)), np.ones((9, 1024))), (np.ones((2, 3)), np.ones((1, 4)))):
                with patch("tools.baseline.native_embedding_adapter.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                    cosine_distances(context, [], a, b)
            for value in (np.inf, -np.inf, 2. ** -101, 2. ** 101):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.cosine_vbx(np.array([[value]]), np.ones((1, 1)))
            with self.assertRaisesRegex(ValueError, "metric_not_supported"):
                cosine_distances(context, [], np.ones((1, 2)), np.ones((1, 2)), metric="euclidean")
            with self.assertRaisesRegex(ValueError, "format_not_supported"):
                cosine_distances(context, [], np.ones((1, 2), np.int32), np.ones((1, 2)))
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.cosine_vbx(np.ones((1, 2), np.float32), np.ones((1, 2)))

    def test_lifetime_capability_and_serialized_calls(self):
        with self.context() as context:
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.cosine_vbx(np.ones((1, 2)), np.ones((1, 2)))
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.cosine_vbx(np.array([[1., 0.]]), np.array([[-1., 0.]]))[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result, [[2.]])
        with self.assertRaisesRegex(ValueError, "closed"):
            context.cosine_vbx(np.ones((1, 2)), np.ones((1, 2)))

    def test_entire_clustering_composes_with_normalization_centroids_and_kmeans(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        norm = np.linalg.norm
        normalization = SimpleNamespace(normalize=lambda values: (values / norm(values, axis=1, keepdims=True), {}))
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for exact in (None, 2):
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                for mask in range(4):
                    distances, centroids, normalizations = [], [], []
                    with native_embedding_normalization(normalization if mask & 1 else None, normalizations, centroid_context=context if mask & 2 else None, centroid_records=centroids, distance_context=context, distance_records=distances):
                        actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    self.assertEqual(len(distances), 1)
                    self.assertEqual(len(centroids), int(bool(mask & 2)))
                    self.assertEqual(len(normalizations), int(bool(mask & 1)))
                    np.testing.assert_array_equal(actual[0], expected[0])
                    for left, right in zip(expected[1:], actual[1:]):
                        np.testing.assert_allclose(left, right, rtol=3e-14, atol=3e-15)
        self.assertIs(VBxClustering.__call__, original)

    def test_other_threads_small_branch_and_failure_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with native_embedding_normalization(None, [], distance_context=context, distance_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                self.assertEqual(records, [])
                instance.filter_embeddings = lambda *a, **k: (embeddings.reshape(-1, 256)[:1], None, None)
                VBxClustering.__call__(instance, embeddings)
                self.assertEqual(records, [])
            embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
            with self.assertRaisesRegex(ValueError, "injected_cosine_failure"):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=(records := [])), patch.object(context, "cosine_vbx", side_effect=ValueError("injected_cosine_failure")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            self.assertEqual(records, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_fused_scores_keep_both_subtractions_bitwise_and_undefined_pairs(self):
        rng = np.random.default_rng(1591)
        with self.context() as context:
            for rows, count, features in ((1, 1, 1), (19, 3, 8), (3, 7, 129), (2, 2, 1024)):
                a, b = rng.normal(size=(rows, features)), rng.normal(size=(count, features))
                a[0] = 0.
                if rows > 1:
                    a[1, 0] = np.nan
                distances, _ = context.cosine_vbx(a, b)
                actual, record = context.cosine_vbx(a, b, soft_scores=True)
                expected = 2. - distances
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                finite = np.isfinite(actual)
                np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))
                self.assertEqual(record["result_kind"], "soft_scores")
                self.assertEqual(record["gpu_kernels"], ["double_vbx_cosine_norms", "double_vbx_cosine_scores"])
                self.assertEqual(record["logical_result_readback_bytes"], actual.nbytes)
            with self.assertRaisesRegex(ValueError, "score_mode_invalid"):
                context.cosine_vbx(np.ones((1, 2)), np.ones((1, 2)), soft_scores=1)
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_cosine=object())), self.assertRaisesRegex(ValueError, "score_abi_unavailable"):
                context.cosine_vbx(np.ones((1, 2)), np.ones((1, 2)), soft_scores=True)

    def test_fused_scores_complete_clustering_and_scoped_failures(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        norm = np.linalg.norm
        normalization = SimpleNamespace(normalize=lambda values: (values / norm(values, axis=1, keepdims=True), {}))
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for exact in (None, 2):
                for mask in range(4):
                    options = {"centroid_context": context if mask & 2 else None, "centroid_records": [], "distance_context": context, "distance_records": []}
                    with native_embedding_normalization(normalization if mask & 1 else None, [], **options):
                        expected = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    records = []
                    options["distance_records"] = records
                    with native_embedding_normalization(normalization if mask & 1 else None, [], soft_scores=True, **options):
                        actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    for left, right in zip(expected, actual):
                        np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))
                    self.assertEqual(records[0]["result_kind"], "soft_scores")
                    self.assertNotIn("soft_score_subtraction", records[0]["cpu_residuals"])
            with self.assertRaisesRegex(ValueError, "require_cosine"):
                with native_embedding_normalization(normalization, [], soft_scores=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_cosine=object())), self.assertRaisesRegex(ValueError, "score_abi_unavailable"):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=[], soft_scores=True):
                    pass
            with self.assertRaisesRegex(ValueError, "injected_score_failure"):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=[], soft_scores=True), patch.object(context, "cosine_vbx", side_effect=ValueError("injected_score_failure")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
        self.assertIs(VBxClustering.__call__, original)

    def test_fused_score_expression_and_single_consumer_pin(self):
        original = VBxClustering.__call__
        source = textwrap.dedent(inspect.getsource(original))
        for changed in (source.replace("2 - e2k_distance", "2.0 - e2k_distance"), source.replace("soft_clusters = 2 - e2k_distance", "unused = e2k_distance\n    soft_clusters = 2 - e2k_distance")):
            with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "soft_score_contract_changed"):
                compile_native_normalization(original, None, distance=object(), soft_scores=True)

    def test_scope_runtime_capability_and_expression_pins(self):
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], distance_context=context):
                    pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=[]):
                    pass
            with patch("tools.baseline.native_embedding_adapter.CDIST_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=[]):
                    pass
        original = VBxClustering.__call__
        changed = textwrap.dedent(inspect.getsource(original)).replace("metric=self.metric", "metric='cosine'")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "distance_contract_changed"):
            compile_native_normalization(original, None, distance=object())


if __name__ == "__main__":
    unittest.main()

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
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization, weighted_centroids
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeCentroidTests(unittest.TestCase):
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

    def reference(self, weights, features):
        result = np.zeros((weights.shape[1], features.shape[1]))
        for frame in range(weights.shape[0]):
            result = result + weights[frame, :, None] * features[frame, None, :].astype(np.float64)
        return result / weights.sum(0, keepdims=True).T

    def test_centroids_bitwise_ordered_reference_layouts_dtypes_and_signed_zero(self):
        rng = np.random.default_rng(1431)
        with self.context() as context:
            for frames, speakers, dimensions in ((1, 1, 1), (19, 3, 8), (127, 7, 3), (128, 7, 3), (129, 7, 3), (4096, 1, 1), (1, 1024, 1), (1, 1, 1024)):
                for order in ("C", "F"):
                    weights = np.array(rng.uniform(.01, 1., (frames, speakers)), order=order)
                    for dtype in (np.float32, np.float64):
                        features = rng.normal(size=(frames, dimensions)).astype(dtype)
                        features[:, 0] = -0.
                        weights.flags.writeable = features.flags.writeable = False
                        records = []
                        actual = weighted_centroids(context, records, weights, features)
                        expected = self.reference(weights, features)
                        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                        np.testing.assert_allclose(actual, (weights.T @ features) / weights.sum(0, keepdims=True).T, rtol=2e-12, atol=2e-12)
                        self.assertEqual(records[0]["mass_reduction_order"], "pairwise" if weights.flags.f_contiguous else "sequential")
                        self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                        self.assertFalse(records[0]["blas_bitwise_certified"])

    def test_extreme_domain_cancellation_and_zero_mass(self):
        with self.context() as context:
            weights = np.array([[2. ** -200], [2. ** 200]])
            features = np.array([[2. ** -100], [-2. ** 100]])
            actual = weighted_centroids(context, [], weights, features)
            np.testing.assert_array_equal(actual.view(np.uint64), self.reference(weights, features).view(np.uint64))
            for invalid in (-1., np.nan, np.inf, 2. ** -201, 2. ** 201, 0.):
                with self.assertRaises(ValueError):
                    context.centroids_vbx(np.full((1, 2), invalid), features)
            for invalid in (np.nan, np.inf, 2. ** -101, 2. ** 101):
                with self.assertRaisesRegex(ValueError, "feature_domain_invalid"):
                    context.centroids_vbx(np.ones((1, 2)), np.full((2, 1), invalid))

    def test_shape_work_validation_precedes_staging_and_supports_readonly(self):
        with self.context() as context:
            for weights, features in ((np.ones((4097, 1)), np.ones((4097, 1))), (np.ones((1, 1025)), np.ones((1, 1))), (np.ones((1, 1)), np.ones((1, 1025))), (np.ones((4096, 9)), np.ones((4096, 1024))), (np.ones((2, 1)), np.ones((3, 1)))):
                with patch("tools.baseline.native_embedding_adapter.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                    weighted_centroids(context, [], weights, features)
            for weights, features in ((np.ones((2, 1), np.float32), np.ones((2, 1))), (np.ones((2, 1)), np.ones((2, 1), np.int32)), (np.ones((2, 1)), np.ones(2)), (np.ones((4, 2))[::2], np.ones((2, 1)))):
                with self.assertRaisesRegex(ValueError, "format_not_supported"):
                    weighted_centroids(context, [], weights, features)
            with self.assertRaises(ValueError):
                context.centroids_vbx(np.ones((1, 2)), np.ones((2, 1)), "invalid")
            with self.assertRaises(ValueError):
                context.centroids_vbx(np.ones((1, 2)), np.ones((2, 1), np.float32))

    def test_native_lifetime_missing_abi_and_serialized_calls(self):
        with self.context() as context:
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.centroids_vbx(np.ones((1, 2)), np.ones((2, 1)))
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.centroids_vbx(np.ones((1, 2)), np.array([[2.], [4.]]))[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result, [[3.]])
        with self.assertRaisesRegex(ValueError, "closed"):
            context.centroids_vbx(np.ones((1, 2)), np.ones((2, 1)))

    def test_whole_clustering_independent_or_combined_embedding_normalization_and_kmeans(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        norm = np.linalg.norm
        embedding_context = SimpleNamespace(normalize=lambda values: (values / norm(values, axis=1, keepdims=True), {}))
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for exact in (None, 2):
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                for with_embedding in (False, True):
                    records, normalization = [], []
                    with native_embedding_normalization(embedding_context if with_embedding else None, normalization, centroid_context=context, centroid_records=records):
                        actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    self.assertEqual(len(records), 1)
                    self.assertEqual(len(normalization), int(with_embedding))
                    for left, right in zip(expected, actual):
                        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
        self.assertIs(VBxClustering.__call__, original)

    def test_other_threads_small_branch_and_native_failure_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with native_embedding_normalization(None, [], centroid_context=context, centroid_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                self.assertEqual(records, [])
                instance.filter_embeddings = lambda *a, **k: (embeddings.reshape(-1, 256)[:1], None, None)
                VBxClustering.__call__(instance, embeddings)
                self.assertEqual(records, [])
            embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
            with self.assertRaisesRegex(ValueError, "injected_centroid_failure"):
                with native_embedding_normalization(None, [], centroid_context=context, centroid_records=(records := [])), patch.object(context, "centroids_vbx", side_effect=ValueError("injected_centroid_failure")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            self.assertEqual(records, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_scope_records_capability_version_and_ast_drift(self):
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], centroid_context=context):
                    pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_embedding_normalization(None, [], centroid_context=context, centroid_records=[]):
                    pass
            with patch("tools.baseline.native_embedding_adapter.np.__version__", "changed"), self.assertRaisesRegex(ValueError, "numpy_not_supported"):
                with native_embedding_normalization(None, [], centroid_context=context, centroid_records=[]):
                    pass
        original = VBxClustering.__call__
        changed = textwrap.dedent(inspect.getsource(original)).replace("W.sum(0, keepdims=True)", "W.sum(axis=0, keepdims=True)")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "centroid_contract_changed"):
            compile_native_normalization(original, None, object())


if __name__ == "__main__":
    unittest.main()

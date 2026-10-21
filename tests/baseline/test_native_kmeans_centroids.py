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
from tools.baseline.native_embedding_adapter import compile_native_normalization, kmeans_centroids, native_embedding_normalization
import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansCentroidTests(unittest.TestCase):
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

    def test_numpy_float32_means_layouts_boundaries_and_cancellation(self):
        rng = np.random.default_rng(1861)
        with self.context() as context:
            for rows, dimensions, clusters in ((1, 2, 1), (19, 3, 7), (127, 7, 1), (128, 2, 3), (129, 256, 3), (501, 17, 9), (4096, 1024, 8), (1024, 2, 1024)):
                labels = rng.permutation(np.arange(rows) % clusters).astype(np.int64)
                features = rng.normal(size=(rows, dimensions)).astype(np.float32)
                features[:, 0] = -0.
                expected = np.vstack([np.mean(features[labels == k], axis=0) for k in range(clusters)])
                features = np.asfortranarray(features)
                features.flags.writeable = labels.flags.writeable = False
                with patch("numpy.mean", side_effect=AssertionError("CPU centroid mean executed")):
                    actual = kmeans_centroids(context, records := [], features, labels, clusters)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                self.assertLessEqual(records[0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for values in ([2. ** 32, 1., -2. ** 32, 3.], [2. ** -32, -2. ** -32, -0., 0.], [1., 2. ** -24, 2. ** -24, -1.]):
                features = np.repeat(np.array(values, np.float32)[:, None], 2, axis=1)
                actual = kmeans_centroids(context, [], features, np.zeros(4, np.int32), 1)
                np.testing.assert_array_equal(actual.view(np.uint32), features.mean(0, keepdims=True).view(np.uint32))

    def test_invalid_domains_shapes_and_lifetime_fail_without_staging(self):
        with self.context() as context:
            features, labels = np.ones((3, 2), np.float32), np.array([0, 1, 0], np.int32)
            for value in (np.inf, np.nan, 2. ** -33, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "feature_domain_invalid"):
                    context.kmeans_centroids_vbx(np.full((3, 2), value, np.float32), labels, 2)
            for wrong in (np.array([-1, 1, 0]), np.array([2 ** 32, 1, 0])):
                with self.assertRaisesRegex(ValueError, "label_not_supported"):
                    kmeans_centroids(context, [], features, wrong, 2)
            with self.assertRaisesRegex(ValueError, "empty_cluster"):
                context.kmeans_centroids_vbx(features, np.zeros(3, np.int32), 2)
            for shape in ((4097, 2), (3, 1), (4096, 1024)):
                data = np.ones(shape, np.float32)
                with patch("tools.baseline.native_embedding_adapter.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                    kmeans_centroids(context, [], data, np.zeros(shape[0], np.int32), 9 if shape[0] == 4096 else 1)
            with self.assertRaisesRegex(ValueError, "format_not_supported"):
                kmeans_centroids(context, [], features.astype(np.float64), labels, 2)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.kmeans_centroids_vbx(features, labels, 2)
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.kmeans_centroids_vbx(features, labels, 2)[0], range(4)))
                for result in results:
                    np.testing.assert_array_equal(result, np.ones((2, 2), np.float32))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_centroids_vbx(features, labels, 2)

    def test_complete_clustering_forced_and_bypassed_branches_and_cosine_composition(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for exact in (None, 1, 2, 3):
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                for native_cosine in (False, True):
                    records = []
                    with native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=records, distance_context=context if native_cosine else None, distance_records=[]):
                        actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    self.assertEqual(len(records), int(exact in (1, 2)))
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[2].view(np.uint8), expected[2].view(np.uint8))
                    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
        self.assertIs(VBxClustering.__call__, original)

    def test_thread_ownership_failure_nesting_and_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=2).result()
                self.assertEqual(records, [])
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_kmeans_failure"):
                with native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=(records := [])), patch.object(context, "kmeans_centroids_vbx", side_effect=ValueError("injected_kmeans_failure")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=2)
            self.assertEqual(records, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_scope_capability_source_and_expression_drift(self):
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], kmeans_context=context):
                    pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=[]):
                    pass
            with patch("tools.baseline.native_embedding_adapter.np.__version__", "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=[]):
                    pass
        source = textwrap.dedent(inspect.getsource(VBxClustering.__call__)).replace("axis=0)", "axis=1)")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=source), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "kmeans_centroid_contract_changed"):
            compile_native_normalization(VBxClustering.__call__, None, kmeans_centroid=object())


if __name__ == "__main__":
    unittest.main()

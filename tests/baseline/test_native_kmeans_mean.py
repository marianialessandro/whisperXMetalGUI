from concurrent.futures import ThreadPoolExecutor
import functools
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
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import native_embedding_normalization
from tools.baseline.native_kmeans_mean import compile_native_mean, feature_mean, native_kmeans_feature_mean
import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKmeansMeanTests(unittest.TestCase):
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

    def features(self):
        return np.random.default_rng(1871).normal(size=(129, 7)).astype(np.float32)

    def test_primitive_float32_mean_and_domains(self):
        features = self.features()
        expected = features.mean(axis=0)
        with self.context() as context:
            actual = feature_mean(context, records := [], features)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(records[0]["host_generated_label_bytes"], len(features) * 4)
            for values in (features.astype(np.float64), features.T, features[:, 0]):
                with self.assertRaisesRegex(ValueError, "operand_not_supported"):
                    feature_mean(context, [], values)
            with self.assertRaisesRegex(ValueError, "shape_not_supported"):
                feature_mean(context, [], np.ones((2, 1), np.float32))

    def test_complete_fit_bitwise_centers_labels_inertia_and_original_input_behavior(self):
        features = self.features()
        weights = np.linspace(.5, 2., len(features), dtype=np.float32)
        original = KMeans.fit
        with self.context() as context:
            for algorithm in ("lloyd", "elkan"):
                for copy_x in (False, True):
                    for init in ("k-means++", features[:3].copy()):
                        options = dict(n_clusters=3, n_init=3, random_state=42, copy_x=copy_x, algorithm=algorithm, init=init)
                        reference_input, actual_input = features.copy(), features.copy()
                        expected = KMeans(**options).fit(reference_input, sample_weight=weights)
                        actual = KMeans(**options)
                        with native_kmeans_feature_mean(context, records := []):
                            self.assertEqual(inspect.signature(KMeans.fit), inspect.signature(original))
                            self.assertIs(actual.fit(actual_input, sample_weight=weights), actual)
                        self.assertEqual(len(records), 1)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        np.testing.assert_array_equal(actual_input.view(np.uint32), reference_input.view(np.uint32))
                        self.assertEqual(actual.inertia_, expected.inertia_)
                        self.assertEqual(actual.n_iter_, expected.n_iter_)
        self.assertIs(KMeans.fit, original)

    def test_fit_predict_composes_with_vbx_centroids_and_cpu_audit(self):
        from pyannote.audio.pipelines.clustering import VBxClustering
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original_fit, original_call = KMeans.fit, VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for forced in (None, 2):
                expected = original_call(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=forced)
                with clustering_cpu_audit(audit := []), native_kmeans_feature_mean(context, means := []), native_embedding_normalization(None, [], kmeans_context=context, kmeans_records=(centroids := [])):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=forced)
                self.assertEqual(len(means), int(forced == 2))
                self.assertEqual(len(centroids), int(forced == 2))
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))
                fit = next(row for row in audit if row["operation"] == "kmeans_fit_predict")
                self.assertEqual(fit["calls_started"], int(forced == 2))
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(VBxClustering.__call__, original_call)

    def test_sparse_invalid_and_background_thread_paths(self):
        features = self.features()
        original = KMeans.fit
        expected = KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(features))
        with self.context() as context, native_kmeans_feature_mean(context, records := []):
            actual = KMeans(n_clusters=3, n_init=1, random_state=42).fit(sparse.csr_matrix(features))
            np.testing.assert_array_equal(actual.cluster_centers_, expected.cluster_centers_)
            self.assertEqual(records, [])
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(KMeans(n_clusters=3, n_init=1, random_state=42).fit, features).result()
            self.assertEqual(records, [])
            with self.assertRaises(ValueError):
                KMeans(n_clusters=len(features) + 1).fit(features)
            self.assertEqual(records, [])
            with self.assertRaisesRegex(ValueError, "operand_not_supported"):
                KMeans(n_clusters=3).fit(features.astype(np.float64))
        self.assertIs(KMeans.fit, original)

    def test_capability_wrappers_nested_failure_and_restore(self):
        original = KMeans.fit
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_kmeans_feature_mean(context, None):
                    pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_kmeans_feature_mean(context, []):
                    pass
            with patch("tools.baseline.native_kmeans_mean.np.__version__", "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_kmeans_feature_mean(context, []):
                    pass
            @functools.wraps(original)
            def foreign(*args, **kwargs):
                return original(*args, **kwargs)
            with patch.object(KMeans, "fit", foreign), self.assertRaisesRegex(ValueError, "fit_wrapper_not_supported"):
                with native_kmeans_feature_mean(context, []):
                    pass
            with native_kmeans_feature_mean(context, []):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_kmeans_feature_mean(context, []):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_mean_failure"):
                with native_kmeans_feature_mean(context, records := []), patch.object(context, "kmeans_centroids_vbx", side_effect=ValueError("injected_mean_failure")):
                    KMeans(n_clusters=3).fit(self.features())
            self.assertEqual(records, [])
        self.assertIs(KMeans.fit, original)

    def test_source_and_mean_or_decorator_expression_drift(self):
        source = inspect.getsource(inspect.unwrap(KMeans.fit))
        for changed in (source.replace("X.mean(axis=0)", "X.mean(axis=1)"), source.replace("prefer_skip_nested_validation=True", "prefer_skip_nested_validation=False")):
            with patch("tools.baseline.native_kmeans_mean.inspect.getsource", return_value=changed), patch("tools.baseline.native_kmeans_mean.FIT_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
                compile_native_mean(KMeans.fit, object())


if __name__ == "__main__":
    unittest.main()

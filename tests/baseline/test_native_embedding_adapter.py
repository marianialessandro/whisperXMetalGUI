from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import textwrap
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization


class NativeEmbeddingAdapterTests(unittest.TestCase):
    def fixture(self):
        embeddings = np.random.default_rng(391).normal(size=(7, 3, 256)).astype(np.float32)
        train = embeddings.reshape(-1, 256)
        instance = SimpleNamespace(constrained_assignment=False, filter_embeddings=lambda *a, **k: (train, None, None), threshold=.6, Fa=.07, Fb=.8, metric="cosine", plda=SimpleNamespace(phi=np.ones(128)))
        class PLDA:
            phi = np.ones(128)
            def __call__(self, values):
                return values[:, :128]
        instance.plda = PLDA()
        q = np.eye(3)[np.arange(21) % 3]
        return embeddings, instance, q

    def test_substitution_preserves_entire_clustering_and_skips_numpy_norm(self):
        embeddings, instance, q = self.fixture()
        original = VBxClustering.__call__
        norm = np.linalg.norm
        context = SimpleNamespace(normalize=lambda values: (values / norm(values, axis=1, keepdims=True), {"rows": len(values)}))
        records = []
        with patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            reference = original(instance, embeddings, min_clusters=1, max_clusters=10)
            with native_embedding_normalization(context, records):
                with patch("numpy.linalg.norm", side_effect=AssertionError("CPU normalization executed")):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
        for expected, result in zip(reference, actual):
            np.testing.assert_array_equal(result.view(np.uint8), expected.view(np.uint8))
        self.assertIs(VBxClustering.__call__, original)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["adapter_scope"], "vbx_training_embeddings_before_centroid_linkage")

    def test_native_failure_does_not_run_cpu_norm_and_scope_restores(self):
        embeddings, instance, q = self.fixture()
        original = VBxClustering.__call__
        def fail(values):
            raise ValueError("injected_native_failure")
        with self.assertRaisesRegex(ValueError, "injected_native_failure"):
            with native_embedding_normalization(SimpleNamespace(normalize=fail), []):
                with patch("numpy.linalg.norm", side_effect=AssertionError("CPU fallback executed")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
        self.assertIs(VBxClustering.__call__, original)

    def test_other_threads_and_small_training_set_keep_original_branch(self):
        embeddings, instance, q = self.fixture()
        context = SimpleNamespace(normalize=lambda values: (_ for _ in ()).throw(AssertionError("unexpected native call")))
        records = []
        original = VBxClustering.__call__
        with patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            reference = original(instance, embeddings, min_clusters=1, max_clusters=10)
            with native_embedding_normalization(context, records):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    actual = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                instance.filter_embeddings = lambda *a, **k: (embeddings.reshape(-1, 256)[:1], None, None)
                small = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                self.assertEqual(small[2].shape, (1, 256))
        for expected, result in zip(reference, actual):
            np.testing.assert_array_equal(result, expected)
        self.assertEqual(records, [])

    def test_source_version_drift_and_nested_scopes_are_rejected(self):
        original = VBxClustering.__call__
        with patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "source_not_supported"):
            compile_native_normalization(original, object())
        with patch("tools.baseline.native_embedding_adapter.importlib.metadata.version", return_value="changed"), self.assertRaisesRegex(ValueError, "version_not_supported"):
            with native_embedding_normalization(object(), []):
                pass
        changed_source = textwrap.dedent(inspect.getsource(original)).replace("keepdims=True", "keepdims=False")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed_source), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed_source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "contract_changed"):
            compile_native_normalization(original, object())
        with native_embedding_normalization(object(), []):
            with self.assertRaisesRegex(ValueError, "already_active"):
                with native_embedding_normalization(object(), []):
                    pass
        self.assertIs(VBxClustering.__call__, original)


if __name__ == "__main__":
    unittest.main()

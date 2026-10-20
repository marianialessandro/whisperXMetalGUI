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
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization, unconstrained_argmax
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeArgmaxTests(unittest.TestCase):
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

    def test_argmax_numpy_exact_arbitrary_binary64_bits_nan_ties_signed_zeros_and_limits(self):
        rng = np.random.default_rng(1627)
        raw = rng.integers(0, np.iinfo(np.uint64).max, size=(2048, 17), dtype=np.uint64)
        values = raw.view(np.float64)
        special = np.array([[1., np.nan, 9., np.nan], [np.inf, -np.inf, np.nan, np.inf], [-0., 0., -0., 0.], [0., -0., 0., -0.], [-np.inf, -np.inf, -np.inf, -np.inf], [3., 3., 3., 3.]])
        with self.context() as context:
            for scores in (values, special, np.ones((1, 1)), np.zeros((4096, 1024))):
                expected = np.argmax(scores, axis=1)
                scores.flags.writeable = False
                actual, record = context.argmax_vbx(scores)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.dtype(np.intp))
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertEqual(record["logical_result_readback_bytes"], len(scores) * 8)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_argmax_adapter_layouts_dtype_shape_guards_and_lifetime(self):
        values = np.arange(72, dtype=np.float64).reshape(4, 3, 6)
        with self.context() as context:
            for scores in (values, np.asfortranarray(values), values[:, :, ::2]):
                records = []
                actual = unconstrained_argmax(context, records, scores)
                np.testing.assert_array_equal(actual, np.argmax(scores, axis=2))
                self.assertEqual(len(records), 1)
            for scores in (np.ones((4097, 1, 1)), np.ones((1, 1, 1025)), np.ones((0, 2, 3))):
                with patch("tools.baseline.native_embedding_adapter.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                    unconstrained_argmax(context, [], scores)
            for scores in (np.ones((2, 3)), np.ones((2, 3, 4), np.float32)):
                with self.assertRaisesRegex(ValueError, "format_not_supported"):
                    unconstrained_argmax(context, [], scores)
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.argmax_vbx(np.ones((2, 3), np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.argmax_vbx(np.ones((2, 3)))
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in executor.map(lambda _: context.argmax_vbx(np.array([[1., 3., 2.]]))[0], range(4)):
                    np.testing.assert_array_equal(result, [1])
        with self.assertRaisesRegex(ValueError, "closed"):
            context.argmax_vbx(np.ones((2, 3)))

    def test_entire_clustering_argmax_composes_with_previous_native_stages_and_kmeans(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        norm = np.linalg.norm
        normalization = SimpleNamespace(normalize=lambda values: (values / norm(values, axis=1, keepdims=True), {}))
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for exact in (None, 2):
                for mask in range(8):
                    options = {"centroid_context": context if mask & 2 else None, "centroid_records": [], "distance_context": context if mask & 4 else None, "distance_records": [], "soft_scores": bool(mask & 4)}
                    native_norm = normalization if mask & 1 else None
                    if not mask:
                        expected = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    else:
                        with native_embedding_normalization(native_norm, [], **options):
                            expected = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    with native_embedding_normalization(native_norm, [], argmax_context=context, argmax_records=(records := []), **options):
                        actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    self.assertEqual(len(records), 1)
                    for left, right in zip(expected, actual):
                        np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))

    def test_constrained_small_and_unrelated_thread_branches_keep_original(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with native_embedding_normalization(None, [], argmax_context=context, argmax_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                self.assertEqual(records, [])
                instance.constrained_assignment = True
                instance.constrained_argmax = lambda values: np.argmax(values, axis=2)
                segmentations = SimpleNamespace(data=np.ones((7, 2, 3)))
                with patch.object(context, "argmax_vbx", side_effect=AssertionError("constrained branch dispatched native argmax")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=segmentations)
                self.assertEqual(records, [])
                instance.filter_embeddings = lambda *a, **k: (embeddings.reshape(-1, 256)[:1], None, None)
                small = VBxClustering.__call__(instance, embeddings)
                self.assertEqual(small[0].dtype, np.int8)
                self.assertEqual(records, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_source_capability_fault_and_restoration(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], argmax_context=context):
                    pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_embedding_normalization(None, [], argmax_context=context, argmax_records=[]):
                    pass
            with patch("tools.baseline.native_embedding_adapter.np.__version__", "changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_embedding_normalization(None, [], argmax_context=context, argmax_records=[]):
                    pass
            with patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))), self.assertRaisesRegex(ValueError, "injected_argmax_failure"):
                with native_embedding_normalization(None, [], argmax_context=context, argmax_records=(records := [])), patch.object(context, "argmax_vbx", side_effect=ValueError("injected_argmax_failure")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            self.assertEqual(records, [])
        self.assertIs(VBxClustering.__call__, original)
        changed = textwrap.dedent(inspect.getsource(original)).replace("np.argmax(soft_clusters, axis=2)", "np.argmax(soft_clusters, axis=-1)")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "argmax_contract_changed"):
            compile_native_normalization(original, None, argmax=object())


if __name__ == "__main__":
    unittest.main()

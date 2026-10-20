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

    def test_resident_primitive_matches_separate_scores_and_argmax_with_nan(self):
        rng = np.random.default_rng(1661)
        with self.context() as context:
            for rows, count, features in ((1, 1, 1), (19, 3, 8), (4096, 1, 1), (2, 2, 1024)):
                a, b = rng.normal(size=(rows, features)), rng.normal(size=(count, features))
                a[0] = 0.
                if rows > 1:
                    a[1, 0] = np.nan
                expected_scores, _ = context.cosine_vbx(a, b, soft_scores=True)
                expected_labels, _ = context.argmax_vbx(expected_scores)
                for _ in range(3):
                    (scores, labels), record = context.cosine_vbx(a, b, soft_scores=True, return_labels=True)
                    np.testing.assert_array_equal(scores.view(np.uint64), expected_scores.view(np.uint64))
                    np.testing.assert_array_equal(labels, expected_labels)
                    self.assertEqual(len(record["gpu_kernels"]), 3)
                    self.assertEqual(record["command_buffers_per_run"], 1)
                    self.assertEqual(record["logical_result_readback_bytes"], scores.nbytes + labels.nbytes)
                    self.assertTrue(record["resident_argmax"])
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            with self.assertRaisesRegex(ValueError, "score_mode_invalid"):
                context.cosine_vbx(a, b, return_labels=True)
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_cosine_scores=object())), self.assertRaisesRegex(ValueError, "cosine_argmax_abi_unavailable"):
                context.cosine_vbx(a, b, soft_scores=True, return_labels=True)

    def test_resident_whole_clustering_reuses_dispatch_multicall_and_constrained_bypass(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for with_centroids in (False, True):
                for exact in (None, 2):
                    options = {"centroid_context": context if with_centroids else None, "centroid_records": [], "distance_context": context, "distance_records": [], "soft_scores": True, "argmax_context": context, "argmax_records": []}
                    with native_embedding_normalization(None, [], **options):
                        expected = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                    distances, assignments = [], []
                    options.update(distance_records=distances, argmax_records=assignments)
                    with native_embedding_normalization(None, [], resident_argmax=True, **options), patch.object(context, "argmax_vbx", side_effect=AssertionError("scores reuploaded for standalone argmax")):
                        for _ in range(2):
                            actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact)
                            for left, right in zip(expected, actual):
                                np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))
                    self.assertEqual(len(distances), 2)
                    self.assertEqual([r["shared_cosine_dispatch_index"] for r in assignments], [0, 1])
                    self.assertTrue(all(r["command_buffers_per_run"] == 0 and r["logical_input_upload_bytes"] == 0 for r in assignments))
            instance.constrained_assignment = True
            instance.constrained_argmax = lambda values: np.argmax(values, axis=2)
            with native_embedding_normalization(None, [], distance_context=context, distance_records=(distances := []), soft_scores=True, argmax_context=context, argmax_records=(assignments := []), resident_argmax=True):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=SimpleNamespace(data=np.ones((7, 2, 3))))
            self.assertEqual(assignments, [])
            self.assertFalse(distances[0]["resident_argmax"])
            self.assertEqual(len(distances[0]["gpu_kernels"]), 2)

    def test_resident_configuration_failure_and_cache_cleanup(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_embedding_normalization(None, [], argmax_context=context, argmax_records=[], resident_argmax=True):
                    pass
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                compile_native_normalization(original, None, resident_argmax=True)
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_argmax=object(), wxdouble_vbx_cosine_scores=object())), self.assertRaisesRegex(ValueError, "cosine_argmax_abi_unavailable"):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=[], soft_scores=True, argmax_context=context, argmax_records=[], resident_argmax=True):
                    pass
            with patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
                with native_embedding_normalization(None, [], distance_context=context, distance_records=[], soft_scores=True, argmax_context=context, argmax_records=[], resident_argmax=True):
                    with patch.object(context, "cosine_vbx", side_effect=ValueError("injected_resident_failure")), self.assertRaisesRegex(ValueError, "injected_resident_failure"):
                        VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                    with patch.object(context, "cosine_vbx", return_value=((np.ones((21, 3)), np.zeros(20, dtype=np.int64)), {})), self.assertRaisesRegex(ValueError, "cache_mismatch"):
                        VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                    self.assertEqual(actual[0].shape, (7, 3))
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

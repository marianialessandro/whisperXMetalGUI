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
from pyannote.audio.pipelines.clustering import BaseClustering, VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization, inactive_speaker_mask
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeInactiveMaskTests(unittest.TestCase):
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

    def reference(self, scores, segments):
        output = scores.copy()
        output[segments.sum(1) == 0] = scores.min() - 1.
        return output

    def compare(self, actual, expected):
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(actual[finite].view(np.uint64), expected[finite].view(np.uint64))

    def test_mask_numpy_exact_layouts_binary_dtypes_and_partial_boundaries(self):
        rng = np.random.default_rng(1709)
        with self.context() as context:
            for chunks, speakers, columns, frames in ((1, 1, 1, 1), (4, 3, 7, 129), (1, 1, 1023, 3), (1, 1, 1024, 3), (5, 1, 205, 3), (4096, 1, 1024, 1)):
                scores = rng.uniform(0., 2., (chunks, speakers, columns))
                scores.reshape(-1)[-1] = 0.
                segments = np.ones((chunks, frames, speakers))
                segments[::2] = 0
                for dtype in (np.bool_, np.int8, np.float32, np.float64):
                    data = np.array(segments, dtype=dtype, order="F")
                    source = np.asfortranarray(scores)
                    source.flags.writeable = data.flags.writeable = False
                    records = []
                    actual = inactive_speaker_mask(context, records, source, data)
                    self.compare(actual, self.reference(source, data))
                    self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                    self.assertEqual(len(records[0]["gpu_kernels"]), 2)
                    self.assertLessEqual(records[0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_nan_signed_zero_and_tiny_minimum_subtraction_rounding(self):
        with self.context() as context:
            scores = np.array([[[np.nan, .2, .4], [0., -0., 1.]], [[1., 2., 1.5], [.1, 0., -0.]]])
            segments = np.array([[[0., 1.], [-0., 1.]], [[1., 0.], [1., -0.]]])
            actual = inactive_speaker_mask(context, [], scores, segments)
            self.compare(actual, self.reference(scores, segments))
            for minimum in (0., -0., np.nextafter(0., 1.), 2. ** -54, np.nextafter(2. ** -54, np.inf), np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 2.):
                scores = np.array([[[minimum, 2.]]])
                data = np.zeros((1, 2, 1))
                self.compare(inactive_speaker_mask(context, [], scores, data), self.reference(scores, data))

    def test_entire_clustering_constrained_matches_reference_and_unconstrained_bypasses(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        instance.constrained_assignment = True
        instance.constrained_argmax = lambda scores: BaseClustering.constrained_argmax(instance, scores)
        data = np.ones((7, 4, 3), dtype=np.float32)
        data[::2, :, 0] = 0
        segmentations = SimpleNamespace(data=data)
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            for exact in (None, 2):
                for combined in (False, True):
                    options = {"distance_context": context if combined else None, "distance_records": [], "soft_scores": combined, "argmax_context": context if combined else None, "argmax_records": [], "resident_argmax": combined}
                    if combined:
                        with native_embedding_normalization(None, [], **options):
                            expected = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact, segmentations=segmentations)
                    else:
                        expected = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact, segmentations=segmentations)
                    with native_embedding_normalization(None, [], inactive_context=context, inactive_records=(records := []), **options):
                        actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, num_clusters=exact, segmentations=segmentations)
                    self.assertEqual(len(records), int(exact is None))
                    for left, right in zip(expected, actual):
                        np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))

    def test_domain_shape_capability_lifetime_and_fault_restore(self):
        original = VBxClustering.__call__
        with self.context() as context:
            for value in (-1., np.inf, 2.1):
                with self.assertRaisesRegex(ValueError, "score_domain_invalid"):
                    context.inactive_mask_vbx(np.full((2, 3), value), np.zeros((2, 1)))
            for value in (-1., .5, np.nan):
                with self.assertRaisesRegex(ValueError, "binary_segments_required"):
                    context.inactive_mask_vbx(np.ones((2, 3)), np.full((2, 1), value))
            with patch("tools.baseline.native_embedding_adapter.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                inactive_speaker_mask(context, [], np.ones((4097, 1, 1)), np.ones((4097, 1, 1)))
            with self.assertRaisesRegex(ValueError, "format_not_supported"):
                inactive_speaker_mask(context, [], np.ones((1, 1, 1)), np.ones((1, 1, 1), dtype=np.complex128))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.inactive_mask_vbx(np.ones((2, 3)), np.ones((2, 1)))
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], inactive_context=context):
                    pass
            embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
            instance.constrained_assignment = True
            with patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))), self.assertRaisesRegex(ValueError, "injected_inactive_failure"):
                with native_embedding_normalization(None, [], inactive_context=context, inactive_records=[]), patch.object(context, "inactive_mask_vbx", side_effect=ValueError("injected_inactive_failure")):
                    VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=SimpleNamespace(data=np.ones((7, 2, 3))))
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in executor.map(lambda _: context.inactive_mask_vbx(np.ones((2, 3)), np.zeros((2, 1)))[0], range(4)):
                    np.testing.assert_array_equal(result, np.zeros((2, 3)))
        self.assertIs(VBxClustering.__call__, original)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.inactive_mask_vbx(np.ones((2, 3)), np.ones((2, 1)))

    def test_expression_pin_and_unrelated_thread(self):
        original = VBxClustering.__call__
        changed = textwrap.dedent(inspect.getsource(original)).replace("soft_clusters.min() - 1.", "soft_clusters.min() - 1")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "inactive_contract_changed"):
            compile_native_normalization(original, None, inactive=object())
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))), native_embedding_normalization(None, [], inactive_context=context, inactive_records=(records := [])):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
            self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()

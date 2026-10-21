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
import warnings

import numpy as np
import torch
from pyannote.audio.pipelines.clustering import BaseClustering, VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tools.baseline.native_resident_assignment import resident_assignment
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeResidentAssignmentTests(unittest.TestCase):
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
        masked = scores.copy()
        masked[segments.sum(1) == 0] = scores.min() - 1.
        return BaseClustering.constrained_argmax(object(), masked), masked

    def compare(self, actual, expected):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(np.isnan(actual[1]), np.isnan(expected[1]))
        finite = np.isfinite(expected[1])
        np.testing.assert_array_equal(actual[1][finite].view(np.uint64), expected[1][finite].view(np.uint64))

    def test_resident_mask_public_scores_ties_nan_rectangles_and_budgets(self):
        rng = np.random.default_rng(8111)
        with self.context() as context:
            for shape in ((51, 3, 3), (7, 2, 3), (7, 3, 2), (1, 16, 64), (64, 64, 16), (4096, 1, 64)):
                scores = rng.integers(0, 3, shape).astype(np.float64)
                segments = rng.integers(0, 2, (shape[0], 4, shape[1]), dtype=np.int8)
                segments[0, :, 0] = 0
                expected = self.reference(scores, segments)
                actual = resident_assignment(context, records := [], scores, segments)
                self.compare(actual, expected)
                record = records[0]
                self.assertEqual(record["command_buffers_per_run"], 1)
                self.assertEqual(len(record["gpu_kernels"]), 5)
                self.assertEqual(record["gpu_blit_encoders"], 1)
                self.assertEqual(record["logical_public_soft_score_readback_bytes"], scores.nbytes)
                self.assertEqual(record["intermediate_score_readback_bytes"], 0)
                self.assertEqual(record["assignment_pair_readback_bytes"], 0)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            scores = np.array([[[np.nan, 1.], [0., -0.]], [[2. ** -64, 2.], [1., 1.]]])
            segments = np.ones((2, 4, 2), dtype=np.bool_)
            segments[0, :, 1] = 0
            expected = self.reference(scores, segments)
            for source in (scores, np.asfortranarray(scores), np.repeat(scores, 2, axis=2)[:, :, ::2]):
                source.flags.writeable = False
                self.compare(resident_assignment(context, [], source, segments), expected)
            for minimum in (0., -0., 2. ** -64, 2. ** -54, np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 2.):
                scores = np.full((2, 2, 2), minimum)
                self.compare(resident_assignment(context, [], scores, segments), self.reference(scores, segments))
            scores = np.ones((1, 2, 3))
            segments = np.ones((1, 2, 4))
            for array in (scores, segments):
                unaligned = np.ndarray(array.shape, dtype=np.float64, buffer=bytearray(array.nbytes + 1), offset=1)
                unaligned[:] = array
                if array is scores:
                    scores = unaligned
                else:
                    segments = unaligned
            (labels, masked), _ = context.inactive_assignment(scores, segments)
            np.testing.assert_array_equal(labels, [[0, 1]])
            np.testing.assert_array_equal(masked, scores)

    def test_domains_all_nan_after_mask_and_capabilities(self):
        with self.context() as context:
            scores, segments = np.ones((2, 3, 3)), np.ones((2, 3, 4))
            for value in (-1., np.inf, -np.inf, 2.01, 2. ** -65):
                with self.assertRaisesRegex(ValueError, "score_domain_invalid"):
                    context.inactive_assignment(np.full(scores.shape, value), segments)
            for value in (-1., .5, np.nan, np.inf, 2.):
                with self.assertRaisesRegex(ValueError, "binary_segments_required"):
                    context.inactive_assignment(scores, np.full(segments.shape, value))
            for shape in ((0, 2, 3), (1, 65, 2), (1, 2, 65), (1, 17, 17), (2049, 2, 1)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.inactive_assignment(np.ones(shape), np.ones((*shape[:2], 1)))
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                context.inactive_assignment(scores, np.ones((2, 3, 4097)))
            for source in (np.full((2, 3, 3), np.nan), np.where(np.indices((2, 3, 3))[2] == 0, np.nan, 1.)):
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    with self.assertRaisesRegex(ValueError, "all_nan_input"):
                        resident_assignment(context, records := [], source, np.zeros((2, 4, 3)))
                self.assertEqual([str(item.message) for item in observed], ["All-NaN slice encountered"])
                self.assertEqual(records, [])
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.inactive_assignment(scores, segments)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.inactive_assignment(scores, segments)

    def fixture(self):
        embeddings, namespace, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        instance = object.__new__(VBxClustering)
        instance.__dict__.update(namespace.__dict__)
        instance.constrained_assignment = True
        return embeddings, instance, q

    def test_whole_vbx_public_result_cpu_solver_displacement_threads_and_restoration(self):
        embeddings, instance, q = self.fixture()
        original = VBxClustering.__call__
        segmentations = SimpleNamespace(data=np.ones((7, 4, 3), np.int8))
        segmentations.data[0, :, 1] = 0
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with clustering_cpu_audit(reference_audit := []):
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=segmentations)
            self.assertEqual(next(row for row in reference_audit if row["operation"] == "linear_sum_assignment")["calls_started"], 7)
            with clustering_cpu_audit(audit := []), native_embedding_normalization(None, [], resident_assignment_context=context, resident_assignment_records=(records := [])):
                actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=segmentations)
                self.compare(actual[:2], expected[:2])
                np.testing.assert_array_equal(actual[2], expected[2])
                self.assertEqual(len(records), 1)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    other = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10, segmentations=segmentations).result()
                self.compare(other[:2], expected[:2])
                self.assertEqual(len(records), 1)
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_embedding_normalization(None, [], resident_assignment_context=context, resident_assignment_records=[]):
                        pass
            self.assertEqual(next(row for row in audit if row["operation"] == "linear_sum_assignment")["calls_started"], 0)
            with self.assertRaisesRegex(ValueError, "injected_resident_failure"), native_embedding_normalization(None, [], resident_assignment_context=context, resident_assignment_records=[]), patch.object(context, "inactive_assignment", side_effect=ValueError("injected_resident_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=segmentations)
            self.assertIs(VBxClustering.__call__, original)
            instance.constrained_argmax = lambda scores: np.zeros(scores.shape[:2], np.int8)
            with self.assertRaisesRegex(ValueError, "instance_not_supported"), native_embedding_normalization(None, [], resident_assignment_context=context, resident_assignment_records=[]):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10, segmentations=segmentations)
        self.assertIs(VBxClustering.__call__, original)

    def test_source_runtime_and_configuration_guards(self):
        original = VBxClustering.__call__
        source = textwrap.dedent(inspect.getsource(original))
        for changed in (source.replace("soft_clusters.min() - 1.", "soft_clusters.min() - 2."), source.replace("self.constrained_argmax(", "self.other_argmax("), source.replace("segmentations.data.sum(1)", "segmentations.data.sum(2)")):
            with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
                compile_native_normalization(original, None, resident_assignment=object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_embedding_normalization(None, [], inactive_context=context, inactive_records=[], resident_assignment_context=context, resident_assignment_records=[]):
                    pass
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_embedding_normalization(None, [], resident_assignment_context=context):
                    pass
            with patch("tools.baseline.native_embedding_adapter.np.__version__", "future"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_embedding_normalization(None, [], resident_assignment_context=context, resident_assignment_records=[]):
                    pass
        self.assertIs(VBxClustering.__call__, original)

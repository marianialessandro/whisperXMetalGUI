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
from pyannote.audio.pipelines.clustering import BaseClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_assignment import batch_labels, native_batched_assignment
from tools.baseline.native_constrained_cleanup import compile_native_cleanup, native_constrained_nan_cleanup
from tools.baseline.native_double import NativeDoubleNormalization, build_library


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAssignmentBatchTests(unittest.TestCase):
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

    def test_global_nan_replacement_labels_ties_rectangles_and_maximum_budgets(self):
        rng = np.random.default_rng(2811)
        original = BaseClustering.constrained_argmax
        with self.context() as context:
            for shape in ((1, 1, 1), (7, 2, 3), (7, 3, 2), (51, 3, 3), (1, 16, 64), (64, 64, 16), (4096, 1, 64)):
                scores = rng.integers(-1, 3, shape).astype(np.float64)
                scores.flat[::11] = np.nan
                scores.flat[-1] = -1.
                expected = original(object(), scores)
                actual, record = context.assignment_batch(scores)
                np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
                self.assertEqual(actual.dtype, np.int8)
                self.assertEqual(record["command_buffers_per_run"], 1)
                self.assertEqual(len(record["gpu_kernels"]), 3)
                self.assertEqual(record["logical_result_readback_bytes"], actual.nbytes + shape[0] * 4)
                self.assertEqual(record["intermediate_score_readback_bytes"], 0)
                self.assertEqual(record["assignment_pair_readback_bytes"], 0)
                self.assertFalse(record["host_numeric_label_scatter"])
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            scores = np.array([[[np.nan, np.nan], [np.nan, np.nan]], [[-1., 2.], [0., 1.]], [[0., -0.], [-0., 0.]]])
            expected = original(object(), scores)
            for source in (np.asfortranarray(scores), np.repeat(scores, 2, axis=2)[:, :, ::2]):
                source.flags.writeable = False
                actual = batch_labels(context, records := [], source)
                np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
                self.assertEqual(records[0]["host_input_staging_copy_bytes"], source.nbytes)
            unaligned = np.ndarray(scores.shape, dtype=np.float64, buffer=bytearray(scores.nbytes + 1), offset=1)
            unaligned[:] = scores
            self.assertFalse(unaligned.flags.aligned)
            actual, _ = context.assignment_batch(unaligned)
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))

    def test_shapes_domains_all_nan_warning_and_closed_context(self):
        with self.context() as context:
            for scores in (np.ones((2, 3, 4), np.float32), np.ones((3, 4)), np.ones((2, 3, 8))[:, :, ::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.assignment_batch(scores)
            for shape in ((0, 2, 3), (4097, 1, 1), (2049, 2, 1), (1, 65, 2), (1, 2, 65), (1, 17, 17), (1, 2, 0)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.assignment_batch(np.ones(shape))
            for value in (-1.01, 2.01, np.inf, -np.inf, 2. ** -65, -2. ** -65):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.assignment_batch(np.full((2, 3, 4), value))
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(ValueError, "all_nan_input"):
                    batch_labels(context, records := [], np.full((2, 3, 4), np.nan))
            self.assertEqual([str(item.message) for item in observed], ["All-NaN slice encountered"])
            self.assertEqual(records, [])
        with self.assertRaisesRegex(ValueError, "closed"):
            context.assignment_batch(np.ones((2, 3, 4)))

    def test_whole_method_displaces_cpu_solver_cleanup_initialization_and_scatter(self):
        original = BaseClustering.constrained_argmax
        scores = np.random.default_rng(2813).integers(-1, 3, (51, 7, 4)).astype(np.float64)
        scores.flat[::11] = np.nan
        scores.flat[-1] = -1.
        with clustering_cpu_audit(reference_audit := []):
            expected = original(object(), scores)
        self.assertEqual(next(row for row in reference_audit if row["operation"] == "linear_sum_assignment")["calls_started"], 51)
        with self.context() as context, clustering_cpu_audit(audit := []), native_batched_assignment(context, records := []), patch("numpy.nanmin", side_effect=AssertionError("CPU nanmin executed")), patch("numpy.nan_to_num", side_effect=AssertionError("CPU nan_to_num executed")), patch("numpy.ones", side_effect=AssertionError("CPU label initialization executed")):
            actual = BaseClustering.constrained_argmax(object(), scores)
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["logical_assignment_matrices"], 51)
        self.assertEqual(next(row for row in audit if row["operation"] == "linear_sum_assignment")["calls_started"], 0)
        self.assertTrue(all(row["observation_complete"] for row in audit))
        self.assertIs(BaseClustering.constrained_argmax, original)

    def test_caps_threads_nested_conflicting_modes_and_failure_restore_original(self):
        original = BaseClustering.constrained_argmax
        scores = np.array([[[np.nan, 2., -1.], [1., np.nan, 0.]]])
        expected = original(object(), scores)
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "batch_records_required"):
                with native_batched_assignment(context, ()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_nan_cleanup=object())), self.assertRaisesRegex(ValueError, "batch_abi_unavailable"):
                with native_batched_assignment(context, []):
                    pass
            with self.assertRaisesRegex(ValueError, "batch_configuration_invalid"):
                with native_constrained_nan_cleanup(context, [], assignment_records=[], batch_records=[]):
                    pass
            with native_batched_assignment(context, records := []):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    actual = executor.submit(BaseClustering.constrained_argmax, object(), scores).result()
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(records, [])
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_batched_assignment(context, []):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_batch_failure"):
                with native_batched_assignment(context, records := []), patch.object(context, "assignment_batch", side_effect=ValueError("injected_batch_failure")):
                    BaseClustering.constrained_argmax(object(), scores)
            self.assertEqual(records, [])
        self.assertIs(BaseClustering.constrained_argmax, original)

    def test_loop_allocation_shape_return_and_runtime_reference_guards(self):
        original = BaseClustering.constrained_argmax
        source = textwrap.dedent(inspect.getsource(original))
        for old, new in (("hard_clusters = -2", "hard_clusters = -1"), ("maximize=True", "maximize=False"), ("return hard_clusters", "return hard_clusters.copy()"), ("soft_clusters.shape", "soft_clusters.shape[::-1]")):
            changed = source.replace(old, new)
            with patch("tools.baseline.native_constrained_cleanup.inspect.getsource", return_value=changed), patch("tools.baseline.native_constrained_cleanup.CONSTRAINED_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "batch_expression_contract_changed"):
                compile_native_cleanup(original, object(), batch=object())
        with patch("tools.baseline.native_assignment.SOLVER_BINARY_SHA256", "changed"), self.context() as context, self.assertRaisesRegex(ValueError, "reference_not_supported"):
            with native_batched_assignment(context, []):
                pass
        with self.context() as context, patch.dict(original.__globals__, {"np": object()}), self.assertRaisesRegex(ValueError, "numpy_reference_changed"):
            with native_batched_assignment(context, []):
                pass
        self.assertIs(BaseClustering.constrained_argmax, original)


if __name__ == "__main__":
    unittest.main()

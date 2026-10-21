from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import itertools
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
from scipy.optimize import linear_sum_assignment
import torch
from pyannote.audio.pipelines.clustering import BaseClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_assignment import assignment_pairs, validate_assignment_reference
from tools.baseline.native_constrained_cleanup import compile_native_cleanup, native_constrained_nan_cleanup
from tools.baseline.native_double import NativeDoubleNormalization, build_library


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAssignmentTests(unittest.TestCase):
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

    def assert_scipy(self, context, scores):
        expected = linear_sum_assignment(scores, maximize=True)
        actual, record = context.linear_assignment_max(scores)
        for left, right in zip(actual, expected):
            self.assertEqual(left.dtype, right.dtype)
            np.testing.assert_array_equal(left, right)
        self.assertGreater(record["computation_gpu_seconds"], 0)
        self.assertEqual(record["logical_result_readback_bytes"], 16 * min(scores.shape) + 4)
        self.assertEqual(record["logical_status_readback_bytes"], 4)
        self.assertFalse(record["scipy_assignment_certified"])
        self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_exhaustive_rectangular_integer_ties_both_orientations_and_float_rounding(self):
        rng = np.random.default_rng(2711)
        with self.context() as context:
            for values in itertools.product((-1., 0., 1.), repeat=6):
                scores = np.array(values).reshape(2, 3)
                self.assert_scipy(context, scores)
                self.assert_scipy(context, scores.T.copy())
            for rows, columns in ((1, 1), (2, 3), (3, 2), (3, 3), (4, 7), (7, 4), (16, 64), (64, 16)):
                for mode in ("zero", "integer", "float", "near_tie", "extremes"):
                    if mode == "zero":
                        scores = np.zeros((rows, columns))
                        scores.flat[::2] = -0.
                    elif mode == "integer":
                        scores = rng.integers(-3, 4, (rows, columns)).astype(np.float64)
                    elif mode == "float":
                        scores = rng.normal(size=(rows, columns))
                    elif mode == "near_tie":
                        scores = np.ones((rows, columns))
                        scores.flat[::3] = np.nextafter(1., 2.)
                    else:
                        scores = rng.choice(np.array([-2. ** 64, -1e10, -2. ** -64, 0., 2. ** -64, 2. ** 64]), size=(rows, columns))
                    self.assert_scipy(context, np.ascontiguousarray(scores))

    def test_layout_staging_domains_shapes_readonly_and_closed_context(self):
        with self.context() as context:
            for scores in (np.asfortranarray(np.arange(12.).reshape(3, 4)), np.arange(24.).reshape(3, 8)[:, ::2]):
                scores.flags.writeable = False
                actual = assignment_pairs(context, records := [], scores)
                expected = linear_sum_assignment(scores, maximize=True)
                for left, right in zip(actual, expected):
                    np.testing.assert_array_equal(left, right)
                self.assertEqual(records[0]["host_input_staging_copy_bytes"], scores.nbytes)
            for scores in (np.ones((2, 3), np.float32), np.ones(3), np.ones((3, 4))[:, ::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.linear_assignment_max(scores)
            for rows, columns in ((0, 1), (1, 0), (65, 2), (2, 65), (17, 17)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.linear_assignment_max(np.ones((rows, columns)))
            for value in (np.nan, np.inf, -np.inf, 2. ** -65, 2. ** 65, -2. ** -65, -2. ** 65):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.linear_assignment_max(np.full((2, 3), value))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.linear_assignment_max(np.ones((2, 3)))

    def test_complete_constrained_labels_match_and_original_cpu_solver_calls_are_displaced(self):
        original = BaseClustering.constrained_argmax
        rng = np.random.default_rng(2713)
        with self.context() as context:
            for speakers, columns in ((1, 1), (2, 3), (3, 2), (3, 3), (4, 7), (7, 4)):
                scores = rng.integers(-1, 3, (40, speakers, columns)).astype(np.float64)
                scores.flat[::11] = np.nan
                scores.flat[-1] = -1.
                with clustering_cpu_audit(reference_audit := []):
                    expected = original(object(), scores)
                self.assertEqual(next(row for row in reference_audit if row["operation"] == "linear_sum_assignment")["calls_started"], 40)
                with clustering_cpu_audit(audit := []), native_constrained_nan_cleanup(context, cleanup := [], assignment_records=(records := [])), patch("numpy.nanmin", side_effect=AssertionError("CPU nanmin executed")), patch("numpy.nan_to_num", side_effect=AssertionError("CPU nan_to_num executed")):
                    actual = BaseClustering.constrained_argmax(object(), scores)
                np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
                self.assertEqual(actual.dtype, np.int8)
                self.assertEqual(len(records), 40)
                self.assertEqual(len(cleanup), 1)
                self.assertNotIn("linear_sum_assignment", cleanup[0]["cpu_residuals"])
                self.assertEqual(next(row for row in audit if row["operation"] == "linear_sum_assignment")["calls_started"], 0)
                self.assertTrue(all(row["observation_complete"] for row in audit))
        self.assertIs(BaseClustering.constrained_argmax, original)

    def test_missing_abi_threads_nested_and_native_failure_restore_method(self):
        original = BaseClustering.constrained_argmax
        scores = np.array([[[1., 0., 1.], [0., 1., 1.]]])
        expected = original(object(), scores)
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "assignment_records_required"):
                with native_constrained_nan_cleanup(context, [], assignment_records=()):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_nan_cleanup=object())), self.assertRaisesRegex(ValueError, "assignment_abi_unavailable"):
                with native_constrained_nan_cleanup(context, [], assignment_records=[]):
                    pass
            with native_constrained_nan_cleanup(context, cleanup := [], assignment_records=(records := [])):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    actual = executor.submit(BaseClustering.constrained_argmax, object(), scores).result()
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(records, [])
                self.assertEqual(cleanup, [])
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_constrained_nan_cleanup(context, [], assignment_records=[]):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_assignment_failure"):
                with native_constrained_nan_cleanup(context, [], assignment_records=(records := [])), patch.object(context, "linear_assignment_max", side_effect=ValueError("injected_assignment_failure")):
                    BaseClustering.constrained_argmax(object(), scores)
            self.assertEqual(records, [])
        self.assertIs(BaseClustering.constrained_argmax, original)

    def test_source_maximize_policy_and_solver_binary_drift_are_rejected(self):
        original = BaseClustering.constrained_argmax
        source = textwrap.dedent(inspect.getsource(original)).replace("maximize=True", "maximize=False")
        with patch("tools.baseline.native_constrained_cleanup.inspect.getsource", return_value=source), patch("tools.baseline.native_constrained_cleanup.CONSTRAINED_SHA256", hashlib.sha256(source.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "assignment_expression_contract_changed"):
            compile_native_cleanup(original, object(), assignment=object())
        with patch("tools.baseline.native_assignment.SOLVER_BINARY_SHA256", "changed"), self.assertRaisesRegex(ValueError, "reference_not_supported"):
            validate_assignment_reference(linear_sum_assignment)

    def test_all_nan_input_keeps_warning_and_fails_closed_before_any_assignment_record(self):
        original = BaseClustering.constrained_argmax
        with self.context() as context, native_constrained_nan_cleanup(context, [], assignment_records=(records := [])), warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                BaseClustering.constrained_argmax(object(), np.full((2, 3, 4), np.nan))
        self.assertTrue(any(str(item.message) == "All-NaN slice encountered" for item in observed))
        self.assertEqual(records, [])
        self.assertIs(BaseClustering.constrained_argmax, original)


if __name__ == "__main__":
    unittest.main()

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
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_constrained_cleanup import compile_native_cleanup, cleanup_scores, native_constrained_nan_cleanup


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeConstrainedCleanupTests(unittest.TestCase):
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

    def test_minimum_and_nan_replacement_numpy_values_boundaries_and_layouts(self):
        rng = np.random.default_rng(1741)
        with self.context() as context:
            for chunks, speakers, columns in ((1, 1, 1), (4, 3, 7), (1, 1, 1024), (5, 1, 205), (4096, 1, 1024)):
                scores = rng.uniform(-1., 2., (chunks, speakers, columns))
                scores.reshape(-1)[::7] = np.nan
                scores.reshape(-1)[-1] = -1.
                expected = np.nan_to_num(scores, nan=np.nanmin(scores))
                source = np.asfortranarray(scores)
                source.flags.writeable = False
                records = []
                actual = cleanup_scores(context, records, source)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(len(records[0]["gpu_kernels"]), 2)
                self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                self.assertLessEqual(records[0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for values in ([np.nan, -0., 0.], [np.nan, np.nextafter(0., 1.), 1.], [np.nan, -np.nextafter(0., 1.), 1.]):
                scores = np.array([[values]])
                actual = cleanup_scores(context, [], scores)
                expected = np.nan_to_num(scores, nan=np.nanmin(scores))
                np.testing.assert_array_equal(actual, expected)

    def test_whole_solver_preserves_assignments_ties_and_avoids_cpu_preparation(self):
        original = BaseClustering.constrained_argmax
        rng = np.random.default_rng(1753)
        with self.context() as context:
            for speakers, columns in ((1, 1), (2, 3), (3, 2), (3, 3), (4, 7), (7, 4)):
                scores = rng.integers(-1, 3, size=(80, speakers, columns)).astype(np.float64)
                scores.reshape(-1)[::11] = np.nan
                scores.reshape(-1)[-1] = -1.
                expected = original(object(), scores)
                records = []
                with native_constrained_nan_cleanup(context, records), patch("numpy.nanmin", side_effect=AssertionError("CPU nanmin executed")), patch("numpy.nan_to_num", side_effect=AssertionError("CPU nan_to_num executed")):
                    actual = BaseClustering.constrained_argmax(object(), scores)
                np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
                self.assertEqual(len(records), 1)
        self.assertIs(BaseClustering.constrained_argmax, original)

    def test_all_nan_warning_and_original_solver_error_are_preserved(self):
        scores = np.full((2, 3, 4), np.nan)
        original = BaseClustering.constrained_argmax
        with warnings.catch_warnings(record=True) as reference_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError) as reference_error:
                original(object(), scores)
        with self.context() as context, native_constrained_nan_cleanup(context, records := []), warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError) as actual_error:
                BaseClustering.constrained_argmax(object(), scores)
        self.assertEqual(str(reference_error.exception), str(actual_error.exception))
        self.assertEqual([str(item.message) for item in reference_warnings], [str(item.message) for item in actual_warnings])
        self.assertTrue(records[0]["all_nan_input"])
        self.assertIs(BaseClustering.constrained_argmax, original)

    def test_domains_shapes_lifetime_and_capability(self):
        with self.context() as context:
            for value in (np.inf, -np.inf, -1.1, 2.1):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.nan_cleanup_vbx(np.full((2, 3), value))
            with patch("tools.baseline.native_constrained_cleanup.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                cleanup_scores(context, [], np.ones((4097, 1, 1)))
            with self.assertRaisesRegex(ValueError, "format_not_supported"):
                cleanup_scores(context, [], np.ones((1, 2, 3), np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.nan_cleanup_vbx(np.ones((2, 3)))
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_constrained_nan_cleanup(context, None):
                    pass
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in executor.map(lambda _: context.nan_cleanup_vbx(np.array([[np.nan, -1.]]))[0], range(4)):
                    np.testing.assert_array_equal(result, [[-1., -1.]])
        with self.assertRaisesRegex(ValueError, "closed"):
            context.nan_cleanup_vbx(np.ones((2, 3)))

    def test_scope_thread_failure_nested_and_source_restore(self):
        original = BaseClustering.constrained_argmax
        values = np.array([[[np.nan, 2., -1.], [1., np.nan, 0.]]])
        expected = original(object(), values)
        with self.context() as context:
            with native_constrained_nan_cleanup(context, records := []):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    actual = executor.submit(BaseClustering.constrained_argmax, object(), values).result()
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(records, [])
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_constrained_nan_cleanup(context, []):
                        pass
            with self.assertRaisesRegex(ValueError, "injected_cleanup_failure"):
                with native_constrained_nan_cleanup(context, records := []), patch.object(context, "nan_cleanup_vbx", side_effect=ValueError("injected_cleanup_failure")):
                    BaseClustering.constrained_argmax(object(), values)
            self.assertEqual(records, [])
        self.assertIs(BaseClustering.constrained_argmax, original)
        changed = textwrap.dedent(inspect.getsource(original)).replace("np.nanmin(soft_clusters)", "np.min(soft_clusters)")
        with patch("tools.baseline.native_constrained_cleanup.inspect.getsource", return_value=changed), patch("tools.baseline.native_constrained_cleanup.CONSTRAINED_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_cleanup(original, object())


if __name__ == "__main__":
    unittest.main()

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pyannote.audio.core.plda import PLDA
import pyannote.audio.utils.vbx as vbx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_plda_adapter import native_plda_normalization, normalize_l2


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativePLDAAdapterTests(unittest.TestCase):
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

    def test_layout_exactness_readonly_and_copy_accounting(self):
        values = np.random.default_rng(784).normal(size=(33, 257))
        with self.context() as context:
            for array in (values, np.asfortranarray(values)):
                expected = vbx.l2_norm(array)
                array.flags.writeable = False
                records = []
                actual = normalize_l2(context, records, array)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(actual.flags.c_contiguous, expected.flags.c_contiguous)
                self.assertEqual(actual.flags.f_contiguous, expected.flags.f_contiguous)
                self.assertEqual(records[0]["host_input_staging_copy_bytes"], 0 if array.flags.c_contiguous else array.nbytes)
                self.assertEqual(records[0]["host_output_layout_copy_bytes"], 0 if array.flags.c_contiguous else array.nbytes)
            for invalid in (values[:, ::2], values.astype(np.float32), values[0]):
                with self.assertRaisesRegex(ValueError, "layout_or_dtype"):
                    normalize_l2(context, [], invalid)

    def test_original_plda_transform_is_bitwise_equal_with_two_native_calls(self):
        root = Path(self.directory.name)
        rng = np.random.default_rng(152)
        transform, model = root / "transform.npz", root / "plda.npz"
        np.savez(transform, mean1=rng.normal(size=16), mean2=rng.normal(size=8).astype(np.float32), lda=rng.normal(size=(16, 8)).astype(np.float32))
        np.savez(model, mu=rng.normal(size=8), tr=np.diag(np.linspace(1., 2., 8)), psi=np.linspace(.4, 2., 8))
        plda = PLDA(transform, model, lda_dimension=8)
        values = rng.normal(size=(17, 16)).astype(np.float32)
        expected = plda(values)
        original = vbx.l2_norm
        records = []
        with self.context() as context:
            with native_plda_normalization(context, records), patch("numpy.linalg.norm", side_effect=AssertionError("CPU norm executed")):
                actual = plda(values)
                keyword_actual = vbx.l2_norm(vec_or_matrix=values.astype(np.float64))
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        self.assertEqual(len(records), 3)
        self.assertEqual([r["input_layout"] for r in records], ["C", "F", "C"])
        np.testing.assert_array_equal(keyword_actual, original(values.astype(np.float64)))
        self.assertIs(vbx.l2_norm, original)

    def test_failures_restore_scope_and_do_not_run_cpu_fallback(self):
        original = vbx.l2_norm
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "zero_row"):
                with native_plda_normalization(context, []), patch("numpy.linalg.norm", side_effect=AssertionError("CPU fallback executed")):
                    vbx.l2_norm(np.zeros((2, 8)))
            self.assertIs(vbx.l2_norm, original)
            with native_plda_normalization(context, []):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_plda_normalization(context, []):
                        pass
            with patch("tools.baseline.native_plda_adapter.L2_NORM_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "source_not_supported"):
                with native_plda_normalization(context, []):
                    pass
            with patch("tools.baseline.native_plda_adapter.VBX_SETUP_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "source_not_supported"):
                with native_plda_normalization(context, []):
                    pass
            with patch("tools.baseline.native_plda_adapter.importlib.metadata.version", return_value="changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_plda_normalization(context, []):
                    pass
        self.assertIs(vbx.l2_norm, original)

    def test_unrelated_threads_use_original_without_native_calls(self):
        values = np.random.default_rng(739).normal(size=(11, 128))
        expected = vbx.l2_norm(values)
        records = []
        with self.context() as context, native_plda_normalization(context, records):
            with ThreadPoolExecutor(max_workers=1) as executor:
                actual = executor.submit(vbx.l2_norm, values).result()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()

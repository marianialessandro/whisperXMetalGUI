from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyannote.audio.core.plda import PLDA
import pyannote.audio.utils.vbx as vbx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_plda_adapter import native_plda_normalization
from tools.baseline.native_plda_matrix import native_plda_matrix_products, prepared_parameters


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativePLDAMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.library = root / "double.dylib"
        cls.identity = build_library(cls.library)
        rng = np.random.default_rng(644)
        cls.transform, cls.model = root / "transform.npz", root / "model.npz"
        np.savez(cls.transform, mean1=rng.normal(size=32), mean2=rng.normal(size=16).astype(np.float32), lda=rng.normal(size=(32, 16)).astype(np.float32))
        np.savez(cls.model, mu=rng.normal(size=16), tr=np.diag(np.linspace(1., 2., 16)), psi=np.linspace(.4, 2., 16))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeDoubleNormalization(self.library, self.identity["library_sha256"])

    def ordered_product(self, left, right):
        output = np.zeros((len(left), right.shape[1]))
        for k in range(left.shape[1]):
            output = np.add(output, np.multiply(left[:, k, None], right[k, None, :]))
        return output

    def ordered_reference(self, model, values):
        x = inspect.getclosurevars(model._xvec_tf).nonlocals
        p = inspect.getclosurevars(model._plda_tf).nonlocals
        scaled = np.sqrt(x["lda"].shape[0]) * vbx.l2_norm(values - x["mean1"]).T
        projected = self.ordered_product(x["lda"].T.astype(np.float64), scaled).T - x["mean2"]
        transformed = np.sqrt(x["lda"].shape[1]) * vbx.l2_norm(projected)
        return self.ordered_product(transformed - p["plda_mu"], p["plda_tr"].T)[:, :model.lda_dimension]

    def test_ordered_reference_original_blas_comparison_truncation_and_records(self):
        model = PLDA(self.transform, self.model, lda_dimension=11)
        values = np.random.default_rng(838).normal(size=(17, 32)).astype(np.float32)
        original_result, expected = model(values), self.ordered_reference(model, values)
        records, norms = [], []
        original = PLDA.__call__
        with self.context() as context, native_plda_normalization(context, norms), native_plda_matrix_products(context, records):
            actual = model(embeddings=values)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_allclose(actual, original_result, rtol=1e-10, atol=1e-10)
        self.assertIs(PLDA.__call__, original)
        self.assertEqual(len(norms), 2)
        self.assertEqual([r["operation"] for r in records], ["lda", "plda"])
        self.assertFalse(records[0]["reference_blas_bitwise_certified"])
        self.assertGreater(records[0]["host_left_staging_copy_bytes"], 0)
        self.assertEqual(set(records[0]["prepared_parameter_sha256"]), {"mean1", "mean2", "lda", "plda_mu", "plda_tr"})

    def test_invalid_parameters_native_failure_and_nested_scope_restore(self):
        model = PLDA(self.transform, self.model, lda_dimension=16)
        values = np.random.default_rng(554).normal(size=(7, 32)).astype(np.float32)
        original = PLDA.__call__
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "injected_native_failure"):
                with native_plda_matrix_products(context, []), patch.object(context, "matmul", side_effect=ValueError("injected_native_failure")):
                    model(values)
            self.assertIs(PLDA.__call__, original)
            with native_plda_matrix_products(context, []):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_plda_matrix_products(context, []):
                        pass
                with self.assertRaisesRegex(ValueError, "embedding_shape"):
                    model(np.ones((2, 31), dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "embedding_shape"):
                    model(np.ones((1025, 32), dtype=np.float32))
            with patch.object(model, "_xvec_tf", lambda x: x), self.assertRaisesRegex(ValueError, "closure_not_supported"):
                prepared_parameters(model, vbx)
            model.lda_dimension = 17
            with self.assertRaisesRegex(ValueError, "dimension_not_supported"):
                prepared_parameters(model, vbx)
        self.assertIs(PLDA.__call__, original)

    def test_parameter_snapshot_is_used_through_both_products(self):
        model = PLDA(self.transform, self.model, lda_dimension=16)
        values = np.random.default_rng(926).normal(size=(7, 32)).astype(np.float32)
        expected = self.ordered_reference(model, values)
        records = []
        with self.context() as context:
            original_product = context.matmul
            calls = []
            def mutate_after_first(left, right):
                output = original_product(left, right)
                calls.append(True)
                if len(calls) == 1:
                    inspect.getclosurevars(model._plda_tf).nonlocals["plda_mu"][:] += 1.
                return output
            with native_plda_matrix_products(context, records), patch.object(context, "matmul", side_effect=mutate_after_first):
                actual = model(values)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        self.assertGreater(records[0]["parameter_snapshot_copy_bytes"], 0)
        self.assertEqual(records[1]["parameter_snapshot_copy_bytes"], 0)

    def test_native_center_scale_matches_previous_policy_without_cpu_square_roots(self):
        model = PLDA(self.transform, self.model, lda_dimension=11)
        values = np.random.default_rng(631).normal(size=(17, 32)).astype(np.float32)
        expected = self.ordered_reference(model, values)
        records, norms, elements = [], [], []
        original = PLDA.__call__
        with self.context() as context, native_plda_normalization(context, norms), native_plda_matrix_products(context, records, centering_scaling=True, elementwise_records=elements):
            with patch("numpy.sqrt", side_effect=AssertionError("CPU square root executed")):
                actual = model(values)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        self.assertEqual(len(records), 2)
        self.assertEqual(len(norms), 2)
        self.assertEqual([r["transform_step"] for r in elements], ["center_input", "scale_before_lda", "center_after_lda", "scale_before_plda", "center_before_plda"])
        self.assertEqual([r["input_layout"] for r in elements], ["C", "F", "F", "F", "F"])
        self.assertTrue(elements[0]["host_dtype_conversion"])
        self.assertNotIn("centering", records[0]["cpu_residuals"])
        self.assertIs(PLDA.__call__, original)

    def test_elementwise_failure_capability_and_records_validation_restore_scope(self):
        model = PLDA(self.transform, self.model, lda_dimension=16)
        values = np.random.default_rng(633).normal(size=(17, 32)).astype(np.float32)
        original = PLDA.__call__
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_plda_matrix_products(context, [], centering_scaling=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "element_abi_unavailable"):
                with native_plda_matrix_products(context, [], centering_scaling=True, elementwise_records=[]):
                    pass
            with self.assertRaisesRegex(ValueError, "injected_element_failure"):
                with native_plda_matrix_products(context, [], centering_scaling=True, elementwise_records=[]), patch.object(context, "center", side_effect=ValueError("injected_element_failure")):
                    model(values)
        self.assertIs(PLDA.__call__, original)

    def test_source_version_and_missing_matrix_capability_are_rejected(self):
        with self.context() as context:
            with patch("tools.baseline.native_plda_matrix.PLDA_CALL_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "source_not_supported"):
                with native_plda_matrix_products(context, []):
                    pass
            with patch("tools.baseline.native_plda_matrix.importlib.metadata.version", return_value="changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_plda_matrix_products(context, []):
                    pass
            with patch.object(context, "library", object()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_plda_matrix_products(context, []):
                    pass

    def test_unrelated_threads_and_parameter_mutation_provenance(self):
        model = PLDA(self.transform, self.model, lda_dimension=16)
        values = np.random.default_rng(244).normal(size=(7, 32)).astype(np.float32)
        expected = model(values)
        records = []
        with self.context() as context, native_plda_matrix_products(context, records):
            with ThreadPoolExecutor(max_workers=1) as executor:
                actual = executor.submit(model, values).result()
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            self.assertEqual(records, [])
            model(values)
            first_hash = records[0]["prepared_parameter_sha256"]["mean1"]
            inspect.getclosurevars(model._xvec_tf).nonlocals["mean1"][0] += .01
            model(values)
            self.assertNotEqual(records[2]["prepared_parameter_sha256"]["mean1"], first_hash)


if __name__ == "__main__":
    unittest.main()

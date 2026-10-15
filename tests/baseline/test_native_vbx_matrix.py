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
import pyannote.audio.utils.vbx as vbx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_vbx_matrix import compile_native_vbx, native_vbx_matrix_products


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeVBXMatrixTests(unittest.TestCase):
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

    def fixture(self):
        rng = np.random.default_rng(991)
        features = rng.normal(scale=.3, size=(19, 8))
        phi = rng.uniform(.4, 2., 8)
        gamma = rng.uniform(.1, 1., (19, 3))
        gamma /= gamma.sum(axis=1, keepdims=True)
        return features, phi, gamma

    def ordered_product(self, left, right):
        result = np.zeros((left.shape[0], right.shape[1]))
        for k in range(left.shape[1]):
            result = np.add(result, np.multiply(left[:, k, None], right[k, None, :]))
        return result, {"test_reference_only": True}

    def compare(self, expected, actual):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(np.asarray(left).view(np.uint64), np.asarray(right).view(np.uint64))

    def test_complete_iteration_matches_independent_ordered_product_reference(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        baseline = original(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
        reference_context = SimpleNamespace(library=SimpleNamespace(wxdouble_matmul=object()), matmul=self.ordered_product)
        with native_vbx_matrix_products(reference_context, [], []):
            expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
        records, calls = [], []
        with self.context() as context, native_vbx_matrix_products(context, records, calls):
            actual = vbx.VBx(X=features, Phi=phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
        self.compare(expected, actual)
        np.testing.assert_allclose(actual[0], baseline[0], rtol=1e-10, atol=1e-10)
        self.assertEqual(calls[0]["completed_iterations"], len(baseline[2]))
        self.assertEqual(calls[0]["native_product_calls"], 3 * len(actual[2]))
        self.assertEqual({r["operation"] for r in records}, {"speaker_statistics", "frame_scores", "speaker_penalty"})
        self.assertTrue(any(r["vector_result"] for r in records))
        self.assertTrue(all(r["computation_gpu_seconds"] > 0 for r in records))
        self.assertIs(vbx.VBx, original)

    def test_optional_model_initialization_random_initialization_and_zero_iterations(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            records, calls = [], []
            with native_vbx_matrix_products(context, records, calls):
                result = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=1, alpha=np.zeros((3, 8)), invL=np.ones((3, 8)), return_model=True)
            self.assertEqual(len(result[2]), 1)
            self.assertEqual(len(records), 2)
            self.assertEqual([r["operation"] for r in records], ["frame_scores", "speaker_penalty"])
            with native_vbx_matrix_products(context, [], calls):
                result = vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=0)
            self.assertEqual(result[2], [])
            self.assertEqual(calls[-1]["native_product_calls"], 0)
            np.random.seed(818)
            expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, maxIters=3)
            np.random.seed(818)
            with native_vbx_matrix_products(context, [], []):
                actual = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, maxIters=3)
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-10)
            self.assertEqual(len(actual[2]), len(expected[2]))

    def test_native_failure_is_not_cpu_fallback_and_hooks_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        records, calls = [], []
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "injected_native_failure"):
                with native_vbx_matrix_products(context, records, calls), patch.object(context, "matmul", side_effect=ValueError("injected_native_failure")):
                    vbx.VBx(features, phi, pi=3, gamma=gamma.copy())
            self.assertEqual(calls[0]["status"], "failed")
            self.assertIsNone(calls[0]["completed_iterations"])
            self.assertEqual(records, [])
            for invalid_gamma, message in ((gamma.astype(np.float32), "operand_format_not_supported"), (np.full_like(gamma, 1e-100), "domain_invalid")):
                with self.assertRaisesRegex(ValueError, message):
                    with native_vbx_matrix_products(context, [], []):
                        vbx.VBx(features, phi, pi=3, gamma=invalid_gamma, maxIters=1)
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_vbx_matrix_products(context, [], None):
                    pass
            with native_vbx_matrix_products(context, [], []):
                with self.assertRaisesRegex(ValueError, "already_active"):
                    with native_vbx_matrix_products(context, [], []):
                        pass
        self.assertIs(vbx.VBx, original)

    def test_source_ast_version_and_capability_drift_are_rejected(self):
        original = vbx.VBx
        with self.context() as context:
            with patch("tools.baseline.native_vbx_matrix.VBX_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "source_not_supported"):
                with native_vbx_matrix_products(context, [], []):
                    pass
            changed = textwrap.dedent(inspect.getsource(original)).replace("gamma.T.dot(rho)", "gamma.T @ rho")
            with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
                compile_native_vbx(original, object())
            with patch("tools.baseline.native_vbx_matrix.importlib.metadata.version", return_value="changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                with native_vbx_matrix_products(context, [], []):
                    pass
            with patch.object(context, "library", object()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_vbx_matrix_products(context, [], []):
                    pass
        self.assertIs(vbx.VBx, original)

    def test_unrelated_threads_and_multiple_calls_keep_separate_metadata(self):
        features, phi, gamma = self.fixture()
        expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=3)
        records, calls = [], []
        with self.context() as context, native_vbx_matrix_products(context, records, calls):
            with ThreadPoolExecutor(max_workers=1) as executor:
                actual = executor.submit(vbx.VBx, features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=3).result()
            self.compare(expected, actual)
            self.assertEqual(records, [])
            self.assertEqual(calls, [])
            for _ in range(2):
                vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=3)
        self.assertEqual([c["call_index"] for c in calls], [0, 1])
        self.assertEqual({r["vbx_call_index"] for r in records}, {0, 1})


if __name__ == "__main__":
    unittest.main()

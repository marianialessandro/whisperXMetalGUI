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
from tools.baseline.native_vbx_matrix import compile_native_vbx, native_vbx_matrix_products, prior_update, elbo_arithmetic


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

    def test_native_precomputed_terms_preserve_complete_results_and_layout(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            for values in (features, np.asfortranarray(features)):
                covariance = phi[::-1]
                with native_vbx_matrix_products(context, [], []):
                    expected = vbx.VBx(values, covariance, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
                records, calls, terms = [], [], []
                with native_vbx_matrix_products(context, records, calls, precompute=True, precompute_records=terms), patch("numpy.sqrt", side_effect=AssertionError("CPU covariance square root executed")):
                    actual = vbx.VBx(values, covariance, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
                self.compare(expected, actual)
                self.assertEqual(len(terms), 1)
                self.assertEqual(terms[0]["input_layout"], "C" if values.flags.c_contiguous else "F")
                self.assertEqual(terms[0]["host_phi_staging_copy_bytes"], covariance.nbytes)
                self.assertEqual(calls[0]["native_precompute_calls"], 1)
                self.assertEqual(calls[0]["native_product_calls"], 3 * len(actual[2]))
            records, calls, terms = [], [], []
            with native_vbx_matrix_products(context, records, calls, precompute=True, precompute_records=terms):
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=0)
            self.assertEqual(records, [])
            self.assertEqual(calls[0]["completed_iterations"], 0)
            self.assertEqual(len(terms), 1)

    def test_precompute_failure_capability_and_assignment_contract_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "precompute_records_required"):
                with native_vbx_matrix_products(context, [], [], precompute=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "precompute_abi_unavailable"):
                with native_vbx_matrix_products(context, [], [], precompute=True, precompute_records=[]):
                    pass
            terms, calls = [], []
            with self.assertRaisesRegex(ValueError, "phi_domain_invalid"):
                with native_vbx_matrix_products(context, [], calls, precompute=True, precompute_records=terms):
                    vbx.VBx(features, -phi, pi=3, gamma=gamma.copy(), maxIters=1)
            self.assertEqual(terms, [])
            self.assertEqual(calls[0]["status"], "failed")
            changed = textwrap.dedent(inspect.getsource(original)).replace("V = np.sqrt(Phi)", "V = Phi ** .5")
            with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
                compile_native_vbx(original, object(), object())
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

    def test_native_statistics_complete_result_layouts_and_skipped_initial_update(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            for layout, speakers in (("C", 3), ("F", 3), ("C", 1)):
                initial_gamma = np.array(gamma[:, :speakers], order=layout)
                with native_vbx_matrix_products(context, [], [], precompute=True, precompute_records=[]):
                    expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=speakers, gamma=initial_gamma.copy(order="K"), maxIters=5, return_model=True)
                records, calls, statistics = [], [], []
                with native_vbx_matrix_products(context, records, calls, precompute=True, precompute_records=[], statistics=True, statistics_records=statistics):
                    actual = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=speakers, gamma=initial_gamma.copy(order="K"), maxIters=5, return_model=True)
                self.compare(expected, actual)
                self.assertEqual(calls[0]["native_statistics_calls"], 2 * len(actual[2]))
                self.assertEqual(len(records), 3 * len(actual[2]))
                self.assertEqual(statistics[0]["reduction_order"], "sequential" if layout == "C" and speakers > 1 else "pairwise")
                self.assertEqual({r["operation"] for r in statistics}, {"invL", "alpha"})
                self.assertTrue(all(r["computation_gpu_seconds"] > 0 for r in statistics))
            statistics, calls = [], []
            with native_vbx_matrix_products(context, [], calls, statistics=True, statistics_records=statistics):
                vbx.VBx(features, phi, pi=3, gamma=gamma, maxIters=1, alpha=np.zeros((3, 8)), invL=np.ones((3, 8)))
                vbx.VBx(features, phi, pi=3, gamma=gamma, maxIters=0)
            self.assertEqual(statistics, [])
            self.assertEqual([c["native_statistics_calls"] for c in calls], [0, 0])

    def test_statistics_failure_capability_and_ast_contract_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "statistics_records_required"):
                with native_vbx_matrix_products(context, [], [], statistics=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "statistics_abi_unavailable"):
                with native_vbx_matrix_products(context, [], [], statistics=True, statistics_records=[]):
                    pass
            calls, statistics = [], []
            with self.assertRaisesRegex(ValueError, "injected_statistics_failure"):
                with native_vbx_matrix_products(context, [], calls, statistics=True, statistics_records=statistics), patch.object(context, "inv_l_vbx", side_effect=ValueError("injected_statistics_failure")):
                    vbx.VBx(features, phi, pi=3, gamma=gamma, maxIters=1)
            self.assertEqual(calls[0]["status"], "failed")
            self.assertEqual(statistics, [])
            self.assertIs(vbx.VBx, original)
            changed = textwrap.dedent(inspect.getsource(original)).replace("Fa / Fb * invL", "(Fa * invL) / Fb")
            with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "statistics_expression_contract_changed"):
                compile_native_vbx(original, object(), statistics=object())

    def test_native_scores_complete_results_options_layout_and_initial_model(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            for precompute, statistics in ((False, False), (True, False), (False, True), (True, True)):
                for layout in ("C", "F"):
                    with native_vbx_matrix_products(context, [], [], precompute=precompute, precompute_records=[], statistics=statistics, statistics_records=[]):
                        expected = vbx.VBx(np.array(features, order=layout), phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
                    products, calls, scores = [], [], []
                    with native_vbx_matrix_products(context, products, calls, precompute=precompute, precompute_records=[], statistics=statistics, statistics_records=[], scores=True, score_records=scores):
                        actual = vbx.VBx(np.array(features, order=layout), phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
                    self.compare(expected, actual)
                    self.assertEqual(calls[0]["native_score_calls"], 2 * len(actual[2]))
                    self.assertEqual(len(products), 3 * len(actual[2]))
                    self.assertEqual({r["operation"] for r in scores}, {"penalty_terms", "frame_scores"})
                    self.assertTrue(all(r["computation_gpu_seconds"] > 0 for r in scores))
                    self.assertTrue(all("score_elementwise_updates" not in r["cpu_residuals"] for r in scores))
            model = {"alpha": np.zeros((3, 8)), "invL": np.ones((3, 8))}
            with native_vbx_matrix_products(context, [], []):
                expected = vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1, return_model=True, **model)
            scores, calls, products = [], [], []
            with native_vbx_matrix_products(context, products, calls, scores=True, score_records=scores):
                actual = vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1, return_model=True, **model)
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=0)
            self.compare(expected, actual)
            self.assertEqual(len(products), 2)
            self.assertEqual([call["native_score_calls"] for call in calls], [2, 0])

    def test_scores_failure_capability_ast_and_unrelated_thread_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "score_records_required"):
                with native_vbx_matrix_products(context, [], [], scores=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "scores_abi_unavailable"):
                with native_vbx_matrix_products(context, [], [], scores=True, score_records=[]):
                    pass
            for method, completed in (("penalty_terms_vbx", 0), ("frame_scores_vbx", 1)):
                calls, scores = [], []
                with self.assertRaisesRegex(ValueError, "injected_scores_failure"):
                    with native_vbx_matrix_products(context, [], calls, scores=True, score_records=scores), patch.object(context, method, side_effect=ValueError("injected_scores_failure")):
                        vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1)
                self.assertEqual(calls[0]["status"], "failed")
                self.assertEqual(calls[0]["native_score_calls"], completed)
                self.assertIs(vbx.VBx, original)
            changed = textwrap.dedent(inspect.getsource(original)).replace("+ G", "+ (G + 0)")
            with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "score_expression_contract_changed"):
                compile_native_vbx(original, object(), score=object())
            expected = original(features, phi, pi=3, gamma=gamma.copy(), maxIters=1)
            calls, scores = [], []
            with native_vbx_matrix_products(context, [], calls, scores=True, score_records=scores), ThreadPoolExecutor(max_workers=1) as executor:
                actual = executor.submit(vbx.VBx, features, phi, pi=3, gamma=gamma.copy(), maxIters=1).result()
            self.compare(expected, actual)
            self.assertEqual((calls, scores), ([], []))

    def test_native_prior_complete_results_flag_combinations_and_layouts(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            for mask in range(8):
                options = {"precompute": bool(mask & 1), "precompute_records": [], "statistics": bool(mask & 2), "statistics_records": [], "scores": bool(mask & 4), "score_records": []}
                with native_vbx_matrix_products(context, [], [], **options):
                    expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
                records, calls, products = [], [], []
                with native_vbx_matrix_products(context, products, calls, prior=True, prior_records=records, **options):
                    actual = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=5, return_model=True)
                self.compare(expected, actual)
                self.assertEqual(calls[0]["native_prior_calls"], len(actual[2]))
                self.assertEqual(len(products), 3 * len(actual[2]))
                self.assertTrue(all(record["computation_gpu_seconds"] > 0 for record in records))
                self.assertTrue(all("posterior_and_prior_updates" not in record["cpu_residuals"] for record in records + products))
            for values in (gamma, np.asfortranarray(gamma), gamma[:, :1].copy()):
                masses = values.sum(axis=0)
                records = []
                actual = prior_update(context, records, values, 0, 0)
                np.testing.assert_array_equal(actual.view(np.uint64), (masses / masses.sum()).view(np.uint64))
                self.assertEqual(records[0]["frame_reduction_order"], "pairwise" if values.flags.f_contiguous else "sequential")
            records, calls = [], []
            with native_vbx_matrix_products(context, [], calls, prior=True, prior_records=records):
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1, alpha=np.zeros((3, 8)), invL=np.ones((3, 8)))
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=0)
            self.assertEqual([call["native_prior_calls"] for call in calls], [1, 0])

    def test_prior_failure_capability_assignment_contract_and_thread_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "prior_records_required"):
                with native_vbx_matrix_products(context, [], [], prior=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "prior_abi_unavailable"):
                with native_vbx_matrix_products(context, [], [], prior=True, prior_records=[]):
                    pass
            records, calls = [], []
            with self.assertRaisesRegex(ValueError, "injected_prior_failure"):
                with native_vbx_matrix_products(context, [], calls, prior=True, prior_records=records), patch.object(context, "prior_vbx", side_effect=ValueError("injected_prior_failure")):
                    vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1)
            self.assertEqual(calls[0]["status"], "failed")
            self.assertEqual(calls[0]["native_prior_calls"], 0)
            self.assertIs(vbx.VBx, original)
            for before, after in (("np.sum(gamma, axis=0)", "gamma.sum(axis=0)"), ("pi = pi / pi.sum()", "pi /= pi.sum()")):
                changed = textwrap.dedent(inspect.getsource(original)).replace(before, after)
                with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "prior_expression_contract_changed"):
                    compile_native_vbx(original, object(), prior=object())
            with self.assertRaisesRegex(ValueError, "operand_format_not_supported"):
                prior_update(context, [], gamma[:, ::2], 0, 0)
            with patch("tools.baseline.native_vbx_matrix.np.ascontiguousarray", side_effect=AssertionError("oversized staging attempted")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                prior_update(context, [], np.ones((4097, 1)), 0, 0)
            expected = original(features, phi, pi=3, gamma=gamma.copy(), maxIters=1)
            records, calls = [], []
            with native_vbx_matrix_products(context, [], calls, prior=True, prior_records=records), ThreadPoolExecutor(max_workers=1) as executor:
                actual = executor.submit(vbx.VBx, features, phi, pi=3, gamma=gamma.copy(), maxIters=1).result()
            self.compare(expected, actual)
            self.assertEqual((records, calls), ([], []))

    def test_native_posterior_complete_results_flag_combinations_and_counts(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            for mask in range(16):
                options = {"precompute": bool(mask & 1), "precompute_records": [], "statistics": bool(mask & 2), "statistics_records": [], "scores": bool(mask & 4), "score_records": [], "prior": bool(mask & 8), "prior_records": []}
                with native_vbx_matrix_products(context, [], [], **options):
                    expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=4, return_model=True)
                records, calls = [], []
                with native_vbx_matrix_products(context, [], calls, posterior=True, posterior_records=records, **options):
                    actual = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=4, return_model=True)
                self.compare(expected, actual)
                self.assertEqual(calls[0]["native_posterior_calls"], 3 * len(actual[2]))
                self.assertEqual([r["operation"] for r in records[:3]], ["log_prior_addition", "log_prior_addition", "posterior_centering"])
                self.assertTrue(all(r["computation_gpu_seconds"] > 0 for r in records))
            records, calls = [], []
            with native_vbx_matrix_products(context, [], calls, posterior=True, posterior_records=records):
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1, alpha=np.zeros((3, 8)), invL=np.ones((3, 8)))
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=0)
            self.assertEqual([call["native_posterior_calls"] for call in calls], [3, 0])

    def test_posterior_failure_capability_and_ast_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "posterior_records_required"):
                with native_vbx_matrix_products(context, [], [], posterior=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "posterior_abi_unavailable"):
                with native_vbx_matrix_products(context, [], [], posterior=True, posterior_records=[]):
                    pass
            records, calls = [], []
            with self.assertRaisesRegex(ValueError, "injected_posterior_failure"):
                with native_vbx_matrix_products(context, [], calls, posterior=True, posterior_records=records), patch.object(context, "posterior_vbx", side_effect=ValueError("injected_posterior_failure")):
                    vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1)
            self.assertEqual(calls[0]["status"], "failed")
            self.assertEqual(records, [])
            self.assertIs(vbx.VBx, original)
            changed = textwrap.dedent(inspect.getsource(original)).replace("log_p_ + lpi - log_p_x[:, None]", "log_p_ - log_p_x[:, None] + lpi")
            with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "posterior_expression_contract_changed"):
                compile_native_vbx(original, object(), posterior=object())

    def test_native_elbo_complete_results_all_flags_and_fortran_model(self):
        features, phi, gamma = self.fixture()
        with self.context() as context:
            for mask in range(32):
                options = {"precompute": bool(mask & 1), "precompute_records": [], "statistics": bool(mask & 2), "statistics_records": [], "scores": bool(mask & 4), "score_records": [], "prior": bool(mask & 8), "prior_records": [], "posterior": bool(mask & 16), "posterior_records": []}
                with native_vbx_matrix_products(context, [], [], **options):
                    expected = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=3, return_model=True)
                records, calls = [], []
                with native_vbx_matrix_products(context, [], calls, elbo=True, elbo_records=records, **options):
                    actual = vbx.VBx(features, phi, Fa=.07, Fb=.8, pi=3, gamma=gamma.copy(), maxIters=3, return_model=True)
                self.compare(expected, actual)
                self.assertEqual(calls[0]["native_elbo_calls"], 2 * len(actual[2]))
                self.assertEqual({r["operation"] for r in records}, {"likelihood_sum", "elbo"})
            rng = np.random.default_rng(929)
            for alpha_layout in ("C", "F"):
                inv_l = np.asfortranarray(rng.uniform(.1, 1., (3, 8)))
                alpha = np.array(rng.normal(size=(3, 8)), order=alpha_layout)
                records = []
                actual = elbo_arithmetic(context, records, np.log(inv_l), inv_l, alpha, np.float64(-2.), .8, 0, "elbo", 0)
                expected = np.float64(-2.) + .8 * .5 * np.sum(np.log(inv_l) - inv_l - alpha**2 + 1)
                self.assertEqual(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(records[0]["reduction_layout"], alpha_layout)
            calls = []
            with native_vbx_matrix_products(context, [], calls, elbo=True, elbo_records=[]):
                vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=0)
            self.assertEqual(calls[0]["native_elbo_calls"], 0)

    def test_elbo_failure_capability_and_source_contract_restore(self):
        features, phi, gamma = self.fixture()
        original = vbx.VBx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "elbo_records_required"):
                with native_vbx_matrix_products(context, [], [], elbo=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_matmul=object())), self.assertRaisesRegex(ValueError, "elbo_abi_unavailable"):
                with native_vbx_matrix_products(context, [], [], elbo=True, elbo_records=[]):
                    pass
            records, calls = [], []
            with self.assertRaisesRegex(ValueError, "injected_elbo_failure"):
                with native_vbx_matrix_products(context, [], calls, elbo=True, elbo_records=records), patch.object(context, "elbo_vbx", side_effect=ValueError("injected_elbo_failure")):
                    vbx.VBx(features, phi, pi=3, gamma=gamma.copy(), maxIters=1)
            self.assertEqual(calls[0]["status"], "failed")
            self.assertEqual(calls[0]["native_elbo_calls"], 1)
            self.assertIs(vbx.VBx, original)
            changed = textwrap.dedent(inspect.getsource(original)).replace("Fb * 0.5 * np.sum", "(0.5 * Fb) * np.sum")
            with patch("tools.baseline.native_vbx_matrix.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_matrix.VBX_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "elbo_expression_contract_changed"):
                compile_native_vbx(original, object(), elbo=object())

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

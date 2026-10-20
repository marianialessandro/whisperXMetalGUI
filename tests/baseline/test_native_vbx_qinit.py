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
from scipy.special import softmax
import pyannote.audio.utils.vbx as vbx
import pyannote.audio.pipelines.clustering as clustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_vbx_matrix import native_vbx_matrix_products
from tools.baseline.native_vbx_qinit import compile_native_cluster, initial_softmax, initial_one_hot, initial_resident_qinit, native_vbx_qinit


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeVBXQinitTests(unittest.TestCase):
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

    def one_hot(self, rows, columns):
        values = np.zeros((rows, columns))
        values[np.arange(rows), np.arange(rows) % columns] = 1.
        return values

    def test_softmax_scipy_boundaries_and_numpy_inputs_preserved(self):
        with self.context() as context:
            for rows, columns in ((1, 1), (19, 3), (19, 7), (19, 8), (19, 127), (19, 128), (19, 129), (64, 1024), (4096, 1)):
                values = self.one_hot(rows, columns)
                values.flags.writeable = False
                for smoothing in (0., -0., 2. ** -200, .1, 7., 60.):
                    actual, record = context.qinit_softmax_vbx(values, smoothing)
                    signed_values = values.copy()
                    signed_values[signed_values == 0] = -0.
                    signed_actual, _ = context.qinit_softmax_vbx(signed_values, smoothing)
                    np.testing.assert_array_equal(actual.view(np.uint64), signed_actual.view(np.uint64))
                    np.testing.assert_allclose(actual, softmax(values * smoothing, axis=1), rtol=3e-15, atol=0.)
                    self.assertEqual(record["gpu_kernels"], ["double_vbx_qinit_exp", "double_vbx_initialize"])
                    self.assertEqual(record["logical_input_upload_bytes"], values.nbytes)
                    self.assertEqual(record["logical_result_readback_bytes"], values.nbytes)
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertFalse(record["scipy_bitwise_certified"])
                np.testing.assert_array_equal(values, self.one_hot(rows, columns))

    def test_softmax_domains_shapes_and_one_hot_invariant(self):
        with self.context() as context:
            values = self.one_hot(3, 8)
            for smoothing in (-1., np.inf, np.nan, 60.01, 2. ** -201, True, "7"):
                with self.assertRaises(ValueError):
                    context.qinit_softmax_vbx(values, smoothing)
            for invalid in (np.zeros_like(values), np.ones_like(values), np.full_like(values, .5), np.full_like(values, np.nan), np.full_like(values, -1.)):
                with self.assertRaisesRegex(ValueError, "one_hot_required"):
                    context.qinit_softmax_vbx(invalid, 7.)
            for invalid in (values.astype(np.float32), values[:, ::2], np.ones((4097, 1)), np.ones((65, 1024)), np.ones((1, 1025)), np.ones(3)):
                with self.assertRaises(ValueError):
                    context.qinit_softmax_vbx(invalid, 7.)
            with self.assertRaisesRegex(ValueError, "shape_not_supported"):
                initial_softmax(context, [], np.ones((4097, 2)), 7.)

    def test_softmax_context_lifetime_capability_and_thread_serialization(self):
        values = self.one_hot(3, 8)
        with self.context() as context:
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.qinit_softmax_vbx(values, 7.)
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.qinit_softmax_vbx(values, 7.)[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result.view(np.uint64), results[0].view(np.uint64))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.qinit_softmax_vbx(values, 7.)

    def test_scope_aliases_bypass_and_dynamic_vbx_lookup(self):
        labels = np.arange(19) % 3
        original = vbx.cluster_vbx
        observed = []

        def capture(*args, **kwargs):
            observed.append(kwargs["gamma"].copy())
            return kwargs["gamma"], np.ones(3) / 3, [], None, None

        with self.context() as context:
            for smoothing in (-1., 0., 7., 60.):
                records = []
                with native_vbx_qinit(context, records), patch.object(vbx, "VBx", side_effect=capture):
                    self.assertIs(vbx.cluster_vbx, clustering.cluster_vbx)
                    actual = clustering.cluster_vbx(labels, None, None, 1., 1., init_smoothing=smoothing)
                expected = self.one_hot(19, 3)
                expected = expected if smoothing < 0 else softmax(expected * smoothing, axis=1)
                np.testing.assert_allclose(actual[0], expected, rtol=3e-15, atol=0.)
                self.assertEqual(len(records), int(smoothing >= 0))
            self.assertEqual(len(observed), 4)
        self.assertIs(vbx.cluster_vbx, original)
        self.assertIs(clustering.cluster_vbx, original)

    def test_complete_cluster_with_all_native_vbx_options_and_determinism(self):
        rng = np.random.default_rng(991)
        features, phi, labels = rng.normal(scale=.3, size=(19, 8)), rng.uniform(.4, 2., 8), np.arange(19) % 3
        options = {"precompute": True, "precompute_records": [], "statistics": True, "statistics_records": [], "scores": True, "score_records": [], "prior": True, "prior_records": [], "posterior": True, "posterior_records": [], "elbo": True, "elbo_records": [], "transcendentals": True, "transcendental_records": [], "logsumexp": True, "logsumexp_records": [], "scalars": True, "scalar_records": [], "initialization": True, "initialization_records": []}
        with self.context() as context, native_vbx_matrix_products(context, [], [], **options):
            expected = vbx.cluster_vbx(labels, features, phi, .07, .8, maxIters=3)
            for repeat in range(3):
                records = []
                with native_vbx_qinit(context, records):
                    actual = vbx.cluster_vbx(labels, features, phi, .07, .8, maxIters=3)
                self.assertEqual(len(records), 1)
                for left, right in zip(expected, actual):
                    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
                if repeat == 0:
                    previous = actual
                else:
                    for left, right in zip(previous, actual):
                        np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))

    def test_unrelated_thread_remains_original_and_nested_scope_rejected(self):
        labels = np.arange(19) % 3
        values = np.ones((19, 8)) * .1
        phi = np.ones(8)
        expected = vbx.cluster_vbx(labels, values, phi, .07, .8, maxIters=0)
        with self.context() as context, native_vbx_qinit(context, records := []):
            with ThreadPoolExecutor(max_workers=1) as executor:
                actual = executor.submit(clustering.cluster_vbx, labels, values, phi, .07, .8, maxIters=0).result()
            self.assertEqual(records, [])
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
            with self.assertRaisesRegex(ValueError, "already_active"):
                with native_vbx_qinit(context, []):
                    pass

    def test_failures_restore_both_aliases_without_fallback(self):
        original = vbx.cluster_vbx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_vbx_qinit(context, None):
                    pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                with native_vbx_qinit(context, []):
                    pass
            with self.assertRaisesRegex(ValueError, "injected_qinit_failure"):
                with native_vbx_qinit(context, records := []), patch.object(context, "qinit_softmax_vbx", side_effect=ValueError("injected_qinit_failure")):
                    vbx.cluster_vbx(np.arange(19) % 3, None, None, 1., 1.)
            self.assertEqual(records, [])
            self.assertIs(vbx.cluster_vbx, original)
            self.assertIs(clustering.cluster_vbx, original)
            with patch.object(clustering, "cluster_vbx", object()), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                with native_vbx_qinit(context, []):
                    pass

    def test_one_hot_adapter_exact_labels_dtypes_strides_and_shape_before_staging(self):
        with self.context() as context:
            for labels in (np.array([0, 3], dtype=np.int32), np.array([0, 3], dtype=np.int64), np.arange(8, dtype=np.int64)[::2]):
                records = []
                actual = initial_one_hot(context, records, labels)
                expected = np.zeros((len(labels), int(labels.max()) + 1))
                expected[np.arange(len(labels)), labels] = 1.
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["label_dtype_conversion"], labels.dtype == np.int32)
            for labels in (np.array([], dtype=np.int64), np.zeros(4097, dtype=np.int64), np.array([1024], dtype=np.int64)):
                with patch("tools.baseline.native_vbx_qinit.np.ascontiguousarray", side_effect=AssertionError("staging before validation")), self.assertRaisesRegex(ValueError, "shape_not_supported"):
                    initial_one_hot(context, [], labels)
            for labels in (np.ones(2), np.ones((2, 1), dtype=np.int64), np.ones(2, dtype=bool)):
                with self.assertRaisesRegex(ValueError, "format_not_supported"):
                    initial_one_hot(context, [], labels)

    def test_one_hot_complete_cluster_composes_with_softmax_and_negative_bypass(self):
        rng = np.random.default_rng(991)
        features, phi, labels = rng.normal(scale=.3, size=(19, 8)), rng.uniform(.4, 2., 8), np.arange(19) % 3
        with self.context() as context:
            for native_softmax in (False, True):
                for smoothing in (-1., 0., 7.):
                    for iterations in (0, 3):
                        if native_softmax:
                            with native_vbx_qinit(context, []):
                                expected = vbx.cluster_vbx(labels, features, phi, .07, .8, maxIters=iterations, init_smoothing=smoothing)
                        else:
                            expected = vbx.cluster_vbx(labels, features, phi, .07, .8, maxIters=iterations, init_smoothing=smoothing)
                        records, softmax_records = [], []
                        with native_vbx_qinit(context, softmax_records, one_hot=True, one_hot_records=records, softmax=native_softmax):
                            actual = clustering.cluster_vbx(labels, features, phi, .07, .8, maxIters=iterations, init_smoothing=smoothing)
                        for left, right in zip(expected, actual):
                            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
                        self.assertEqual(len(records), 1)
                        self.assertEqual(len(softmax_records), int(native_softmax and smoothing >= 0))

    def test_one_hot_scope_capability_fault_and_thread_isolation(self):
        original = vbx.cluster_vbx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "one_hot_records_required"):
                with native_vbx_qinit(context, [], one_hot=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_qinit=object())), self.assertRaisesRegex(ValueError, "one_hot_abi_unavailable"):
                with native_vbx_qinit(context, [], one_hot=True, one_hot_records=[]):
                    pass
            with self.assertRaisesRegex(ValueError, "stage_required"):
                with native_vbx_qinit(context, [], softmax=False):
                    pass
            with native_vbx_qinit(context, [], one_hot=True, one_hot_records=(records := []), softmax=False):
                with patch.object(vbx, "VBx", return_value=(None, None, None, None, None)), ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(vbx.cluster_vbx, np.array([0, 2]), None, None, 1., 1.).result()
                self.assertEqual(records, [])
            with self.assertRaisesRegex(ValueError, "injected_one_hot_failure"):
                with native_vbx_qinit(context, [], one_hot=True, one_hot_records=(records := [])), patch.object(context, "one_hot_vbx", side_effect=ValueError("injected_one_hot_failure")):
                    vbx.cluster_vbx(np.array([0, 2]), None, None, 1., 1.)
            self.assertEqual(records, [])
        self.assertIs(vbx.cluster_vbx, original)
        self.assertIs(clustering.cluster_vbx, original)

    def test_one_hot_ast_pin_rejects_changed_scatter(self):
        source = textwrap.dedent(inspect.getsource(vbx.cluster_vbx))
        changed = source.replace("ahc_init.astype(int)", "ahc_init.astype(np.int64)")
        with patch("tools.baseline.native_vbx_qinit.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_qinit.CLUSTER_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "one_hot_expression_contract_changed"):
            compile_native_cluster(vbx.cluster_vbx, None, object(), one_hot=object())

    def test_resident_initialization_primitive_exact_separate_policy_and_domains(self):
        with self.context() as context:
            for rows, columns in ((1, 1), (19, 3), (19, 129), (64, 1024), (4096, 1)):
                labels = np.arange(rows, dtype=np.int64) % columns
                labels.flags.writeable = False
                one_hot, _ = context.one_hot_vbx(labels, columns)
                for smoothing in (0., -0., 7., 60.):
                    expected, _ = context.qinit_softmax_vbx(one_hot, smoothing)
                    actual, record = context.qinit_from_labels_vbx(labels, columns, smoothing)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(len(record["gpu_kernels"]), 3)
                    self.assertEqual(record["command_buffers_per_run"], 1)
                    self.assertEqual(record["logical_input_upload_bytes"], labels.nbytes)
                    self.assertEqual(record["logical_one_hot_boundary_bytes_avoided"], 2 * one_hot.nbytes)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for labels, columns, smoothing in ((np.array([-1], np.int64), 3, 7.), (np.array([3], np.int64), 3, 7.), (np.zeros(65, np.int64), 1024, 7.), (np.zeros(4097, np.int64), 1, 7.), (np.zeros(3, np.int64), 3, -1.), (np.zeros(3, np.int64), 3, np.nan)):
                with self.assertRaises(ValueError):
                    context.qinit_from_labels_vbx(labels, columns, smoothing)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "resident_initialization_abi_unavailable"):
                context.qinit_from_labels_vbx(np.zeros(3, np.int64), 3, 7.)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.qinit_from_labels_vbx(np.zeros(3, np.int64), 3, 7.)

    def test_resident_initialization_whole_cluster_and_negative_smoothing_bypass(self):
        rng = np.random.default_rng(991)
        features, phi, labels = rng.normal(scale=.3, size=(19, 8)), rng.uniform(.4, 2., 8), np.arange(19) % 3
        with self.context() as context:
            for smoothing in (-1., 0., 7.):
                for iterations in (0, 3):
                    with native_vbx_qinit(context, [], one_hot=True, one_hot_records=[]):
                        expected = vbx.cluster_vbx(labels, features, phi, .07, .8, maxIters=iterations, init_smoothing=smoothing)
                    one_hot_records, records = [], []
                    with native_vbx_qinit(context, records, one_hot=True, one_hot_records=one_hot_records, resident_initialization=True):
                        actual = clustering.cluster_vbx(labels, features, phi, .07, .8, maxIters=iterations, init_smoothing=smoothing)
                    for left, right in zip(expected, actual):
                        np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
                    self.assertEqual(len(one_hot_records), 1)
                    self.assertEqual(len(records), int(smoothing >= 0))
                    self.assertEqual(one_hot_records[0]["command_buffers_per_run"], int(smoothing < 0))
                    if smoothing >= 0:
                        self.assertEqual(one_hot_records[0]["shared_softmax_dispatch_index"], 0)
                        self.assertTrue(records[0]["resident_initialization"])

    def test_resident_initialization_configuration_source_fault_and_restore(self):
        original = vbx.cluster_vbx
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_vbx_qinit(context, [], resident_initialization=True):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_one_hot=object(), wxdouble_vbx_qinit=object())), self.assertRaisesRegex(ValueError, "resident_initialization_abi_unavailable"):
                with native_vbx_qinit(context, [], one_hot=True, one_hot_records=[], resident_initialization=True):
                    pass
            with self.assertRaisesRegex(ValueError, "injected_resident_initialization_failure"):
                with native_vbx_qinit(context, (records := []), one_hot=True, one_hot_records=(one_hot_records := []), resident_initialization=True), patch.object(context, "qinit_from_labels_vbx", side_effect=ValueError("injected_resident_initialization_failure")):
                    vbx.cluster_vbx(np.array([0, 2]), None, None, 1., 1.)
            self.assertEqual(records, [])
            self.assertEqual(one_hot_records, [])
        self.assertIs(vbx.cluster_vbx, original)
        self.assertIs(clustering.cluster_vbx, original)
        changed = textwrap.dedent(inspect.getsource(original)).replace("init_smoothing < 0", "init_smoothing <= -1")
        with patch("tools.baseline.native_vbx_qinit.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_qinit.CLUSTER_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "resident_initialization_expression_contract_changed"):
            compile_native_cluster(original, object(), object(), one_hot=object(), resident_initialize=object())

    def test_source_and_expression_pins_and_runtime(self):
        source = textwrap.dedent(inspect.getsource(vbx.cluster_vbx))
        with patch("tools.baseline.native_vbx_qinit.inspect.getsource", return_value=source + "\n"), self.assertRaisesRegex(ValueError, "source_not_supported"):
            compile_native_cluster(vbx.cluster_vbx, object(), object())
        changed = source.replace("qinit * init_smoothing", "init_smoothing * qinit")
        with patch("tools.baseline.native_vbx_qinit.inspect.getsource", return_value=changed), patch("tools.baseline.native_vbx_qinit.CLUSTER_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_cluster(vbx.cluster_vbx, object(), object())
        with self.context() as context, patch("tools.baseline.native_vbx_qinit.importlib.metadata.version", return_value="unsupported"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
            with native_vbx_qinit(context, []):
                pass


if __name__ == "__main__":
    unittest.main()

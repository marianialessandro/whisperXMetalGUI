from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
import contextlib
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_lloyd import compile_native_lloyd, shift_total


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansLloydShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "cpu.dylib"
        cls.identity = build_library(cls.library)
        cls.double = Path(cls.directory.name) / "double.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def test_raw_matches_numpy_pairwise_boundaries_and_float32_type(self):
        rng = np.random.default_rng(1063)
        with self.context() as context:
            for count in (*range(1, 261), 511, 512, 513, 1023, 1024):
                shifts = np.exp2(rng.uniform(-60, 32, count)).astype(np.float32)
                actual, record = context.lloyd_shift_total(shifts)
                self.assertIs(type(actual), np.float32)
                self.assertEqual(actual.view(np.uint32), ((shifts ** 2).sum()).view(np.uint32))
                self.assertEqual(record["execution_device"], "cpu")
                self.assertFalse(record["lloyd_iteration_replaced"])
                self.assertFalse(record["python_numeric_shift_reduction"])
                self.assertEqual(record["gpu_kernels"], [])
                self.assertEqual(record["command_buffers_per_run"], 0)
                self.assertIsNone(record["computation_gpu_seconds"])
            for value in (0., -0., 2. ** -60, 2. ** 32):
                shifts = np.full(1024, value, np.float32)
                actual, _ = context.lloyd_shift_total(shifts)
                self.assertEqual(actual.view(np.uint32), ((shifts ** 2).sum()).view(np.uint32))
            shifts = np.ndarray((129,), dtype=np.float32, buffer=bytearray(517), offset=1)
            shifts[:] = rng.uniform(size=129); shifts.flags.writeable = False
            expected = (shifts ** 2).sum()
            with ThreadPoolExecutor(max_workers=2) as executor:
                actuals = list(executor.map(lambda _: context.lloyd_shift_total(shifts)[0], range(4)))
            for actual in actuals:
                self.assertEqual(actual.view(np.uint32), expected.view(np.uint32))

    def test_domains_closed_abi_buffer_and_compiler_guards(self):
        with self.context() as context:
            for shifts in (np.ones(3), np.ones((1, 3), np.float32), np.ones(7, np.float32)[::2]):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    context.lloyd_shift_total(shifts)
            for count in (0, 1025):
                with self.assertRaisesRegex(ValueError, "count_invalid"):
                    context.lloyd_shift_total(np.ones(count, np.float32))
            for value in (-1., np.nan, np.inf, -np.inf, 2. ** -61, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "value_domain"):
                    context.lloyd_shift_total(np.full(3, value, np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.lloyd_shift_total(np.ones(3, np.float32))
            original_buffer = np.getbufsize()
            try:
                np.setbufsize(32)
                with self.assertRaisesRegex(ValueError, "numpy_buffer_not_supported"):
                    shift_total(context, [], np.ones(3, np.float32))
            finally:
                np.setbufsize(original_buffer)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.lloyd_shift_total(np.ones(3, np.float32))
        original = kmeans._kmeans_single_lloyd; inner = inspect.unwrap(original); getsource = inspect.getsource
        changed = getsource(inner).replace("center_shift**2", "center_shift**3")
        with patch("tools.baseline.native_kmeans_lloyd.inspect.getsource", side_effect=lambda function: changed if function is inner else getsource(function)), patch("tools.baseline.native_kmeans_lloyd.LLOYD_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_lloyd(original, object())
        with patch("tools.baseline.native_kmeans_lloyd.THREADPOOL_DECORATOR_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "wrapper_not_supported"):
            compile_native_lloyd(original, object())
        with self.assertRaisesRegex(ValueError, "reference_not_supported"):
            compile_native_lloyd(lambda: None, object())

    def test_direct_lloyd_verbose_tolerance_and_original_global_binding(self):
        values = np.random.default_rng(1069).normal(size=(42, 17)).astype(np.float32)
        original = kmeans._kmeans_single_lloyd
        with self.context() as context:
            candidate = compile_native_lloyd(original, lambda shifts: shift_total(context, records, shifts))
            for tolerance, max_iter in ((0., 20), (1e6, 20), (1e-4, 1)):
                options = dict(max_iter=max_iter, verbose=True, tol=tolerance, n_threads=1)
                weights, centers = np.linspace(.5, 2., len(values), dtype=np.float32), values[[0, 7, 13]].copy()
                stdout_expected, stdout_actual = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout_expected):
                    expected = original(values, weights, centers.copy(), **options)
                records = []
                with contextlib.redirect_stdout(stdout_actual):
                    actual = candidate(values, weights, centers.copy(), **options)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[2].view(np.uint32), expected[2].view(np.uint32))
                self.assertEqual((actual[1], actual[3]), (expected[1], expected[3]))
                self.assertEqual(stdout_actual.getvalue(), stdout_expected.getvalue())
                self.assertTrue(records)
                self.assertIs(kmeans._kmeans_single_lloyd, original)

    def test_fit_composition_sparse_elkan_thread_failure_and_restore(self):
        values = np.random.default_rng(1087).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_lloyd = KMeans.fit, kmeans._kmeans_single_lloyd
        def scope(gpu, cpu, records):
            return native_kmeans_feature_mean(gpu, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True, potential_context=cpu, potential_records=[], product_records=[], chunk_records=[], lloyd_shift_records=records)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            for clusters in (1, 2, 4):
                for weighted in (False, True):
                    weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    with scope(gpu, cpu, records := []):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertTrue(records); count = len(records)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(values.copy())).result()
                        self.assertEqual(len(records), count)
                        self.assertIs(kmeans._kmeans_single_lloyd, original_lloyd)
            with scope(gpu, cpu, skipped := []):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, algorithm="elkan", random_state=17).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope(gpu, cpu, []), patch.object(cpu, "lloyd_shift_total", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(gpu, [], lloyd_shift_records=[]):
                    pass
        self.assertIs(KMeans.fit, original_fit); self.assertIs(kmeans._kmeans_single_lloyd, original_lloyd)

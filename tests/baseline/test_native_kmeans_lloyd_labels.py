from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
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
from tools.baseline.native_kmeans_lloyd import compile_native_lloyd, labels_equal


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple native core required")
class NativeKMeansLloydLabelTests(unittest.TestCase):
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

    def test_raw_equal_different_sentinels_boundaries_and_readonly_unaligned(self):
        rng = np.random.default_rng(1091)
        with self.context() as context:
            for count in (1, 7, 42, 255, 256, 257, 4096):
                values = rng.integers(0, 4096, count, dtype=np.int32)
                for previous in (values, values.copy(), np.full(count, -1, np.int32)):
                    actual, record = context.lloyd_labels_equal(values, previous)
                    self.assertIs(type(actual), bool)
                    self.assertEqual(actual, np.array_equal(values, previous))
                    self.assertEqual(record["execution_device"], "cpu")
                    self.assertEqual(record["command_buffers_per_run"], 0)
                    self.assertEqual(record["gpu_kernels"], [])
                    self.assertIsNone(record["computation_gpu_seconds"])
                    self.assertFalse(record["lloyd_iteration_replaced"])
                for index in (0, count // 2, count - 1):
                    previous = values.copy(); previous[index] = -1
                    self.assertFalse(context.lloyd_labels_equal(values, previous)[0])
            values = np.ndarray((42,), dtype=np.int32, buffer=bytearray(169), offset=1)
            values[:] = rng.integers(-1, 4096, 42); values.flags.writeable = False
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: context.lloyd_labels_equal(values, values)[0], range(4)))
            self.assertEqual(outputs, [True] * 4)

    def test_validation_domains_closed_and_abi(self):
        values = np.arange(7, dtype=np.int32)
        with self.context() as context:
            for previous in (values.astype(np.int64), values[None, :], values[::-1]):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    context.lloyd_labels_equal(values, previous)
            for count in (0, 4097):
                with self.assertRaisesRegex(ValueError, "count_invalid"):
                    context.lloyd_labels_equal(np.zeros(count, np.int32), np.zeros(count, np.int32))
            with self.assertRaisesRegex(ValueError, "count_invalid"):
                context.lloyd_labels_equal(values, values[:3])
            for value in (-2, 4096, np.iinfo(np.int32).max, np.iinfo(np.int32).min):
                invalid = values.copy(); invalid[-1] = value
                with self.assertRaisesRegex(ValueError, "value_domain"):
                    context.lloyd_labels_equal(values, invalid)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.lloyd_labels_equal(values, values)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.lloyd_labels_equal(values, values)

    def test_compiler_displaces_numpy_comparison_preserves_lloyd_and_guards(self):
        values = np.random.default_rng(1093).normal(size=(42, 17)).astype(np.float32)
        original = kmeans._kmeans_single_lloyd
        with self.context() as context:
            candidate = compile_native_lloyd(original, labels_equal=lambda a, b: labels_equal(context, records, a, b))
            inner = inspect.unwrap(candidate)
            proxy = SimpleNamespace(**{name: getattr(np, name) for name in ("zeros_like", "full", "int32", "zeros")}, array_equal=lambda *args: self.fail("NumPy comparison used"))
            for tol, max_iter in ((0., 20), (1e6, 20), (1e-4, 1)):
                weights, centers = np.ones(len(values), np.float32), values[[0, 7, 13]].copy()
                expected = original(values, weights, centers.copy(), tol=tol, max_iter=max_iter)
                records = []
                with patch.dict(inner.__globals__, {"np": proxy}):
                    actual = candidate(values, weights, centers.copy(), tol=tol, max_iter=max_iter)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[2].view(np.uint32), expected[2].view(np.uint32))
                self.assertEqual((actual[1], actual[3]), (expected[1], expected[3]))
                self.assertEqual(len(records), actual[3])
            with patch("tools.baseline.native_kmeans_lloyd.ARRAY_EQUAL_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "labels_reference_not_supported"):
                compile_native_lloyd(original, labels_equal=object())
            getsource = inspect.getsource; inner_original = inspect.unwrap(original)
            changed = getsource(inner_original).replace("np.array_equal(labels, labels_old)", "np.array_equal(labels.copy(), labels_old)")
            with patch("tools.baseline.native_kmeans_lloyd.inspect.getsource", side_effect=lambda function: changed if function is inner_original else getsource(function)), patch("tools.baseline.native_kmeans_lloyd.LLOYD_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "labels_expression_contract_changed"):
                compile_native_lloyd(original, labels_equal=object())

    def test_fits_alone_composed_bypass_and_failure_restore(self):
        values = np.random.default_rng(1097).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_lloyd = KMeans.fit, kmeans._kmeans_single_lloyd
        def scope(gpu, cpu, records, composed=True):
            options = dict(shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True, product_records=[], chunk_records=[], lloyd_shift_records=[]) if composed else {}
            return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=[], lloyd_label_records=records, **options)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            for composed in (False, True):
                for clusters in (1, 2, 4):
                    weights = np.linspace(.5, 2., len(values), dtype=np.float32)
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    with scope(gpu, cpu, records := [], composed):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        count = len(records); self.assertTrue(count)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(values.copy())).result()
                        self.assertEqual(count, len(records))
                        self.assertIs(kmeans._kmeans_single_lloyd, original_lloyd)
            with scope(gpu, cpu, skipped := []):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, algorithm="elkan", random_state=17).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope(gpu, cpu, []), patch.object(cpu, "lloyd_labels_equal", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(gpu, [], lloyd_label_records=[]):
                    pass
        self.assertIs(KMeans.fit, original_fit); self.assertIs(kmeans._kmeans_single_lloyd, original_lloyd)

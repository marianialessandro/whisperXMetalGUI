from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import inspect
from pathlib import Path
import platform
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans

from tools.baseline.native_double import NativeDoubleNormalization, build_library as build_double
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_potential import NativeKMeansPotentials, PotentialMetrics, build_library, weighted_potentials
from tools.baseline.native_kmeans_prefix import compile_native_prefix


@unittest.skipUnless(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Accelerate required")
class NativeKMeansPotentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "potential.dylib"
        cls.identity = build_library(cls.library)
        cls.double = Path(cls.directory.name) / "double.dylib"
        cls.double_identity = build_double(cls.double)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeKMeansPotentials(self.library, self.identity["library_sha256"])

    def test_raw_exact_numpy_dispatch_shapes_domains_and_maximum(self):
        rng = np.random.default_rng(997)
        with self.context() as context:
            for rows, columns in ((1, 1), (1, 7), (1, 4096), (2, 1), (7, 1), (2, 42), (3, 257), (64, 4096), (1024, 1024)):
                distances, weights = rng.uniform(size=(rows, columns)).astype(np.float32), rng.uniform(size=columns).astype(np.float32)
                for right in (weights, weights[:, None]):
                    expected = distances @ right
                    actual, record = context.weighted_potentials(distances, right)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(record["execution_device"], "cpu")
                    self.assertEqual(record["command_buffers_per_run"], 0)
                    self.assertEqual(record["gpu_kernels"], [])
                    self.assertIsNone(record["computation_gpu_seconds"])
                    self.assertEqual(record["native_blas_calls"], 0 if rows > 1 and columns == 1 else 1)
                    self.assertFalse(record["python_numeric_potential_computation"])
            for value, weight in ((0., -0.), (-0., 1.), (2. ** -64, 2. ** -32), (2. ** 32, 2. ** 32)):
                for rows in (1, 3):
                    distances, weights = np.full((rows, 4096), value, np.float32), np.full(4096, weight, np.float32)
                    actual, _ = context.weighted_potentials(distances, weights)
                    np.testing.assert_array_equal(actual.view(np.uint32), (distances @ weights).view(np.uint32))

    def test_validation_hash_abi_build_preservation_and_closed(self):
        distances, weights = np.ones((3, 7), np.float32), np.ones(7, np.float32)
        with self.context() as context:
            for left, right in ((distances.astype(np.float64), weights), (distances[:, ::2], weights), (distances, weights.astype(np.float64)), (distances, np.ones((1, 7), np.float32)), (distances[0], weights)):
                with self.assertRaisesRegex(ValueError, "operand_format"):
                    context.weighted_potentials(left, right)
            for rows, columns in ((0, 1), (1025, 1), (1, 0), (1, 4097), (1024, 4096)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.weighted_potentials(np.zeros((rows, columns), np.float32), np.ones(columns, np.float32))
            with self.assertRaisesRegex(ValueError, "shape_invalid"):
                context.weighted_potentials(distances, weights[:3])
            for value in (-1., np.nan, np.inf, 2. ** -65, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "distance_domain"):
                    context.weighted_potentials(np.full((3, 7), value, np.float32), weights)
            for value in (-1., np.nan, np.inf, 2. ** -33, 2. ** 33):
                with self.assertRaisesRegex(ValueError, "weight_domain"):
                    context.weighted_potentials(distances, np.full(7, value, np.float32))
            with patch.object(context, "native", return_value=0), self.assertRaisesRegex(ValueError, "native_result_invalid"):
                context.weighted_potentials(distances, weights)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.weighted_potentials(distances, weights)
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            NativeKMeansPotentials(self.library, "0" * 64)
        with patch("tools.baseline.native_kmeans_potential.ct.CDLL", return_value=SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
            self.context()
        with self.assertRaisesRegex(ValueError, "output_exists"):
            build_library(self.library)

    def test_readonly_unaligned_serialization_and_staging(self):
        rng = np.random.default_rng(1009)
        original = rng.uniform(size=(3, 42)).astype(np.float32)
        distances = np.ndarray(original.shape, dtype=np.float32, buffer=bytearray(original.nbytes + 1), offset=1)
        distances[:] = original; distances.flags.writeable = False
        weights = rng.uniform(size=42).astype(np.float32); weights.flags.writeable = False
        expected = original @ weights
        with self.context() as context:
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(lambda _: context.weighted_potentials(distances, weights)[0], range(4)))
            for actual in outputs:
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            actual = weighted_potentials(context, records := [], original, weights)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(records[0]["host_input_staging_copy_bytes"], 0)
            with self.assertRaisesRegex(ValueError, "operand_format"):
                weighted_potentials(context, [], original[:, ::-1], weights[::-1])

    def test_source_guard_and_full_fit_composition_rng_bypass_and_restore(self):
        values = np.random.default_rng(1013).normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        def scope(gpu, cpu, records, composed):
            options = dict(shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[], resident_distances=True, resident_norms=True, resident_minimum=True) if composed else {}
            return native_kmeans_feature_mean(gpu, [], potential_context=cpu, potential_records=records, **options)
        with self.context() as cpu, NativeDoubleNormalization(self.double, self.double_identity["library_sha256"]) as gpu:
            for composed in (False, True):
                for clusters in (1, 2, 4):
                    weights = np.linspace(.5, 2., len(values), dtype=np.float32)
                    random_expected, random_actual = np.random.RandomState(17), np.random.RandomState(17)
                    options = dict(n_clusters=clusters, n_init=2, copy_x=False)
                    expected = KMeans(**options, random_state=random_expected).fit(values.copy(), sample_weight=weights)
                    with scope(gpu, cpu, records := [], composed):
                        actual = KMeans(**options, random_state=random_actual).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        count = len(records); self.assertEqual(count, 2 * clusters)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options, random_state=17).fit(values.copy())).result()
                        self.assertEqual(len(records), count)
                    np.testing.assert_array_equal(random_actual.uniform(size=32), random_expected.uniform(size=32))
            with scope(gpu, cpu, skipped := [], False):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(skipped, [])
            with self.assertRaisesRegex(ValueError, "injected"), scope(gpu, cpu, [], True), patch.object(cpu, "weighted_potentials", side_effect=ValueError("injected")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration"):
                with native_kmeans_feature_mean(gpu, [], potential_context=cpu):
                    pass
        self.assertIs(KMeans.fit, original_fit); self.assertIs(kmeans._kmeans_plusplus, original_plus)
        getsource = inspect.getsource
        changed = getsource(original_plus).replace("closest_dist_sq @ sample_weight", "closest_dist_sq.copy() @ sample_weight")
        with patch("tools.baseline.native_kmeans_prefix.inspect.getsource", side_effect=lambda function: changed if function is original_plus else getsource(function)), patch("tools.baseline.native_kmeans_prefix.PLUSPLUS_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "potential_expression"):
            compile_native_prefix(original_plus, None, potential=object())

    def test_native_abi_address_and_undefined_sanitizers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "potential-sanitizers"
            command = ["xcrun", "clang++", "-std=c++20", "-O1", "-g", "-ffp-contract=off", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(root / "tools/baseline/native_kmeans_potential.cpp"), str(root / "tests/fixtures/baseline/native_kmeans_potential_abi.cpp"), "-framework", "Accelerate", "-o", str(binary)]
            subprocess.run(command, capture_output=True, check=True, timeout=60)
            subprocess.run([str(binary)], capture_output=True, check=True, timeout=60)

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as kmeans
import sklearn.metrics.pairwise as pairwise

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
from tools.baseline.native_kmeans_upcast_norms import clamp_distances, compile_upcast_distances, finish_distances, squared_upcast_norms


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKMeansDistanceClampTests(unittest.TestCase):
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

    def oracle(self, values, diagonal=False):
        output = np.maximum(values, np.float32(0))
        if diagonal:
            np.fill_diagonal(output, 0)
        return output

    def test_finite_bit_patterns_signed_zero_subnormals_and_identity_diagonal(self):
        rng = np.random.default_rng(731)
        words = rng.integers(0, 2 ** 32, size=1024 * 1024, dtype=np.uint32)
        words[(words & 0x7fffffff) >= 0x7f800000] &= np.uint32(0xff7fffff)
        words[:65536] = np.arange(65536, dtype=np.uint32)
        words[65536:131072] = np.arange(65536, dtype=np.uint32) | np.uint32(0x80000000)
        words[-9:] = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x7f7fffff, 0xff7fffff]
        values = words.view(np.float32).reshape(1024, 1024)
        values.flags.writeable = False
        with self.context() as context:
            for diagonal in (False, True):
                actual, record = context.kmeans_distance_clamp(values, diagonal)
                np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(values, diagonal).view(np.uint32))
                self.assertEqual(record["gpu_kernels"], ["float_kmeans_distance_clamp"])
                self.assertEqual(record["command_buffers_per_run"], 1)
                self.assertEqual(record["logical_input_upload_bytes"], values.nbytes)
                self.assertFalse(record["host_numeric_clamping"])
                self.assertFalse(record["host_diagonal_fill"])
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_layouts_unaligned_serialization_domains_shapes_and_missing_abi(self):
        values = np.array([[-0., -1., 2.], [np.nextafter(np.float32(0), np.float32(1)), -3., 4.]], np.float32)
        with self.context() as context:
            for source in (np.asfortranarray(values), np.repeat(values, 2, axis=1)[:, ::2]):
                actual = clamp_distances(context, [], source, False)
                np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(values).view(np.uint32))
            unaligned = np.ndarray(values.shape, dtype=np.float32, buffer=bytearray(values.nbytes + 1), offset=1)
            unaligned[:] = values
            with ThreadPoolExecutor(max_workers=2) as executor:
                actuals = list(executor.map(lambda _: context.kmeans_distance_clamp(unaligned)[0], range(4)))
            for actual in actuals:
                np.testing.assert_array_equal(actual.view(np.uint32), self.oracle(values).view(np.uint32))
            for shape in ((0, 2), (4097, 2), (2, 4097), (1025, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_or_policy_invalid"):
                    context.kmeans_distance_clamp(np.zeros(shape, np.float32))
            for policy in (True, 1, np.bool_(False)):
                with self.assertRaisesRegex(ValueError, "shape_or_policy_invalid"):
                    context.kmeans_distance_clamp(values, policy)
            for value in (np.nan, np.inf, -np.inf):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_distance_clamp(np.full((2, 3), value, np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.kmeans_distance_clamp(values)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_distance_clamp(values)

    def test_whole_distance_helper_displaces_cpu_clamp_and_diagonal_with_policy_and_source_guards(self):
        rng = np.random.default_rng(733)
        X, Y = rng.normal(size=(13, 17)).astype(np.float32), rng.normal(size=(19, 17)).astype(np.float32)
        original = pairwise._euclidean_distances
        with self.context() as context:
            records = []
            norms = lambda values, squared=False: squared_upcast_norms(context, [], values, squared)
            finish = lambda products, left, right: finish_distances(context, [], products, left, right)
            clamp = lambda values, diagonal: clamp_distances(context, records, values, diagonal)
            candidate = compile_upcast_distances(original, norms, finish=finish, clamp=clamp)
            for other in (X, X.copy(), Y):
                expected = original(X, other, squared=True)
                with patch.object(pairwise, "_modify_in_place_if_numpy", side_effect=AssertionError("CPU maximum")), patch.object(pairwise, "_fill_diagonal", side_effect=AssertionError("CPU diagonal")):
                    actual = candidate(X, other, squared=True)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(records[-1]["zero_diagonal"], X is other)
            for inputs in ((X, Y, False), (X.astype(np.float64), Y.astype(np.float64), True)):
                with self.assertRaisesRegex(ValueError, "squared_float32_policy_required"):
                    candidate(inputs[0], inputs[1], squared=inputs[2])
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                compile_upcast_distances(original, norms, clamp=clamp)
            source = inspect.getsource(original)
            changed = source.replace("_fill_diagonal(distances, 0, xp=xp)", "_fill_diagonal(distances, 1, xp=xp)")
            getsource = inspect.getsource
            with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is original else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.EUCLIDEAN_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "clamp_expression_contract_changed"):
                compile_upcast_distances(original, norms, finish=finish, clamp=clamp)
        self.assertIs(pairwise._euclidean_distances, original)

    def test_complete_fits_composition_owner_sparse_explicit_and_failure_restore(self):
        rng = np.random.default_rng(739)
        values = rng.normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as context:
            for weighted in (False, True):
                weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                for clusters in (1, 2, 4):
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    records, norms, finishes = [], [], []
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=norms, distance_finish_records=finishes, distance_clamp_records=records):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
                        np.testing.assert_array_equal(actual.labels_, expected.labels_)
                        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))
                        self.assertTrue(records)
                        self.assertTrue(all("distance_clamping_and_optional_sqrt_diagonal_fill" not in row["cpu_residuals"] for row in norms + finishes))
                        before = len(records)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            executor.submit(lambda: KMeans(**options).fit(values.copy(), sample_weight=weights)).result()
                        self.assertEqual(len(records), before)
            records = []
            with native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=records):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(records, [])
            with self.assertRaisesRegex(ValueError, "injected_clamp_failure"), native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[]), patch.object(context, "kmeans_distance_clamp", side_effect=ValueError("injected_clamp_failure")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_kmeans_feature_mean(context, [], distance_clamp_records=[]):
                    pass
            with patch.object(context, "library", SimpleNamespace(wxdouble_vbx_kmeans_centroids=object(), wxdouble_kmeans_distance_finish=object())), self.assertRaisesRegex(ValueError, "distance_clamp_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], upcast_norm_records=[], distance_finish_records=[], distance_clamp_records=[]):
                    pass
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

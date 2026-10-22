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
from tools.baseline.native_kmeans_upcast_norms import compile_upcast_distances, squared_upcast_norms


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeKMeansUpcastNormTests(unittest.TestCase):
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

    def compare_fit(self, actual, expected):
        np.testing.assert_array_equal(actual.cluster_centers_.view(np.uint32), expected.cluster_centers_.view(np.uint32))
        np.testing.assert_array_equal(actual.labels_, expected.labels_)
        self.assertEqual((actual.inertia_, actual.n_iter_, actual._tol), (expected.inertia_, expected.n_iter_, expected._tol))

    def test_numpy_order_boundaries_fused_rounding_and_extreme_exponents(self):
        rng = np.random.default_rng(291)
        with self.context() as context:
            for dimensions in (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 255, 256, 257, 1024):
                for values in (rng.normal(size=(17, dimensions)), rng.normal(size=(17, dimensions)).astype(np.float32).astype(np.float64), np.ldexp(rng.uniform(.5, 1., (17, dimensions)) * rng.choice((-1., 1.), (17, dimensions)), rng.integers(-199, 201, (17, dimensions)))):
                    expected = np.einsum("ij,ij->i", values, values)
                    actual, record = context.kmeans_upcast_norms(values)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            for value in (-0., 0., 2. ** -200, -(2. ** 200)):
                values = np.full((3, 1024), value)
                actual, _ = context.kmeans_upcast_norms(values)
                np.testing.assert_array_equal(actual.view(np.uint64), np.einsum("ij,ij->i", values, values).view(np.uint64))
            self.assertFalse(record["host_numeric_row_norms"])
            self.assertEqual(record["gpu_kernels"], ["double_kmeans_upcast_norms"])
            self.assertEqual(record["command_buffers_per_run"], 1)
            self.assertGreater(record["computation_gpu_seconds"], 0)

    def test_maximum_budget_readonly_unaligned_thread_serialization_and_domain_guards(self):
        rng = np.random.default_rng(293)
        values = rng.normal(size=(4096, 1024))
        values.flags.writeable = False
        with self.context() as context:
            actual, record = context.kmeans_upcast_norms(values)
            np.testing.assert_array_equal(actual.view(np.uint64), np.einsum("ij,ij->i", values, values).view(np.uint64))
            self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            self.assertEqual(record["logical_input_upload_bytes"], values.nbytes)
            unaligned = np.ndarray((8, 7), dtype=np.float64, buffer=bytearray(8 * 7 * 8 + 1), offset=1)
            unaligned[:] = values[:8, :7]
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.kmeans_upcast_norms(unaligned)[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result.view(np.uint64), np.einsum("ij,ij->i", unaligned, unaligned).view(np.uint64))
            for shape in ((0, 7), (4097, 7), (2, 0), (2, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.kmeans_upcast_norms(np.ones(shape))
            for value in (np.nan, np.inf, np.nextafter(2. ** -200, 0.), np.nextafter(2. ** 200, np.inf)):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.kmeans_upcast_norms(np.full((2, 3), value))
            for source in (np.ones((2, 3), np.float32), np.ones((2, 6))[:, ::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.kmeans_upcast_norms(source)
            with self.assertRaisesRegex(ValueError, "policy_not_supported"):
                squared_upcast_norms(context, [], np.ones((2, 3)), squared=False)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.kmeans_upcast_norms(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.kmeans_upcast_norms(np.ones((2, 3)))

    def test_complete_distance_chunks_symmetry_provided_norms_and_original_api_displacement(self):
        rng = np.random.default_rng(296)
        X, Y = rng.normal(size=(7, 17)).astype(np.float32), rng.normal(size=(11, 17)).astype(np.float32)
        original, upcast = pairwise._euclidean_distances, pairwise._euclidean_distances_upcast
        with self.context() as context:
            records = []
            candidate = compile_upcast_distances(original, lambda values, squared=False: squared_upcast_norms(context, records, values, squared))
            for B in (Y, X):
                for squared in (True, False):
                    expected = original(X, B, squared=squared)
                    with patch.object(pairwise, "row_norms", side_effect=AssertionError("original CPU row norms")):
                        actual = candidate(X, B, squared=squared)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            compiled_upcast = candidate.__globals__["_euclidean_distances_upcast"]
            for batch in (1, 2, 5):
                for B in (X, Y):
                    actual = compiled_upcast(X, Y=B, batch_size=batch)
                    expected = upcast(X, Y=B, batch_size=batch)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            before = len(records)
            XX, YY = np.sum(X.astype(np.float64) ** 2, axis=1)[:, None], np.sum(Y.astype(np.float64) ** 2, axis=1)[None, :]
            np.testing.assert_array_equal(compiled_upcast(X, XX, Y, YY).view(np.uint32), upcast(X, XX, Y, YY).view(np.uint32))
            self.assertEqual(len(records), before)
            self.assertIs(pairwise._euclidean_distances, original)
            self.assertIs(pairwise._euclidean_distances_upcast, upcast)

    def test_full_fits_composition_thread_bypass_failure_restore_sparse_and_explicit_init(self):
        rng = np.random.default_rng(295)
        values = rng.normal(size=(42, 17)).astype(np.float32)
        original_fit, original_plus = KMeans.fit, kmeans._kmeans_plusplus
        with self.context() as context:
            for weighted in (False, True):
                weights = np.linspace(.5, 2., len(values), dtype=np.float32) if weighted else None
                for clusters in (1, 2, 4):
                    options = dict(n_clusters=clusters, n_init=2, random_state=17, copy_x=False)
                    expected = KMeans(**options).fit(values.copy(), sample_weight=weights)
                    records = []
                    with native_kmeans_feature_mean(context, [], shift_records=[], tolerance_records=[], norm_records=[], prefix_records=[], minimum_records=[], probability_records=[], search_records=[], best_records=[], query_records=[], upcast_norm_records=records):
                        actual = KMeans(**options).fit(values.copy(), sample_weight=weights)
                        self.compare_fit(actual, expected)
                        self.assertGreater(len(records), 0)
                        count = len(records)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            other = executor.submit(lambda: KMeans(**options).fit(values.copy(), sample_weight=weights)).result()
                        self.compare_fit(other, expected)
                        self.assertEqual(len(records), count)
            records = []
            with native_kmeans_feature_mean(context, [], upcast_norm_records=records):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(sparse.csr_matrix(values))
                KMeans(n_clusters=2, n_init=1, init=values[:2]).fit(values)
            self.assertEqual(records, [])
            with self.assertRaisesRegex(ValueError, "injected_upcast_failure"), native_kmeans_feature_mean(context, [], upcast_norm_records=[]), patch.object(context, "kmeans_upcast_norms", side_effect=ValueError("injected_upcast_failure")):
                KMeans(n_clusters=2, n_init=1, random_state=17).fit(values)
        self.assertIs(KMeans.fit, original_fit)
        self.assertIs(kmeans._kmeans_plusplus, original_plus)

    def test_source_runtime_configuration_and_missing_abi_guards(self):
        original = pairwise._euclidean_distances
        with patch("tools.baseline.native_kmeans_upcast_norms.EUCLIDEAN_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
            compile_upcast_distances(original, object())
        source = inspect.getsource(pairwise._euclidean_distances_upcast)
        changed = source.replace("row_norms(X_chunk, squared=True)", "row_norms(X_chunk, squared=False)")
        getsource = inspect.getsource
        with patch("tools.baseline.native_kmeans_upcast_norms.inspect.getsource", side_effect=lambda function: changed if function is pairwise._euclidean_distances_upcast else getsource(function)), patch("tools.baseline.native_kmeans_upcast_norms.UPCAST_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_upcast_distances(original, object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_kmeans_feature_mean(context, [], upcast_norm_records=object()):
                    pass
            library = SimpleNamespace(wxdouble_vbx_kmeans_centroids=object())
            with patch.object(context, "library", library), self.assertRaisesRegex(ValueError, "upcast_norms_abi_unavailable"):
                with native_kmeans_feature_mean(context, [], upcast_norm_records=[]):
                    pass

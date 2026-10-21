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
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_ahc_distances import condensed_distances, validate_ahc_reference
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAHCDistanceTests(unittest.TestCase):
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

    def test_condensed_order_exact_rounding_extremes_layouts_and_maximum_budget(self):
        rng = np.random.default_rng(8211)
        with self.context() as context:
            for dtype in (np.float32, np.float64):
                for dimensions in (1, 2, 3, 4, 7, 8, 16, 31, 32, 63, 64, 255, 256, 257, 1024):
                    values = rng.normal(size=(8, dimensions)).astype(dtype)
                    expected = pdist(values, metric="euclidean")
                    actual = condensed_distances(context, records := [], values)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(records[0]["pairs"], 28)
                    self.assertEqual(records[0]["command_buffers_per_run"], 1)
                    self.assertEqual(records[0]["gpu_kernels"], ["double_pdist_euclidean"])
                    self.assertGreater(records[0]["computation_gpu_seconds"], 0)
            for values in (np.zeros((4, 7)), np.full((4, 7), -0.), np.array([[2. ** 32, 2. ** -64, -2. ** 32], [np.nextafter(2. ** 32, 0.), np.nextafter(2. ** -64, 1.), -np.nextafter(2. ** 32, 0.)]]), rng.normal(size=(256, 1024))):
                expected = pdist(values)
                actual, record = context.pdist_euclidean(values)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(record["logical_result_readback_bytes"], expected.nbytes)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            values = rng.normal(size=(16, 256))
            expected = pdist(values)
            for source in (np.asfortranarray(values), np.repeat(values, 2, axis=1)[:, ::2]):
                source.flags.writeable = False
                actual = condensed_distances(context, [], source)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            unaligned = np.ndarray(values.shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1)
            unaligned[:] = values
            actual, _ = context.pdist_euclidean(unaligned)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            original_linkage = linkage(values, method="centroid", metric="euclidean")
            native_linkage = linkage(actual, method="centroid", metric="euclidean")
            np.testing.assert_array_equal(native_linkage.view(np.uint64), original_linkage.view(np.uint64))

    def test_domains_capabilities_and_failure_are_explicit(self):
        with self.context() as context:
            for shape in ((0, 2), (1, 2), (257, 2), (2, 0), (2, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.pdist_euclidean(np.ones(shape))
            for value in (np.inf, -np.inf, np.nan, 2. ** -65, -2. ** -65, np.nextafter(2. ** 32, np.inf)):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.pdist_euclidean(np.full((2, 3), value))
            for values in (np.ones((3, 3), np.int32), np.ones((3, 3, 3)), np.ones((3, 3), np.float16)):
                with self.assertRaisesRegex(ValueError, "format_not_supported"):
                    condensed_distances(context, [], values)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.pdist_euclidean(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.pdist_euclidean(np.ones((2, 3)))

    def test_whole_vbx_bypasses_cpu_pdist_preserves_results_threads_and_restoration(self):
        import scipy.spatial.distance as distance
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with patch.object(distance, "pdist", wraps=pdist) as observed:
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            self.assertEqual(observed.call_count, 1)
            with native_embedding_normalization(None, [], ahc_context=context, ahc_records=(records := [])):
                with patch.object(distance, "pdist", side_effect=AssertionError("CPU pdist executed")):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(records), 1)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    other = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                for a, b in zip(other, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(records), 1)
            with self.assertRaisesRegex(ValueError, "injected_pdist_failure"), native_embedding_normalization(None, [], ahc_context=context, ahc_records=[]), patch.object(context, "pdist_euclidean", side_effect=ValueError("injected_pdist_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
        self.assertIs(VBxClustering.__call__, original)

    def test_source_runtime_registry_and_configuration_guards(self):
        original = VBxClustering.__call__
        source = textwrap.dedent(inspect.getsource(original))
        changed = source.replace('method="centroid", metric="euclidean"', 'method="single", metric="euclidean"')
        self.assertNotEqual(changed, source)
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_normalization(original, None, ahc_distance=object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], ahc_context=context):
                    pass
            with patch("tools.baseline.native_ahc_distances.PDIST_BINARY_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                with native_embedding_normalization(None, [], ahc_context=context, ahc_records=[]):
                    pass
            with patch("tools.baseline.native_ahc_distances.np.__version__", "future"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
                validate_ahc_reference(original)
        self.assertIs(VBxClustering.__call__, original)

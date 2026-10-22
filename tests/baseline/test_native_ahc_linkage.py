from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import itertools
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
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_ahc_linkage import centroid_linkage, validate_linkage_reference
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAHCLinkageTests(unittest.TestCase):
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

    def test_exhaustive_ties_and_signed_zero_heap_order(self):
        with self.context() as context:
            for values in itertools.product((0., 1., 2.), repeat=6):
                distances = np.array(values)
                expected = linkage(distances, method="centroid")
                actual, record = context.centroid_linkage(distances)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            for count in (1, 3, 6, 28):
                distances = np.zeros(count)
                distances[::2] = -0.
                actual, _ = context.centroid_linkage(distances)
                np.testing.assert_array_equal(actual.view(np.uint64), linkage(distances, method="centroid").view(np.uint64))
            self.assertFalse(record["host_numeric_merge_search"])
            self.assertFalse(record["host_numeric_distance_update"])

    def test_random_extremes_maximum_budget_layouts_and_unaligned(self):
        rng = np.random.default_rng(9011)
        with self.context() as context:
            for leaves in (2, 3, 4, 8, 16, 32, 64, 256):
                for dimensions in ((1, 3, 7, 64) if leaves <= 32 else (7,)):
                    distances = pdist(rng.normal(size=(leaves, dimensions)))
                    actual, record = context.centroid_linkage(distances)
                    np.testing.assert_array_equal(actual.view(np.uint64), linkage(distances, method="centroid").view(np.uint64))
                    self.assertEqual(record["logical_result_readback_bytes"], actual.nbytes + 4)
                    self.assertEqual(record["command_buffers_per_run"], 1)
                    self.assertGreater(record["computation_gpu_seconds"], 0)
                    self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            for distance in (0., 2. ** -32, 2. ** 32):
                distances = np.full(256 * 255 // 2, distance)
                actual, _ = context.centroid_linkage(distances)
                np.testing.assert_array_equal(actual.view(np.uint64), linkage(distances, method="centroid").view(np.uint64))
            distances = pdist(rng.normal(size=(8, 7)))
            strided = np.repeat(distances, 2)[::2]
            strided.flags.writeable = False
            actual = centroid_linkage(context, [], strided)
            np.testing.assert_array_equal(actual.view(np.uint64), linkage(distances, method="centroid").view(np.uint64))
            unaligned = np.ndarray(distances.shape, dtype=np.float64, buffer=bytearray(distances.nbytes + 1), offset=1)
            unaligned[:] = distances
            actual, _ = context.centroid_linkage(unaligned)
            np.testing.assert_array_equal(actual.view(np.uint64), linkage(distances, method="centroid").view(np.uint64))

    def test_shapes_domains_caps_and_closed_context(self):
        with self.context() as context:
            for count in (0, 2, 4, 5, 257 * 256 // 2):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.centroid_linkage(np.ones(count))
            for value in (-1., np.inf, np.nan, 2. ** -33, np.nextafter(2. ** 32, np.inf)):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.centroid_linkage(np.full(3, value))
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.centroid_linkage(np.ones(3, np.float32))
            with self.assertRaisesRegex(ValueError, "policy_not_supported"):
                centroid_linkage(context, [], np.ones(3), method="single")
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.centroid_linkage(np.ones(3))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.centroid_linkage(np.ones(3))

    def test_whole_vbx_linkage_displacement_composition_threads_and_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            for native_cut in (False, True):
                with clustering_cpu_audit(audit := [], include_pairwise=True), native_embedding_normalization(None, [], ahc_context=context, ahc_records=(distances := []), linkage_context=context, linkage_records=(trees := []), cut_context=context if native_cut else None, cut_records=(cuts := []), reindex_context=context if native_cut else None, reindex_records=(ranks := []), resident_cut=native_cut):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                    self.assertEqual(len(trees), 1)
                    self.assertNotIn("original_centroid_linkage", distances[0]["cpu_residuals"])
                    if native_cut:
                        self.assertNotIn("centroid_linkage", cuts[0]["cpu_residuals"])
                        self.assertNotIn("centroid_linkage", ranks[0]["cpu_residuals"])
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        other = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                    for a, b in zip(other, expected):
                        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                    self.assertEqual(len(trees), 1)
                observed = {row["operation"]: row["calls_started"] for row in audit}
                self.assertEqual(observed["ahc_linkage"], 0)
                self.assertEqual(observed["ahc_pairwise_distances"], 0)
                self.assertEqual(observed["ahc_fcluster"], int(not native_cut))
            with self.assertRaisesRegex(ValueError, "injected_linkage_failure"), native_embedding_normalization(None, [], ahc_context=context, ahc_records=[], linkage_context=context, linkage_records=[]), patch.object(context, "centroid_linkage", side_effect=ValueError("injected_linkage_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            train = embeddings.reshape(-1, 256)[:1]
            instance.filter_embeddings = lambda *a, **k: (train, None, None)
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            with native_embedding_normalization(None, [], ahc_context=context, ahc_records=(early_distances := []), linkage_context=context, linkage_records=(early_trees := [])):
                actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
            self.assertEqual(early_distances, [])
            self.assertEqual(early_trees, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_configuration_source_and_reference_guards(self):
        original = VBxClustering.__call__
        with self.assertRaisesRegex(ValueError, "requires_native_distances"):
            compile_native_normalization(original, None, ahc_linkage=object())
        source = textwrap.dedent(inspect.getsource(original))
        changed = source.replace('method="centroid", metric="euclidean"', 'method="single", metric="euclidean"')
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_normalization(original, None, ahc_linkage=object(), ahc_distance=object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                with native_embedding_normalization(None, [], linkage_context=context, linkage_records=[]):
                    pass
            with patch("tools.baseline.native_ahc_linkage.HIERARCHY_BINARY_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                with native_embedding_normalization(None, [], ahc_context=context, ahc_records=[], linkage_context=context, linkage_records=[]):
                    pass
            with patch("tools.baseline.native_ahc_linkage.np.__version__", "future"), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                validate_linkage_reference(original)
        self.assertIs(VBxClustering.__call__, original)

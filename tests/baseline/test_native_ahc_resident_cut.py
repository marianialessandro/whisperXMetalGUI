from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_ahc_cut import resident_cut_tree
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAHCResidentCutTests(unittest.TestCase):
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

    def test_exhaustive_identity_inverse_label_order_dtype_and_residency(self):
        with self.context() as context:
            for pair in itertools.combinations(range(3), 2):
                remaining = next(value for value in range(3) if value not in pair)
                for swap_first, swap_root in itertools.product((False, True), repeat=2):
                    children = list(pair[::-1] if swap_first else pair)
                    root = [3, remaining] if swap_root else [remaining, 3]
                    for first_height, root_height in itertools.product((0., 1., 2.), repeat=2):
                        tree = np.array([[*children, first_height, 2.], [*root, root_height, 3.]])
                        for cutoff in (-1., -0., 0., np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 2., 3.):
                            labels = fcluster(tree, cutoff, criterion="distance") - 1
                            expected = np.unique(labels, return_inverse=True)[1]
                            np.testing.assert_array_equal(labels, expected)
                            actual, record = context.fcluster_reindex(tree, cutoff)
                            np.testing.assert_array_equal(actual, expected)
                            self.assertEqual(actual.dtype, np.int64)
                            self.assertEqual(record["gpu_kernels"], ["double_fcluster_distance", "integer_cut_labels_i64"])
                            self.assertEqual(record["command_buffers_per_run"], 1)
                            self.assertEqual(record["intermediate_label_readback_bytes"], 0)
                            self.assertEqual(record["logical_result_readback_bytes"], 24)
                            self.assertFalse(record["host_label_dtype_conversion"])

    def test_methods_maximum_depth_layouts_unaligned_and_native_failure_guards(self):
        rng = np.random.default_rng(8511)
        with self.context() as context:
            for method in ("single", "complete", "average", "weighted", "centroid", "median", "ward"):
                tree = linkage(rng.normal(size=(16, 7)), method=method)
                for cutoff in (-1., 0., .5, 1., 2., 4., float(tree[7, 2])):
                    actual = resident_cut_tree(context, records := [], tree, cutoff)
                    np.testing.assert_array_equal(actual, np.unique(fcluster(tree, cutoff, criterion="distance") - 1, return_inverse=True)[1])
                    self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                    self.assertLessEqual(records[0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            tree = np.array([[0., 1., 2. ** 64, 2.], *[[256. + row - 1, row + 1., 0., row + 2.] for row in range(1, 255)]])
            for source in (tree, np.asfortranarray(tree), np.repeat(tree, 2, axis=1)[:, ::2]):
                source.flags.writeable = False
                actual = resident_cut_tree(context, [], source, 0.)
                np.testing.assert_array_equal(actual, np.unique(fcluster(tree, 0., criterion="distance") - 1, return_inverse=True)[1])
            unaligned = np.ndarray(tree.shape, dtype=np.float64, buffer=bytearray(tree.nbytes + 1), offset=1)
            unaligned[:] = tree
            actual, _ = context.fcluster_reindex(unaligned, 0.)
            np.testing.assert_array_equal(actual, np.arange(256, dtype=np.int64))
            bad_tree = tree.copy()
            bad_tree[0, 0] = 1.
            with self.assertRaisesRegex(ValueError, "topology_invalid"):
                context.fcluster_reindex(bad_tree, 0.)
            for cutoff in (np.nan, np.inf, 2. ** -101):
                with self.assertRaisesRegex(ValueError, "cutoff_domain_invalid"):
                    context.fcluster_reindex(tree, cutoff)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.fcluster_reindex(tree, 0.)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.fcluster_reindex(tree, 0.)

    def test_full_vbx_fused_path_no_standalone_bridge_unique_or_reindex_and_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            with native_embedding_normalization(None, [], cut_context=context, cut_records=(cuts := []), reindex_context=context, reindex_records=(reindex := []), resident_cut=True):
                with patch.object(context, "fcluster_distance", side_effect=AssertionError("standalone cut bridge")), patch.object(context, "ahc_reindex", side_effect=AssertionError("standalone rank bridge")), patch("numpy.unique", side_effect=AssertionError("CPU unique")):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(cuts), 1)
                self.assertEqual(len(reindex), 1)
                self.assertTrue(reindex[0]["resident_label_reuse"])
                self.assertEqual(reindex[0]["command_buffers_per_run"], 0)
                self.assertEqual(reindex[0]["logical_input_upload_bytes"], 0)
                self.assertEqual(reindex[0]["logical_result_readback_bytes"], 0)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    other = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                for a, b in zip(other, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(cuts), 1)
            with self.assertRaisesRegex(ValueError, "injected_resident_cut_failure"), native_embedding_normalization(None, [], cut_context=context, cut_records=[], reindex_context=context, reindex_records=[], resident_cut=True), patch.object(context, "fcluster_reindex", side_effect=ValueError("injected_resident_cut_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            train = embeddings.reshape(-1, 256)[:1]
            instance.filter_embeddings = lambda *a, **k: (train, None, None)
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            with native_embedding_normalization(None, [], cut_context=context, cut_records=(early_cuts := []), reindex_context=context, reindex_records=(early_reindex := []), resident_cut=True):
                actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
            self.assertEqual(early_cuts, [])
            self.assertEqual(early_reindex, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_required_compiler_contexts_same_library_and_abi_guards(self):
        original = VBxClustering.__call__
        for policy in (True, 1):
            with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                compile_native_normalization(original, None, resident_cut=policy)
        with self.context() as context:
            for cut_context, reindex_context in ((None, context), (context, None), (context, SimpleNamespace())):
                with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                    with native_embedding_normalization(None, [], cut_context=cut_context, cut_records=[], reindex_context=reindex_context, reindex_records=[], resident_cut=True):
                        pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "fcluster_reindex_abi_unavailable"):
                with native_embedding_normalization(None, [], cut_context=context, cut_records=[], reindex_context=context, reindex_records=[], resident_cut=True):
                    pass
        self.assertIs(VBxClustering.__call__, original)

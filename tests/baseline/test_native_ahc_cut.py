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
from scipy.cluster.hierarchy import fcluster, linkage
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_ahc_cut import cut_tree, validate_cut_reference
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAHCCutTests(unittest.TestCase):
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

    def test_exhaustive_three_leaf_topologies_orientations_heights_ties_and_label_order(self):
        with self.context() as context:
            for pair in itertools.combinations(range(3), 2):
                remaining = next(value for value in range(3) if value not in pair)
                for swap_first, swap_root in itertools.product((False, True), repeat=2):
                    children = list(pair[::-1] if swap_first else pair)
                    root = [3, remaining] if swap_root else [remaining, 3]
                    for first_height, root_height in itertools.product((0., 1., 2.), repeat=2):
                        tree = np.array([[*children, first_height, 2.], [*root, root_height, 3.]])
                        for cutoff in (-1., -0., 0., np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 2., 3.):
                            expected = fcluster(tree, cutoff, criterion="distance")
                            for zero_based in (False, True):
                                actual, record = context.fcluster_distance(tree, cutoff, zero_based)
                                np.testing.assert_array_equal(actual, expected - int(zero_based))
                                self.assertEqual(actual.dtype, np.int32)
                                self.assertEqual(record["command_buffers_per_run"], 1)
                                self.assertFalse(record["host_numeric_tree_cut"])
                                self.assertFalse(record["host_label_offset"])

    def test_real_linkage_methods_maximum_depth_inversions_extremes_and_layouts(self):
        rng = np.random.default_rng(8411)
        with self.context() as context:
            for method in ("single", "complete", "average", "weighted", "centroid", "median", "ward"):
                tree = linkage(rng.normal(size=(32, 7)), method=method)
                for cutoff in (-1., 0., .5, 1., 2., 4., float(tree[7, 2]), np.nextafter(tree[7, 2], 0.), np.nextafter(tree[7, 2], np.inf)):
                    actual = cut_tree(context, records := [], tree, cutoff)
                    np.testing.assert_array_equal(actual, fcluster(tree, cutoff, criterion="distance") - 1)
                    self.assertEqual(records[0]["gpu_kernels"], ["double_fcluster_distance"])
                    self.assertGreater(records[0]["computation_gpu_seconds"], 0)
                    self.assertLessEqual(records[0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            tree = np.array([[0., 1., 2. ** 64, 2.], *[[256. + row - 1, row + 1., 0., row + 2.] for row in range(1, 255)]])
            for cutoff in (0., 2. ** -100, np.nextafter(2. ** 64, 0.), 2. ** 64):
                actual, _ = context.fcluster_distance(tree, cutoff)
                np.testing.assert_array_equal(actual, fcluster(tree, cutoff, criterion="distance"))
            for source in (np.asfortranarray(tree), np.repeat(tree, 2, axis=1)[:, ::2]):
                source.flags.writeable = False
                actual = cut_tree(context, [], source, 0.)
                np.testing.assert_array_equal(actual, fcluster(tree, 0., criterion="distance") - 1)
            unaligned = np.ndarray(tree.shape, dtype=np.float64, buffer=bytearray(tree.nbytes + 1), offset=1)
            unaligned[:] = tree
            actual, _ = context.fcluster_distance(unaligned, 0.)
            np.testing.assert_array_equal(actual, fcluster(tree, 0., criterion="distance"))
            for height in (0., -0., 2. ** -100, 2. ** 64):
                simple = np.array([[-0., 1., height, 2.]])
                actual, _ = context.fcluster_distance(simple, height)
                np.testing.assert_array_equal(actual, [1, 1])

    def test_input_topology_domains_closed_context_and_missing_capability(self):
        tree = np.array([[0., 1., .5, 2.], [2., 3., 1., 3.]])
        with self.context() as context:
            for source in (tree.astype(np.float32), tree.ravel(), np.repeat(tree, 2, axis=1)[:, ::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.fcluster_distance(source, 1.)
            for shape in ((0, 4), (256, 4), (2, 3)):
                with self.assertRaisesRegex(ValueError, "shape_or_policy_invalid"):
                    context.fcluster_distance(np.ones(shape), 1.)
            for cutoff in (np.nan, np.inf, -np.inf, 2. ** 65, 2. ** -101):
                with self.assertRaisesRegex(ValueError, "cutoff_domain_invalid"):
                    context.fcluster_distance(tree, cutoff)
            for row, column, value in ((0, 0, 3.), (0, 0, -.5), (0, 0, .5), (0, 0, 2. ** -101), (0, 1, 0.), (1, 0, 0.), (0, 3, 3.), (0, 2, -1.), (0, 2, np.nan), (0, 2, np.inf), (0, 2, 2. ** -101)):
                changed = tree.copy()
                changed[row, column] = value
                with self.assertRaises(ValueError):
                    context.fcluster_distance(changed, 1.)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.fcluster_distance(tree, 1.)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.fcluster_distance(tree, 1.)

    def test_whole_vbx_cpu_cut_displacement_composition_threads_and_fault_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with clustering_cpu_audit(reference_audit := [], include_pairwise=True):
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            self.assertEqual(next(row for row in reference_audit if row["operation"] == "ahc_fcluster")["calls_started"], 1)
            with clustering_cpu_audit(audit := [], include_pairwise=True), native_embedding_normalization(None, [], cut_context=context, cut_records=(records := []), ahc_context=context, ahc_records=(distance_records := []), reindex_context=context, reindex_records=(reindex_records := [])):
                actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(records), 1)
                self.assertNotIn("original_fcluster", distance_records[0]["cpu_residuals"])
                self.assertNotIn("fcluster", reindex_records[0]["cpu_residuals"])
                with ThreadPoolExecutor(max_workers=1) as executor:
                    other = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                for a, b in zip(other, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(records), 1)
            observed = {row["operation"]: row for row in audit}
            self.assertEqual(observed["ahc_fcluster"]["calls_started"], 0)
            self.assertEqual(observed["ahc_pairwise_distances"]["calls_started"], 0)
            self.assertEqual(observed["ahc_linkage"]["calls_started"], 1)
            with self.assertRaisesRegex(ValueError, "injected_cut_failure"), native_embedding_normalization(None, [], cut_context=context, cut_records=[]), patch.object(context, "fcluster_distance", side_effect=ValueError("injected_cut_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            train = embeddings.reshape(-1, 256)[:1]
            instance.filter_embeddings = lambda *a, **k: (train, None, None)
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            with native_embedding_normalization(None, [], cut_context=context, cut_records=(early_records := [])):
                actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
            self.assertEqual(early_records, [])
        self.assertIs(VBxClustering.__call__, original)

    def test_source_runtime_and_records_guards(self):
        original = VBxClustering.__call__
        source = textwrap.dedent(inspect.getsource(original))
        for changed in (source.replace('criterion="distance") - 1', 'criterion="distance") - 2'), source.replace('criterion="distance") - 1', 'criterion="maxclust") - 1')):
            self.assertNotEqual(changed, source)
            with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
                compile_native_normalization(original, None, ahc_cut=object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], cut_context=context):
                    pass
            with patch("tools.baseline.native_ahc_cut.HIERARCHY_BINARY_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                with native_embedding_normalization(None, [], cut_context=context, cut_records=[]):
                    pass
            with patch("tools.baseline.native_ahc_cut.np.__version__", "future"), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                validate_cut_reference(original)
        self.assertIs(VBxClustering.__call__, original)

from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
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
from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
from tools.baseline.native_ahc_resident import resident_pipeline_labels
from tools.baseline.native_double import Metrics, NativeDoubleNormalization, build_library, shader_source
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAHCResidentTests(unittest.TestCase):
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

    def options(self, context, records):
        return dict(ahc_context=context, ahc_records=records[0], linkage_context=context, linkage_records=records[1], cut_context=context, cut_records=records[2], reindex_context=context, reindex_records=records[3], resident_cut=True, resident_pipeline=True, pipeline_records=records[4])

    def expected(self, values, cutoff):
        return np.unique(fcluster(linkage(values, method="centroid"), cutoff, criterion="distance") - 1, return_inverse=True)[1]

    def test_exhaustive_ties_zero_and_cut_boundaries(self):
        with self.context() as context:
            for points in itertools.product((0., 1., 2.), repeat=4):
                values = np.array(points).reshape(4, 1)
                tree = linkage(values, method="centroid")
                for cutoff in (-1., -0., 0., *tree[:, 2], *np.nextafter(tree[:, 2], -np.inf), 4.):
                    if 0 < abs(cutoff) < 2. ** -100:
                        continue
                    actual, record = context.ahc_resident(values, float(cutoff))
                    np.testing.assert_array_equal(actual, self.expected(values, cutoff))
                    self.assertEqual(actual.dtype, np.int64)
            self.assertEqual(record["command_buffers_per_run"], 1)
            self.assertEqual(len(record["gpu_kernels"]), 4)
            self.assertEqual(record["logical_result_readback_bytes"], 36)
            for key in ("intermediate_distance_readback_bytes", "intermediate_tree_readback_bytes", "intermediate_label_readback_bytes"):
                self.assertEqual(record[key], 0)

    def test_random_maximum_budget_layouts_unaligned_and_buffer_reuse(self):
        rng = np.random.default_rng(2931)
        with self.context() as context:
            for rows, features in ((2, 1), (3, 7), (16, 64), (42, 256), (256, 1024)):
                values = rng.normal(size=(rows, features))
                expected = self.expected(values, 1.)
                for source in (values, np.asfortranarray(values), np.repeat(values, 2, axis=1)[:, ::2]):
                    source.flags.writeable = False
                    records = [[] for _ in range(5)]
                    actual = resident_pipeline_labels(context, records[4], *records[:4], source, 1.)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertGreater(records[4][0]["computation_gpu_seconds"], 0)
                    self.assertLessEqual(records[4][0]["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
                    self.assertEqual(sum(row[0]["command_buffers_per_run"] for row in records), 1)
                    self.assertEqual(sum(row[0]["logical_input_upload_bytes"] for row in records), values.nbytes)
                    self.assertEqual(sum(row[0]["logical_result_readback_bytes"] for row in records), rows * 8 + 4)
            values = rng.normal(size=(8, 7))
            unaligned = np.ndarray(values.shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1)
            unaligned[:] = values
            actual, _ = context.ahc_resident(unaligned, 1.)
            np.testing.assert_array_equal(actual, self.expected(values, 1.))
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: context.ahc_resident(values, 1.)[0], range(4)))
            for result in results:
                np.testing.assert_array_equal(result, actual)

    def test_domains_generated_distance_rejection_and_closed_context(self):
        with self.context() as context:
            for shape in ((0, 7), (1, 7), (257, 7), (2, 0), (2, 1025)):
                with self.assertRaisesRegex(ValueError, "shape_or_policy_invalid"):
                    context.ahc_resident(np.ones(shape), 1.)
            for value in (np.nan, np.inf, 2. ** -65, np.nextafter(2. ** 32, np.inf)):
                with self.assertRaisesRegex(ValueError, "value_domain_invalid"):
                    context.ahc_resident(np.full((2, 1), value), 1.)
            for values in (np.array([[0.], [2. ** -33]]), np.array([[-2. ** 32], [2. ** 32]])):
                with self.assertRaisesRegex(ValueError, "distance_domain_invalid"):
                    context.ahc_resident(values, 1.)
                actual, _ = context.ahc_resident(np.zeros((2, 1)), 0.)
                np.testing.assert_array_equal(actual, [0, 0])
            for cutoff in (np.nan, np.inf, 2. ** -101, 2. ** 65):
                with self.assertRaisesRegex(ValueError, "cutoff_domain_invalid"):
                    context.ahc_resident(np.ones((2, 1)), cutoff)
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.ahc_resident(np.ones((2, 1), np.float32), 1.)
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.ahc_resident(np.ones((2, 1)), 1.)
        with self.assertRaisesRegex(ValueError, "closed"):
            context.ahc_resident(np.ones((2, 1)), 1.)

    def test_gpu_failure_propagation_blocks_cut_and_preserves_caller(self):
        source = shader_source()
        replacements = ((b"centroid_linkage_body(distances, d, tree, status);\n}", b"centroid_linkage_body(distances, d, tree, status); status[0] = 3;\n}", "negative_variance"), (b"    fcluster_distance_body(tree, d, output);\n    for", b"    fcluster_distance_body(tree, d, output); output[0] = -1;\n    for", "labels_invalid"), (b"    if (sizes[2 * d.leaves - 2] != d.leaves)", b"    if (true || sizes[2 * d.leaves - 2] != d.leaves)", "tree_invalid"))
        producer = b"    centroid_linkage_body(distances, d, tree, status);\n}\n\n// Validate"
        for corruption in (b"tree[0] = 0x7ff0000000000000ul;", b"tree[1] = tree[0];", b"tree[2] = 0xbff0000000000000ul;", b"tree[3] = 0;"):
            replacements += ((producer, b"    centroid_linkage_body(distances, d, tree, status); " + corruption + b"\n}\n\n// Validate", "tree_invalid"),)
        for old, new, error in replacements:
            self.assertIn(old, source)
            with patch("tools.baseline.native_double.shader_source", return_value=source.replace(old, new)), self.context() as context:
                values, output, metrics, message = np.ones((3, 2)), np.full(3, 91, dtype=np.int64), Metrics(), ct.create_string_buffer(128)
                native = context.library.wxdouble_ahc_resident
                native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_double, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
                self.assertNotEqual(native(context.context, values.ctypes.data, 3, 2, 1., output.ctypes.data, ct.byref(metrics), message, len(message)), 0)
                self.assertIn(error, message.value.decode())
                np.testing.assert_array_equal(output, [91, 91, 91])

    def test_whole_vbx_no_intermediate_bridges_thread_restore_and_early_branch(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            records = [[] for _ in range(5)]
            with clustering_cpu_audit(audit := [], include_pairwise=True), native_embedding_normalization(None, [], **self.options(context, records)):
                with patch.object(context, "pdist_euclidean", side_effect=AssertionError("pdist bridge")), patch.object(context, "centroid_linkage", side_effect=AssertionError("tree bridge")), patch.object(context, "fcluster_reindex", side_effect=AssertionError("cut bridge")), patch("numpy.unique", side_effect=AssertionError("unique")):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                self.assertEqual([len(items) for items in records], [1] * 5)
            self.assertTrue(all(row["calls_started"] == 0 for row in audit))
            with self.assertRaisesRegex(ValueError, "injected_resident_failure"), native_embedding_normalization(None, [], **self.options(context, [[] for _ in range(5)])), patch.object(context, "ahc_resident", side_effect=ValueError("injected_resident_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            instance.filter_embeddings = lambda *a, **k: (embeddings.reshape(-1, 256)[:1], None, None)
            expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            records = [[] for _ in range(5)]
            with native_embedding_normalization(None, [], **self.options(context, records)):
                actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
            self.assertEqual(records, [[] for _ in range(5)])
        self.assertIs(VBxClustering.__call__, original)

    def test_configuration_and_abi_guards(self):
        original = VBxClustering.__call__
        with self.assertRaisesRegex(ValueError, "configuration_invalid"):
            compile_native_normalization(original, None, ahc_resident=object())
        with self.context() as context:
            for changes in ({"resident_cut": False}, {"linkage_context": None}, {"cut_context": object()}, {"pipeline_records": None}, {"resident_pipeline": 1}):
                options = self.options(context, [[] for _ in range(5)])
                options.update(changes)
                with self.assertRaisesRegex(ValueError, "configuration_invalid"):
                    with native_embedding_normalization(None, [], **options):
                        pass
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "ahc_resident_abi_unavailable"):
                with native_embedding_normalization(None, [], **self.options(context, [[] for _ in range(5)])):
                    pass
        self.assertIs(VBxClustering.__call__, original)

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
from pyannote.audio.pipelines.clustering import VBxClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_ahc_reindex import reindex_labels, validate_reindex_reference
from tools.baseline.native_double import NativeDoubleNormalization, build_library
from tools.baseline.native_embedding_adapter import compile_native_normalization, native_embedding_normalization
from tests.baseline import test_native_embedding_adapter as embedding_tests


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativeAHCReindexTests(unittest.TestCase):
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

    def test_exhaustive_ties_sparse_classes_layouts_dtype_and_maximum(self):
        rng = np.random.default_rng(8311)
        with self.context() as context:
            for sequence in itertools.product((0, 63, 4095), repeat=6):
                labels = np.array(sequence, dtype=np.int64)
                expected = np.unique(labels, return_inverse=True)[1]
                actual, record = context.ahc_reindex(labels)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
            for labels in (np.array([4095]), np.array([64, 63, 128, 127, 0, 4095, 64]), np.arange(4096, dtype=np.int64)[::-1].copy(), rng.integers(0, 4096, 4096, dtype=np.int64)):
                actual, record = context.ahc_reindex(labels)
                np.testing.assert_array_equal(actual, np.unique(labels, return_inverse=True)[1])
                self.assertEqual(record["command_buffers_per_run"], 1)
                self.assertEqual(record["gpu_kernels"], ["integer_ahc_reindex"])
                self.assertEqual(record["logical_result_readback_bytes"], labels.nbytes)
                self.assertGreater(record["computation_gpu_seconds"], 0)
                self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)
            labels = rng.integers(0, 4096, 42, dtype=np.int64)
            expected = np.unique(labels, return_inverse=True)[1]
            for source in (labels.astype(np.int32), np.repeat(labels, 2)[::2]):
                source.flags.writeable = False
                actual = reindex_labels(context, records := [], source)
                np.testing.assert_array_equal(actual, expected)
            unaligned = np.ndarray(labels.shape, dtype=np.int64, buffer=bytearray(labels.nbytes + 1), offset=1)
            unaligned[:] = labels
            actual, _ = context.ahc_reindex(unaligned)
            np.testing.assert_array_equal(actual, expected)

    def test_domains_lifetime_capability_and_format_guards(self):
        with self.context() as context:
            for labels in (np.ones(3), np.ones(3, np.int32), np.ones((2, 2), np.int64), np.ones(6, np.int64)[::2]):
                with self.assertRaisesRegex(ValueError, "format_invalid"):
                    context.ahc_reindex(labels)
            for count in (0, 4097):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.ahc_reindex(np.ones(count, np.int64))
            for value in (-1, 4096, np.iinfo(np.int64).max):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.ahc_reindex(np.array([value], np.int64))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.ahc_reindex(np.ones(3, np.int64))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.ahc_reindex(np.ones(3, np.int64))

    def test_whole_vbx_skips_cpu_unique_preserves_results_thread_and_failure_restore(self):
        embeddings, instance, q = embedding_tests.NativeEmbeddingAdapterTests().fixture()
        original = VBxClustering.__call__
        unique = np.unique
        with self.context() as context, patch("pyannote.audio.pipelines.clustering.cluster_vbx", return_value=(q, np.ones(3))):
            with patch("numpy.unique", wraps=unique) as observed:
                expected = original(instance, embeddings, min_clusters=1, max_clusters=10)
            self.assertEqual(observed.call_count, 1)
            with native_embedding_normalization(None, [], reindex_context=context, reindex_records=(records := [])):
                with patch("numpy.unique", side_effect=AssertionError("CPU unique executed")):
                    actual = VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(records), 1)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    other = executor.submit(VBxClustering.__call__, instance, embeddings, min_clusters=1, max_clusters=10).result()
                for a, b in zip(other, expected):
                    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
                self.assertEqual(len(records), 1)
            with self.assertRaisesRegex(ValueError, "injected_reindex_failure"), native_embedding_normalization(None, [], reindex_context=context, reindex_records=[]), patch.object(context, "ahc_reindex", side_effect=ValueError("injected_reindex_failure")):
                VBxClustering.__call__(instance, embeddings, min_clusters=1, max_clusters=10)
        self.assertIs(VBxClustering.__call__, original)

    def test_source_reference_and_configuration_guards(self):
        original = VBxClustering.__call__
        source = textwrap.dedent(inspect.getsource(original))
        changed = source.replace("np.unique(ahc_clusters, return_inverse=True)", "np.unique(ahc_clusters, return_inverse=False)")
        with patch("tools.baseline.native_embedding_adapter.inspect.getsource", return_value=changed), patch("tools.baseline.native_embedding_adapter.VBX_CALL_SHA256", hashlib.sha256(changed.encode()).hexdigest()), self.assertRaisesRegex(ValueError, "expression_contract_changed"):
            compile_native_normalization(original, None, ahc_reindex=object())
        with self.context() as context:
            with self.assertRaisesRegex(ValueError, "records_required"):
                with native_embedding_normalization(None, [], reindex_context=context):
                    pass
            with patch("tools.baseline.native_ahc_reindex.UNIQUE_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                with native_embedding_normalization(None, [], reindex_context=context, reindex_records=[]):
                    pass
            with patch("tools.baseline.native_ahc_reindex.np.__version__", "future"), self.assertRaisesRegex(ValueError, "reference_not_supported"):
                validate_reindex_reference(original)
        self.assertIs(VBxClustering.__call__, original)

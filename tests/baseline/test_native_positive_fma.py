from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.native_double import NativeDoubleNormalization, build_library


@unittest.skipUnless(platform.system() == "Darwin" and torch.backends.mps.is_available(), "Apple Metal required")
class NativePositiveFmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.library = Path(cls.directory.name) / "double.dylib"
        cls.identity = build_library(cls.library)
        cls.fma = ct.CDLL(None).fma
        cls.fma.argtypes = [ct.c_double] * 3
        cls.fma.restype = ct.c_double

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def context(self):
        return NativeDoubleNormalization(self.library, self.identity["library_sha256"])

    def reference(self, values):
        return np.array([self.fma(*row) for row in values])

    def test_exact_fused_rounding_ties_carry_wide_exponent_alignment_and_random(self):
        rng = np.random.default_rng(8911)
        special = [[0., 0., 0.], [1., 1., 2. ** -53], [np.nextafter(1., 2.), 1., 2. ** -53], [np.nextafter(2., 0.), 1., 2. ** -52], [1. + 2. ** -52, 1. + 2. ** -52, 2. ** -53], [2. ** -200, 2. ** -200, 2. ** 200], [2. ** 200, 2. ** 200, 2. ** -200]]
        for a, b in ((1., 1.), (1.5, 1.5)):
            for exponent in (-200, -129, -128, -127, -65, -64, -63, 0, 63, 64, 127, 128, 129, 199):
                special.append([a, b, np.ldexp(1.25, exponent)])
        with self.context() as context:
            values = np.array(special)
            actual, record = context.positive_fma(values)
            np.testing.assert_array_equal(actual.view(np.uint64), self.reference(values).view(np.uint64))
            self.assertNotEqual(actual[4], values[4, 0] * values[4, 1] + values[4, 2])
            mismatches_to_nonfused = 0
            for _ in range(8):
                values = np.ldexp(rng.uniform(1, 2, (4096, 3)), rng.integers(-200, 200, (4096, 3)))
                values[::17, 0] = 0.; values[::19, 1] = 0.; values[::23, 2] = 0.
                expected = self.reference(values)
                actual, record = context.positive_fma(values)
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                mismatches_to_nonfused += np.count_nonzero(expected.view(np.uint64) != (values[:, 0] * values[:, 1] + values[:, 2]).view(np.uint64))
            self.assertGreater(mismatches_to_nonfused, 0)
            self.assertEqual(record["gpu_kernels"], ["double_positive_fma"])
            self.assertEqual(record["command_buffers_per_run"], 1)
            self.assertGreater(record["computation_gpu_seconds"], 0)
            self.assertLessEqual(record["allocated_shared_buffer_bytes"], 72 * 1024 * 1024 + 40 * 1024)

    def test_domains_shapes_and_lifetime(self):
        with self.context() as context:
            for value in (-0., -1., np.inf, np.nan, 2. ** -201, np.nextafter(2. ** 200, np.inf)):
                with self.assertRaisesRegex(ValueError, "domain_invalid"):
                    context.positive_fma(np.full((1, 3), value))
            for shape in ((0, 3), (4097, 3), (2, 2)):
                with self.assertRaisesRegex(ValueError, "shape_invalid"):
                    context.positive_fma(np.ones(shape))
            with self.assertRaisesRegex(ValueError, "format_invalid"):
                context.positive_fma(np.ones((2, 3), np.float32))
            with patch.object(context, "library", SimpleNamespace()), self.assertRaisesRegex(ValueError, "abi_unavailable"):
                context.positive_fma(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "closed"):
            context.positive_fma(np.ones((2, 3)))

    def test_unaligned_readonly_and_serialized_thread_calls(self):
        values = np.array([[1. + 2. ** -52, 1. + 2. ** -52, 2. ** -53], [2., 3., 4.]])
        unaligned = np.ndarray(values.shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1)
        unaligned[:] = values
        unaligned.flags.writeable = False
        expected = self.reference(values)
        with self.context() as context, ThreadPoolExecutor(max_workers=2) as executor:
            for actual in executor.map(lambda _: context.positive_fma(unaligned)[0], range(8)):
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

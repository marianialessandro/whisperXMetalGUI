from pathlib import Path
import sys
import unittest
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import mlx.core as mx
from tools.baseline.reproducibility import seed_process


class OptInReproducibilityTests(unittest.TestCase):
    def sample(self):
        return random.random(), np.random.random(3).tolist(), torch.rand(3).tolist(), mx.random.uniform(shape=(3,)).tolist()

    def test_requested_seed_replays_process_rngs_without_forcing_deterministic_algorithms(self):
        policy = torch.are_deterministic_algorithms_enabled()
        report = seed_process(17)
        first = self.sample()
        seed_process(17)
        self.assertEqual(first, self.sample())
        self.assertEqual(policy, torch.are_deterministic_algorithms_enabled())
        self.assertFalse(report["deterministic_algorithms_forced"])

    def test_invalid_seed_rejected_before_import_or_rng_mutation(self):
        seed_process(17)
        expected = self.sample()
        for seed in (-1, 2 ** 32, True, "17"):
            with self.subTest(seed=seed):
                seed_process(17)
                with self.assertRaises(ValueError): seed_process(seed)
                self.assertEqual(expected, self.sample())


if __name__ == "__main__":
    unittest.main()

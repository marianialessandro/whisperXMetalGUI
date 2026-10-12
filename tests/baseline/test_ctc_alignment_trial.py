from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
import torch
import whisperx.alignment as alignment
from tools.baseline.ctc_alignment_trial import metal_ctc


@unittest.skipUnless(mx.metal.is_available(), "Metal hardware required")
class ScopedAlignmentTrialTests(unittest.TestCase):
    def test_trial_returns_original_point_shape_and_restores_functions(self):
        trellis_function, backtrack_function = alignment.get_trellis, alignment.backtrack
        emission = torch.tensor([[-3., -.1], [-.2, -2.], [-.1, -3.]], dtype=torch.float32)
        expected = backtrack_function(trellis_function(emission, [1]), emission, [1])
        spans = []
        with metal_ctc(alignment, spans):
            trellis = alignment.get_trellis(emission, [1])
            self.assertEqual(trellis.size(0), 4)
            self.assertEqual(trellis.size(1), 2)
            result = alignment.backtrack(trellis, emission, [1])
            self.assertEqual([(point.token_index, point.time_index) for point in result], [(point.token_index, point.time_index) for point in expected])
        self.assertIs(alignment.get_trellis, trellis_function)
        self.assertIs(alignment.backtrack, backtrack_function)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["logical_cpu_to_mlx_emission_bytes"], 24)

    def test_failure_restores_hooks_and_does_not_fall_back_to_original_cpu(self):
        trellis_function, backtrack_function = alignment.get_trellis, alignment.backtrack
        with self.assertRaises(ValueError):
            with metal_ctc(alignment, []):
                alignment.get_trellis(torch.zeros((3, 2), dtype=torch.float32), [])
        self.assertIs(alignment.get_trellis, trellis_function)
        self.assertIs(alignment.backtrack, backtrack_function)


if __name__ == "__main__":
    unittest.main()

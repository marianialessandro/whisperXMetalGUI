from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import metrics


class BundledDiarizationMetricTests(unittest.TestCase):
    def test_actual_bundled_der_jer_preserve_permutation_overlap_and_uem(self):
        reference = [{"start": 0, "end": 2, "speaker": "A"}, {"start": 1, "end": 3, "speaker": "B"}]
        equivalent = [{"start": 0, "end": 2, "speaker": "X"}, {"start": 1, "end": 3, "speaker": "Y"}]
        full = metrics.diarization_scores(reference, equivalent, [[0, 3]])
        self.assertEqual(full["der"], 0)
        self.assertEqual(full["jer"], 0)
        self.assertEqual(full["mapping"], {"X": "A", "Y": "B"})
        missed = metrics.diarization_scores(reference, equivalent[:1], [[0, 3]])
        self.assertAlmostEqual(missed["der"], 0.5)
        self.assertAlmostEqual(missed["jer"], 0.5)
        cropped = metrics.diarization_scores(reference, equivalent[:1], [[0, 1]])
        self.assertEqual(cropped["der"], 0)
        self.assertEqual(cropped["speaker_count_error"], 0)


if __name__ == "__main__":
    unittest.main()

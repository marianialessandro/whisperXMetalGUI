from pathlib import Path
import sys
import unittest
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import metrics


class MetricTests(unittest.TestCase):
    def test_edit_counts_are_analytical_and_deterministic(self):
        report = metrics.edit_distance(["a", "b", "c"], ["a", "x", "c", "d"])
        self.assertEqual((report["substitutions"], report["insertions"], report["deletions"]), (1, 1, 0))
        self.assertAlmostEqual(report["rate"], 2 / 3)
        self.assertEqual(metrics.edit_distance(["a", "b"], ["a"])["deletions"], 1)
        self.assertEqual(report, metrics.edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]))

    def test_unicode_normalization_keeps_numerals_and_separates_raw(self):
        report = metrics.text_scores("Caffè, 99!", "caffe\u0300 99", "it")
        self.assertEqual(report["normalized"]["wer"]["rate"], 0)
        self.assertGreater(report["raw"]["wer"]["rate"], 0)
        self.assertGreater(metrics.text_scores("99", "ninety nine", "en")["normalized"]["wer"]["rate"], 0)

    def test_no_space_language_uses_cer_and_no_invented_word_tokenizer(self):
        report = metrics.text_scores("你好", "你号", "zh")
        self.assertEqual(report["primary_metric"], "cer")
        self.assertEqual(report["normalized"]["cer"]["rate"], 0.5)
        self.assertIsNone(report["normalized"]["wer"])

    def test_silence_never_has_fabricated_zero_wer_and_overlap_time_is_union(self):
        report = metrics.text_scores("", "hello", "en")
        self.assertIsNone(report["normalized"]["wer"]["rate"])
        self.assertEqual(report["normalized"]["wer"]["insertions"], 1)
        self.assertEqual(metrics.hallucinated_seconds([{"start": 0, "end": 2}, {"start": 1, "end": 3}]), 3)

    def test_timestamp_denominator_includes_unmatched_and_untimed_words(self):
        reference = [{"word": "one", "start": 0, "end": 1}, {"word": "99", "start": 1, "end": 2}, {"word": "three", "start": 2, "end": 3}]
        hypothesis = [{"word": "one", "start": 0.1, "end": 1.2}, {"word": "99"}]
        report = metrics.timestamp_scores(reference, hypothesis)
        self.assertEqual(report["reference_words"], 3)
        self.assertEqual(report["coverage"], 1 / 3)
        self.assertEqual(report["untimed_hypothesis_matches"], 1)
        self.assertEqual(report["unmatched_reference"], 1)
        self.assertAlmostEqual(report["start_mae"], 0.1)
        self.assertAlmostEqual(report["end_p95"], 0.2)

    def test_speaker_mapping_is_reused_and_unassigned_words_reduce_coverage(self):
        reference = [{"word": "one", "speaker": "A"}, {"word": "two", "speaker": "B"}, {"word": "three", "speakers": ["A", "B"]}]
        hypothesis = [{"word": "one", "speaker": "X"}, {"word": "two"}, {"word": "three", "speaker": "Y"}]
        report = metrics.word_speaker_scores(reference, hypothesis, {"X": "A", "Y": "B"})
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["assigned_coverage"], 0.5)
        self.assertEqual(report["ambiguous_reference"], 1)

    def test_nonfinite_or_reversed_timestamps_are_rejected(self):
        for interval in ({"start": float("nan"), "end": 1}, {"start": 2, "end": 1}):
            with self.assertRaises(metrics.MetricError):
                metrics.hallucinated_seconds([interval])

    def test_long_transcript_uses_bounded_memory_and_keeps_exact_edits(self):
        reference = [str(index) for index in range(4096)]
        hypothesis = reference.copy()
        hypothesis[501] = "replacement"
        del hypothesis[100]
        hypothesis.append("extra")
        report = metrics.edit_distance(reference, hypothesis)
        self.assertEqual((report["substitutions"], report["insertions"], report["deletions"]), (1, 1, 1))
        self.assertEqual(len(report["matches"]), 4094)

    def test_bit_vector_rows_agree_with_analytical_small_matrix_paths(self):
        rng = random.Random(17)
        for _ in range(80):
            reference = [rng.choice("abc") for _ in range(rng.randrange(1, 30))]
            hypothesis = [rng.choice("abc") for _ in range(rng.randrange(1, 30))]
            expected = metrics.edit_distance(reference, hypothesis)
            self.assertEqual(metrics.distance_row(reference, hypothesis)[-1], expected["substitutions"] + expected["insertions"] + expected["deletions"])

    def test_parity_audio_cannot_be_scored_as_independent_accuracy(self):
        with self.assertRaises(metrics.MetricError) as error:
            metrics.evaluate_case({"cases": [{"id": "parity_it_1", "type": "parity_only"}]}, {"licenses": []}, Path("."), "parity_it_1", "missing-result.json")
        self.assertEqual(str(error.exception), "independent_ground_truth_required")


if __name__ == "__main__":
    unittest.main()

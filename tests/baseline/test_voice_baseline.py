from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.evaluation import voice_baseline


class VoiceBaselineTests(unittest.TestCase):
    def samples(self):
        jobs, rows = [], []
        for index in range(6):
            split = "calibration" if index < 3 else "evaluation"
            for replica in range(3):
                jobs.append({"id": f"job-{index}-{replica}", "asset_id": f"recording-{index}", "quality": "Rapida", "split": split, "replica": replica})
                rows.append({"score": {"text": {"normalized": {"cer": {"rate": 0.1}, "wer": {"rate": 0.2}}}, "dataset_timestamp_agreement": {"timed_matches": 10, "start_mae": 0.2, "end_mae": 0.3, "coverage": 0.5}, "dataset_turn_agreement": {"der": 0.1 if split == "calibration" else 0.9, "jer": 0.2, "speaker_count_error": 0}, "dataset_word_speaker_agreement": {"accuracy": 0.8 if split == "calibration" else 0.7}}})
        return jobs, rows

    def test_voice_bounds_use_recording_means_and_never_retune_from_heldout(self):
        jobs, rows = self.samples()
        report = voice_baseline.summarize(jobs, rows)["Rapida"]
        self.assertAlmostEqual(report["der"]["bound"], 0.1)
        self.assertAlmostEqual(report["der"]["evaluation_mean"], 0.9)
        self.assertFalse(report["der"]["heldout_within_calibration_bound"])
        self.assertEqual(report["der"]["degradation_delta"], 0)
        self.assertAlmostEqual(report["word_speaker_accuracy"]["bound"], 0.8)
        self.assertEqual(report["word_speaker_accuracy"]["direction"], "lower")
        self.assertFalse(report["word_speaker_accuracy"]["heldout_within_calibration_bound"])
        self.assertEqual(report["der"]["calibration_recordings"], 3)

    def test_missing_replica_duplicate_or_nonfinite_metric_is_rejected(self):
        jobs, rows = self.samples()
        with self.assertRaisesRegex(ValueError, "replica_missing"): voice_baseline.summarize(jobs[1:], rows[1:])
        jobs[1]["id"] = jobs[0]["id"]
        with self.assertRaisesRegex(ValueError, "measurements_missing"): voice_baseline.summarize(jobs, rows)
        jobs, rows = self.samples()
        rows[0]["score"]["dataset_turn_agreement"]["der"] = float("nan")
        with self.assertRaisesRegex(ValueError, "metric_missing"): voice_baseline.summarize(jobs, rows)
        jobs, rows = self.samples()
        rows[0]["score"]["dataset_word_speaker_agreement"]["accuracy"] = None
        with self.assertRaisesRegex(ValueError, "metric_missing"): voice_baseline.summarize(jobs, rows)


if __name__ == "__main__":
    unittest.main()

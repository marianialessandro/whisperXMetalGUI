import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import freeze, metrics, profiling, corpus


class FreezeTests(unittest.TestCase):
    def test_draft_or_null_nan_and_candidate_derived_thresholds_are_rejected(self):
        for contract in [{"state": "draft"}, {"state": "calibrated", "thresholds": None}, {"state": "calibrated", "thresholds": {"wer": float("nan")}}, {"state": "calibrated", "calibration_role": "candidate"}]:
            with self.assertRaises(freeze.FreezeError):
                freeze.validate_contract(contract)

    def test_exclusive_seal_preserves_existing_evidence_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "external.txt").write_text("preserve")
            payload = {"kind": "unit_fixture", "policy": metrics.POLICY}
            digest = freeze.seal(root, "fixture.json", payload)
            self.assertTrue(freeze.verify_seal(root, "fixture.json", digest))
            with self.assertRaises(freeze.FreezeError):
                freeze.seal(root, "fixture.json", {"changed": True})
            self.assertEqual((root / "external.txt").read_text(), "preserve")
            (root / "fixture.json").write_text('{"changed":true}')
            with self.assertRaises(freeze.FreezeError):
                freeze.verify_seal(root, "fixture.json", digest)

    def test_uncovered_case_and_operator_cannot_satisfy_gate(self):
        root = Path(__file__).resolve().parents[2]
        coverage = json.loads((root / "evaluation/baseline/coverage.json").read_text())
        report = freeze.evidence_gate({"cases": [], "operators": [], "prerequisites": []}, coverage)
        self.assertEqual(report["status"], "prerequisite_missing")
        self.assertTrue(any(item["code"] == "required_case_missing" for item in report["missing"]))
        self.assertTrue(any(item["code"] == "operator_observation_missing" for item in report["missing"]))
        coverage["required_cases"].pop()
        with self.assertRaises(freeze.FreezeError):
            freeze.evidence_gate({}, coverage)

    def test_bootstrap_is_recording_level_seeded_and_deterministic_zero_stays_zero(self):
        result = freeze.bootstrap_bound([0.0, 0.0, 0.0], "upper", seed=17)
        self.assertEqual(result, 0)
        self.assertEqual(freeze.bootstrap_bound([0.1, 0.2, 0.3], "upper", seed=17), freeze.bootstrap_bound([0.1, 0.2, 0.3], "upper", seed=17))
        with self.assertRaises(freeze.FreezeError):
            freeze.bootstrap_bound([0.1, 0.2], "upper", seed=17)

    def test_calibration_rejects_parity_only_and_candidate_inputs(self):
        for patch in ({"role": "candidate"}, {"type": "parity_only"}):
            with self.assertRaises(freeze.FreezeError):
                freeze.calibrate([{"role": "baseline", "type": "natural_annotated", **patch}], {"state": "draft", "policy": metrics.POLICY})

    def test_complete_analytical_calibration_keeps_not_applicable_and_zero_deltas(self):
        root = Path(__file__).resolve().parents[2]
        strata = sorted(case["id"] for case in corpus.expected_coverage(root)["required_cases"] if case["requires_natural_quality"])
        draft = {"state": "draft", "policy": metrics.POLICY, "calibration_policy": freeze.CALIBRATION_POLICY, "required_strata": strata, "token": "sentinel-private-credential"}
        rows = []
        for split in ("calibration", "evaluation"):
            for case in strata + ["silence"]:
                for recording in range(3):
                    identity = split + "_" + case + "_" + str(recording)
                    language = case[len("language_"):] if case.startswith("language_") else "it"
                    silence = case == "silence"
                    voices = case in ("voices_auto", "voices_exact", "it_overlap", "en_overlap")
                    values = {name: 0.0 for name in freeze.REQUIRED_METRICS}
                    values.update(timestamp_coverage=1.0, word_speaker_accuracy=1.0)
                    for replica in range(3):
                        rows.append({"role": "baseline", "type": "natural_annotated", "independent_annotation_verified": True, "measured": True, "split": split, "case_id": case, "recording_id": identity, "source_group": identity, "speaker_ids": [identity], "language": language, "reference_word_count": 0 if silence else 1, "reference_char_count": 0 if silence else 1, "reference_speaker_word_count": 1 if voices else 0, "voices_requested": voices, "source_digest": "a" * 64, "runtime_digest": "b" * 64, "model_digest": "c" * 64, "hardware_digest": "d" * 64, "run_id": str(uuid.uuid4()), "metrics": values, "resources": {"completion_wall_ns": 10, "rss_peak_process_bytes": 100}})
        result = freeze.calibrate(rows, draft)
        freeze.validate_contract(result)
        self.assertNotIn("sentinel-private-credential", json.dumps(result))
        with self.assertRaises(freeze.FreezeError):
            freeze.validate_contract(dict(result, token="sentinel-private-credential"))
        self.assertEqual(set(result["thresholds_by_stratum"]), set(strata))
        self.assertEqual(result["thresholds_by_stratum"]["language_ja"]["wer"]["status"], "not_applicable")
        self.assertTrue(all(delta == 0 for delta in result["degradation_delta"].values()))
        # Analytical observations test the mechanism only; missing licensed files and traces still prevent any freeze.
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(freeze.FreezeError):
            freeze.freeze_contract(result, {"cases": [], "operators": [], "prerequisites": []}, corpus.expected_coverage(root), Path(directory), "unearned-freeze.json")

    def test_trace_verified_flag_without_artifact_does_not_pass_gate(self):
        root = Path(__file__).resolve().parents[2]
        entries = [{"operator": row["id"], "backend": "gpu", "observed": True, "reviewed": True, "trace_verified": True, "evidence_kind": "trace"} for row in profiling.expected_operators()]
        report = freeze.evidence_gate({"cases": [], "operators": entries, "prerequisites": []}, corpus.expected_coverage(root))
        self.assertEqual(len(report["placement"]["missing"]), len(profiling.expected_operators()))


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, freeze, manifest, released_baseline as baseline


class ReleasedBaselineTests(unittest.TestCase):
    def catalog(self):
        assets = [{"id": f"fleurs-{language}-{index}", "language": language, "sha256": f"{language}-{index}", "path": f"audio/{language}-{index}.wav"} for language in corpus.REPRESENTATIVES for index in range(6)]
        return {"reference_policy": "released-dataset", "assets": assets, "cases": [{"id": asset["id"], "asset_id": asset["id"], "type": "released_dataset", "reference_path": "refs/" + asset["id"] + ".json"} for asset in assets]}

    def fixture(self, root):
        catalog = self.catalog()
        for asset in catalog["assets"]:
            path = root / asset["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(asset["id"].encode())
            asset["sha256"] = manifest.sha256(path)
        for case in catalog["cases"]:
            path = root / case["reference_path"]
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}")
        (root / "catalog.json").write_text(json.dumps(catalog))
        (root / "licenses.json").write_text('{}')
        with patch.object(baseline, "identity", return_value=({"fixture": True}, {})):
            document = baseline.prepare(root, "snapshot", root / "catalog.json", root / "licenses.json", ["Rapida"], 3, root, root, root, [])
        return document

    def measured(self, root, document):
        for job in document["jobs"]:
            pair = "snapshot/jobs/" + job["id"] + "/attempts/attempt-fixture/pair"
            paths = [pair + "/" + suffix for suffix in ("original/result.json", "instrumented/result.json", "spans.json")]
            score = {"audio_sha256": job["audio_sha256"], "reference_policy": "released-dataset", "text": {"normalized": {"cer": {"rate": 0.1 if job["split"] == "calibration" else 0.8}, "wer": {"rate": 0.2}}}, "dataset_timestamp_agreement": {"status": "not_available"}}
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(score))
            row = {"status": "measured", "role": "baseline", "job": job, "identity": document["identity"], "paired_execution": {"status": "paired_equivalent", "warnings": []}, "pair_wall_ns": 100, "pair_path": pair, "score": score, "artifacts": {name: manifest.sha256(root / name) for name in paths}}
            freeze.seal(root, "snapshot/jobs/" + job["id"] + "/measurement.json", row)

    def test_plan_has_recording_splits_and_replicas_without_invented_speaker_identity(self):
        plan = baseline.plan(self.catalog(), ["Rapida"])
        self.assertEqual(len(plan["jobs"]), 162)
        for language in corpus.REPRESENTATIVES:
            rows = [job for job in plan["jobs"] if job["settings"]["language"] == language]
            self.assertEqual({job["replica"] for job in rows}, {0, 1, 2})
            left = {job["asset_id"] for job in rows if job["split"] == "calibration"}
            right = {job["asset_id"] for job in rows if job["split"] == "evaluation"}
            self.assertEqual((len(left), len(right)), (3, 3))
            self.assertFalse(left & right)
        self.assertFalse(plan["local_human_review_required"])
        with self.assertRaises(ValueError): baseline.plan(self.catalog(), ["Rapida"], 2)

    def test_missing_language_duplicate_and_audio_leakage_rejected(self):
        catalog = self.catalog()
        catalog["assets"][3]["sha256"] = catalog["assets"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "audio_split_leakage"): baseline.plan(catalog, ["Rapida"])
        catalog = self.catalog()
        catalog["cases"].append(copy.deepcopy(catalog["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_identity"): baseline.plan(catalog, ["Rapida"])
        catalog = self.catalog()
        catalog["cases"] = [case for case in catalog["cases"] if not case["id"].startswith("fleurs-no-")]
        with self.assertRaisesRegex(ValueError, "languages_incomplete"): baseline.plan(catalog, ["Rapida"])

    def test_prepare_failure_is_atomic_and_plan_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self.fixture(root)
            (root / "snapshot/plan.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "freeze_mutated"): baseline.load(root, "snapshot")
            with patch.object(baseline, "identity", side_effect=ValueError("failed")):
                with self.assertRaises(ValueError): baseline.prepare(root, "other", root / "catalog.json", root / "licenses.json", ["Rapida"], 3, root, root, root, [])
            self.assertFalse((root / "other").exists())

    def test_calibration_excludes_heldout_and_seal_rejects_changed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self.fixture(root)
            self.measured(root, document)
            def evaluate(root, catalog, licenses, case_id, result): return json.loads((root / result).read_text())
            with patch.object(baseline.dataset_reference, "evaluate", side_effect=evaluate):
                report = baseline.calibrate(root, "snapshot")
                for stratum in report["strata"].values():
                    self.assertAlmostEqual(stratum["cer"]["bound"], 0.1)
                    self.assertAlmostEqual(stratum["cer"]["evaluation_mean"], 0.8)
                    self.assertFalse(stratum["cer"]["heldout_within_calibration_bound"])
                    self.assertEqual(stratum["cer"]["degradation_delta"], 0)
                checksum = baseline.seal(root, "snapshot")
                self.assertEqual(baseline.verify(root, "snapshot", checksum)["jobs"], 162)
                pair = "snapshot/jobs/" + document["jobs"][0]["id"] + "/attempts/attempt-fixture/pair"
                (root / pair / "original/result.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "artifact_changed"): baseline.verify(root, "snapshot", checksum)

    def test_missing_replica_candidate_and_fabricated_score_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self.fixture(root)
            self.measured(root, document)
            job = document["jobs"][0]
            path = root / "snapshot/jobs" / job["id"] / "measurement.json"
            row = json.loads(path.read_text())
            row["role"] = "candidate"
            with self.assertRaisesRegex(ValueError, "job_identity_invalid"): baseline.verify_job(root, row, job)
            row["role"] = "baseline"
            row["score"]["text"]["normalized"]["cer"]["rate"] = 0
            with patch.object(baseline.dataset_reference, "evaluate", side_effect=lambda root, catalog, licenses, case_id, result: json.loads((root / result).read_text())):
                with self.assertRaisesRegex(ValueError, "score_changed"): baseline.verify_job(root, row, job, {}, {})
            path.unlink()
            with self.assertRaises(FileNotFoundError): baseline.calibrate(root, "snapshot")


    def test_resume_preserves_incomplete_attempt_and_skips_completed_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            document = self.fixture(root)
            document["jobs"] = document["jobs"][:1]
            catalog = json.loads((root / "catalog.json").read_text())
            job = document["jobs"][0]
            score = {"audio_sha256": job["audio_sha256"], "reference_policy": "released-dataset"}
            def interrupted(runtime, worker, source, audio, settings, pair, **kwargs):
                pair.mkdir(parents=True)
                (pair / "partial.txt").write_text("interrupted evidence")
                raise RuntimeError("interrupted")
            def completed(runtime, worker, source, audio, settings, pair, **kwargs):
                for suffix in ("original/result.json", "instrumented/result.json", "spans.json"):
                    path = pair / suffix
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}")
                return {"status": "paired_equivalent", "warnings": []}
            with patch.object(baseline, "load", return_value=(document, catalog, {})), patch.object(baseline, "identity", return_value=(document["identity"], {})), patch.object(baseline.run, "model_cache_roots", return_value=(root, root)), patch.object(baseline.dataset_reference, "evaluate", return_value=score):
                with patch.object(baseline.run, "paired_run", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"): baseline.execute(root, "snapshot", root, root, root, [])
                with patch.object(baseline.run, "paired_run", side_effect=completed) as worker:
                    baseline.execute(root, "snapshot", root, root, root, [])
                    self.assertEqual(worker.call_count, 1)
                    baseline.execute(root, "snapshot", root, root, root, [])
                    self.assertEqual(worker.call_count, 1)
                attempts = root / "snapshot/jobs" / job["id"] / "attempts"
                self.assertEqual(len(list(attempts.iterdir())), 2)
                self.assertEqual(len(list(attempts.glob("*/pair/partial.txt"))), 1)


if __name__ == "__main__":
    unittest.main()

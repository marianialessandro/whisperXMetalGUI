import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import freeze, manifest, profiling
from tools.certification import resources


class ResourceObservationTests(unittest.TestCase):
    def spans(self):
        samples = [{"sample_time_ns": 1, "rss_peak_process_bytes": 10, "mps_allocated_bytes": 20}, {"sample_time_ns": 2, "rss_peak_process_bytes": 15, "mps_allocated_bytes": 25}]
        row = {"pid": 41, "status": "synchronized", "drain_calls": 2, "api_wall_ns": 7, "post_return_wait_ns": 3, "completion_wall_ns": 10, "cpu_process_ns": 4, "sample_cadence_ns": 50000000, "memory_samples": samples, "memory": profiling.memory_summary(samples)}
        return {"pid": 41, "run_id": "fixture", "timing_kind": "synchronized_completion", "spans": [{**copy.deepcopy(row), "stage": stage} for stage in ("decode", "asr_mlx", "alignment", "export_write", "export_write")]}

    def rows(self):
        return [{"stratum": "Rapida/fixture/it", "asset_id": split + str(index), "split": split, "replica": replica, "values": {"alignment/completion_wall_ns": value}} for split, value in (("calibration", 10), ("evaluation", 1000)) for index in range(3) for replica in range(3)]

    def test_repeated_stage_timing_sums_but_memory_domains_do_not(self):
        result = resources.measurements(self.spans())
        values = result["values"]
        self.assertEqual(values["export_write/completion_wall_ns"], 20)
        self.assertEqual(values["export_write/rss_peak_process_sampled_max_bytes"], 15)
        self.assertEqual(values["export_write/mps_allocated_sampled_max_bytes"], 25)
        self.assertFalse(any("total" in key for key in values))

    def test_unsynchronized_foreign_clock_and_summary_changes_fail(self):
        for name, value in (("pid", 42), ("api_wall_ns", True), ("completion_wall_ns", 99), ("status", "returned"), ("drain_calls", 1), ("sample_cadence_ns", 0)):
            document = self.spans()
            document["spans"][0][name] = value
            with self.assertRaises(ValueError): resources.measurements(document)
        document = self.spans()
        document["spans"][0]["memory"]["rss_peak_process_sampled_max_bytes"] = 999
        with self.assertRaises(ValueError): resources.measurements(document)
        document = self.spans()
        document["spans"][0]["memory_samples"][0]["sample_time_ns"] = 3
        with self.assertRaises(ValueError): resources.measurements(document)

    def test_required_voice_phases_and_unavailable_memory_are_explicit(self):
        document = self.spans()
        with self.assertRaises(ValueError): resources.measurements(document, voices=True)
        for stage in ("diarization", "speaker_assignment"):
            row = copy.deepcopy(document["spans"][0])
            row["stage"] = stage
            row["memory_samples"].append({"sample_time_ns": 3, "sample_unavailable": True})
            row["memory"] = profiling.memory_summary(row["memory_samples"])
            document["spans"].append(row)
        self.assertEqual(resources.measurements(document, voices=True)["unavailable_memory_samples"], 2)

    def test_heldout_data_never_retunes_calibration_maximum(self):
        result = resources.summarize(self.rows())["Rapida/fixture/it"]["alignment/completion_wall_ns"]
        self.assertEqual(result["calibration_observed_maximum"], 10)
        self.assertEqual(result["evaluation_observed_maximum"], 1000)
        self.assertEqual(result["evaluation_recordings_exceeding_calibration_maximum"], 3)
        self.assertEqual(result["calibration_recordings"], 3)

    def test_missing_duplicate_replica_overlap_and_metric_coverage_fail(self):
        rows = self.rows()
        with self.assertRaises(ValueError): resources.summarize(rows[:-1])
        with self.assertRaises(ValueError): resources.summarize(rows + [rows[0]])
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["asset_id"] == "evaluation0": row["asset_id"] = "calibration0"
        with self.assertRaises(ValueError): resources.summarize(changed)
        changed = copy.deepcopy(rows)
        changed[0]["values"] = {}
        with self.assertRaises(ValueError): resources.summarize(changed)
        with self.assertRaises(ValueError): resources.summarize([])

    def test_build_uses_only_sealed_span_inputs_and_report_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bench/jobs").mkdir(parents=True)
            artifacts, jobs = {}, []
            for index, row in enumerate(self.rows()):
                job = {key: value for key, value in row.items() if key != "values"}
                job["id"] = str(index)
                jobs.append(job)
                name = "bench/run-" + str(index) + "/spans.json"
                (root / name).parent.mkdir()
                (root / name).write_text(json.dumps(self.spans()))
                artifacts[name] = manifest.sha256(root / name)
                path = "bench/jobs/" + str(index) + "/measurement.json"
                (root / path).parent.mkdir()
                (root / path).write_text(json.dumps({"artifacts": {name: artifacts[name]}}))
                artifacts[path] = manifest.sha256(root / path)
            archive = {"artifacts": artifacts}
            benchmark_checksum = freeze.seal(root, "bench/sealed.json", archive)
            with patch.object(resources.released_baseline, "verify", return_value=archive), patch.object(resources.released_baseline, "load", return_value=({"jobs": jobs}, {}, {})):
                report = resources.build(root, "bench", benchmark_checksum, "text")
                self.assertEqual(report["paired_jobs"], 18)
                self.assertFalse(report["resource_budgets_certified"])
                checksum = freeze.seal(root, "report.json", report)
                self.assertEqual(resources.verify(root, "report.json", checksum), report)
                (root / "bench/run-0/spans.json").write_text('{}')
                with self.assertRaises(ValueError): resources.verify(root, "report.json", checksum)


if __name__ == "__main__":
    unittest.main()

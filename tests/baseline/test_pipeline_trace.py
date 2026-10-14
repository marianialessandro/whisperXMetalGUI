from pathlib import Path
import sys
import unittest
import copy
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.tracing import pipeline_review


class PipelineTraceTests(unittest.TestCase):
    def fixture(self):
        toc = ET.fromstring('<trace-toc><run><info><target><process pid="401" type="attached"/></target></info></run></trace-toc>')
        columns = "start duration process channel-name state".split()
        schema = "<schema name='metal-gpu-intervals'>" + "".join("<col><mnemonic>" + column + "</mnemonic></col>" for column in columns) + "</schema>"
        foreign = '<row><start id="1">10</start><duration id="2">20</duration><process id="3"><pid id="4">602</pid></process><channel id="5">Compute</channel><state id="6">Active</state></row>'
        target = '<row><start ref="1"/><duration ref="2"/><process><pid>401</pid></process><channel ref="5"/><state ref="6"/></row>'
        gpu = ET.fromstring("<trace-query-result><node>" + schema + foreign + target + "</node></trace-query-result>")
        spans = {"run_id": "actual", "pid": 401, "timing_kind": "synchronized_completion", "spans": [{"stage": stage, "status": "synchronized"} for stage in ("decode", "asr_mlx", "alignment", "export_write", "diarization", "speaker_assignment")]}
        controller = {"status": "diagnostic_pipeline_completed", "pid": 401, "child_returncode": 0, "run_id": "actual", "voices": True}
        return toc, gpu, spans, controller

    def test_pipeline_process_activity_never_certifies_operators_or_other_apps(self):
        result = pipeline_review.review(*self.fixture())
        self.assertTrue(result["gpu_activity_observed"])
        self.assertEqual(result["target_compute_intervals"], [{"start_ns": 10, "duration_ns": 20}])
        self.assertEqual(result["foreign_process_rows_excluded"], 1)
        self.assertFalse(result["individual_operator_dispatch_verified"])
        self.assertFalse(result["phase_interval_attribution_verified"])
        self.assertTrue(all(item["status"] == "unverified" for item in result["operator_ledger"]))
        self.assertEqual(len(result["operator_ledger"]), 26)

    def test_wrong_target_phase_run_or_failed_child_is_rejected(self):
        for kind in ("pid", "phase", "run", "exit"):
            toc, gpu, spans, controller = self.fixture()
            if kind == "pid": controller["pid"] = 999
            if kind == "phase": spans["spans"] = spans["spans"][:-1]
            if kind == "run": controller["run_id"] = "other"
            if kind == "exit": controller["child_returncode"] = 2
            with self.assertRaises(ValueError): pipeline_review.review(toc, gpu, spans, controller)

    def test_foreign_only_activity_and_broken_references_fail_closed(self):
        toc, gpu, spans, controller = self.fixture()
        gpu.findall(".//row")[1][2].find("pid").text = "602"
        self.assertFalse(pipeline_review.review(toc, gpu, spans, controller)["gpu_activity_observed"])
        gpu.findall(".//row")[1][2].find("pid").text = "401"
        gpu.findall(".//row")[1][0].set("ref", "missing")
        with self.assertRaises(ValueError): pipeline_review.review(toc, gpu, spans, controller)


    def test_native_unattributed_sentinel_is_excluded_and_missing_process_pid_still_rejected(self):
        toc, gpu, spans, controller = self.fixture()
        row = copy.deepcopy(gpu.findall(".//row")[1])
        row[2] = ET.Element("sentinel")
        gpu.find("node").append(row)
        report = pipeline_review.review(toc, gpu, spans, controller)
        self.assertEqual(report["unattributed_process_rows_excluded"], 1)
        self.assertEqual(len(report["target_compute_intervals"]), 1)
        row[2] = ET.Element("process")
        with self.assertRaisesRegex(ValueError, "reference_invalid"): pipeline_review.review(toc, gpu, spans, controller)


if __name__ == "__main__":
    unittest.main()

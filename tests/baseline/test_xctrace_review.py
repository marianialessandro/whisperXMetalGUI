import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import xctrace_review


class HeadlessTraceReviewTests(unittest.TestCase):
    def fixtures(self, target_pid=401):
        toc = ET.fromstring('<trace-toc><run><info><target><process pid="401" return-exit-status="0"/></target></info></run></trace-toc>')
        columns = "start duration process channel-name state".split()
        schema = "<schema name='metal-gpu-intervals'>" + "".join("<col><mnemonic>" + column + "</mnemonic></col>" for column in columns) + "</schema>"
        foreign = '<row><start id="1">10</start><duration id="2">20</duration><process id="3"><pid id="4">602</pid></process><channel id="5">Compute</channel><state id="6">Active</state></row>'
        target = '<row><start ref="1"/><duration ref="2"/><process><pid>' + str(target_pid) + '</pid></process><channel ref="5"/><state ref="6"/></row>'
        gpu = ET.fromstring("<trace-query-result><node>" + schema + foreign + target + "</node></trace-query-result>")
        return toc, gpu, {"pid": 401, "run_id": "fixture-run", "status": "primitive_parity_passed"}

    def test_only_exact_pid_active_compute_can_be_reported(self):
        toc, gpu, report = self.fixtures()
        result = xctrace_review.review(toc, gpu, report)
        self.assertTrue(result["gpu_work_observed"])
        self.assertEqual(result["target_active_compute_intervals"], [{"start_ns": 10, "duration_ns": 20}])
        self.assertEqual(result["other_process_rows_excluded"], 1)
        self.assertFalse(result["individual_operator_dispatch_verified"])
        self.assertFalse(result["full_baseline_gate_passed"])

    def test_other_application_activity_does_not_certify_target(self):
        toc, gpu, report = self.fixtures(target_pid=602)
        result = xctrace_review.review(toc, gpu, report)
        self.assertFalse(result["gpu_work_observed"])
        self.assertEqual(result["status"], "target_gpu_work_missing")

    def test_mismatched_pid_and_missing_reference_fail_closed(self):
        toc, gpu, report = self.fixtures()
        with self.assertRaises(ValueError): xctrace_review.review(toc, gpu, dict(report, pid=602))
        gpu.findall(".//row")[1][0].set("ref", "missing")
        with self.assertRaises(ValueError): xctrace_review.review(toc, gpu, report)

    def test_reference_cycles_are_rejected_without_hanging(self):
        toc, gpu, report = self.fixtures()
        first = gpu.findall(".//row")[0][0]
        first.set("ref", "1")
        with self.assertRaises(ValueError): xctrace_review.review(toc, gpu, report)


if __name__ == "__main__":
    unittest.main()

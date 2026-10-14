from pathlib import Path
import sys
import unittest
import json
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import freeze, manifest
from tools.tracing import isolated_operator


class IsolatedOperatorTests(unittest.TestCase):
    def fixture(self):
        toc = ET.fromstring('<trace-toc><run><info><target><process pid="401"/></target></info></run></trace-toc>')
        columns = "start duration process channel-name state".split()
        schema = '<schema name="metal-gpu-intervals">' + ''.join('<col><mnemonic>' + name + '</mnemonic></col>' for name in columns) + '</schema>'
        gpu = ET.fromstring('<trace-query-result><node>' + schema + '<row><start>1</start><duration>2</duration><process><pid>401</pid></process><channel>Compute</channel><state>Active</state></row></node></trace-query-result>')
        operator = {"status": "isolated_operator_completed", "operator": "alignment_log_softmax", "pid": 401, "run_id": "isolated", "output_identical": True, "original_output_sha256": "a" * 64, "observed_output_sha256": "a" * 64, "output_device": "mps:0", "setup_and_original_reference_before_trace": True, "result_readback_after_trace_saved": True, "mps_fallback_allowed": False, "dispatch_audit": {"aten_operations": {"aten.log_softmax.int": 1}, "tensor_device_operations": {"mps:0": 1}, "numerical_values_retained": False, "scalar_readbacks": 0}}
        controller = {"operator": "alignment_log_softmax", "pid": 401, "run_id": "isolated", "child_returncode": 0, "recorder_returncode": 0}
        return toc, gpu, operator, controller

    def test_named_operation_gpu_work_is_distinct_from_complete_numerical_gpu_certification(self):
        report = isolated_operator.review(*self.fixture())
        self.assertTrue(report["individual_operator_dispatch_verified"])
        self.assertFalse(report["all_numerical_gpu_verified"])
        self.assertFalse(report["placement_gate_eligible"])
        self.assertFalse(report["gpu_placement_certified"])
        self.assertEqual(report["observed_cpu_tensor_operations"], 0)
        toc, gpu, operator, controller = self.fixture()
        operator["output_device"] = "mps"
        operator["dispatch_audit"]["tensor_device_operations"] = {"mps": 1, "cpu": 2}
        self.assertEqual(isolated_operator.review(toc, gpu, operator, controller)["observed_cpu_tensor_operations"], 2)

    def test_output_mutation_bad_identity_or_readback_inside_capture_is_rejected(self):
        for field, value in (("observed_output_sha256", "b" * 64), ("output_device", "cpu"), ("result_readback_after_trace_saved", False), ("mps_fallback_allowed", True), ("pid", True)):
            toc, gpu, operator, controller = self.fixture()
            operator[field] = value
            with self.assertRaises(ValueError): isolated_operator.review(toc, gpu, operator, controller)

    def test_unrelated_process_activity_never_proves_operator_dispatch(self):
        toc, gpu, operator, controller = self.fixture()
        gpu.find('.//row/process/pid').text = "999"
        result = isolated_operator.review(toc, gpu, operator, controller)
        self.assertFalse(result["gpu_activity_observed"])
        self.assertFalse(result["individual_operator_dispatch_verified"])


    def test_cpu_ctc_dispatch_is_not_relabelled_as_gpu(self):
        toc, gpu, operator, controller = self.fixture()
        operator.update(operator="ctc_trellis", output_device="cpu")
        controller["operator"] = "ctc_trellis"
        operator["dispatch_audit"].update(aten_operations={"aten.maximum.default": 10}, tensor_device_operations={"cpu": 10})
        gpu.find('.//row/process/pid').text = "999"
        report = isolated_operator.review(toc, gpu, operator, controller)
        self.assertEqual(report["status"], "isolated_operator_cpu_dispatch_observed")
        self.assertEqual(report["observed_tensor_backend"], "cpu")
        self.assertTrue(report["individual_operator_dispatch_verified"])
        self.assertFalse(report["gpu_activity_observed"])
        self.assertFalse(report["gpu_placement_certified"])


    def test_archive_verification_and_output_mutation_detection_preserve_legacy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            folder = root / "snapshot"
            folder.mkdir()
            toc, gpu, operator, controller = self.fixture()
            ET.ElementTree(toc).write(folder / "toc.xml")
            ET.ElementTree(gpu).write(folder / "gpu.xml")
            freeze.seal(folder, "operator.json", operator)
            freeze.seal(folder, "controller.json", controller)
            source = "historical producer before tensor backend field"
            import hashlib
            digest = hashlib.sha256(source.encode()).hexdigest()
            name = "tools/tracing/isolated_operator.py"
            freeze.seal(folder, "context.json", {"isolated_harness": {"files": [{"name": name, "sha256": digest}]}})
            freeze.seal(folder, "executed-harness.json", {name: {"sha256": digest, "content": source}})
            report = isolated_operator.review(toc, gpu, operator, controller)
            report.pop("observed_tensor_backend")
            report["artifacts"] = {str(path.relative_to(root)): manifest.sha256(path) for path in folder.iterdir()}
            checksum = freeze.seal(folder, "review.json", report)
            self.assertTrue(isolated_operator.verify(root, "snapshot", checksum)["individual_operator_dispatch_verified"])
            (folder / "operator.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact_changed"): isolated_operator.verify(root, "snapshot", checksum)


if __name__ == "__main__":
    unittest.main()

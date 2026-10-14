from contextvars import ContextVar
import copy
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from tools.observability.python_readbacks import PythonReadbacks, summarize


class PythonReadbackTests(unittest.TestCase):
    def test_original_cpu_identity_numpy_view_and_inference_tensors_preserved(self):
        original_cpu, original_numpy = torch.Tensor.cpu, torch.Tensor.numpy
        probe = PythonReadbacks(ContextVar("stage", default="alignment"))
        with probe.observe():
            with torch.inference_mode(): source = torch.tensor([2.0, 4.0])
            self.assertIs(source.cpu(), source)
            result = source.numpy()
            self.assertEqual(result.tolist(), [2.0, 4.0])
            result[0] = 9.0
            self.assertEqual(source.tolist(), [9.0, 4.0])
        self.assertIs(torch.Tensor.cpu, original_cpu)
        self.assertIs(torch.Tensor.numpy, original_numpy)
        report = probe.report()
        self.assertEqual(sum(row["source_tensor_logical_bytes"] for row in report["records"]), 16)
        summary = summarize(report, {"spans": [{"stage": "alignment"}]})
        self.assertEqual(summary["cpu_method_same_object_calls"], 1)
        self.assertEqual(summary["numpy_method_calls"], 1)
        self.assertEqual(summary["cpu_method_device_to_host_calls"], 0)
        self.assertFalse(summary["aten_counters_additive"])

    def test_failed_numpy_call_restores_methods_and_keeps_no_error_or_values(self):
        original = torch.Tensor.numpy
        probe = PythonReadbacks(ContextVar("stage", default="alignment"))
        with self.assertRaises(RuntimeError):
            with probe.observe(): torch.tensor([1.0], requires_grad=True).numpy()
        self.assertIs(torch.Tensor.numpy, original)
        self.assertEqual(probe.report()["failed_calls"], 1)
        self.assertEqual(probe.report()["records"], [])
        self.assertNotIn("requires_grad", json.dumps(probe.report()))

    def test_other_threads_are_not_counted_under_the_main_phase(self):
        probe = PythonReadbacks(ContextVar("stage", default="alignment"))
        with probe.observe():
            thread = threading.Thread(target=lambda: torch.ones(1).cpu())
            thread.start()
            thread.join()
        self.assertEqual(probe.report()["records"], [])

    def test_foreign_phase_duplicate_and_negative_counters_rejected(self):
        probe = PythonReadbacks(ContextVar("stage", default="alignment"))
        with probe.observe(): torch.ones(1).cpu()
        report = probe.report()
        spans = {"spans": [{"stage": "alignment"}]}
        for key, value in (("stage", "foreign"), ("calls", True), ("source_tensor_logical_bytes", -1), ("identity_returns", 2)):
            invalid = copy.deepcopy(report)
            invalid["records"][0][key] = value
            with self.assertRaises(ValueError): summarize(invalid, spans)
        report["records"] *= 2
        with self.assertRaises(ValueError): summarize(report, spans)


if __name__ == "__main__":
    unittest.main()

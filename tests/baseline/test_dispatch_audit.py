from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from tools.tracing.dispatch_audit import DispatchAudit, devices


class DispatchAuditTests(unittest.TestCase):
    def test_passthrough_keeps_outputs_and_retains_only_device_counts(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.log_softmax(data, dim=-1)
        audit = DispatchAudit()
        with audit: observed = torch.log_softmax(data, dim=-1)
        self.assertTrue(torch.equal(expected, observed))
        report = audit.report()
        self.assertGreater(report["tensor_device_operations"]["cpu"], 0)
        self.assertFalse(report["numerical_values_retained"])
        self.assertFalse(report["internal_backend_cpu_work_certified_absent"])
        self.assertEqual(devices({"tensor": data, "private": "fixture-secret"}), {"cpu"})
        self.assertNotIn("fixture-secret", str(report))

    def test_scalar_readback_is_visible_without_storing_its_value(self):
        audit = DispatchAudit()
        with audit: value = torch.tensor([2.0]).sum().item()
        self.assertEqual(value, 2)
        self.assertGreater(audit.report()["scalar_readbacks"], 0)


if __name__ == "__main__":
    unittest.main()

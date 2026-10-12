import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import profiling


class ProfilingTests(unittest.TestCase):
    def test_ordering_evaluates_lazy_output_before_completion_drain(self):
        order = []
        clock = iter([10, 20, 50])
        cpu = iter([5, 15])
        def call():
            order.append("call")
            return "lazy"
        result, report = profiling.measure(call, lambda: order.append("drain"), lambda result: order.append("eval:" + result), clock=lambda: next(clock), cpu_clock=lambda: next(cpu))
        self.assertEqual(order, ["drain", "call", "eval:lazy", "drain"])
        self.assertEqual(result, "lazy")
        self.assertEqual(report["api_wall_ns"], 10)
        self.assertEqual(report["post_return_wait_ns"], 30)
        self.assertEqual(report["completion_wall_ns"], 40)
        self.assertEqual(report["cpu_process_ns"], 10)
        self.assertFalse(report["gpu_dispatch_observed"])

    def test_drain_failure_cannot_report_success_or_leak_exception(self):
        calls = []
        def broken():
            raise RuntimeError("sentinel-private-credential")
        with self.assertRaises(profiling.MeasurementError) as error:
            profiling.measure(lambda: calls.append("call"), broken)
        self.assertEqual(calls, [])
        self.assertEqual(str(error.exception), "synchronization_failed")

    def test_post_return_drain_failure_is_not_success_or_double_execution(self):
        calls, drains = [], []
        def drain():
            drains.append("drain")
            if len(drains) == 2: raise RuntimeError("sentinel-private-credential")
        with self.assertRaises(profiling.MeasurementError):
            profiling.measure(lambda: calls.append("once"), drain)
        self.assertEqual(calls, ["once"])

    def test_required_operator_omission_and_cpu_residue_block_target_gate(self):
        expected = profiling.expected_operators()
        entries = [{"operator": row["id"], "backend": "cpu", "observed": True, "reviewed": True, "evidence_kind": "trace", "trace_verified": True} for row in expected]
        self.assertEqual(profiling.placement_gate(entries, expected, target="baseline")["status"], "observed")
        self.assertEqual(profiling.placement_gate(entries[:-1], expected, target="baseline")["status"], "incomplete")
        self.assertEqual(profiling.placement_gate(entries, expected, target="native_gpu")["status"], "incomplete")

    def test_memory_domains_are_reported_separately_and_peak_is_lower_bound(self):
        report = profiling.memory_summary([{"rss_peak_process_bytes": 100, "mlx_active_bytes": 50, "mps_allocated_bytes": 60}, {"rss_peak_process_bytes": 110, "mlx_active_bytes": 40, "mps_allocated_bytes": 90}])
        self.assertEqual(report["mps_allocated_sampled_max_bytes"], 90)
        self.assertTrue(report["mps_peak_is_lower_bound"])
        self.assertNotIn("total_bytes", report)

    def test_trace_identity_requires_same_run_pid_phase_and_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.bin").write_bytes(b"synthetic trace fixture")
            digest = profiling.artifact_identity(root, "trace.bin")
            descriptor = {"path": "trace.bin", "sha256": digest, "run_id": "run-1", "pid": 12, "phases": ["asr"], "kind": "diagnostic"}
            profiling.verify_trace(descriptor, root, "run-1", 12, "asr")
            for run_id, pid, phase in [("other", 12, "asr"), ("run-1", 13, "asr"), ("run-1", 12, "align")]:
                with self.assertRaises(profiling.MeasurementError):
                    profiling.verify_trace(descriptor, root, run_id, pid, phase)
            (root / "trace.bin").write_bytes(b"changed")
            with self.assertRaises(profiling.MeasurementError):
                profiling.verify_trace(descriptor, root, "run-1", 12, "asr")

    def test_trace_internal_deduplication_link_is_allowed_external_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace"
            trace.mkdir()
            (trace / "buffer").write_bytes(b"synthetic")
            (trace / "alias").symlink_to("buffer")
            self.assertEqual(len(profiling.artifact_identity(root, "trace")), 64)
            (root / "private").write_bytes(b"preserve")
            (trace / "escape").symlink_to(root / "private")
            with self.assertRaises(profiling.MeasurementError):
                profiling.artifact_identity(root, "trace")
            self.assertEqual((root / "private").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()

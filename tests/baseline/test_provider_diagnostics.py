import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.baseline import profiling
from tools.tracing import provider_diagnostics as diagnostics


class ProviderDiagnosticsTests(unittest.TestCase):
    def fixture(self, root):
        trace = root / "snapshot/pipeline.trace"
        trace.mkdir(parents=True)
        (trace / "data").write_bytes(b"original trace")
        return profiling.artifact_identity(root, "snapshot/pipeline.trace")

    def export(self, trace, schema, output, log):
        log.write_text("export completed")
        output.write_text('<trace-query-result><node><schema name="' + schema + '"/><row><value>private-value</value></row></node></trace-query-result>')
        return 0

    def test_positive_table_counts_preserve_archive_and_never_certify_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checksum = self.fixture(root)
            report = diagnostics.diagnose(root, "snapshot", checksum, "diagnostics", self.export)
            self.assertEqual(report["status"], "provider_tables_present")
            self.assertTrue(report["original_trace_unchanged"])
            self.assertEqual([row["rows_all_processes"] for row in report["tables"]], [1] * 4)
            self.assertEqual(profiling.artifact_identity(root, "snapshot/pipeline.trace"), checksum)
            for name in ("process_attribution_verified", "gpu_dispatch_certified", "full_pipeline_gpu_certified", "quality_certified", "promotion_allowed", "new_capture_performed"):
                self.assertFalse(report[name])
            self.assertNotIn("private-value", json.dumps(report))
            self.assertEqual(json.loads((root / "diagnostics/report.json").read_text()), report)

    def test_completed_buffers_without_gpu_intervals_remain_incomplete(self):
        def export(trace, schema, output, log):
            result = self.export(trace, schema, output, log)
            if schema == "metal-gpu-intervals":
                output.write_text('<trace-query-result><node><schema name="metal-gpu-intervals"/></node></trace-query-result>')
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checksum = self.fixture(root)
            report = diagnostics.diagnose(root, "snapshot", checksum, "diagnostics", export)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["tables"][-1]["rows_all_processes"], 0)
            self.assertEqual(report["tables"][1]["rows_all_processes"], 1)

    def test_failed_missing_malformed_and_timed_out_exports_receive_no_credit(self):
        for cause in ("returncode", "missing", "malformed", "wrong-schema", "timeout"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checksum = self.fixture(root)
                def export(trace, schema, output, log):
                    if schema != "metal-gpu-info":
                        return self.export(trace, schema, output, log)
                    if cause == "returncode":
                        return 1
                    if cause == "timeout":
                        raise subprocess.TimeoutExpired("private-command", 45)
                    if cause == "malformed":
                        output.write_text("<private-malformed")
                    elif cause == "wrong-schema":
                        output.write_text('<root><schema name="other"/><row/></root>')
                    return 0
                report = diagnostics.diagnose(root, "snapshot", checksum, "diagnostics", export)
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["tables"][0]["status"], "export_failed")
                self.assertIsNone(report["tables"][0]["rows_all_processes"])
                self.assertNotIn("private", json.dumps(report))
                self.assertTrue(report["original_trace_unchanged"])

    def test_invalid_pin_overlap_and_existing_destination_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checksum = self.fixture(root)
            for snapshot, pin, destination in (("../escape", checksum, "new"), ("snapshot", "bad", "new"), ("snapshot", "0" * 64, "new"), ("snapshot", checksum, "snapshot/inside"), ("snapshot", checksum, "."), ("snapshot", checksum, "../escape")):
                with self.assertRaises(ValueError):
                    diagnostics.diagnose(root, snapshot, pin, destination, self.export)
            self.assertFalse((root / "new").exists())
            self.assertFalse((root / "snapshot/inside").exists())
            (root / "existing").mkdir()
            (root / "existing/sentinel").write_text("preserve")
            with self.assertRaises(FileExistsError):
                diagnostics.diagnose(root, "snapshot", checksum, "existing", self.export)
            self.assertEqual((root / "existing/sentinel").read_text(), "preserve")

    def test_external_trace_change_invalidates_observations(self):
        def export(trace, schema, output, log):
            result = self.export(trace, schema, output, log)
            (trace / "data").write_bytes(b"external concurrent change")
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checksum = self.fixture(root)
            report = diagnostics.diagnose(root, "snapshot", checksum, "diagnostics", export)
            self.assertFalse(report["original_trace_unchanged"])
            self.assertEqual(report["status"], "incomplete")

    def test_export_size_budget_and_duplicate_schemas_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.xml"
            with path.open("wb") as stream:
                stream.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "budget_exceeded"):
                diagnostics.count_rows(path, "schema")
            path.write_text('<root><schema name="schema"/><schema name="schema"/><row/></root>')
            with self.assertRaisesRegex(ValueError, "schema_missing_or_invalid"):
                diagnostics.count_rows(path, "schema")

    def test_cli_invalid_evidence_does_not_emit_private_paths(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            code = diagnostics.main(["--evidence-root", directory, "--snapshot", "private-snapshot", "--trace-sha256", "0" * 64, "--output-directory", "diagnostics"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue()), {"status": "provider_diagnostics_invalid", "promotion_allowed": False})


if __name__ == "__main__":
    unittest.main()

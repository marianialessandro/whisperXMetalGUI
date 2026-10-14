import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from tools.baseline import freeze, released_baseline, run
from tools.observability import pair, worker
from tools.observability.transfers import TransferAudit, phases


class TransferAuditTests(unittest.TestCase):
    def fixture(self):
        audit = {"status": "transfer_audit_completed", "pid": 41, **TransferAudit().report()}
        return audit, {"pid": 41, "spans": [{"stage": "alignment"}]}

    def test_original_copy_and_scalar_results_are_preserved_without_numerical_retention(self):
        source = torch.tensor([2.0, 7.0], dtype=torch.float32)
        audit = TransferAudit()
        with audit:
            result = source.to(torch.float64)
            value = result[0].item()
        self.assertTrue(torch.equal(result, source.to(torch.float64)))
        self.assertEqual(value, 2.0)
        report = audit.report()
        copied = next(row for row in report["records"] if row["operation"] == "aten._to_copy.default")
        self.assertEqual(copied["source_tensor_logical_bytes"], 8)
        self.assertEqual(copied["destination_tensor_logical_bytes"], 16)
        self.assertEqual(copied["calls"], 1)
        self.assertEqual(next(row for row in report["records"] if row["destination_device"] == "host_scalar")["source_device"], "cpu")
        self.assertFalse(report["numerical_values_retained"])
        self.assertNotIn("values", report)
        self.assertIsNone(report["physical_copy_bytes"])

    def test_broadcast_copy_has_separate_source_and_destination_logical_sizes(self):
        source = torch.tensor([1.0, 2.0])
        destination = torch.empty((3, 2), dtype=torch.float64)
        audit = TransferAudit()
        with audit: returned = destination.copy_(source)
        self.assertIs(returned, destination)
        row = audit.report()["records"][0]
        self.assertEqual(row["source_tensor_logical_bytes"], 8)
        self.assertEqual(row["destination_tensor_logical_bytes"], 48)

    def test_nested_phase_context_and_original_functions_restore_on_exception(self):
        owner = SimpleNamespace(inner=lambda: torch.ones(1).to(torch.float64))
        owner.outer = lambda: owner.inner()
        inner, outer = owner.inner, owner.outer
        adapter = SimpleNamespace(boundaries=lambda request: [(owner, "inner", "inner"), (owner, "outer", "outer")])
        boundaries = adapter.boundaries
        audit = TransferAudit()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with phases(adapter, audit), audit:
                adapter.boundaries({})
                owner.outer()
                self.assertEqual(audit.stage.get(), "outside_observed_phases")
                raise RuntimeError("fixture")
        self.assertIs(owner.inner, inner)
        self.assertIs(owner.outer, outer)
        self.assertIs(adapter.boundaries, boundaries)
        self.assertEqual({row["stage"] for row in audit.report()["records"]}, {"inner"})

    def test_failed_dispatch_retains_no_error_message_or_false_copy(self):
        audit = TransferAudit()
        def fail(*args, **kwargs): raise RuntimeError("private-sentinel")
        with self.assertRaises(RuntimeError): audit.__torch_dispatch__(fail, (), ())
        self.assertEqual(audit.report()["failed_dispatches"], 1)
        self.assertEqual(audit.report()["records"], [])
        self.assertNotIn("private-sentinel", json.dumps(audit.report()))

    def test_fixture_host_device_and_scalar_counts_remain_distinct(self):
        audit, spans = self.fixture()
        base = {"stage": "alignment", "source_dtype": "torch.float32", "destination_dtype": "torch.float32", "operation": "aten._to_copy.default", "source_tensor_logical_bytes": 24, "destination_tensor_logical_bytes": 24, "calls": 2}
        audit["records"] = [{**base, "source_device": "mps", "destination_device": "cpu"}, {**base, "source_device": "cpu", "destination_device": "mps"}, {**base, "source_device": "mps", "destination_device": "host_scalar", "operation": "aten._local_scalar_dense.default", "source_tensor_logical_bytes": 4, "destination_tensor_logical_bytes": 0, "calls": 1}]
        report = pair.summarize(audit, spans)
        self.assertEqual(report["device_to_host_source_logical_bytes"], 24)
        self.assertEqual(report["host_to_device_source_logical_bytes"], 24)
        self.assertEqual(report["device_scalar_source_logical_bytes"], 4)
        self.assertEqual(report["device_scalar_readback_calls"], 1)
        self.assertIsNone(report["physical_copy_bytes"])
        self.assertFalse(report["full_pipeline_gpu_certified"])

    def test_foreign_pid_bad_counters_and_duplicate_boundaries_rejected(self):
        audit, spans = self.fixture()
        invalid = copy.deepcopy(audit)
        invalid["pid"] = 42
        with self.assertRaises(ValueError): pair.summarize(invalid, spans)
        row = {"stage": "alignment", "source_device": "cpu", "destination_device": "mps", "operation": "aten._to_copy.default", "source_dtype": "torch.float32", "destination_dtype": "torch.float32", "calls": 1, "source_tensor_logical_bytes": 4, "destination_tensor_logical_bytes": 4}
        for key, value in (("stage", "unrecorded"), ("calls", True), ("source_tensor_logical_bytes", -1), ("destination_device", "host_scalar")):
            invalid = copy.deepcopy(audit)
            invalid["records"] = [{**row, key: value}]
            with self.assertRaises(ValueError): pair.summarize(invalid, spans)
        audit["records"] = [row, row]
        with self.assertRaises(ValueError): pair.summarize(audit, spans)

    def test_worker_failure_and_existing_destination_never_create_or_replace_report(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "transfers.json"
            with patch.object(worker.adapter, "main", return_value=2):
                self.assertEqual(worker.main(["--transfer-report", str(target)]), 2)
            self.assertFalse(target.exists())
            target.write_text("preserved")
            with patch.object(worker.adapter, "main") as main:
                with self.assertRaises(ValueError): worker.main(["--transfer-report", str(target)])
                main.assert_not_called()
            self.assertEqual(target.read_text(), "preserved")

    def test_archive_verification_reconstructs_exports_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "fixture"
            folder.mkdir()
            semantics = {"basename": "fixture"}
            for name in ("original", "observed"):
                output = folder / name
                output.mkdir()
                for filename in ("original.json", "result.json"):
                    (output / filename).write_text(json.dumps({"segments": [], "language": None}))
                for extension in ("txt", "srt", "vtt"): (output / ("fixture." + extension)).write_text("")
                events = [{"event": "completed", "warnings": [], "output": str(output), "result": str(output / "result.json")}]
                semantics[name] = run.semantic_bundle(output, events, "fixture")
            audit, spans = self.fixture()
            spans.update(run_id="fixture", timing_kind="synchronized_completion")
            source = "synthetic source archive fixture"
            checksum = hashlib.sha256(source.encode()).hexdigest()
            context = {"transfer_harness": {"status": "identified", "files": [{"name": "fixture.py", "sha256": checksum}]}}
            for name, body in (("context.json", context), ("executed-harness.json", {"fixture.py": {"sha256": checksum, "content": source}}), ("spans.json", spans), ("transfers.json", audit), ("controller.json", {"pid": 41, "returncode": 0}), ("pair-semantics.json", semantics)):
                freeze.seal(root, "fixture/" + name, body)
            report = {"schema_version": 1, "status": "paired_transfer_audit_verified", "paired_equivalent": True, "snapshot": "fixture", **pair.summarize(audit, spans), "artifacts": released_baseline.artifact_hashes(root, "fixture")}
            checksum = freeze.seal(root, "fixture/review.json", report)
            self.assertEqual(pair.verify(root, "fixture", checksum), report)
            (folder / "observed/fixture.txt").write_text("changed")
            with self.assertRaises(ValueError): pair.verify(root, "fixture", checksum)

    def test_verified_community_roles_are_used_and_cache_gate_precedes_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = {"turns": [], "words": []}
            (root / "reference.json").write_text(json.dumps(reference))
            case = {"id": "case", "asset_id": "asset", "reference_path": "reference.json", "reference_sha256": hashlib.sha256((root / "reference.json").read_bytes()).hexdigest()}
            asset = {"id": "asset", "language": "en", "license_id": "license"}
            specs = [{"role": "alignment_en", "repository": "mlx-community/whisper-base-mlx"}]
            context = {"model_inventory": [{"roles": ["speaker_segmentation", "speaker_embedding", "speaker_plda"]}]}
            with patch.object(pair.corpus, "verify_license"), patch.object(pair.dataset_reference, "reference", return_value=reference), patch.object(pair.released_baseline, "identity", return_value=("identity", context)), patch.object(pair.run, "model_cache_roots", return_value=(None, None)), patch.object(pair.provision, "read_credential") as credential:
                with self.assertRaisesRegex(ValueError, "transfer_cache_missing"):
                    pair.capture(root, "new", {"cases": [case], "assets": [asset]}, {"licenses": [{"id": "license"}]}, "case", "Rapida", True, root, root, root, specs, root / "bridge")
                credential.assert_not_called()
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contract_audit, contract_prerequisites, corpus, freeze, manifest, run


class ContractPrerequisiteTests(unittest.TestCase):
    def snapshot(self, root):
        directory = root / "snapshot"
        directory.mkdir()
        source = root / "source"
        source.mkdir()
        (source / "worker.py").write_text("unchanged oracle")
        archive = directory / "harness-source"
        archive.mkdir()
        (archive / "entry.py").write_text("executed harness")
        identities = {"source": manifest.source_identity(source, ["worker.py"]), "harness": manifest.source_identity(archive, ["entry.py"])}
        (directory / "execution-identities.json").write_text(json.dumps(identities))
        (root / "runtime").write_bytes(b"runtime")
        (directory / "audio.wav").write_bytes(b"audio")
        (root / "license.txt").write_bytes(b"test license")
        asset = {"id": "asset", "path": "snapshot/audio.wav", "sha256": manifest.sha256(directory / "audio.wav"), "license_id": "test"}
        license = {"id": "test", "verified": True, "revision": "test", "source": "https://example.test/license", "text_path": "license.txt", "text_sha256": manifest.sha256(root / "license.txt"), "attribution": "Test fixture", "rights": {"download": True}, "asset_ids": ["asset"]}
        rows, cases = [], []
        for identifier in corpus.CONTRACT_CASES:
            outcome = "failed" if identifier in ("invalid_token", "model_failure") else identifier if identifier in ("cancelled", "interrupted") else "completed"
            scope = "alignment_export_boundary" if identifier in ("unknown_chars", "out_of_range") else "startup_termination" if identifier in ("cancelled", "interrupted") else "original_worker"
            row = {"case_id": identifier, "status": "contract_verified", "measured": True, "oracle_adapter_equivalent": True, "quality_certified": False, "gpu_placement_certified": False, "outcome": outcome, "execution_scope": scope, "semantic_digest": "a" * 64, "asset_sha256": asset["sha256"], "processes": [{"process_group_alive": False, "stdin_closed": True, "stdout_closed": True, "returncode": 0 if outcome == "completed" else 1}] * 2, "source_digest": freeze.digest(identities["source"]), "harness_digest": freeze.digest(identities["harness"]), "runtime_digest": manifest.sha256(root / "runtime"), "model_digest": freeze.digest([])}
            row.update(asr_performed=scope == "original_worker" and outcome == "completed", termination_scope="startup_deadline_not_all_pipeline_stages" if identifier in ("cancelled", "interrupted") else None, invalid_token_scope="empty_credential_guard_not_http_authentication" if identifier == "invalid_token" else None)
            rows.append(row)
            cases.append({"id": identifier, "asset_id": "asset", "execution_scope": scope})
            freeze.seal(directory, "results/" + identifier + ".json", row)
            if outcome == "completed":
                for mode in ("original", "instrumented"):
                    pair = directory / "pairs" / identifier / mode
                    pair.mkdir(parents=True)
                    for name in ("original.json", "aligned.json", "result.json"): (pair / name).write_text(json.dumps({"segments": [], "language": "it"}))
                    for extension in ("txt", "srt", "vtt"): (pair / ("audio." + extension)).write_text("")
        freeze.seal(directory, "fixture-inventory.json", {"cases": cases, "assets": [asset], "licenses": {"licenses": [license]}})
        freeze.seal(directory, "contract-evidence.json", {"cases": rows, "required_cases": list(corpus.CONTRACT_CASES), "quality_certified": False, "full_baseline_gate_passed": False})
        return source

    def test_sealed_artifacts_detect_export_audio_and_harness_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.snapshot(root)
            report = contract_audit.audit(root, "snapshot", root / "runtime", source, [])
            freeze.seal(root, "snapshot/artifact-audit.json", report)
            self.assertEqual(contract_audit.audit(root, "snapshot", root / "runtime", source, []), report)
            export = root / "snapshot/pairs/silence/original/audio.txt"
            export.write_text("changed export")
            with self.assertRaisesRegex(ValueError, "contract_artifacts_changed"): contract_audit.audit(root, "snapshot", root / "runtime", source, [])
            export.write_text("")
            (root / "snapshot/audio.wav").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "fixture_hash_mismatch"): contract_audit.audit(root, "snapshot", root / "runtime", source, [])
            (root / "snapshot/audio.wav").write_bytes(b"audio")
            (root / "snapshot/harness-source/entry.py").write_text("changed harness")
            with self.assertRaisesRegex(ValueError, "execution_source_changed"): contract_audit.audit(root, "snapshot", root / "runtime", source, [])

    def test_completed_measurement_resumes_by_auditing_without_relaunch_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.snapshot(root)
            with patch.object(contract_prerequisites, "execute_pair", side_effect=AssertionError("must not relaunch")):
                rows = contract_prerequisites.measure(root, "snapshot", root / "runtime", source, [])
            self.assertEqual(len(rows), 14)
            changed = json.loads((root / "snapshot/contract-evidence.json").read_text())
            changed["cases"][0]["processes"][0]["process_group_alive"] = True
            (root / "snapshot/contract-evidence.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "contract_record_changed"): contract_prerequisites.measure(root, "snapshot", root / "runtime", source, [])

    def test_stronger_preflight_requires_seal_and_matching_contract_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.snapshot(root)
            inventory = json.loads((root / "snapshot/fixture-inventory.json").read_text())
            catalog = {"cases": inventory["cases"], "assets": inventory["assets"]}
            arguments = (catalog, corpus.expected_coverage(Path(__file__).resolve().parents[2]), inventory["licenses"], root, [])
            error = {"code": "contract_evidence_unverified"}
            self.assertNotIn(error, run.preflight(*arguments)["missing"])
            self.assertIn(error, run.preflight(*arguments, contract_snapshot="snapshot", runtime=root / "runtime", source_root=source)["missing"])
            freeze.seal(root, "snapshot/artifact-audit.json", contract_audit.audit(root, "snapshot", root / "runtime", source, []))
            report = run.preflight(*arguments, contract_snapshot="snapshot", runtime=root / "runtime", source_root=source)
            self.assertNotIn(error, report["missing"])
            self.assertEqual(len(report["contracts"]["cases"]), 14)
            catalog["cases"][0]["execution_scope"] = "unmeasured"
            self.assertIn(error, run.preflight(*arguments, contract_snapshot="snapshot", runtime=root / "runtime", source_root=source)["missing"])

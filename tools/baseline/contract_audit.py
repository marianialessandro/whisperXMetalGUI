"""Recheck private contract artifacts against their archived execution provenance without rerunning workers."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, freeze, manifest, provision, run


def require(condition, code):
    if not condition: raise corpus.CorpusError(code)


def audit(root, snapshot, runtime, source_root, specs):
    directory = corpus.safe_target(root, snapshot)
    evidence = json.loads((directory / "contract-evidence.json").read_text())
    inventory = json.loads((directory / "fixture-inventory.json").read_text())
    identities = json.loads((directory / "execution-identities.json").read_text())
    require(evidence.get("quality_certified") is False and evidence.get("full_baseline_gate_passed") is False, "contract_scope_invalid")
    require(evidence["required_cases"] == list(corpus.CONTRACT_CASES) and [row["case_id"] for row in evidence["cases"]] == list(corpus.CONTRACT_CASES), "contract_coverage_invalid")
    require([case["id"] for case in inventory["cases"]] == list(corpus.CONTRACT_CASES) and len({asset["id"] for asset in inventory["assets"]}) == len(inventory["assets"]), "fixture_inventory_invalid")
    for identity, base in ((identities["source"], source_root), (identities["harness"], directory / "harness-source")):
        require(identity["status"] == "identified", "execution_source_unidentified")
        current = manifest.source_identity(base, [item["name"] for item in identity["files"]])
        require(current["files"] == identity["files"] and current["digest"] == identity["digest"], "execution_source_changed")
    models = [provision.inspect_spec(spec) for spec in specs]
    require(all(model["status"] == "files_verified" for model in models), "models_not_verified")
    expected = {"source_digest": freeze.digest(identities["source"]), "harness_digest": freeze.digest(identities["harness"]), "runtime_digest": manifest.sha256(Path(runtime).resolve()), "model_digest": freeze.digest(models)}
    artifact_hashes = {}
    for case, record in zip(inventory["cases"], evidence["cases"]):
        identifier = case["id"]
        require(record == json.loads((directory / "results" / (identifier + ".json")).read_text()), "contract_record_changed")
        require(all(record.get(key) == value for key, value in expected.items()), "contract_execution_identity_changed")
        require(record.get("status") == "contract_verified" and record.get("measured") is True and record.get("oracle_adapter_equivalent") is True and record.get("quality_certified") is False and record.get("gpu_placement_certified") is False, "contract_measurement_invalid")
        outcome = "failed" if identifier in ("invalid_token", "model_failure") else identifier if identifier in ("cancelled", "interrupted") else "completed"
        scope = "alignment_export_boundary" if identifier in ("unknown_chars", "out_of_range") else "startup_termination" if identifier in ("cancelled", "interrupted") else "original_worker"
        require(record.get("outcome") == outcome and record.get("execution_scope") == scope and case.get("execution_scope") == scope and corpus.valid_digest(record.get("semantic_digest")), "contract_outcome_invalid")
        require(len(record.get("processes", [])) == 2 and all(item.get("process_group_alive") is False and item.get("stdin_closed") is True and item.get("stdout_closed") is True and type(item.get("returncode")) is int for item in record["processes"]), "contract_cleanup_unverified")
        require(all((item["returncode"] == 0) == (outcome == "completed") for item in record["processes"]), "contract_exit_status_invalid")
        if identifier in ("cancelled", "interrupted"):
            require(record.get("termination_scope") == "startup_deadline_not_all_pipeline_stages", "contract_termination_scope_invalid")
        if identifier == "invalid_token": require(record.get("invalid_token_scope") == "empty_credential_guard_not_http_authentication", "contract_credential_scope_invalid")
        require(record.get("asr_performed") is (scope == "original_worker" and outcome == "completed"), "contract_asr_scope_invalid")
        asset = next(item for item in inventory["assets"] if item["id"] == case["asset_id"])
        legal = next(item for item in inventory["licenses"]["licenses"] if item["id"] == asset["license_id"])
        corpus.verify_license(asset, legal, root, "download")
        require(record["asset_sha256"] == asset["sha256"] == manifest.sha256(corpus.safe_target(root, asset["path"])), "fixture_hash_mismatch")
        if outcome == "completed":
            pair = directory / "pairs" / identifier
            bundles = [run.semantic_bundle(pair / mode, [], Path(asset["path"]).stem) for mode in ("original", "instrumented")]
            require(bundles[0] == bundles[1], "contract_artifacts_changed")
            if identifier == "empty": require(bundles[0]["artifacts"]["result.json"]["segments"] == [], "empty_contract_mismatch")
    for path in directory.rglob("*"):
        if path.is_file() and path.name != "artifact-audit.json":
            name = str(path.relative_to(directory))
            artifact_hashes[name] = manifest.sha256(corpus.safe_target(directory, name))
    report = {"status": "contract_artifacts_verified", "cases": list(corpus.CONTRACT_CASES), "execution_identities": expected, "artifact_sha256": artifact_hashes, "event_scope": "execution_verified_events_not_retained_for_replay", "quality_certified": False, "full_baseline_gate_passed": False}
    sealed = directory / "artifact-audit.json"
    if sealed.exists(): require(json.loads(sealed.read_text()) == report, "contract_snapshot_mutated")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-specs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true", help="Seal the first verified artifact inventory without overwriting; subsequent audits compare every artifact hash to it.")
    args = parser.parse_args(argv)
    try:
        report = audit(args.evidence_root.resolve(strict=True), args.snapshot, args.runtime, args.source_root.resolve(strict=True), json.loads(args.model_specs.read_text()))
        destination = args.snapshot + "/artifact-audit.json"
        if args.seal and not corpus.safe_target(args.evidence_root, destination).exists(): freeze.seal(args.evidence_root, destination, report)
        print(json.dumps({key: value for key, value in report.items() if key != "artifact_sha256"}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "prerequisite_missing", "code": "contract_audit_failed", "full_baseline_gate_passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Recheck pinned released benchmarks and diagnostic traces without inference."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, freeze, released_baseline
from tools.evaluation import voice_baseline
from tools.tracing import isolated_operator, pipeline_trace


ROLES = {"text": "text", "voices": "voices", "pipeline": "pipeline", "wav2vec2": "isolated", "alignment_log_softmax": "isolated", "ctc_trellis": "isolated", "ctc_backtrack": "isolated"}
VERIFIERS = {"text": released_baseline.verify, "voices": voice_baseline.verify, "pipeline": pipeline_trace.verify, "isolated": isolated_operator.verify}


def validate(document):
    corpus.require(isinstance(document, dict), "prerequisites_manifest_invalid")
    corpus.require(document.get("schema_version") == 1 and document.get("reference_policy") == "released-dataset", "prerequisites_policy_invalid")
    rows = document.get("evidence")
    corpus.require(isinstance(rows, list) and len(rows) == len(ROLES) and all(isinstance(row, dict) for row in rows), "prerequisites_evidence_missing")
    corpus.require({row.get("id") for row in rows} == set(ROLES), "prerequisites_roles_invalid")
    for row in rows:
        corpus.require(row.get("kind") == ROLES[row["id"]], "prerequisites_kind_invalid")
        name = row.get("snapshot")
        corpus.require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name, "prerequisites_snapshot_invalid")
        corpus.require(row.get("sha256") is None or corpus.valid_digest(row["sha256"]), "prerequisites_checksum_invalid")
    corpus.require(len({row["snapshot"] for row in rows}) == len(rows), "prerequisites_snapshots_duplicate")


def inspect(root, document):
    validate(document)
    results = []
    for row in sorted(document["evidence"], key=lambda item: item["id"]):
        result = {**row, "status": "pending_checksum"}
        if row["sha256"] is not None:
            try:
                evidence = VERIFIERS[row["kind"]](root, row["snapshot"], row["sha256"])
                if row["kind"] in ("text", "voices"):
                    corpus.require(set(evidence.get("profiles", [])) == {"Rapida", "Bilanciata", "Massima"} and evidence.get("jobs") == (540 if row["kind"] == "text" else 54), "prerequisites_benchmark_coverage_missing")
                    result["paired_jobs"] = evidence["jobs"]
                if row["kind"] == "isolated":
                    corpus.require(evidence.get("operator") == row["id"] and evidence.get("individual_operator_dispatch_verified") is True, "prerequisites_operator_mismatch")
                    result.update(observed_tensor_backend="cpu" if row["id"].startswith("ctc_") else "mps", gpu_activity_observed=evidence["gpu_activity_observed"], placement_gate_eligible=False)
                if row["kind"] == "pipeline":
                    context = json.loads(corpus.safe_target(root, row["snapshot"] + "/context.json").read_text())
                    corpus.require(context.get("settings", {}).get("voices") is True and evidence.get("gpu_activity_observed") is True, "prerequisites_voice_trace_missing")
                    result["input_scope"] = evidence.get("input_scope", {"kind": "full_source_recording"})
                result["status"] = "verified"
            except (ValueError, OSError, KeyError, TypeError):
                result["status"] = "invalid_or_missing"
        results.append(result)
    ready = all(row["status"] == "verified" for row in results)
    return {"schema_version": 1, "scope": "released_benchmarks_and_scoped_dispatch", "status": "verified" if ready else "incomplete", "manifest": document, "evidence": results, "local_human_review_required": False, "current_environment_verified": False, "full_pipeline_gpu_certified": False, "all_numerical_gpu_verified": False, "promotion_allowed": False, "remaining_requirements": ["complete_operator_placement_evidence", "complete_corpus_coverage", "isolated_resource_budgets"]}


def seal(root, destination, document):
    report = inspect(root, document)
    corpus.require(report["status"] == "verified", "prerequisites_incomplete")
    return freeze.seal(root, destination, report)


def verify(root, destination, checksum):
    freeze.verify_seal(root, destination, checksum)
    report = json.loads(corpus.safe_target(root, destination).read_text())
    corpus.require(report == inspect(root, report["manifest"]) and report["status"] == "verified", "prerequisites_report_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Seven pinned evidence roles; null checksums are pending and cannot be sealed.")
    parser.add_argument("--report", help="Exclusive report destination relative to evidence root.")
    parser.add_argument("--verify-sha256", help="Read-only verification of a sealed report and every underlying archive.")
    args = parser.parse_args(argv)
    if args.verify_sha256 and (not args.report or args.manifest): parser.error("--verify-sha256 requires --report and excludes --manifest")
    if not args.verify_sha256 and not args.manifest: parser.error("--manifest is required for inspection or sealing")
    try:
        if args.verify_sha256:
            report = verify(args.evidence_root, args.report, args.verify_sha256)
        else:
            document = json.loads(args.manifest.read_text())
            report = inspect(args.evidence_root, document)
            if args.report:
                corpus.require(report["status"] == "verified", "prerequisites_incomplete")
                checksum = freeze.seal(args.evidence_root, args.report, report)
                report = {"status": "sealed", "sha256": checksum, "promotion_allowed": False}
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] in ("verified", "sealed") else 2
    except (ValueError, OSError, KeyError, TypeError):
        print(json.dumps({"status": "prerequisites_invalid", "promotion_allowed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

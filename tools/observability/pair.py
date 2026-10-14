"""Source-verified paired transfer diagnostics using released licensed audio."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, dataset_reference, freeze, manifest, provision, released_baseline, run


def summarize(audit, spans):
    corpus.require(audit.get("status") == "transfer_audit_completed" and type(audit.get("pid")) is int and audit["pid"] > 0 and audit["pid"] == spans.get("pid") and audit.get("scope") == "python_thread_aten_copy_and_scalar_boundaries", "transfer_audit_identity_invalid")
    corpus.require(type(audit.get("failed_dispatches")) is int and audit["failed_dispatches"] >= 0 and audit.get("numerical_values_retained") is False and audit.get("physical_copy_bytes") is None and audit.get("compiled_backend_internal_transfers_complete") is False and audit.get("other_threads_observed") is False and audit.get("mlx_transfers_observed") is False, "transfer_audit_scope_invalid")
    stages = {row["stage"] for row in spans["spans"]} | {"outside_observed_phases"}
    rows = audit.get("records")
    corpus.require(isinstance(rows, list), "transfer_audit_rows_invalid")
    counts = {"device_to_host_tensor_calls": 0, "device_to_host_source_logical_bytes": 0, "host_to_device_tensor_calls": 0, "host_to_device_source_logical_bytes": 0, "device_scalar_readback_calls": 0, "device_scalar_source_logical_bytes": 0, "cpu_scalar_calls": 0, "same_device_copy_calls": 0}
    identities = set()
    for row in rows:
        corpus.require(row.get("stage") in stages and row.get("operation") in ("aten._to_copy.default", "aten.copy_.default", "aten._local_scalar_dense.default"), "transfer_audit_boundary_invalid")
        corpus.require(all(type(row.get(key)) is int and row[key] >= 0 for key in ("calls", "source_tensor_logical_bytes", "destination_tensor_logical_bytes")) and row["calls"] > 0, "transfer_audit_counters_invalid")
        source, target = row["source_device"], row["destination_device"]
        corpus.require(source in ("cpu", "mps", "cuda", "xpu", "meta") and target in ("cpu", "mps", "cuda", "xpu", "meta", "host_scalar"), "transfer_audit_device_invalid")
        corpus.require(all(isinstance(row.get(key), str) and row[key].startswith("torch.") for key in ("source_dtype", "destination_dtype")), "transfer_audit_dtype_invalid")
        key = tuple(row[name] for name in ("stage", "operation", "source_device", "destination_device", "source_dtype", "destination_dtype"))
        corpus.require(key not in identities and (target == "host_scalar") == (row["operation"] == "aten._local_scalar_dense.default"), "transfer_audit_duplicate_or_kind_invalid")
        identities.add(key)
        if target == "host_scalar":
            corpus.require(row["destination_tensor_logical_bytes"] == 0, "transfer_audit_scalar_invalid")
            if source == "cpu": counts["cpu_scalar_calls"] += row["calls"]
            elif source != "meta":
                counts["device_scalar_readback_calls"] += row["calls"]
                counts["device_scalar_source_logical_bytes"] += row["source_tensor_logical_bytes"]
        elif source != "cpu" and source != "meta" and target == "cpu":
            counts["device_to_host_tensor_calls"] += row["calls"]
            counts["device_to_host_source_logical_bytes"] += row["source_tensor_logical_bytes"]
        elif source == "cpu" and target not in ("cpu", "meta"):
            counts["host_to_device_tensor_calls"] += row["calls"]
            counts["host_to_device_source_logical_bytes"] += row["source_tensor_logical_bytes"]
        elif source == target: counts["same_device_copy_calls"] += row["calls"]
    return {"scope": audit["scope"], **counts, "failed_dispatches": audit["failed_dispatches"], "physical_copy_bytes": None, "mlx_transfer_coverage": "unobserved", "compiled_backend_internal_transfer_coverage": "unobserved", "other_thread_transfer_coverage": "unobserved", "numerical_values_retained": False, "benchmark_timing_eligible": False, "full_pipeline_gpu_certified": False}


def capture(root, snapshot, catalog, licenses, case_id, quality, voices, source, runtime, reference_root, specs, bridge=None, timeout=1800, python_readbacks=False):
    corpus.require(Path(snapshot).name == snapshot and snapshot not in ("", ".", "..") and 30 <= timeout <= 1800, "transfer_settings_invalid")
    folder = corpus.safe_target(root, snapshot)
    corpus.require(not folder.exists(), "transfer_destination_exists")
    case = next(item for item in catalog["cases"] if item["id"] == case_id)
    asset = next(item for item in catalog["assets"] if item["id"] == case["asset_id"])
    legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
    corpus.verify_license(asset, legal, root, "download")
    reference = dataset_reference.reference(root, asset)
    corpus.require(manifest.sha256(corpus.safe_target(root, case["reference_path"])) == case["reference_sha256"] and json.loads(corpus.safe_target(root, case["reference_path"]).read_text()) == reference, "transfer_reference_invalid")
    fingerprint, context = released_baseline.identity(source, runtime, reference_root, specs)
    roles = {role for model in context["model_inventory"] for role in model.get("roles", [])}
    roles.update(spec.get("role") for spec in specs)
    required = {"alignment_" + asset["language"]}
    if voices: required.update(("speaker_segmentation", "speaker_embedding", "speaker_plda"))
    corpus.require(required.issubset(roles) and len([spec for spec in specs if spec.get("repository") == contracts.PROFILES[quality]["mlx"]]) == 1 and (not voices or bridge is not None), "transfer_models_or_bridge_missing")
    hf_cache, torch_cache = run.model_cache_roots(specs)
    corpus.require(hf_cache is not None and torch_cache is not None, "transfer_cache_missing")
    names = [str(path.relative_to(source)) for path in (source / "tools/observability").glob("*.py")]
    harness = manifest.source_identity(source, names)
    corpus.require(harness["status"] == "identified", "transfer_harness_missing")
    settings = {"quality": quality, "language": asset["language"], "voices": voices, "hints": ""}
    contracts.validate_settings(settings)
    context.update(transfer_harness=harness, settings=settings, case_id=case_id, audio_sha256=asset["sha256"], diagnostic_timing_only=True)
    freeze.seal(root, snapshot + "/context.json", context)
    freeze.seal(root, snapshot + "/executed-harness.json", {item["name"]: {"sha256": item["sha256"], "content": (source / item["name"]).read_text()} for item in harness["files"]})
    original, observed = folder / "original", folder / "observed"
    original.mkdir()
    observed.mkdir()
    token = None
    request = {}
    owned = {}
    observer_options = ["--python-readback-report", folder / "python-readbacks.json"] if python_readbacks else []
    try:
        token = provision.read_credential(bridge) if voices else None
        request = dict(settings, audio=str(corpus.safe_target(root, asset["path"])), token=token)
        common = {"cache_root": hf_cache, "torch_home": torch_cache}
        first = run.execute(runtime, [source / "tools/baseline/offline_worker.py", "--worker", source / "macOS/Worker/transcribe_worker.py", "--source-root", source, "--seed", "1729"], dict(request, output=str(original)), timeout, **common)
        second = run.execute(runtime, [source / "tools/observability/worker.py", "--transfer-report", folder / "transfers.json", *observer_options, "--worker", source / "macOS/Worker/transcribe_worker.py", "--source-root", source, "--spans", folder / "spans.json", "--run-id", snapshot, "--synchronized", "--seed", "1729"], dict(request, output=str(observed)), timeout, process_audit=owned, **common)
    finally:
        request.clear()
        token = None
    basename = Path(asset["path"]).stem
    corpus.require(contracts.equivalent(run.semantic_bundle(original, first, basename), run.semantic_bundle(observed, second, basename), str(original), str(observed)) and not first[-1]["warnings"] and not second[-1]["warnings"], "transfer_pair_mismatch")
    freeze.seal(root, snapshot + "/pair-semantics.json", {"basename": basename, "original": run.semantic_bundle(original, first, basename), "observed": run.semantic_bundle(observed, second, basename)})
    corpus.require(released_baseline.identity(source, runtime, reference_root, specs)[0] == fingerprint and manifest.source_identity(source, names)["digest"] == harness["digest"], "transfer_execution_identity_changed")
    spans = json.loads((folder / "spans.json").read_text())
    corpus.require(spans["run_id"] == snapshot and spans["timing_kind"] == "synchronized_completion" and owned.get("pid") == spans["pid"] and owned.get("returncode") == 0, "transfer_span_identity_invalid")
    freeze.seal(root, snapshot + "/controller.json", owned)
    report = {"schema_version": 1, "status": "paired_transfer_audit_verified", "paired_equivalent": True, "snapshot": snapshot, **summarize(json.loads((folder / "transfers.json").read_text()), spans)}
    if python_readbacks:
        report["python_method_readbacks"] = method_summary(folder, spans)
    report["artifacts"] = released_baseline.artifact_hashes(root, snapshot)
    return report, freeze.seal(root, snapshot + "/review.json", report)


def method_summary(folder, spans):
    from tools.observability import python_readbacks
    report = json.loads((folder / "python-readbacks.json").read_text())
    corpus.require(report.get("pid") == spans["pid"] and report.get("status") == "python_readback_audit_completed", "python_readback_identity_invalid")
    return python_readbacks.summarize(report, spans)


def verify(root, snapshot, checksum):
    freeze.verify_seal(root, snapshot + "/review.json", checksum)
    folder = corpus.safe_target(root, snapshot)
    report = json.loads((folder / "review.json").read_text())
    corpus.require(report.get("status") == "paired_transfer_audit_verified" and report.get("paired_equivalent") is True and report.get("snapshot") == snapshot, "transfer_review_invalid")
    corpus.require(report.get("artifacts") and all(manifest.sha256(corpus.safe_target(root, name)) == digest for name, digest in report["artifacts"].items()), "transfer_artifact_changed")
    context = json.loads((folder / "context.json").read_text())
    archive = json.loads((folder / "executed-harness.json").read_text())
    corpus.require(context["transfer_harness"].get("status") == "identified" and bool(archive) and set(archive) == {item["name"] for item in context["transfer_harness"]["files"]}, "transfer_harness_invalid")
    for item in context["transfer_harness"]["files"]:
        retained = archive[item["name"]]
        corpus.require(retained["sha256"] == item["sha256"] and hashlib.sha256(retained["content"].encode()).hexdigest() == item["sha256"], "transfer_harness_changed")
    spans = json.loads((folder / "spans.json").read_text())
    owned = json.loads((folder / "controller.json").read_text())
    corpus.require(spans["run_id"] == snapshot and spans["timing_kind"] == "synchronized_completion" and owned.get("pid") == spans["pid"] and owned.get("returncode") == 0, "transfer_span_identity_invalid")
    semantics = json.loads((folder / "pair-semantics.json").read_text())
    for name in ("original", "observed"):
        contracts.validate_events(semantics[name]["events"], "completed")
        corpus.require(run.semantic_bundle(folder / name, semantics[name]["events"], semantics["basename"]) == semantics[name], "transfer_pair_artifacts_changed")
    corpus.require(not semantics["original"]["events"][-1]["warnings"] and not semantics["observed"]["events"][-1]["warnings"] and contracts.equivalent(semantics["original"], semantics["observed"], semantics["original"]["events"][-1]["output"], semantics["observed"]["events"][-1]["output"]), "transfer_pair_semantics_changed")
    rebuilt = summarize(json.loads((folder / "transfers.json").read_text()), spans)
    if "python_method_readbacks" in report:
        corpus.require(snapshot + "/python-readbacks.json" in report["artifacts"], "python_readback_artifact_missing")
        rebuilt["python_method_readbacks"] = method_summary(folder, spans)
    corpus.require(all(report.get(key) == value for key, value in rebuilt.items()), "transfer_review_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-root", "catalog", "licenses", "source-root", "runtime", "reference-root", "model-specs"): parser.add_argument("--" + name, type=Path, required=name == "evidence-root")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--quality", choices=tuple(contracts.PROFILES), default="Rapida")
    parser.add_argument("--voices", action="store_true")
    parser.add_argument("--python-readbacks", action="store_true", help="Optional separate Tensor.cpu/numpy method observations, including inference tensor API boundaries.")
    parser.add_argument("--bridge", type=Path, help="Private credential bridge for optional voice diagnostics; credentials only in memory/stdin.")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--verify-sha256", help="Archive-only verification; requires just evidence-root and snapshot.")
    args = parser.parse_args(argv)
    if not args.verify_sha256:
        if any(getattr(args, key) is None for key in ("catalog", "licenses", "source_root", "runtime", "reference_root", "model_specs", "case_id")): parser.error("capture requires all source/model/catalog arguments")
        if args.voices and args.bridge is None: parser.error("voices requires bridge")
    try:
        if args.verify_sha256:
            report = verify(args.evidence_root, args.snapshot, args.verify_sha256)
            checksum = args.verify_sha256
        else:
            report, checksum = capture(args.evidence_root.resolve(strict=True), args.snapshot, json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text()), args.case_id, args.quality, args.voices, args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text()), args.bridge, args.timeout, args.python_readbacks)
        summary = {"status": report["status"], "sha256": checksum, "device_to_host_source_logical_bytes": report["device_to_host_source_logical_bytes"], "full_pipeline_gpu_certified": False}
        if "python_method_readbacks" in report: summary["python_method_readbacks"] = report["python_method_readbacks"]
        print(json.dumps(summary))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "transfer_audit_incomplete", "full_pipeline_gpu_certified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

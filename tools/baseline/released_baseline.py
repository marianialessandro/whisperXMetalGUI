"""Reproducible released-dataset baseline runs, recording-level calibration and scoped immutable sealing."""

import argparse
from collections import defaultdict
import json
import os
import shutil
import tempfile
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, dataset_reference, freeze, manifest, provision, run

POLICY = {"version": 1, "reference_policy": "released-dataset", "minimum_recordings_per_stratum_per_split": 3, "minimum_replicates": 3, "bootstrap_seed": 17, "bootstrap_replicates": 2000, "confidence": 0.95, "replica_seed": 1729, "voices": False, "profile_timing": "synchronized_instrumented_api_and_total_pair_wall", "sampling_unit": "recording_mean", "speaker_disjointness": "verified_for_AMI_unavailable_for_FLEURS", "candidate_inputs": "forbidden", "scope": "text_and_available_dataset_timestamp_agreement"}


def require(condition, code):
    if not condition: raise corpus.CorpusError(code)


def identity(source, runtime, reference_root, specs):
    models = [provision.inspect_spec(spec) for spec in specs]
    require(models and all(item["status"] == "files_verified" for item in models), "released_models_unverified")
    for spec in specs:
        if spec.get("transport") not in ("torch_hub", "torch_hub_source"):
            ref = manifest.contained(Path(spec["cache_root"]), "models--" + spec["repository"].replace("/", "--") + "/refs/main")
            require(ref.is_file() and ref.read_text().strip() == spec["revision"], "released_model_ref_changed")
    context = manifest.collect("gui-checkout", source, runtime, reference_root, specs)
    require(context["status"] == "identified", "released_runtime_source_unidentified")
    harness = manifest.source_identity(source, [str(path.relative_to(source)) for path in (source / "tools/baseline").glob("*.py")])
    fingerprints = {"source_digest": context["source"]["digest"], "runtime_digest": freeze.digest(context["runtime"]), "hardware_digest": freeze.digest(context["hardware"]), "model_digest": freeze.digest(models), "harness_digest": harness["digest"]}
    return fingerprints, {"manifest": context, "model_inventory": models, "harness": harness}


def plan(catalog, profiles, replicates=3):
    require(catalog.get("reference_policy") == "released-dataset" and type(replicates) is int and replicates >= POLICY["minimum_replicates"], "released_plan_policy_invalid")
    require(profiles and len(set(profiles)) == len(profiles) and all(profile in contracts.PROFILES for profile in profiles), "released_profiles_invalid")
    assets = {item["id"]: item for item in catalog["assets"]}
    require(len(assets) == len(catalog["assets"]) and len({case["id"] for case in catalog["cases"]}) == len(catalog["cases"]), "released_duplicate_identity")
    groups = defaultdict(list)
    for case in catalog["cases"]:
        asset = assets[case["asset_id"]]
        require(case.get("type") == "released_dataset", "released_case_type_invalid")
        dataset = "AMI" if asset["id"].startswith("ami-") else "FLEURS" if asset["id"].startswith("fleurs-") else None
        require(dataset is not None, "released_dataset_unsupported")
        groups[(dataset, asset["language"])].append((case, asset))
    require({language for dataset, language in groups if dataset == "FLEURS"} == set(corpus.REPRESENTATIVES), "released_languages_incomplete")
    entries, ami_splits, audio_splits = [], [], {}
    for (dataset, language), records in sorted(groups.items()):
        records.sort(key=lambda pair: pair[1]["id"])
        require(len(records) >= 6, "released_recordings_insufficient")
        for index, (case, asset) in enumerate(records):
            split = asset.get("proposed_split") if dataset == "AMI" else "calibration" if index < len(records) // 2 else "evaluation"
            require(split in ("calibration", "evaluation"), "released_split_missing")
            require(asset["sha256"] not in audio_splits or audio_splits[asset["sha256"]] == split, "released_audio_split_leakage")
            audio_splits[asset["sha256"]] = split
            if dataset == "AMI": ami_splits.append(dict(asset, split=split))
            for profile in profiles:
                for replica in range(replicates):
                    entries.append({"id": uuid.uuid4().hex, "case_id": case["id"], "asset_id": asset["id"], "audio_sha256": asset["sha256"], "split": split, "replica": replica, "stratum": profile + "/" + dataset + "/" + language, "settings": {"quality": profile, "language": language, "voices": False, "hints": ""}})
        require(all(sum((asset.get("proposed_split") if dataset == "AMI" else "calibration" if index < len(records) // 2 else "evaluation") == split for index, (case, asset) in enumerate(records)) >= 3 for split in ("calibration", "evaluation")), "released_split_recordings_insufficient")
    corpus.validate_splits(ami_splits)
    return {"schema_version": 1, "policy": POLICY, "profiles": profiles, "replicates": replicates, "jobs": entries, "local_human_review_required": False, "gpu_placement_certified": False}


def prepare(root, snapshot, catalog_path, license_path, profiles, replicates, source, runtime, reference_root, specs):
    require(Path(snapshot).name == snapshot and snapshot not in ("", ".", ".."), "released_snapshot_invalid")
    target = corpus.safe_target(root, snapshot)
    require(not target.exists(), "destination_exists")
    catalog = json.loads(catalog_path.read_text())
    document = plan(catalog, profiles, replicates)
    fingerprints, context = identity(source, runtime, reference_root, specs)
    document.update(identity=fingerprints, catalog_path=str(catalog_path.relative_to(root)), catalog_sha256=manifest.sha256(catalog_path), licenses_path=str(license_path.relative_to(root)), licenses_sha256=manifest.sha256(license_path))
    temporary = Path(tempfile.mkdtemp(prefix=".released-plan-", dir=root))
    try:
        freeze.seal(temporary, "context.json", context)
        checksum = freeze.seal(temporary, "plan.json", document)
        freeze.seal(temporary, "plan-receipt.json", {"sha256": checksum})
        require(not target.exists(), "destination_exists")
        os.rename(temporary, target)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return document


def load(root, snapshot):
    receipt = json.loads(corpus.safe_target(root, snapshot + "/plan-receipt.json").read_text())
    freeze.verify_seal(root, snapshot + "/plan.json", receipt["sha256"])
    document = json.loads(corpus.safe_target(root, snapshot + "/plan.json").read_text())
    require(document.get("policy") == POLICY, "released_policy_changed")
    catalog_path, licenses_path = corpus.safe_target(root, document["catalog_path"]), corpus.safe_target(root, document["licenses_path"])
    require(manifest.sha256(catalog_path) == document["catalog_sha256"] and manifest.sha256(licenses_path) == document["licenses_sha256"], "released_inputs_changed")
    catalog = json.loads(catalog_path.read_text())
    rebuilt = plan(catalog, document["profiles"], document["replicates"])
    jobs = document["jobs"]
    require(len(jobs) == len(rebuilt["jobs"]) and len({job["id"] for job in jobs}) == len(jobs) and all(isinstance(job["id"], str) and len(job["id"]) == 32 and all(character in "0123456789abcdef" for character in job["id"]) for job in jobs), "released_plan_jobs_invalid")
    require([{key: value for key, value in job.items() if key != "id"} for job in jobs] == [{key: value for key, value in job.items() if key != "id"} for job in rebuilt["jobs"]], "released_plan_jobs_changed")
    return document, catalog, json.loads(licenses_path.read_text())


def artifact_hashes(root, relative):
    root = root.resolve(strict=True)
    folder = corpus.safe_target(root, relative)
    return {str(path.relative_to(root)): manifest.sha256(corpus.safe_target(root, path.relative_to(root))) for path in sorted(folder.rglob("*")) if path.is_file()}


def verify_job(root, row, expected, catalog=None, licenses=None):
    require(row.get("status") == "measured" and row.get("role") == "baseline" and row.get("job") == expected, "released_job_identity_invalid")
    require(row.get("paired_execution", {}).get("status") == "paired_equivalent" and not row["paired_execution"].get("warnings"), "released_pair_unverified")
    require(row.get("artifacts") and all(manifest.sha256(corpus.safe_target(root, path)) == checksum for path, checksum in row["artifacts"].items()), "released_job_artifact_changed")
    pair = row["pair_path"]
    require(len(Path(pair).parts) == 6 and Path(pair).parts[1:4] == ("jobs", expected["id"], "attempts") and Path(pair).name == "pair" and Path(pair).parent.name.startswith("attempt-") and all(path.startswith(pair + "/") for path in row["artifacts"]), "released_pair_path_invalid")
    require(all(pair + "/" + name in row["artifacts"] for name in ("original/result.json", "instrumented/result.json", "spans.json")), "released_pair_artifacts_missing")
    require(freeze.finite(row["pair_wall_ns"]) and row["pair_wall_ns"] > 0, "released_pair_timing_invalid")
    require(row["score"]["audio_sha256"] == expected["audio_sha256"] and row["score"]["reference_policy"] == "released-dataset", "released_job_reference_mismatch")

    if catalog is not None:
        score = dataset_reference.evaluate(root, catalog, licenses, expected["case_id"], pair + "/original/result.json")
        require(score == row["score"], "released_score_changed")


def execute(root, snapshot, source, runtime, reference_root, specs, timeout=1800):
    document, catalog, licenses = load(root, snapshot)
    fingerprints, _ = identity(source, runtime, reference_root, specs)
    require(fingerprints == document["identity"], "released_execution_identity_changed")
    hf_cache, torch_cache = run.model_cache_roots(specs)
    require(hf_cache is not None and torch_cache is not None, "released_model_cache_missing")
    assets = {asset["id"]: asset for asset in catalog["assets"]}
    for number, job in enumerate(document["jobs"], 1):
        base = snapshot + "/jobs/" + job["id"]
        record_path = corpus.safe_target(root, base + "/measurement.json")
        if record_path.exists():
            row = json.loads(record_path.read_text())
            verify_job(root, row, job, catalog, licenses)
            require(row.get("identity") == fingerprints, "released_resume_identity_changed")
        else:
            pair_relative = base + "/attempts/attempt-" + uuid.uuid4().hex + "/pair"
            pair = corpus.safe_target(root, pair_relative)
            asset = assets[job["asset_id"]]
            require(manifest.sha256(corpus.safe_target(root, asset["path"])) == job["audio_sha256"], "released_audio_changed")
            started = time.perf_counter_ns()
            comparison = run.paired_run(runtime, source / "macOS/Worker/transcribe_worker.py", source, corpus.safe_target(root, asset["path"]), job["settings"], pair, timeout=timeout, cache_root=hf_cache, torch_home=torch_cache, synchronized=True, seed=POLICY["replica_seed"])
            elapsed = time.perf_counter_ns() - started
            require(not comparison["warnings"], "released_pipeline_warning")
            first = dataset_reference.evaluate(root, catalog, licenses, job["case_id"], pair_relative + "/original/result.json")
            second = dataset_reference.evaluate(root, catalog, licenses, job["case_id"], pair_relative + "/instrumented/result.json")
            require({key: value for key, value in first.items() if key != "result_sha256"} == {key: value for key, value in second.items() if key != "result_sha256"}, "released_score_mismatch")
            row = {"status": "measured", "role": "baseline", "job": job, "identity": fingerprints, "paired_execution": comparison, "pair_wall_ns": elapsed, "score": first, "pair_path": pair_relative, "artifacts": artifact_hashes(root, pair_relative)}
            freeze.seal(root, base + "/measurement.json", row)
        print(json.dumps({"completed": number, "total": len(document["jobs"]), "stratum": job["stratum"], "split": job["split"], "replica": job["replica"], "status": row["status"]}), flush=True)
    require(identity(source, runtime, reference_root, specs)[0] == fingerprints, "released_execution_identity_changed")


def metric_values(row):
    score = row["score"]
    values = {"cer": score["text"]["normalized"]["cer"]["rate"]}
    if score["text"]["normalized"]["wer"] is not None: values["wer"] = score["text"]["normalized"]["wer"]["rate"]
    timestamp = score["dataset_timestamp_agreement"]
    if timestamp.get("timed_matches", 0): values.update(dataset_start_mae=timestamp["start_mae"], dataset_end_mae=timestamp["end_mae"], dataset_timestamp_coverage=timestamp["coverage"])
    require(all(freeze.finite(value) and value >= 0 for value in values.values()), "released_metric_missing")
    return values


def calibrate(root, snapshot):
    document, catalog, licenses = load(root, snapshot)
    groups = defaultdict(list)
    for job in document["jobs"]:
        row = json.loads(corpus.safe_target(root, snapshot + "/jobs/" + job["id"] + "/measurement.json").read_text())
        verify_job(root, row, job, catalog, licenses)
        require(row["identity"] == document["identity"], "released_calibration_identity_changed")
        groups[(job["stratum"], job["split"], job["asset_id"])].append(row)
    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    variability = defaultdict(list)
    for (stratum, split, asset), rows in sorted(groups.items()):
        require(len(rows) == document["replicates"] and len({row["job"]["replica"] for row in rows}) == document["replicates"], "released_replica_missing")
        samples = [metric_values(row) for row in rows]
        require(all(set(sample) == set(samples[0]) for sample in samples), "released_metric_applicability_changed")
        for metric in samples[0]:
            values = [sample[metric] for sample in samples]
            strata[stratum][metric][split].append(sum(values) / len(values))
            if split == "calibration": variability[(stratum, metric)].append(max(values) - min(values))
    results = {}
    for stratum, metrics_by_name in sorted(strata.items()):
        results[stratum] = {}
        for metric, splits in metrics_by_name.items():
            require(set(splits) == {"calibration", "evaluation"} and all(len(values) >= 3 for values in splits.values()), "released_calibration_recordings_missing")
            direction = "lower" if metric == "dataset_timestamp_coverage" else "upper"
            bound = freeze.bootstrap_bound(splits["calibration"], direction)
            heldout = sum(splits["evaluation"]) / len(splits["evaluation"])
            results[stratum][metric] = {"direction": direction, "bound": bound, "degradation_delta": freeze.bootstrap_bound(variability[(stratum, metric)], "upper"), "calibration_recordings": len(splits["calibration"]), "evaluation_recordings": len(splits["evaluation"]), "calibration_mean": sum(splits["calibration"]) / len(splits["calibration"]), "evaluation_mean": heldout, "heldout_within_calibration_bound": heldout >= bound if direction == "lower" else heldout <= bound}
    return {"schema_version": 1, "state": "calibrated_released_dataset_benchmark", "policy": POLICY, "identity": document["identity"], "profiles": document["profiles"], "jobs": len(document["jobs"]), "strata": results, "local_human_review_required": False, "independent_timestamp_accuracy_certified": False, "full_pipeline_gpu_certified": False, "heldout_policy": "diagnostic_only_never_select_or_retune_from_heldout", "resource_scope": "pair_wall_not_single_worker_latency"}


def seal(root, snapshot):
    report = calibrate(root, snapshot)
    require(not corpus.safe_target(root, snapshot + "/sealed.json").exists(), "destination_exists")
    report["state"] = "sealed_released_dataset_benchmark"
    report["artifacts"] = artifact_hashes(root, snapshot)
    document, catalog, licenses = load(root, snapshot)
    for name in (document["catalog_path"], document["licenses_path"]): report["artifacts"][name] = manifest.sha256(corpus.safe_target(root, name))
    for asset in catalog["assets"]: report["artifacts"][asset["path"]] = manifest.sha256(corpus.safe_target(root, asset["path"]))
    for case in catalog["cases"]: report["artifacts"][case["reference_path"]] = manifest.sha256(corpus.safe_target(root, case["reference_path"]))
    return freeze.seal(root, snapshot + "/sealed.json", report)


def verify(root, snapshot, checksum):
    freeze.verify_seal(root, snapshot + "/sealed.json", checksum)
    report = json.loads(corpus.safe_target(root, snapshot + "/sealed.json").read_text())
    require(report["state"] == "sealed_released_dataset_benchmark" and report["policy"] == POLICY and report["full_pipeline_gpu_certified"] is False, "released_seal_scope_invalid")
    require(all(manifest.sha256(corpus.safe_target(root, path)) == digest for path, digest in report["artifacts"].items()), "released_seal_artifact_changed")
    calibrated = calibrate(root, snapshot)
    require(calibrated["strata"] == report["strata"] and calibrated["identity"] == report["identity"], "released_seal_calibration_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan", help="Write a new recording split plan before running any oracle; no overwrite.")
    create.add_argument("--catalog", type=Path, required=True)
    create.add_argument("--licenses", type=Path, required=True)
    create.add_argument("--profiles", nargs="+", choices=tuple(contracts.PROFILES), default=["Rapida"])
    create.add_argument("--replicates", type=int, default=3)
    execute_command = commands.add_parser("run", help="Execute or resume the exact plan; networking denied, separate cold original/adapter processes, synchronized instrumented measurements.")
    execute_command.add_argument("--timeout", type=int, default=1800)
    for command in (create, execute_command):
        for name in ("source-root", "runtime", "reference-root", "model-specs"): command.add_argument("--" + name, type=Path, required=True)
    commands.add_parser("calibrate", help="Baseline-only recording means and repeat variability; heldout data never sets thresholds.")
    commands.add_parser("seal", help="Seal a complete scoped released benchmark with artifact digests; does not certify complete GPU placement.")
    check = commands.add_parser("verify")
    check.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        root = args.evidence_root.resolve(strict=True)
        if args.command == "plan":
            document = prepare(root, args.snapshot, args.catalog.resolve(strict=True), args.licenses.resolve(strict=True), args.profiles, args.replicates, args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text()))
            print(json.dumps({"status": "released_baseline_plan_ready", "jobs": len(document["jobs"]), "local_human_review_required": False}))
        elif args.command == "run": execute(root, args.snapshot, args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text()), args.timeout)
        elif args.command == "calibrate": print(json.dumps(calibrate(root, args.snapshot)))
        elif args.command == "seal": print(json.dumps({"status": "released_baseline_sealed", "sha256": seal(root, args.snapshot), "full_pipeline_gpu_certified": False}))
        else:
            report = verify(root, args.snapshot, args.sha256)
            print(json.dumps({"status": "released_baseline_verified", "jobs": report["jobs"], "full_pipeline_gpu_certified": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        print(json.dumps({"status": "released_baseline_incomplete", "reason": str(error), "local_human_review_required": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

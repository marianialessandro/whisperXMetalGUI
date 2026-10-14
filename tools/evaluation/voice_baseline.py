"""Resumable released AMI voice calibration with disjoint recordings and actual turn evidence."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, dataset_reference, freeze, manifest, provision, released_baseline
from tools.evaluation import voice_pair

POLICY = {"reference_policy": "released-dataset", "minimum_recordings_per_split": 3, "replicates": 3, "seed": 1729, "bootstrap_seed": 17, "bootstrap_replicates": 2000, "confidence": 0.95, "collar": 0.0, "skip_overlap": False, "uem": "full_recording_dataset_convention", "heldout_policy": "diagnostic_only_never_retune", "candidate_inputs": "forbidden", "scope": "released_AMI_voice_agreement"}


def harness(source):
    return manifest.source_identity(source, [str(path.relative_to(source)) for path in (source / "tools/evaluation").glob("*.py")])


def prepare(root, snapshot, catalog_path, license_path, profiles, mode, source, runtime, reference_root, specs):
    corpus.require(Path(snapshot).name == snapshot and snapshot not in ("", ".", ".."), "voice_snapshot_invalid")
    corpus.require(not corpus.safe_target(root, snapshot).exists(), "voice_destination_exists")
    corpus.require(profiles and len(set(profiles)) == len(profiles) and all(profile in contracts.PROFILES for profile in profiles) and mode in ("auto", "exact"), "voice_plan_settings_invalid")
    catalog = json.loads(catalog_path.read_text())
    corpus.require(catalog.get("reference_policy") == "released-dataset", "voice_reference_policy_invalid")
    assets = [asset for asset in catalog["assets"] if asset["id"].startswith("ami-")]
    corpus.require(len({asset["id"] for asset in assets}) == len(assets) and all(sum(asset.get("proposed_split") == split for asset in assets) >= 3 for split in ("calibration", "evaluation")), "voice_recordings_missing")
    corpus.validate_splits([dict(asset, split=asset.get("proposed_split")) for asset in assets])
    jobs = []
    for asset in sorted(assets, key=lambda item: item["id"]):
        case = next(item for item in catalog["cases"] if item["asset_id"] == asset["id"])
        reference = dataset_reference.reference(root, asset)
        count = len({turn["speaker"] for turn in reference["turns"]})
        corpus.require(count > 0, "voice_reference_speakers_missing")
        for quality in profiles:
            for replica in range(POLICY["replicates"]):
                jobs.append({"id": uuid.uuid4().hex, "asset_id": asset["id"], "case_id": case["id"], "quality": quality, "split": asset["proposed_split"], "replica": replica, "speaker_count": count if mode == "exact" else None, "audio_sha256": asset["sha256"]})
    identity, context = released_baseline.identity(source, runtime, reference_root, specs)
    voice_harness = harness(source)
    corpus.require(voice_harness["status"] == "identified", "voice_harness_missing")
    document = {"policy": POLICY, "profiles": profiles, "speaker_mode": mode, "identity": identity, "voice_harness_digest": voice_harness["digest"], "jobs": jobs, "catalog_path": str(catalog_path.relative_to(root)), "catalog_sha256": manifest.sha256(catalog_path), "licenses_path": str(license_path.relative_to(root)), "licenses_sha256": manifest.sha256(license_path)}
    context["voice_harness"] = voice_harness
    with tempfile.TemporaryDirectory(prefix=".voice-plan-", dir=root) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        freeze.seal(staging, "context.json", context)
        checksum = freeze.seal(staging, "plan.json", document)
        freeze.seal(staging, "plan-receipt.json", {"sha256": checksum})
        provision.rename_directory_exclusive(staging, corpus.safe_target(root, snapshot))
    return document


def load(root, snapshot):
    receipt = json.loads(corpus.safe_target(root, snapshot + "/plan-receipt.json").read_text())
    freeze.verify_seal(root, snapshot + "/plan.json", receipt["sha256"])
    document = json.loads(corpus.safe_target(root, snapshot + "/plan.json").read_text())
    corpus.require(document["policy"] == POLICY, "voice_policy_changed")
    catalog_path, licenses_path = (corpus.safe_target(root, document[name + "_path"]) for name in ("catalog", "licenses"))
    corpus.require(manifest.sha256(catalog_path) == document["catalog_sha256"] and manifest.sha256(licenses_path) == document["licenses_sha256"], "voice_inputs_changed")
    return document, json.loads(catalog_path.read_text()), json.loads(licenses_path.read_text())


def measurement(root, snapshot, job, catalog, licenses):
    receipt = json.loads(corpus.safe_target(root, snapshot + "/receipts/" + job["id"] + ".json").read_text())
    corpus.require(receipt["job"] == job, "voice_job_changed")
    freeze.verify_seal(root, receipt["measurement_path"], receipt["sha256"])
    row = json.loads(corpus.safe_target(root, receipt["measurement_path"]).read_text())
    corpus.require(row["status"] == "released_voice_pair_measured" and row["case_id"] == job["case_id"] and row["settings"] == {"quality": job["quality"], "language": "en", "voices": True, "hints": "", **({"speaker_count": job["speaker_count"]} if job["speaker_count"] is not None else {})} and row["comparison"]["status"] == "paired_equivalent", "voice_measurement_invalid")
    corpus.require(all(manifest.sha256(corpus.safe_target(root, name)) == checksum for name, checksum in row["artifacts"].items()), "voice_artifact_changed")
    base = str(Path(receipt["measurement_path"]).parent)
    score = dataset_reference.evaluate(root, catalog, licenses, job["case_id"], base + "/pair/instrumented/result.json", base + "/pair/turns.json")
    result = json.loads(corpus.safe_target(root, base + "/pair/instrumented/result.json").read_text())
    words = result.get("word_segments", [word for segment in result["segments"] for word in segment.get("words", [])])
    asset = next(asset for asset in catalog["assets"] if asset["id"] == job["asset_id"])
    reference = dataset_reference.reference(root, asset)
    score["dataset_word_speaker_agreement"] = voice_pair.metrics.word_speaker_scores(reference["words"], words, score["dataset_turn_agreement"]["mapping"])
    corpus.require(score == row["score"] and score["audio_sha256"] == job["audio_sha256"], "voice_score_changed")
    return row, receipt


def execute(root, snapshot, source, runtime, reference_root, specs, bridge, timeout):
    document, catalog, licenses = load(root, snapshot)
    corpus.require(released_baseline.identity(source, runtime, reference_root, specs)[0] == document["identity"] and harness(source)["digest"] == document["voice_harness_digest"], "voice_execution_identity_changed")
    for index, job in enumerate(document["jobs"], 1):
        receipt_path = snapshot + "/receipts/" + job["id"] + ".json"
        if corpus.safe_target(root, receipt_path).exists():
            row, receipt = measurement(root, snapshot, job, catalog, licenses)
            corpus.require(row["identity"] == document["identity"], "voice_resume_identity_changed")
        else:
            attempt = snapshot + "-attempt-" + uuid.uuid4().hex
            row, checksum = voice_pair.execute(root, attempt, catalog, licenses, job["case_id"], job["quality"], job["speaker_count"], source, runtime, reference_root, specs, bridge, timeout)
            corpus.require(row["identity"] == document["identity"] and harness(source)["digest"] == document["voice_harness_digest"], "voice_execution_identity_changed")
            freeze.seal(root, receipt_path, {"job": job, "measurement_path": attempt + "/measurement.json", "sha256": checksum})
        print(json.dumps({"completed": index, "total": len(document["jobs"]), "quality": job["quality"], "case_id": job["case_id"], "replica": job["replica"], "status": "voice_measured"}), flush=True)
    return calibrate(root, snapshot)


def values(row):
    score = row["score"]
    result = released_baseline.metric_values(row)
    result.update(der=score["dataset_turn_agreement"]["der"], jer=score["dataset_turn_agreement"]["jer"], speaker_count_error=score["dataset_turn_agreement"]["speaker_count_error"], word_speaker_accuracy=score["dataset_word_speaker_agreement"]["accuracy"])
    corpus.require(all(freeze.finite(value) and value >= 0 for value in result.values()), "voice_metric_missing")
    return result


def summarize(jobs, measured):
    corpus.require(len(jobs) == len(measured) and len({job["id"] for job in jobs}) == len(jobs), "voice_measurements_missing")
    recordings = defaultdict(list)
    for job, row in zip(jobs, measured): recordings[(job["quality"], job["split"], job["asset_id"])].append((job["replica"], values(row)))
    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    variability = defaultdict(list)
    for (quality, split, asset), rows in sorted(recordings.items()):
        corpus.require(len(rows) == 3 and {replica for replica, row in rows} == {0, 1, 2}, "voice_replica_missing")
        samples = [row for replica, row in rows]
        corpus.require(all(set(row) == set(samples[0]) for row in samples), "voice_metric_applicability_changed")
        for name in samples[0]:
            observed = [row[name] for row in samples]
            strata[quality][name][split].append(sum(observed) / len(observed))
            if split == "calibration": variability[(quality, name)].append(max(observed) - min(observed))
    result = {}
    for quality, names in sorted(strata.items()):
        result[quality] = {}
        for name, splits in names.items():
            corpus.require(set(splits) == {"calibration", "evaluation"} and all(len(samples) >= 3 for samples in splits.values()), "voice_calibration_recordings_missing")
            direction = "lower" if name in ("word_speaker_accuracy", "dataset_timestamp_coverage") else "upper"
            bound = freeze.bootstrap_bound(splits["calibration"], direction)
            heldout = sum(splits["evaluation"]) / len(splits["evaluation"])
            result[quality][name] = {"bound": bound, "direction": direction, "degradation_delta": freeze.bootstrap_bound(variability[(quality, name)], "upper"), "evaluation_mean": heldout, "heldout_within_calibration_bound": heldout >= bound if direction == "lower" else heldout <= bound, "calibration_recordings": len(splits["calibration"]), "evaluation_recordings": len(splits["evaluation"])}
    return result


def calibrate(root, snapshot):
    document, catalog, licenses = load(root, snapshot)
    rows = [measurement(root, snapshot, job, catalog, licenses)[0] for job in document["jobs"]]
    corpus.require(all(row["identity"] == document["identity"] for row in rows), "voice_calibration_identity_changed")
    return {"state": "calibrated_released_voice_benchmark", "policy": POLICY, "profiles": document["profiles"], "speaker_mode": document["speaker_mode"], "jobs": len(document["jobs"]), "strata": summarize(document["jobs"], rows), "local_human_review_required": False, "independent_timestamp_accuracy_certified": False, "full_pipeline_gpu_certified": False}


def seal(root, snapshot):
    report = calibrate(root, snapshot)
    corpus.require(not corpus.safe_target(root, snapshot + "/sealed.json").exists(), "voice_destination_exists")
    report["state"] = "sealed_released_voice_benchmark"
    report["artifacts"] = released_baseline.artifact_hashes(root, snapshot)
    document, catalog, licenses = load(root, snapshot)
    for job in document["jobs"]:
        row, receipt = measurement(root, snapshot, job, catalog, licenses)
        report["artifacts"].update(row["artifacts"])
        report["artifacts"][receipt["measurement_path"]] = receipt["sha256"]
    for name in (document["catalog_path"], document["licenses_path"]): report["artifacts"][name] = manifest.sha256(corpus.safe_target(root, name))
    for case in catalog["cases"]:
        if case["asset_id"].startswith("ami-"): report["artifacts"][case["reference_path"]] = manifest.sha256(corpus.safe_target(root, case["reference_path"]))
    for asset in catalog["assets"]:
        if asset["id"].startswith("ami-"): report["artifacts"][asset["path"]] = manifest.sha256(corpus.safe_target(root, asset["path"]))
    return freeze.seal(root, snapshot + "/sealed.json", report)


def verify(root, snapshot, checksum):
    freeze.verify_seal(root, snapshot + "/sealed.json", checksum)
    report = json.loads(corpus.safe_target(root, snapshot + "/sealed.json").read_text())
    corpus.require(report["state"] == "sealed_released_voice_benchmark" and report["policy"] == POLICY and report["full_pipeline_gpu_certified"] is False, "voice_seal_scope_invalid")
    corpus.require(all(manifest.sha256(corpus.safe_target(root, path)) == checksum for path, checksum in report["artifacts"].items()), "voice_seal_artifact_changed")
    corpus.require(calibrate(root, snapshot)["strata"] == report["strata"], "voice_seal_calibration_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan")
    create.add_argument("--catalog", type=Path, required=True)
    create.add_argument("--licenses", type=Path, required=True)
    create.add_argument("--profiles", nargs="+", choices=tuple(contracts.PROFILES), default=["Rapida"])
    create.add_argument("--speaker-mode", choices=("auto", "exact"), default="auto")
    execute_command = commands.add_parser("run", help="Resume verified receipts; preserve incomplete attempts; private Keychain credential only after reference/model gates.")
    execute_command.add_argument("--bridge", type=Path, required=True)
    execute_command.add_argument("--timeout", type=int, default=1800)
    for command in (create, execute_command):
        for name in ("source-root", "runtime", "reference-root", "model-specs"): command.add_argument("--" + name, type=Path, required=True)
    commands.add_parser("calibrate")
    commands.add_parser("seal")
    check = commands.add_parser("verify")
    check.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        root = args.evidence_root.resolve(strict=True)
        if args.command in ("plan", "run"):
            context = [args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text())]
            if args.command == "plan":
                document = prepare(root, args.snapshot, args.catalog.resolve(strict=True), args.licenses.resolve(strict=True), args.profiles, args.speaker_mode, *context)
                print(json.dumps({"status": "released_voice_plan_ready", "jobs": len(document["jobs"])}))
            else: execute(root, args.snapshot, *context, args.bridge, args.timeout)
        elif args.command == "calibrate": print(json.dumps(calibrate(root, args.snapshot)))
        elif args.command == "seal": print(json.dumps({"status": "released_voice_sealed", "sha256": seal(root, args.snapshot)}))
        else:
            report = verify(root, args.snapshot, args.sha256)
            print(json.dumps({"status": "released_voice_verified", "jobs": report["jobs"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "released_voice_benchmark_incomplete", "local_human_review_required": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

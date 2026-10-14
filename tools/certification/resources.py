"""Archive-only phase/resource envelopes; no promotion budgets or speedup claims."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, freeze, profiling, released_baseline
from tools.evaluation import voice_baseline


def nonnegative(value):
    return type(value) is int and value >= 0


def measurements(document, voices=False):
    pid = document.get("pid")
    corpus.require(type(pid) is int and pid > 0 and isinstance(document.get("run_id"), str) and bool(document["run_id"]) and document.get("timing_kind") == "synchronized_completion", "resources_span_identity_invalid")
    rows = document.get("spans", [])
    corpus.require(isinstance(rows, list) and bool(rows), "resources_spans_missing")
    values = {}
    unavailable = 0
    for span in rows:
        stage = span.get("stage")
        corpus.require(isinstance(stage, str) and bool(stage) and span.get("pid") == pid and span.get("status") == "synchronized" and span.get("drain_calls") == 2, "resources_span_invalid")
        clocks = [span.get(name) for name in ("api_wall_ns", "post_return_wait_ns", "completion_wall_ns", "cpu_process_ns")]
        corpus.require(all(nonnegative(value) for value in clocks) and clocks[0] + clocks[1] == clocks[2], "resources_clock_invalid")
        for name, value in zip(("api_wall_ns", "post_return_wait_ns", "completion_wall_ns", "cpu_process_ns"), clocks):
            key = stage + "/" + name
            values[key] = values.get(key, 0) + value
        samples = span.get("memory_samples")
        corpus.require(isinstance(samples, list) and bool(samples) and nonnegative(span.get("sample_cadence_ns")) and span["sample_cadence_ns"] > 0, "resources_memory_samples_missing")
        for sample in samples:
            corpus.require(nonnegative(sample.get("sample_time_ns")) and all(nonnegative(value) for name, value in sample.items() if name.endswith("_bytes")), "resources_memory_sample_invalid")
        corpus.require(all(left["sample_time_ns"] <= right["sample_time_ns"] for left, right in zip(samples, samples[1:])), "resources_sample_clock_invalid")
        memory = profiling.memory_summary(samples)
        corpus.require(memory == span.get("memory"), "resources_memory_summary_changed")
        unavailable += memory["unavailable_samples"]
        for name, value in memory.items():
            if name.endswith("_bytes") and value is not None:
                key = stage + "/" + name
                values[key] = max(values.get(key, 0), value)
    required = {"decode", "asr_mlx", "alignment", "export_write"}
    if voices: required.update(("diarization", "speaker_assignment"))
    corpus.require(required.issubset({span["stage"] for span in rows}), "resources_required_phase_missing")
    return {"values": values, "unavailable_memory_samples": unavailable}


def summarize(rows):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        corpus.require(row["split"] in ("calibration", "evaluation") and type(row["replica"]) is int and row["replica"] in range(3), "resources_split_or_replica_invalid")
        samples = grouped[row["stratum"]][(row["split"], row["asset_id"])]
        corpus.require(row["replica"] not in samples, "resources_duplicate_replica")
        samples[row["replica"]] = row["values"]
    result = {}
    for stratum, recordings in sorted(grouped.items()):
        corpus.require(all(set(samples) == {0, 1, 2} for samples in recordings.values()), "resources_replica_missing")
        corpus.require(all(len([key for key in recordings if key[0] == split]) >= 3 for split in ("calibration", "evaluation")), "resources_recordings_missing")
        identities = defaultdict(set)
        for split, asset in recordings: identities[asset].add(split)
        corpus.require(all(len(splits) == 1 for splits in identities.values()), "resources_recording_split_overlap")
        keys = [set(sample) for samples in recordings.values() for sample in samples.values()]
        corpus.require(bool(keys[0]) and all(key == keys[0] for key in keys), "resources_metric_coverage_changed")
        metrics = {}
        for metric in sorted(keys[0]):
            partitions = defaultdict(list)
            for (split, asset), samples in recordings.items():
                values = [sample[metric] for sample in samples.values()]
                corpus.require(all(nonnegative(value) for value in values), "resources_value_invalid")
                partitions[split].append({"mean": sum(values) / len(values), "maximum": max(values)})
            calibration = partitions["calibration"]
            evaluation = partitions["evaluation"]
            bound = max(row["maximum"] for row in calibration)
            metrics[metric] = {"calibration_recordings": len(calibration), "evaluation_recordings": len(evaluation), "calibration_observed_maximum": bound, "calibration_recording_mean": sum(row["mean"] for row in calibration) / len(calibration), "evaluation_observed_maximum": max(row["maximum"] for row in evaluation), "evaluation_recording_mean": sum(row["mean"] for row in evaluation) / len(evaluation), "evaluation_recordings_exceeding_calibration_maximum": sum(row["maximum"] > bound for row in evaluation)}
        result[stratum] = metrics
    corpus.require(bool(result), "resources_rows_missing")
    return result


def build(root, snapshot, checksum, kind):
    corpus.require(kind in ("text", "voices"), "resources_kind_invalid")
    module = released_baseline if kind == "text" else voice_baseline
    archive = module.verify(root, snapshot, checksum)
    plan, _, _ = module.load(root, snapshot)
    def retained(name):
        path = corpus.safe_target(root, name)
        payload = path.read_bytes()
        corpus.require(archive["artifacts"].get(name) == hashlib.sha256(payload).hexdigest(), "resources_input_not_sealed")
        return json.loads(payload)
    rows, unavailable = [], 0
    for job in plan["jobs"]:
        if kind == "text":
            measurement = retained(snapshot + "/jobs/" + job["id"] + "/measurement.json")
            stratum = job["stratum"]
        else:
            receipt = retained(snapshot + "/receipts/" + job["id"] + ".json")
            measurement = retained(receipt["measurement_path"])
            stratum = job["quality"] + "/AMI/en"
        span_names = [name for name in measurement["artifacts"] if name.endswith("/spans.json")]
        corpus.require(len(span_names) == 1, "resources_span_artifact_ambiguous")
        observed = measurements(retained(span_names[0]), voices=kind == "voices")
        rows.append({"asset_id": job["asset_id"], "stratum": stratum, "split": job["split"], "replica": job["replica"], "values": observed["values"]})
        unavailable += observed["unavailable_memory_samples"]
    strata = summarize(rows)
    freeze.verify_seal(root, snapshot + "/sealed.json", checksum)
    return {"schema_version": 1, "scope": "instrumented_phase_resource_observations", "kind": kind, "benchmark_snapshot": snapshot, "benchmark_sha256": checksum, "paired_jobs": len(rows), "strata": strata, "unavailable_memory_samples": unavailable, "policy": {"replicates": 3, "recording_weight": "equal", "calibration_statistic": "observed_maximum_only", "evaluation_policy": "diagnostic_never_retune", "repeated_stage_timing": "sum_calls_within_stage_only", "memory_policy": "separate_domains_never_sum", "rss_scope": "process_lifetime_high_water", "mps_peak_is_lower_bound": True, "nested_phase_times_additive": False}, "isolation_verified": False, "stream_inventory_complete": False, "resource_budgets_certified": False, "speedup_comparison_eligible": False, "promotion_allowed": False}


def verify(root, destination, checksum):
    freeze.verify_seal(root, destination, checksum)
    report = json.loads(corpus.safe_target(root, destination).read_text())
    corpus.require(report == build(root, report["benchmark_snapshot"], report["benchmark_sha256"], report["kind"]), "resources_report_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--snapshot", help="Sealed benchmark snapshot.")
    parser.add_argument("--benchmark-sha256", help="Externally pinned benchmark checksum.")
    parser.add_argument("--kind", choices=("text", "voices"))
    parser.add_argument("--report", required=True, help="Exclusive output relative to evidence root.")
    parser.add_argument("--verify-sha256", help="Verify an existing report and reconstruct its resource observations.")
    args = parser.parse_args(argv)
    if args.verify_sha256 and any((args.snapshot, args.benchmark_sha256, args.kind)): parser.error("verification excludes collection arguments")
    if not args.verify_sha256 and not all((args.snapshot, args.benchmark_sha256, args.kind)): parser.error("collection requires --snapshot, --benchmark-sha256 and --kind")
    try:
        if args.verify_sha256:
            verify(args.evidence_root, args.report, args.verify_sha256)
            print(json.dumps({"status": "resource_observations_verified", "resource_budgets_certified": False}))
        else:
            report = build(args.evidence_root, args.snapshot, args.benchmark_sha256, args.kind)
            checksum = freeze.seal(args.evidence_root, args.report, report)
            print(json.dumps({"status": "resource_observations_sealed", "sha256": checksum, "paired_jobs": report["paired_jobs"], "resource_budgets_certified": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError):
        print(json.dumps({"status": "resource_observations_invalid", "resource_budgets_certified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Attribute a separate trace to one original alignment operation; internal CPU work remains a separate question."""

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, dataset_reference, freeze, manifest, released_baseline, run, xctrace_review
from tools.tracing import pipeline_review

OPERATORS = ("wav2vec2", "alignment_log_softmax", "ctc_trellis", "ctc_backtrack")


def line(child, timeout):
    corpus.require(select.select([child.stdout], [], [], timeout)[0], "isolated_protocol_timeout")
    raw = child.stdout.readline(65537)
    corpus.require(len(raw) <= 65536 and raw.endswith(b"\n"), "isolated_protocol_invalid")
    return json.loads(raw)


def record(runtime, source, folder, request, hf_cache, torch_cache, timeout):
    environment = manifest.clean_environment(runtime.resolve().parent.parent, os.environ)
    environment.update(HF_HUB_CACHE=str(hf_cache), TORCH_HOME=str(torch_cache), PYTORCH_ENABLE_MPS_FALLBACK="0")
    child = subprocess.Popen([str(runtime), "-I", "-B", "-u", str(source / "tools/tracing/isolated_entry.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, process_group=0)
    tracer = None
    try:
        child.stdin.write(json.dumps(request).encode() + b"\n")
        child.stdin.flush()
        corpus.require(line(child, timeout) == {"ready": True, "pid": child.pid, "run_id": request["run_id"]}, "isolated_ready_invalid")
        with (folder / "recorder.log").open("xb") as log:
            tracer = subprocess.Popen(["/usr/bin/xcrun", "xctrace", "record", "--template", "Metal System Trace", "--no-prompt", "--run-name", request["run_id"], "--output", str(folder / "operator.trace"), "--time-limit", str(timeout) + "s", "--attach", str(child.pid)], stdout=log, stderr=subprocess.STDOUT, env=environment, process_group=0)
            corpus.require(line(child, timeout) == {"operator_completed": request["operator"], "pid": child.pid, "run_id": request["run_id"]}, "isolated_completion_invalid")
            if tracer.poll() is None: tracer.send_signal(signal.SIGINT)
            tracer.wait(timeout=60)
        corpus.require(tracer.returncode == 0, "isolated_recorder_failed")
        child.stdin.write(b"trace_saved\n")
        child.stdin.close()
        child.stdin = None
        output, _ = child.communicate(timeout=60)
        corpus.require(child.returncode == 0 and not output, "isolated_child_failed")
        freeze.seal(folder, "controller.json", {"pid": child.pid, "run_id": request["run_id"], "operator": request["operator"], "child_returncode": 0, "recorder_returncode": 0})
    finally:
        run.stop_child(child)
        if tracer is not None: run.stop_child(tracer)


def review(toc, gpu, operator, controller):
    corpus.require(operator.get("status") == "isolated_operator_completed" and operator.get("operator") in OPERATORS and controller.get("operator") == operator["operator"] and controller.get("pid") == operator.get("pid") and controller.get("run_id") == operator.get("run_id") and controller.get("child_returncode") == 0 and controller.get("recorder_returncode") == 0, "isolated_trace_identity_invalid")
    expected_backend = "cpu" if operator["operator"] in ("ctc_trellis", "ctc_backtrack") else "mps"
    corpus.require(operator.get("output_identical") is True and operator["original_output_sha256"] == operator["observed_output_sha256"] and operator.get("output_device") in (("cpu", "cpu:0") if expected_backend == "cpu" else ("mps", "mps:0")) and operator.get("setup_and_original_reference_before_trace") is True and operator.get("result_readback_after_trace_saved") is True and operator.get("mps_fallback_allowed") is False, "isolated_operator_scope_invalid")
    intervals, foreign, unattributed = pipeline_review.extract_intervals(toc, gpu, operator["pid"])
    audit = operator["dispatch_audit"]
    mps = sum(audit["tensor_device_operations"].get(name, 0) for name in ("mps", "mps:0"))
    cpu = sum(audit["tensor_device_operations"].get(name, 0) for name in ("cpu", "cpu:0"))
    corpus.require((cpu > 0 if expected_backend == "cpu" else mps > 0) and audit["aten_operations"] and audit["numerical_values_retained"] is False, "isolated_tensor_dispatch_missing")
    status = "isolated_operator_cpu_dispatch_observed" if expected_backend == "cpu" and cpu > 0 else "isolated_operator_gpu_activity_observed" if intervals else "isolated_operator_gpu_activity_missing"
    return {"scope": "selected_original_operator_workload", "status": status, "observed_tensor_backend": expected_backend, "operator": operator["operator"], "run_id": operator["run_id"], "pid": operator["pid"], "target_compute_intervals": intervals, "foreign_process_rows_excluded": foreign, "unattributed_process_rows_excluded": unattributed, "gpu_activity_observed": bool(intervals), "individual_operator_dispatch_verified": cpu > 0 and mps == 0 if expected_backend == "cpu" else bool(intervals), "observed_cpu_tensor_operations": cpu, "observed_scalar_readbacks": audit["scalar_readbacks"], "all_numerical_gpu_verified": False, "placement_gate_eligible": False, "gpu_placement_certified": False, "quality_certified": False, "benchmark_timing_eligible": False}


def capture(root, snapshot, catalog, licenses, case_id, operator, source, runtime, reference_root, specs, timeout=180):
    corpus.require(operator in OPERATORS and Path(snapshot).name == snapshot and snapshot not in ("", ".", "..") and 30 <= timeout <= 600, "isolated_settings_invalid")
    folder = corpus.safe_target(root, snapshot)
    corpus.require(not folder.exists(), "isolated_destination_exists")
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    asset = next(asset for asset in catalog["assets"] if asset["id"] == case["asset_id"])
    legal = next(legal for legal in licenses["licenses"] if legal["id"] == asset["license_id"])
    corpus.verify_license(asset, legal, root, "download")
    annotation = dataset_reference.reference(root, asset)
    corpus.require(manifest.sha256(corpus.safe_target(root, case["reference_path"])) == case["reference_sha256"] and json.loads(corpus.safe_target(root, case["reference_path"]).read_text()) == annotation, "isolated_reference_invalid")
    fingerprint, context = released_baseline.identity(source, runtime, reference_root, specs)
    corpus.require(len([spec for spec in specs if spec.get("role") == "alignment_" + asset["language"]]) == 1, "isolated_model_role_missing")
    hf_cache, torch_cache = run.model_cache_roots(specs)
    corpus.require(hf_cache is not None and torch_cache is not None, "isolated_cache_missing")
    harness = manifest.source_identity(source, [str(path.relative_to(source)) for path in (source / "tools/tracing").glob("*.py")])
    corpus.require(harness["status"] == "identified", "isolated_harness_missing")
    run_id = str(uuid.uuid4())
    context.update(isolated_harness=harness, run_id=run_id, operator=operator, case_id=case_id, audio_sha256=asset["sha256"], input_scope="first_up_to_eight_seconds_before_trace", timing_scope="warm_diagnostic_not_pipeline_replica")
    freeze.seal(root, snapshot + "/context.json", context)
    freeze.seal(root, snapshot + "/executed-harness.json", {item["name"]: {"sha256": item["sha256"], "content": (source / item["name"]).read_text()} for item in harness["files"]})
    request = {"run_id": run_id, "operator": operator, "source_root": str(source), "folder": str(folder), "audio": str(corpus.safe_target(root, asset["path"])), "language": asset["language"], "reference_text": annotation["transcript"]}
    record(runtime, source, folder, request, hf_cache, torch_cache, timeout)
    corpus.require(released_baseline.identity(source, runtime, reference_root, specs)[0] == fingerprint and manifest.source_identity(source, [item["name"] for item in harness["files"]])["digest"] == harness["digest"], "isolated_execution_identity_changed")
    with (folder / "export.log").open("xb") as log:
        for arguments in (["--toc", "--output", str(folder / "toc.xml")], ["--xpath", '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]', "--output", str(folder / "gpu.xml")]):
            subprocess.run(["/usr/bin/xcrun", "xctrace", "export", "--input", str(folder / "operator.trace"), *arguments], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    result = review(xctrace_review.parse_xml(folder / "toc.xml"), xctrace_review.parse_xml(folder / "gpu.xml"), json.loads((folder / "operator.json").read_text()), json.loads((folder / "controller.json").read_text()))
    result["artifacts"] = released_baseline.artifact_hashes(root, snapshot)
    return result, freeze.seal(root, snapshot + "/review.json", result)



def verify(root, snapshot, checksum):
    freeze.verify_seal(root, snapshot + "/review.json", checksum)
    folder = corpus.safe_target(root, snapshot)
    report = json.loads((folder / "review.json").read_text())
    corpus.require(report.get("scope") == "selected_original_operator_workload" and report.get("all_numerical_gpu_verified") is False and report.get("placement_gate_eligible") is False, "isolated_review_scope_invalid")
    corpus.require(report.get("artifacts") and all(manifest.sha256(corpus.safe_target(root, name)) == digest for name, digest in report["artifacts"].items()), "isolated_artifact_changed")
    context = json.loads((folder / "context.json").read_text())
    archive = json.loads((folder / "executed-harness.json").read_text())
    corpus.require(set(archive) == {item["name"] for item in context["isolated_harness"]["files"]}, "isolated_harness_archive_invalid")
    import hashlib
    for item in context["isolated_harness"]["files"]:
        retained = archive[item["name"]]
        corpus.require(retained["sha256"] == item["sha256"] and hashlib.sha256(retained["content"].encode()).hexdigest() == item["sha256"], "isolated_harness_archive_changed")
    rebuilt = review(xctrace_review.parse_xml(folder / "toc.xml"), xctrace_review.parse_xml(folder / "gpu.xml"), json.loads((folder / "operator.json").read_text()), json.loads((folder / "controller.json").read_text()))
    if "observed_tensor_backend" not in report:
        producer = archive["tools/tracing/isolated_operator.py"]["content"]
        corpus.require(report["operator"] in ("wav2vec2", "alignment_log_softmax") and "observed_tensor_backend" not in producer, "isolated_legacy_schema_invalid")
        rebuilt.pop("observed_tensor_backend")
    corpus.require(all(report.get(key) == value for key, value in rebuilt.items()), "isolated_review_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-root", "catalog", "licenses", "source-root", "runtime", "reference-root", "model-specs"): parser.add_argument("--" + name, type=Path, required=name == "evidence-root")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--operator", choices=OPERATORS)
    parser.add_argument("--verify-sha256", help="Optional read-only archived verification; needs only evidence-root and snapshot.")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if not args.verify_sha256:
        missing = [name for name in ("catalog", "licenses", "source_root", "runtime", "reference_root", "model_specs", "case_id", "operator") if getattr(args, name) is None]
        if missing: parser.error("capture requires " + ", ".join(name.replace("_", "-") for name in missing))
    try:
        if args.verify_sha256:
            report = verify(args.evidence_root.resolve(strict=True), args.snapshot, args.verify_sha256)
            print(json.dumps({"status": "isolated_operator_review_verified", "operator": report["operator"], "all_numerical_gpu_verified": False}))
            return 0
        report, checksum = capture(args.evidence_root.resolve(strict=True), args.snapshot, json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text()), args.case_id, args.operator, args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text()), args.timeout)
        print(json.dumps({"status": report["status"], "sha256": checksum, "operator": args.operator, "target_compute_intervals": len(report["target_compute_intervals"]), "all_numerical_gpu_verified": False}))
        return 0 if report["individual_operator_dispatch_verified"] else 2
    except (ValueError, OSError, KeyError, TypeError, StopIteration, subprocess.SubprocessError):
        print(json.dumps({"status": "isolated_operator_incomplete", "gpu_placement_certified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Capture and review a separate CLI Metal System Trace for one source-verified pipeline process."""

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, dataset_reference, freeze, manifest, profiling, provision, released_baseline, run, xctrace_review
from tools.tracing import audio_clip, pipeline_review
from tools.tracing.start_notification import StartNotification


def finish_recorder(tracer, grace=60, target_alive=False):
    if target_alive:
        if tracer.poll() is None: tracer.send_signal(signal.SIGINT)
        return tracer.wait(timeout=60)
    try:
        return tracer.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if tracer.poll() is None: tracer.send_signal(signal.SIGINT)
        return tracer.wait(timeout=60)


def wait_recorder_ready(tracer, path, timeout=60, clock=time.monotonic, pause=time.sleep):
    deadline = clock() + timeout
    while clock() < deadline:
        corpus.require(tracer.poll() is None, "pipeline_trace_recorder_exited_before_attach")
        if "Ctrl-C to stop the recording" in path.read_text(): return
        pause(0.1)
    raise corpus.CorpusError("pipeline_trace_recorder_attach_timeout")


def record(runtime, source, settings, audio, folder, run_id, hf_cache, torch_cache, token, timeout):
    runtime = runtime.resolve(strict=True)
    environment = manifest.clean_environment(runtime.resolve().parent.parent, os.environ)
    environment.update(HF_HUB_CACHE=str(hf_cache), TORCH_HOME=str(torch_cache))
    arguments = [str(runtime), "-I", "-B", "-u", str(source / "tools/tracing/pipeline_entry.py"), "--source-root", str(source), "--spans", str(folder / "spans.json"), "--run-id", run_id]
    arguments.append("--controlled-start")
    if settings["voices"]: arguments.extend(["--turns", str(folder / "turns.json")])
    child = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, process_group=0, bufsize=0)
    tracer = None
    request = dict(settings, audio=str(audio), output=str(folder / "output"))
    if token is not None: request["token"] = token
    try:
        child.stdin.write(json.dumps(request).encode() + b"\n")
        child.stdin.flush()
        request.clear()
        corpus.require(select.select([child.stdout], [], [], 20)[0], "pipeline_trace_ready_timeout")
        ready = json.loads(child.stdout.readline())
        corpus.require(ready == {"trace_ready": True, "pid": child.pid, "run_id": run_id}, "pipeline_trace_ready_invalid")
        with (folder / "recorder.log").open("xb") as log, StartNotification("org.whisperx.trace." + run_id) as notification:
            tracer = subprocess.Popen(["/usr/bin/xcrun", "xctrace", "record", "--template", "Metal System Trace", "--no-prompt", "--run-name", run_id, "--output", str(folder / "pipeline.trace"), "--time-limit", str(timeout) + "s", "--notify-tracing-started", notification.name, "--attach", str(child.pid)], stdout=log, stderr=subprocess.STDOUT, env=environment, process_group=0)
            notification.wait()
            wait_recorder_ready(tracer, folder / "recorder.log")
            child.stdin.write(b"trace_attached\n")
            child.stdin.flush()
            events, size = [], 0
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                corpus.require(remaining > 0 and select.select([child.stdout], [], [], remaining)[0], "pipeline_trace_completion_timeout")
                line = child.stdout.readline(65537)
                size += len(line)
                corpus.require(line.endswith(b"\n") and len(line) <= 65536 and size <= 1024 * 1024, "pipeline_trace_child_protocol_invalid")
                event = json.loads(line)
                if "trace_worker_completed" in event:
                    corpus.require(event == {"trace_worker_completed": 0, "pid": child.pid, "run_id": run_id}, "pipeline_trace_child_failed")
                    break
                events.append(event)
            contracts.validate_events(events, "completed", str(folder / "output"))
            corpus.require(not events[-1]["warnings"], "pipeline_trace_pipeline_warning")
            freeze.seal(folder, "events.json", events)
            finish_recorder(tracer, target_alive=True)
        corpus.require(tracer.returncode == 0, "pipeline_trace_recorder_failed")
        child.stdin.write(b"trace_saved\n")
        child.stdin.close()
        child.stdin = None
        output, _ = child.communicate(timeout=60)
        corpus.require(child.returncode == 0 and not output, "pipeline_trace_child_failed")
        freeze.seal(folder, "controller.json", {"status": "diagnostic_pipeline_completed", "pid": child.pid, "child_returncode": child.returncode, "run_id": run_id, "voices": settings["voices"], "trace_mode": "owned_process_attach", "benchmark_timing_eligible": False})
    finally:
        request.clear()
        try:
            run.stop_child(child)
            if tracer is not None and tracer.poll() is None:
                tracer.terminate()
                tracer.wait(timeout=60)
        finally:
            freeze.seal(folder, "capture-outcome.json", {"pid": child.pid, "run_id": run_id, "child_returncode": child.returncode, "recorder_returncode": tracer.returncode if tracer is not None else None, "controller_completed": (folder / "controller.json").is_file()})


def capture(root, snapshot, catalog, licenses, case_id, quality, voices, source, runtime, reference_root, specs, bridge=None, timeout=600, max_audio_seconds=None):
    corpus.require(max_audio_seconds is None or type(max_audio_seconds) is int and 1 <= max_audio_seconds <= 120, "trace_audio_prefix_limit_invalid")
    corpus.require(Path(snapshot).name == snapshot and snapshot not in ("", ".", ".."), "pipeline_trace_snapshot_invalid")
    folder = corpus.safe_target(root, snapshot)
    corpus.require(not folder.exists() and 30 <= timeout <= 1800, "pipeline_trace_settings_invalid")
    case = next(item for item in catalog["cases"] if item["id"] == case_id)
    asset = next(item for item in catalog["assets"] if item["id"] == case["asset_id"])
    legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
    corpus.verify_license(asset, legal, root, "download")
    annotation = dataset_reference.reference(root, asset)
    corpus.require(manifest.sha256(corpus.safe_target(root, case["reference_path"])) == case["reference_sha256"] and json.loads(corpus.safe_target(root, case["reference_path"]).read_text()) == annotation and (not voices or bool(annotation["turns"])), "pipeline_trace_reference_invalid")
    fingerprint, context = released_baseline.identity(source, runtime, reference_root, specs)
    roles = {role for model in context["model_inventory"] for role in model.get("roles", [])}
    roles.update(spec.get("role") for spec in specs)
    required_roles = {"alignment_" + asset["language"]}
    if voices: required_roles.update(("speaker_segmentation", "speaker_embedding", "speaker_plda"))
    corpus.require(required_roles.issubset(roles) and len([spec for spec in specs if spec.get("repository") == contracts.PROFILES[quality]["mlx"]]) == 1, "pipeline_trace_models_missing")
    trace_harness = manifest.source_identity(source, [str(path.relative_to(source)) for path in (source / "tools/tracing").glob("*.py")] + ["tools/evaluation/voice_adapter.py"])
    corpus.require(trace_harness["status"] == "identified" and (not voices or bridge is not None), "pipeline_trace_harness_or_bridge_missing")
    hf_cache, torch_cache = run.model_cache_roots(specs)
    corpus.require(hf_cache is not None and torch_cache is not None, "pipeline_trace_cache_missing")
    settings = {"quality": quality, "language": asset["language"], "voices": voices, "hints": ""}
    contracts.validate_settings(settings)
    run_id = str(uuid.uuid4())
    context.update(trace_harness=trace_harness, run_id=run_id, settings=settings, case_id=case_id, audio_path=asset["path"], audio_sha256=asset["sha256"])
    freeze.seal(root, snapshot + "/executed-harness.json", {item["name"]: {"sha256": item["sha256"], "content": (source / item["name"]).read_text()} for item in trace_harness["files"]})
    selected_audio = corpus.safe_target(root, asset["path"])
    if max_audio_seconds is not None:
        selected_audio = folder / "diagnostic-input.wav"
        context["input_scope"] = audio_clip.prefix(corpus.safe_target(root, asset["path"]), selected_audio, max_audio_seconds)
    freeze.seal(root, snapshot + "/context.json", context)
    token = None
    try:
        token = provision.read_credential(bridge) if voices else None
        record(runtime, source, settings, selected_audio, folder, run_id, hf_cache, torch_cache, token, timeout)
    finally:
        token = None
    corpus.require(released_baseline.identity(source, runtime, reference_root, specs)[0] == fingerprint and manifest.source_identity(source, [item["name"] for item in trace_harness["files"]])["digest"] == trace_harness["digest"], "pipeline_trace_execution_identity_changed")
    with (folder / "export.log").open("xb") as log:
        for arguments in (["--toc", "--output", str(folder / "toc.xml")], ["--xpath", '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]', "--output", str(folder / "gpu.xml")]):
            subprocess.run(["/usr/bin/xcrun", "xctrace", "export", "--input", str(folder / "pipeline.trace"), *arguments], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120, env=manifest.clean_environment(runtime.resolve().parent.parent, os.environ))
    controller = json.loads((folder / "controller.json").read_text())
    spans = json.loads((folder / "spans.json").read_text())
    report = pipeline_review.review(xctrace_review.parse_xml(folder / "toc.xml"), xctrace_review.parse_xml(folder / "gpu.xml"), spans, controller)
    report.update(reference_policy="released-dataset", local_human_review_required=False, audio_sha256=asset["sha256"], trace_sha256=profiling.artifact_identity(root, snapshot + "/pipeline.trace"))
    if max_audio_seconds is not None:
        report.update(input_scope=context["input_scope"], source_audio_sha256=asset["sha256"], audio_sha256=context["input_scope"]["sha256"])
    report["artifacts"] = released_baseline.artifact_hashes(root, snapshot)
    checksum = freeze.seal(root, snapshot + "/review.json", report)
    return report, checksum



def verify(root, snapshot, checksum):
    freeze.verify_seal(root, snapshot + "/review.json", checksum)
    folder = corpus.safe_target(root, snapshot)
    report = json.loads((folder / "review.json").read_text())
    corpus.require(report.get("scope") == "selected_pipeline_process_only" and report.get("individual_operator_dispatch_verified") is False and report.get("gpu_placement_certified") is False, "pipeline_trace_review_scope_invalid")
    corpus.require(report.get("artifacts") and all(manifest.sha256(corpus.safe_target(root, name)) == digest for name, digest in report["artifacts"].items()), "pipeline_trace_artifact_changed")
    context = json.loads((folder / "context.json").read_text())
    archive = json.loads((folder / "executed-harness.json").read_text())
    corpus.require(set(archive) == {item["name"] for item in context["trace_harness"]["files"]}, "pipeline_trace_harness_archive_invalid")
    import hashlib
    for item in context["trace_harness"]["files"]:
        retained = archive[item["name"]]
        corpus.require(retained["sha256"] == item["sha256"] and hashlib.sha256(retained["content"].encode()).hexdigest() == item["sha256"], "pipeline_trace_harness_archive_changed")
    rebuilt = pipeline_review.review(xctrace_review.parse_xml(folder / "toc.xml"), xctrace_review.parse_xml(folder / "gpu.xml"), json.loads((folder / "spans.json").read_text()), json.loads((folder / "controller.json").read_text()))
    corpus.require(all(report.get(key) == value for key, value in rebuilt.items()), "pipeline_trace_review_changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-root", "catalog", "licenses", "source-root", "runtime", "reference-root", "model-specs"): parser.add_argument("--" + name, type=Path, required=name == "evidence-root")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--verify-sha256", help="Optional archived review verification; needs only evidence-root and snapshot, without inference or live-environment certification.")
    parser.add_argument("--quality", choices=tuple(contracts.PROFILES), default="Rapida")
    parser.add_argument("--voices", action="store_true", help="Optional actual voice API phases and returned turns; requires released speaker reference and private bridge.")
    parser.add_argument("--max-audio-seconds", type=int, help="Optional 1..120 second prefix of verified mono PCM16/16kHz audio for a separate bounded trace; source audio and benchmark replicas stay intact.")
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--timeout", type=int, default=600, help="30 to 1800 seconds for the separate diagnostic capture, not a benchmark replica.")
    args = parser.parse_args(argv)
    if not args.verify_sha256:
        missing = [name for name in ("catalog", "licenses", "source_root", "runtime", "reference_root", "model_specs", "case_id") if getattr(args, name) is None]
        if missing: parser.error("capture requires " + ", ".join(name.replace("_", "-") for name in missing))
        if args.voices and args.bridge is None: parser.error("voices requires bridge")
    try:
        if args.verify_sha256:
            report = verify(args.evidence_root.resolve(strict=True), args.snapshot, args.verify_sha256)
            print(json.dumps({"status": "pipeline_trace_review_verified", "target_compute_intervals": len(report["target_compute_intervals"]), "individual_operator_dispatch_verified": False}))
            return 0
        report, checksum = capture(args.evidence_root.resolve(strict=True), args.snapshot, json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text()), args.case_id, args.quality, args.voices, args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text()), args.bridge, args.timeout, args.max_audio_seconds)
        print(json.dumps({"status": report["status"], "sha256": checksum, "target_compute_intervals": len(report["target_compute_intervals"]), "individual_operator_dispatch_verified": False, "gpu_placement_certified": False}))
        return 0 if report["gpu_activity_observed"] else 2
    except (ValueError, OSError, KeyError, TypeError, StopIteration, subprocess.SubprocessError):
        print(json.dumps({"status": "pipeline_trace_incomplete", "gpu_placement_certified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

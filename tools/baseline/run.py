"""Offline full-gate preflight and paired original/adapter GUI worker runs in private sandboxes."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, manifest, provision, metrics, freeze, alignment_inventory, vad_inventory


class RunError(ValueError):
    pass


def model_cache_roots(model_specs):
    roots = {"huggingface": set(), "torch_hub": set()}
    for spec in model_specs:
        transport = spec.get("transport", "huggingface")
        if transport == "torch_hub_source": transport = "torch_hub"
        if transport not in roots:
            raise RunError("model_cache_transport_unsupported")
        roots[transport].add(Path(spec["cache_root"]).resolve())
    if any(len(group) > 1 for group in roots.values()):
        raise RunError("model_cache_topology_unresolved")
    return tuple(next(iter(roots[name]), None) for name in ("huggingface", "torch_hub"))


def preflight(catalog, coverage, licenses, evidence_root, model_specs, alignment_selection=None, verify_hf_alignment=False, vad_selection=None):
    corpus_report = corpus.preflight(catalog, coverage, licenses, evidence_root, evidence_root)
    models = [provision.inspect_spec(spec) for spec in model_specs]
    missing = list(corpus_report["missing"])
    if not models:
        missing.append({"code": "model_inventory_missing"})
    for model in models:
        if model["status"] != "files_verified":
            missing.append({"code": model.get("code", "model_not_verified")})
    required_repositories = {mapping["mlx"] for mapping in contracts.PROFILES.values()} | {"pyannote/speaker-diarization-community-1"}
    declared_repositories = {spec.get("repository") for spec in model_specs}
    for repository in sorted(required_repositories - declared_repositories):
        missing.append({"code": "required_model_undeclared", "repository": repository})
    required_roles = {"alignment_" + language for language in corpus.REPRESENTATIVES} | {"speaker_segmentation", "speaker_embedding", "speaker_plda", "vad"}
    verified_roles = set()
    for spec, model in zip(model_specs, models):
        if model["status"] == "files_verified":
            verified_roles.update(model.get("roles", [spec["role"]] if isinstance(spec.get("role"), str) else []))
    for role in sorted(required_roles - verified_roles):
        missing.append({"code": "required_model_role_undeclared", "role": role})
    for spec in model_specs:
        if spec.get("license_verified") is not True or not spec.get("license_source", "").startswith("https://"):
            missing.append({"code": "model_license_unverified", "repository": spec["repository"]})
        if spec.get("transport") == "torch_hub_source":
            if not vad_inventory.matches_spec(spec, vad_selection): missing.append({"code": "worker_vad_selection_unverified", "repository": spec["repository"]})
            continue
        if spec.get("transport") == "torch_hub":
            if not alignment_inventory.matches_spec(spec, alignment_selection):
                missing.append({"code": "worker_alignment_selection_unverified", "repository": spec["repository"]})
            continue
        if verify_hf_alignment and str(spec.get("role", "")).startswith("alignment_") and not alignment_inventory.matches_spec(spec, alignment_selection):
            missing.append({"code": "worker_alignment_selection_unverified", "repository": spec["repository"]})
        try:
            reference = manifest.contained(Path(spec["cache_root"]), Path("models--" + spec["repository"].replace("/", "--")) / "refs/main")
            verified = reference.is_file() and reference.read_text().strip() == spec["revision"]
        except (OSError, manifest.BaselineError):
            verified = False
        if not verified:
            missing.append({"code": "worker_default_revision_unverified", "repository": spec["repository"]})
    try:
        model_cache_roots(model_specs)
    except RunError as error:
        missing.append({"code": str(error)})
    report = {"status": "prerequisite_missing" if missing else "prerequisites_verified", "missing": missing, "models": models, "gpu_placement_certified": False}
    if alignment_selection is not None: report["alignment_selection"] = alignment_selection
    if vad_selection is not None: report["vad_selection"] = vad_selection
    return report


def stop_child(child):
    try:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            if child.poll() is None: child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                child.kill()
            child.wait()
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    finally:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(child, name, None)
            if stream is not None:
                try: stream.close()
                except OSError: pass


def execute(runtime, arguments, request, timeout, cache_root=None, torch_home=None):
    runtime = Path(runtime).resolve()
    environment = manifest.clean_environment(runtime.parent.parent, os.environ)
    if cache_root is not None:
        environment["HF_HUB_CACHE"] = str(cache_root)
    if torch_home is not None:
        environment["TORCH_HOME"] = str(torch_home)
    child = subprocess.Popen([str(runtime), "-I", "-B", "-u", *map(str, arguments)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, process_group=0)
    try:
        output, _ = child.communicate(json.dumps(request).encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_child(child)
        raise RunError("job_interrupted") from None
    except KeyboardInterrupt:
        stop_child(child)
        raise RunError("job_cancelled") from None
    if len(output) > 1024 * 1024:
        raise RunError("worker_event_budget_exceeded")
    try:
        events = [json.loads(line) for line in output.decode().splitlines()]
        outcome = "completed" if child.returncode == 0 else "failed"
        contracts.validate_events(events, outcome, request["output"])
    except (ValueError, UnicodeError):
        raise RunError("worker_protocol_invalid") from None
    if outcome != "completed":
        raise RunError("worker_failed")
    return events


def semantic_bundle(output, events, basename):
    artifacts = {}
    for name in ("original.json", "aligned.json", "result.json"):
        path = corpus.safe_target(output, name)
        if path.exists():
            if path.stat().st_size > 64 * 1024 * 1024:
                raise RunError("result_budget_exceeded")
            artifacts[name] = json.loads(path.read_text())
    contracts.validate_artifacts(artifacts)
    exports = {}
    for extension in ("txt", "srt", "vtt"):
        path = corpus.safe_target(output, basename + "." + extension)
        if path.stat().st_size > 64 * 1024 * 1024:
            raise RunError("export_budget_exceeded")
        exports[extension] = path.read_bytes().hex()
    return {"artifacts": artifacts, "events": events, "exports": exports}


def paired_run(runtime, worker, source_root, asset_path, settings, run_root, token=None, timeout=3600, cache_root=None, synchronized=False, torch_home=None, seed=None):
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 32):
        raise RunError("diagnostic_seed_invalid")
    contracts.validate_settings(settings)
    run_root = Path(run_root)
    if run_root.exists():
        raise RunError("run_destination_exists")
    run_root.mkdir(parents=True)
    original, instrumented = run_root / "original", run_root / "instrumented"
    original.mkdir()
    instrumented.mkdir()
    base = {**settings, "audio": str(asset_path)}
    if settings["voices"]:
        if token is None:
            raise RunError("keychain_unavailable")
        base["token"] = token
    entrypoint = Path(__file__).with_name("offline_worker.py")
    seed_arguments = ["--seed", str(seed)] if seed is not None else []
    extra = {"torch_home": torch_home} if torch_home is not None else {}
    first = execute(runtime, [entrypoint, "--worker", worker, "--source-root", source_root, *seed_arguments], {**base, "output": str(original)}, timeout, cache_root, **extra)
    adapter = Path(__file__).with_name("adapter.py")
    adapter_arguments = [adapter, "--worker", worker, "--source-root", source_root, "--spans", run_root / "spans.json", "--run-id", run_root.parent.name]
    adapter_arguments.extend(seed_arguments)
    if synchronized:
        adapter_arguments.append("--synchronized")
    second = execute(runtime, adapter_arguments, {**base, "output": str(instrumented)}, timeout, cache_root, **extra)
    base.clear()
    left = semantic_bundle(original, first, asset_path.stem)
    right = semantic_bundle(instrumented, second, asset_path.stem)
    if not contracts.equivalent(left, right, str(original), str(instrumented)):
        raise RunError("paired_semantic_mismatch")
    backends = sorted({event["backend"] for event in first if "backend" in event})
    if "MLX / Metal" not in backends:
        raise RunError("required_mlx_backend_unobserved")
    return {"status": "paired_equivalent", "gpu_placement_certified": False, "quality_certified": False, "timing_kind": "cold_process_warm_cache", "warnings": first[-1]["warnings"], "observed_backends": backends, "pipeline_without_warnings": not first[-1]["warnings"], "seed": seed, "deterministic_algorithms_forced": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", choices=("gui-checkout", "gui-bundle"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True, help="Existing private corpus/evidence directory; no personal catalog or outputs.")
    parser.add_argument("--model-specs", type=Path, required=True)
    parser.add_argument("--verify-hf-alignment", action="store_true", help="Optional stronger offline gate: verify declared HF alignment defaults, pinned processor metadata and installed Transformers version in the selected runtime. No neural checkpoints are loaded; legacy Torch-only inspection stays unchanged without this flag.")
    parser.add_argument("--bridge", type=Path, help="Private Keychain pipe for voices; never token argv/env/file.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Offline corpus/model gate and Torch default/config inspection in the selected runtime; no checkpoint loading, network or Keychain.")
    sub.add_parser("provision", help="Explicit pinned download separate from measured runs; individual credentials via bridge only.")
    run = sub.add_parser("run", help="Full gate must pass before selected case; original and adapter in separate cold processes, shared warm disk cache, never warm-model claim.")
    run.add_argument("--case", required=True)
    run.add_argument("--timeout", type=int, default=3600)
    run.add_argument("--seed", type=int, help="Optional explicit paired bootstrap RNG seed; default legacy policy remains unchanged, and freeze must retain the selected seed policy.")
    run.add_argument("--synchronized", action="store_true", help="Instrumented process drains registered MLX/MPS work; dispatch still requires reviewed trace evidence.")
    evaluate = sub.add_parser("evaluate", help="Score a licensed natural_annotated case; parity_only is never natural accuracy.")
    evaluate.add_argument("--case-id", required=True)
    evaluate.add_argument("--result", required=True, help="Result JSON path relative to evidence root.")
    evaluate.add_argument("--speaker-turns", help="Optional actual pyannote turn intermediate, never inferred from dominant transcript labels.")
    evaluate.add_argument("--output", required=True)
    calibration = sub.add_parser("calibrate", help="Baseline-only seeded recording calibration; full corpus/trace gate required, candidate input rejected.")
    calibration.add_argument("--observations", required=True)
    calibration.add_argument("--contract", type=Path, required=True)
    calibration.add_argument("--evidence", required=True)
    calibration.add_argument("--output", required=True)
    seal = sub.add_parser("freeze", help="Write-once calibrated contract plus verified evidence identity; draft/missing evidence fails.")
    seal.add_argument("--contract", type=Path, required=True)
    seal.add_argument("--evidence", required=True)
    seal.add_argument("--output", required=True)
    for name in ("verify-freeze", "full-gate"):
        verify = sub.add_parser(name, help="Recheck frozen content and artifacts; full-gate also reruns live corpus/model prerequisites.")
        verify.add_argument("--frozen", required=True)
        verify.add_argument("--sha256", required=True)
        if name == "full-gate": verify.add_argument("--target", choices=("baseline", "native_gpu"), default="baseline")
    args = parser.parse_args(argv)
    if args.command == "run" and args.seed is not None and not 0 <= args.seed < 2 ** 32: parser.error("seed must be between 0 and 2**32-1")
    try:
        catalog, coverage, licenses, specs = [json.loads(path.read_text()) for path in (args.corpus, args.coverage, args.licenses, args.model_specs)]
        selection = None
        vad_selection = None
        if args.command in ("preflight", "run", "full-gate", "calibrate", "freeze") and (any(spec.get("transport") == "torch_hub" for spec in specs) or args.verify_hf_alignment):
            imports = args.source_root if args.baseline_source == "gui-checkout" else args.source_root / "Contents/Resources"
            selection = alignment_inventory.inspect_runtime(args.runtime, imports, args.model_specs, include_hf=True) if args.verify_hf_alignment else alignment_inventory.inspect_runtime(args.runtime, imports, args.model_specs)
        if args.command in ("preflight", "run", "full-gate", "calibrate", "freeze") and any(spec.get("transport") == "torch_hub_source" for spec in specs):
            imports = args.source_root if args.baseline_source == "gui-checkout" else args.source_root / "Contents/Resources"
            vad_selection = vad_inventory.inspect_runtime(args.runtime, imports, args.model_specs)
        if args.command == "evaluate":
            result = metrics.evaluate_case(catalog, licenses, args.evidence_root, args.case_id, args.result, args.speaker_turns)
            identity = freeze.seal(args.evidence_root, args.output, result)
            print(json.dumps({"status": "evaluated_not_certified", "case_id": args.case_id, "sha256": identity}))
            return 0
        if args.command in ("calibrate", "freeze"):
            prerequisites = preflight(catalog, coverage, licenses, args.evidence_root, specs, selection, verify_hf_alignment=args.verify_hf_alignment, vad_selection=vad_selection)
            if prerequisites["missing"]:
                print(json.dumps(prerequisites))
                return 2
            evidence = json.loads(corpus.safe_target(args.evidence_root, args.evidence).read_text())
            gate = freeze.evidence_gate(evidence, coverage, args.evidence_root)
            if gate["missing"]:
                print(json.dumps(gate))
                return 2
            contract = json.loads(args.contract.read_text())
            if args.command == "calibrate":
                observations = json.loads(corpus.safe_target(args.evidence_root, args.observations).read_text())
                result = freeze.calibrate(observations, contract)
                print(json.dumps({"state": result["state"], "contract_id": result["contract_id"], "sha256": freeze.seal(args.evidence_root, args.output, result)}))
            else:
                live = manifest.collect(args.baseline_source, args.source_root.resolve(), args.runtime, args.reference_root, specs)
                if live["source"]["digest"] != contract.get("source_digest") or freeze.digest(live["runtime"]) != contract.get("runtime_digest") or freeze.digest(live["hardware"]) != contract.get("hardware_digest"):
                    raise RunError("live_baseline_identity_mismatch")
                print(json.dumps(freeze.freeze_contract(contract, evidence, coverage, args.evidence_root, args.output)))
            return 0
        if args.command == "verify-freeze":
            print(json.dumps(freeze.verify_frozen(args.evidence_root, args.frozen, args.sha256, coverage)))
            return 0
        if args.command == "provision":
            token = provision.read_credential(args.bridge) if args.bridge and any(spec.get("gated") for spec in specs) else None
            reports = [provision.provision_spec(spec, token) for spec in specs]
            print(json.dumps(manifest.sanitize({"status": "files_verified", "models": reports}, secrets=[token] if token else [])))
            return 0
        report = preflight(catalog, coverage, licenses, args.evidence_root, specs, selection, verify_hf_alignment=args.verify_hf_alignment, vad_selection=vad_selection)
        if args.command in ("preflight", "full-gate") or report["missing"]:
            if args.command == "full-gate" and not report["missing"]:
                frozen = freeze.verify_frozen(args.evidence_root, args.frozen, args.sha256, coverage)
                if args.target == "native_gpu" and not frozen["complete_gpu"]:
                    raise RunError("native_gpu_placement_incomplete")
                print(json.dumps(frozen))
                return 0
            print(json.dumps(report))
            return 2 if report["missing"] else 0
        case = next(case for case in catalog["cases"] if case["id"] == args.case)
        asset = next(asset for asset in catalog["assets"] if asset["id"] == case["asset_id"])
        source = args.source_root.resolve(strict=True)
        worker = source / ("macOS/Worker/transcribe_worker.py" if args.baseline_source == "gui-checkout" else "Contents/Resources/transcribe_worker.py")
        imports = source if args.baseline_source == "gui-checkout" else source / "Contents/Resources"
        token = provision.read_credential(args.bridge) if case["settings"]["voices"] and args.bridge else None
        run_id = str(uuid.uuid4())
        identity = manifest.collect(args.baseline_source, source, args.runtime, args.reference_root, specs, case["settings"])
        if args.seed is not None: identity["settings"]["seed"] = args.seed
        harness_root = Path(__file__).resolve().parents[2]
        identity["harness"] = manifest.source_identity(harness_root, [str(path.relative_to(harness_root)) for path in Path(__file__).parent.glob("*.py")])
        if identity["prerequisites"]:
            raise RunError("source_identity_incomplete")
        manifest.write_evidence(args.evidence_root, "runs/" + run_id + "/manifest.json", identity)
        cache_root, torch_home = model_cache_roots(specs)
        result = paired_run(args.runtime, worker, imports, corpus.safe_target(args.evidence_root, asset["path"]), case["settings"], args.evidence_root / "runs" / run_id / "pair", token, args.timeout, cache_root, args.synchronized, torch_home=torch_home, seed=args.seed)
        with corpus.safe_target(args.evidence_root, "runs/" + run_id + "/comparison.json").open("x") as output:
            json.dump(result, output)
        print(json.dumps({"run_id": run_id, "status": result["status"], "gpu_placement_certified": False, "quality_certified": False}))
        return 0
    except (RunError, provision.ProvisionError, metrics.MetricError, freeze.FreezeError) as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "prerequisite_missing", "code": "baseline_run_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

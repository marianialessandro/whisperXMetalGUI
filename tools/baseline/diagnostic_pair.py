"""Run an explicitly partial original/adapter MLX diagnostic; never passes the full baseline gate."""

import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, manifest, provision, run


def prerequisites(asset, license, root, spec, settings):
    contracts.validate_settings(settings)
    if settings["voices"]:
        raise run.RunError("diagnostic_voices_not_supported")
    corpus.verify_license(asset, license, root, "download")
    path = corpus.safe_target(root, asset["path"])
    if not path.is_file() or manifest.sha256(path) != asset["sha256"]:
        raise run.RunError("asset_hash_mismatch")
    if spec.get("license_verified") is not True or not spec.get("license_source", "").startswith("https://"):
        raise run.RunError("model_license_unverified")
    if spec["repository"] != contracts.PROFILES[settings["quality"]]["mlx"]:
        raise run.RunError("diagnostic_profile_model_mismatch")
    if provision.inspect_spec(spec)["status"] != "files_verified":
        raise run.RunError("diagnostic_model_not_verified")
    reference = manifest.contained(Path(spec["cache_root"]), Path("models--" + spec["repository"].replace("/", "--")) / "refs/main")
    if not reference.is_file() or reference.read_text().strip() != spec["revision"]:
        raise run.RunError("worker_default_revision_unverified")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="GUI checkout, unchanged original worker and WhisperX imports.")
    parser.add_argument("--runtime", type=Path, required=True, help="Existing bundled Python; no dependency installation.")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    parser.add_argument("--model-specs", type=Path, required=True)
    parser.add_argument("--torch-specs", type=Path, help="Optional hash-verified torchaudio alignment checkpoints in a private TORCH_HOME; no downloads in diagnostic runs.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--asset", required=True, help="Licensed hash-verified audio; parity-only clips are admitted solely for this diagnostic.")
    parser.add_argument("--quality", choices=tuple(contracts.PROFILES), default="Rapida")
    parser.add_argument("--language", default="it")
    parser.add_argument("--seed", type=int, help="Optional explicit bootstrap RNG seed for both workers; records sampling policy and never changes default decoder temperatures.")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--synchronized", action="store_true", help="Drain MLX/MPS work at observed API boundaries; dispatch review remains separate.")
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2 ** 32: parser.error("seed must be between 0 and 2**32-1")
    try:
        catalog = json.loads(args.corpus.read_text())
        licenses = json.loads(args.licenses.read_text())["licenses"]
        specs = json.loads(args.model_specs.read_text())
        asset = next(asset for asset in catalog["assets"] if asset["id"] == args.asset)
        license = next(license for license in licenses if license["id"] == asset["license_id"])
        spec = next(spec for spec in specs if spec["repository"] == contracts.PROFILES[args.quality]["mlx"])
        settings = {"language": args.language, "quality": args.quality, "voices": False, "hints": ""}
        root = args.evidence_root.resolve(strict=True)
        path = prerequisites(asset, license, root, spec, settings)
        torch_specs = json.loads(args.torch_specs.read_text()) if args.torch_specs else []
        torch_home = None
        if torch_specs:
            chosen = next(item for item in torch_specs if item.get("role") == "alignment_" + args.language)
            if chosen.get("transport") != "torch_hub" or chosen.get("license_verified") is not True or not chosen.get("license_source", "").startswith("https://") or provision.inspect_spec(chosen)["status"] != "files_verified":
                raise run.RunError("diagnostic_alignment_model_not_verified")
            torch_home = Path(chosen["cache_root"]).resolve(strict=True)
        source = args.source_root.resolve(strict=True)
        run_id = str(uuid.uuid4())
        identity = manifest.collect("gui-checkout", source, args.runtime, args.reference_root, [spec], settings)
        if args.seed is not None: identity["settings"]["seed"] = args.seed
        if torch_specs: identity["models"].append(provision.inspect_spec(chosen))
        identity["harness"] = manifest.source_identity(Path(__file__).resolve().parents[2], [str(path.relative_to(Path(__file__).resolve().parents[2])) for path in Path(__file__).parent.glob("*.py")])
        manifest.write_evidence(root, "diagnostic-pairs/" + run_id + "/manifest.json", identity)
        report = {"run_id": run_id, "scope": "diagnostic_partial_pipeline", "full_baseline_gate_passed": False, "quality_certified": False, "gpu_placement_certified": False, "asset_sha256": asset["sha256"], "model_revision": spec["revision"], "independent_word_speaker_ground_truth": False, "seed": args.seed, "deterministic_algorithms_forced": False}
        try:
            report.update(run.paired_run(args.runtime, source / "macOS/Worker/transcribe_worker.py", source, path, settings, root / "diagnostic-pairs" / run_id / "pair", timeout=args.timeout, cache_root=Path(spec["cache_root"]), synchronized=args.synchronized, torch_home=torch_home, seed=args.seed))
        except run.RunError as error:
            report.update({"status": "diagnostic_failed", "code": str(error)})
        destination = corpus.safe_target(root, "diagnostic-pairs/" + run_id + "/comparison.json")
        with destination.open("x") as output: json.dump(report, output, indent=2)
        print(json.dumps(report))
        return 0 if report["status"] == "paired_equivalent" else 2
    except (run.RunError, corpus.CorpusError, provision.ProvisionError) as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error), "full_baseline_gate_passed": False}))
        return 2
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "diagnostic_failed", "code": "diagnostic_inputs_invalid", "full_baseline_gate_passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

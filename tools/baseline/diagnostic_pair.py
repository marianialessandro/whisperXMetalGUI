"""Run an explicitly partial original/adapter MLX diagnostic; never passes the full baseline gate."""

import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, manifest, provision, run, alignment_inventory


def prerequisites(asset, license, root, spec, settings, diarization_verified=False):
    contracts.validate_settings(settings)
    if settings["voices"] and not diarization_verified:
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


def diarization_prerequisites(specs, asr_spec):
    choices = [spec for spec in specs if spec.get("repository") == "pyannote/speaker-diarization-community-1"]
    if len(choices) != 1: raise run.RunError("diagnostic_diarization_inventory_invalid")
    spec = choices[0]
    if spec.get("license_verified") is not True or not spec.get("license_source", "").startswith("https://") or Path(spec["cache_root"]).resolve() != Path(asr_spec["cache_root"]).resolve(): raise run.RunError("diagnostic_diarization_license_or_cache_invalid")
    identity = provision.inspect_spec(spec)
    if identity["status"] != "files_verified" or identity.get("components_verified") is not True or {"speaker_segmentation", "speaker_embedding", "speaker_plda"} - set(identity.get("roles", [])): raise run.RunError("diagnostic_diarization_components_not_verified")
    reference = manifest.contained(Path(spec["cache_root"]), "models--pyannote--speaker-diarization-community-1/refs/main")
    if not reference.is_file() or reference.read_text().strip() != spec["revision"]: raise run.RunError("worker_default_revision_unverified")
    return spec


def voice_completion(pair_root, warnings):
    spans = json.loads((pair_root / "spans.json").read_text())["spans"]
    observed = {span["stage"] for span in spans if span.get("status") in ("returned", "synchronized")}
    result = json.loads((pair_root / "instrumented/result.json").read_text())
    rows = list(result.get("segments", [])) + list(result.get("word_segments", []))
    speakers = {row["speaker"] for row in rows if isinstance(row.get("speaker"), str) and row["speaker"]}
    return {"voice_pipeline_completed": not warnings and {"diarization", "speaker_assignment"}.issubset(observed) and bool(speakers), "observed_speaker_count": len(speakers)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="GUI checkout, unchanged original worker and WhisperX imports.")
    parser.add_argument("--runtime", type=Path, required=True, help="Existing bundled Python; no dependency installation.")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    parser.add_argument("--model-specs", type=Path, required=True)
    parser.add_argument("--torch-specs", type=Path, help="Optional hash-verified torchaudio alignment checkpoints in a private TORCH_HOME; no downloads in diagnostic runs.")
    parser.add_argument("--voices", action="store_true", help="Optional paired original-worker diarization diagnostic; requires verified --diarization-specs, --torch-specs and private --bridge. No independent quality claim.")
    parser.add_argument("--speaker-count", type=int, help="Optional exact speaker bound with --voices; default preserves automatic speaker estimation.")
    parser.add_argument("--diarization-specs", type=Path, help="Pinned community-1 snapshot with verified segmentation, embedding and PLDA component mappings; same HF cache as ASR.")
    parser.add_argument("--bridge", type=Path, help="Existing private Keychain pipe; credential only in memory/stdin, never argv/env/evidence. Used only with --voices after prerequisite checks.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--asset", required=True, help="Licensed hash-verified audio; parity-only clips are admitted solely for this diagnostic.")
    parser.add_argument("--quality", choices=tuple(contracts.PROFILES), default="Rapida")
    parser.add_argument("--language", default="it")
    parser.add_argument("--seed", type=int, help="Optional explicit bootstrap RNG seed for both workers; records sampling policy and never changes default decoder temperatures.")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--synchronized", action="store_true", help="Drain MLX/MPS work at observed API boundaries; dispatch review remains separate.")
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2 ** 32: parser.error("seed must be between 0 and 2**32-1")
    if args.speaker_count is not None and (not args.voices or args.speaker_count <= 0): parser.error("speaker-count requires --voices and a positive value")
    if args.voices and not (args.diarization_specs and args.torch_specs and args.bridge): parser.error("voices requires diarization-specs, torch-specs and bridge")
    try:
        catalog = json.loads(args.corpus.read_text())
        licenses = json.loads(args.licenses.read_text())["licenses"]
        specs = json.loads(args.model_specs.read_text())
        asset = next(asset for asset in catalog["assets"] if asset["id"] == args.asset)
        license = next(license for license in licenses if license["id"] == asset["license_id"])
        spec = next(spec for spec in specs if spec["repository"] == contracts.PROFILES[args.quality]["mlx"])
        settings = {"language": args.language, "quality": args.quality, "voices": args.voices, "hints": ""}
        if args.speaker_count is not None: settings["speaker_count"] = args.speaker_count
        diarization_spec = diarization_prerequisites(json.loads(args.diarization_specs.read_text()), spec) if args.voices else None
        root = args.evidence_root.resolve(strict=True)
        path = prerequisites(asset, license, root, spec, settings, diarization_verified=diarization_spec is not None)
        torch_specs = json.loads(args.torch_specs.read_text()) if args.torch_specs else []
        torch_home = None
        if torch_specs:
            chosen = next(item for item in torch_specs if item.get("role") == "alignment_" + args.language)
            if chosen.get("transport") != "torch_hub" or chosen.get("license_verified") is not True or not chosen.get("license_source", "").startswith("https://") or provision.inspect_spec(chosen)["status"] != "files_verified":
                raise run.RunError("diagnostic_alignment_model_not_verified")
            torch_home = Path(chosen["cache_root"]).resolve(strict=True)
        source = args.source_root.resolve(strict=True)
        selection = alignment_inventory.inspect_runtime(args.runtime, source, args.torch_specs) if args.voices else None
        if args.voices and not alignment_inventory.matches_spec(chosen, selection): raise run.RunError("worker_alignment_selection_unverified")
        run_id = str(uuid.uuid4())
        identity = manifest.collect("gui-checkout", source, args.runtime, args.reference_root, [spec] + ([diarization_spec] if diarization_spec is not None else []), settings)
        if args.seed is not None: identity["settings"]["seed"] = args.seed
        if torch_specs: identity["models"].append(provision.inspect_spec(chosen))
        identity["harness"] = manifest.source_identity(Path(__file__).resolve().parents[2], [str(path.relative_to(Path(__file__).resolve().parents[2])) for path in Path(__file__).parent.glob("*.py")])
        manifest.write_evidence(root, "diagnostic-pairs/" + run_id + "/manifest.json", identity)
        report = {"run_id": run_id, "scope": "diagnostic_partial_pipeline", "full_baseline_gate_passed": False, "quality_certified": False, "gpu_placement_certified": False, "asset_sha256": asset["sha256"], "model_revision": spec["revision"], "independent_word_speaker_ground_truth": False, "seed": args.seed, "deterministic_algorithms_forced": False}
        if args.voices: report.update(voices_requested=True, speaker_count_requested=args.speaker_count, alignment_selection_verified=True, alignment_source_sha256=selection["alignment_source_sha256"])
        token = None
        try:
            token = provision.read_credential(args.bridge) if args.voices else None
            pair_root = root / "diagnostic-pairs" / run_id / "pair"
            report.update(run.paired_run(args.runtime, source / "macOS/Worker/transcribe_worker.py", source, path, settings, pair_root, token=token, timeout=args.timeout, cache_root=Path(spec["cache_root"]), synchronized=args.synchronized, torch_home=torch_home, seed=args.seed))
            if args.voices:
                report.update(voice_completion(pair_root, report["warnings"]))
                if not report["voice_pipeline_completed"]: report.update(status="diagnostic_failed", code="diagnostic_voice_pipeline_incomplete")
        except run.RunError as error:
            report.update({"status": "diagnostic_failed", "code": str(error)})
        finally:
            token = None
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

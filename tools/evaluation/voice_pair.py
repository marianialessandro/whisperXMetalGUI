"""Paired unchanged-worker voice benchmark using actual turns and released AMI references."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, dataset_reference, freeze, manifest, metrics, provision, released_baseline, run


def paired(runtime, source, audio, settings, destination, token, hf_cache, torch_cache, timeout):
    corpus.require(not destination.exists(), "voice_destination_exists")
    destination.mkdir(parents=True)
    original, instrumented = destination / "original", destination / "instrumented"
    original.mkdir()
    instrumented.mkdir()
    request = dict(settings, audio=str(audio), token=token)
    common = {"cache_root": hf_cache, "torch_home": torch_cache}
    try:
        first = run.execute(runtime, [source / "tools/baseline/offline_worker.py", "--worker", source / "macOS/Worker/transcribe_worker.py", "--source-root", source, "--seed", str(released_baseline.POLICY["replica_seed"])], dict(request, output=str(original)), timeout, **common)
        second = run.execute(runtime, [source / "tools/evaluation/voice_adapter.py", "--speaker-turns", destination / "turns.json", "--worker", source / "macOS/Worker/transcribe_worker.py", "--source-root", source, "--spans", destination / "spans.json", "--run-id", destination.parent.name, "--synchronized", "--seed", str(released_baseline.POLICY["replica_seed"])], dict(request, output=str(instrumented)), timeout, **common)
    finally:
        request.clear()
    corpus.require(contracts.equivalent(run.semantic_bundle(original, first, audio.stem), run.semantic_bundle(instrumented, second, audio.stem), str(original), str(instrumented)), "voice_pair_mismatch")
    corpus.require(not first[-1]["warnings"] and not second[-1]["warnings"] and any(event.get("backend") == "MLX / Metal" for event in first), "voice_pipeline_incomplete")
    turns = json.loads((destination / "turns.json").read_text())
    corpus.require(bool(turns), "voice_turns_missing")
    return {"status": "paired_equivalent", "warnings": [], "actual_turns_retained": True, "original_turn_parity_measured": False, "quality_certified": False, "full_pipeline_gpu_certified": False}


def execute(root, snapshot, catalog, licenses, case_id, quality, exact_count, source, runtime, reference_root, specs, bridge, timeout):
    corpus.require(Path(snapshot).name == snapshot and snapshot not in ("", ".", ".."), "voice_snapshot_invalid")
    target = corpus.safe_target(root, snapshot)
    corpus.require(not target.exists(), "voice_destination_exists")
    case = next(item for item in catalog["cases"] if item["id"] == case_id)
    asset = next(item for item in catalog["assets"] if item["id"] == case["asset_id"])
    corpus.require(asset["id"].startswith("ami-") and asset["language"] == "en", "voice_released_reference_unsupported")
    legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
    corpus.verify_license(asset, legal, root, "download")
    reference = dataset_reference.reference(root, asset)
    corpus.require(reference["turns"] and manifest.sha256(corpus.safe_target(root, case["reference_path"])) == case["reference_sha256"] and json.loads(corpus.safe_target(root, case["reference_path"]).read_text()) == reference, "voice_reference_invalid")
    fingerprint, context = released_baseline.identity(source, runtime, reference_root, specs)
    roles = {role for model in context["model_inventory"] for role in model.get("roles", [])}
    roles.update(spec.get("role") for spec in specs)
    corpus.require({"speaker_segmentation", "speaker_embedding", "speaker_plda", "alignment_en"}.issubset(roles), "voice_models_missing")
    extra = manifest.source_identity(source, [str(path.relative_to(source)) for path in (source / "tools/evaluation").glob("*.py")])
    corpus.require(extra["status"] == "identified", "voice_harness_missing")
    hf_cache, torch_cache = run.model_cache_roots(specs)
    corpus.require(hf_cache is not None and torch_cache is not None, "voice_cache_missing")
    settings = {"quality": quality, "language": "en", "voices": True, "hints": ""}
    if exact_count is not None: settings["speaker_count"] = exact_count
    contracts.validate_settings(settings)
    context.update(voice_harness=extra, voice_settings=settings, corpus_case=case, asset_sha256=asset["sha256"])
    freeze.seal(root, snapshot + "/context.json", context)
    token = None
    try:
        token = provision.read_credential(bridge)
        started = time.perf_counter_ns()
        comparison = paired(runtime, source, corpus.safe_target(root, asset["path"]), settings, target / "pair", token, hf_cache, torch_cache, timeout)
        elapsed = time.perf_counter_ns() - started
    finally:
        token = None
    corpus.require(released_baseline.identity(source, runtime, reference_root, specs)[0] == fingerprint and manifest.source_identity(source, [item["name"] for item in extra["files"]])["digest"] == extra["digest"], "voice_execution_identity_changed")
    score = dataset_reference.evaluate(root, catalog, licenses, case_id, snapshot + "/pair/instrumented/result.json", snapshot + "/pair/turns.json")
    result = json.loads((target / "pair/instrumented/result.json").read_text())
    words = result.get("word_segments", [word for segment in result["segments"] for word in segment.get("words", [])])
    score["dataset_word_speaker_agreement"] = metrics.word_speaker_scores(reference["words"], words, score["dataset_turn_agreement"]["mapping"])
    report = {"schema_version": 1, "status": "released_voice_pair_measured", "scope": "released_AMI_turn_and_word_speaker_agreement", "case_id": case_id, "settings": settings, "identity": fingerprint, "comparison": comparison, "pair_wall_ns": elapsed, "score": score, "local_human_review_required": False, "gpu_placement_certified": False, "quality_certified": False, "artifacts": released_baseline.artifact_hashes(root, snapshot)}
    checksum = freeze.seal(root, snapshot + "/measurement.json", report)
    return report, checksum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-root", "catalog", "licenses", "source-root", "runtime", "reference-root", "model-specs", "bridge"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--snapshot", required=True, help="New private snapshot, preserved on failure; no overwrite.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--quality", choices=tuple(contracts.PROFILES), default="Rapida")
    parser.add_argument("--speaker-count", type=int, help="Optional exact speaker count; default automatic estimation is preserved.")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.speaker_count is not None and args.speaker_count <= 0: parser.error("speaker-count must be positive")
    try:
        report, checksum = execute(args.evidence_root.resolve(strict=True), args.snapshot, json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text()), args.case_id, args.quality, args.speaker_count, args.source_root.resolve(strict=True), args.runtime.resolve(strict=True), args.reference_root.resolve(strict=True), json.loads(args.model_specs.read_text()), args.bridge, args.timeout)
        scores = report["score"]["dataset_turn_agreement"]
        print(json.dumps({"status": report["status"], "sha256": checksum, "case_id": args.case_id, "quality": args.quality, "der": scores["der"], "jer": scores["jer"], "local_human_review_required": False, "quality_certified": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "released_voice_pair_incomplete", "local_human_review_required": False, "quality_certified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

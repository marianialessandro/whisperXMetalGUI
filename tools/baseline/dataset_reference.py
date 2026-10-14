"""Use released FLEURS/AMI references directly, retaining dataset provenance and metric-specific limits."""

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import acquire_fleurs, contracts, corpus, freeze, import_ami, manifest, metrics, provision


def reference(root, asset):
    audio = corpus.safe_target(root, asset["path"])
    corpus.require(manifest.sha256(audio) == asset["sha256"], "dataset_audio_changed")
    language = asset["language"]
    if asset["id"].startswith("fleurs-"):
        inventory_path = corpus.safe_target(root, "corpus-fleurs/" + language + "-acquisition.json")
        inventory = json.loads(inventory_path.read_text())
        metadata = corpus.safe_target(root, "corpus-fleurs/metadata/" + language + "-dev.tsv")
        card = corpus.safe_target(root, "corpus-fleurs/licenses/fleurs-card.md")
        corpus.require(asset.get("type") == "parity_only" and asset.get("revision") == acquire_fleurs.REVISION and asset.get("archive_sha256") == inventory["archive"]["sha256"] and inventory["archive"]["revision"] == acquire_fleurs.REVISION and manifest.sha256(metadata) == inventory["metadata_sha256"] and manifest.sha256(card) == acquire_fleurs.CARD_HASH, "fleurs_reference_identity_changed")
        with metadata.open(encoding="utf-8") as source:
            selected = [row for row in csv.reader(source, delimiter="\t") if len(row) >= 4 and row[1] == audio.name]
        corpus.require(len(selected) == 1 and asset["transcript"] == selected[0][2], "fleurs_reference_text_changed")
        annotation = {"transcript": selected[0][2], "words": [], "turns": [], "uem": [[0, asset["duration"]]]}
        origin = {"dataset": "google/fleurs", "revision": acquire_fleurs.REVISION, "metadata_sha256": inventory["metadata_sha256"], "dataset_card_sha256": acquire_fleurs.CARD_HASH, "upstream": inventory["license"]["source"], "timing_basis": "not_supplied", "speaker_basis": "not_supplied"}
    elif asset["id"].startswith("ami-"):
        source_path = corpus.safe_target(root, asset["review_reference_path"])
        corpus.require(manifest.sha256(source_path) == asset["review_reference_sha256"], "ami_reference_changed")
        annotation = json.loads(source_path.read_text())
        archive = corpus.safe_target(root, "corpus-ami/manual-1.6.2.zip")
        meeting = annotation["recording_id"].removeprefix("ami_")
        original = import_ami.import_manual(archive, annotation["archive_sha256"], meeting, asset["sha256"], asset["duration"])
        corpus.require(annotation == original, "ami_reference_not_original_release")
        origin = {"dataset": "AMI", "revision": "manual-1.6.2", "archive_sha256": annotation["archive_sha256"], "upstream": annotation["source"], "timing_basis": "dataset_supplied_not_locally_adjudicated", "speaker_basis": "manual_dataset_transcriber_segments", "uem_basis": "full_recording_reference_convention"}
    else:
        raise corpus.CorpusError("released_dataset_unsupported")
    return {"schema_version": 1, "asset_id": asset["id"], "audio_sha256": asset["sha256"], "language": language, "duration": asset["duration"], "reference_policy": "released-dataset", "origin": origin, "transcript": annotation["transcript"], "words": annotation["words"], "turns": annotation["turns"], "uem": annotation["uem"], "local_human_review_required": False, "local_human_review_performed": False, "independent_word_boundary_review_verified": False}


def prepare(root, snapshot, catalog, licenses):
    corpus.require(Path(snapshot).name == snapshot and snapshot not in ("", ".", ".."), "dataset_snapshot_invalid")
    destination = corpus.safe_target(root, snapshot)
    corpus.require(not destination.exists(), "destination_exists")
    assets = catalog["assets"]
    corpus.require(assets and len({item["id"] for item in assets}) == len(assets), "dataset_inventory_invalid")
    with tempfile.TemporaryDirectory(prefix=".dataset-reference-", dir=root) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        cases = []
        for asset in assets:
            legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
            corpus.verify_license(asset, legal, root, "download")
            annotation = reference(root, asset)
            name = "references/" + asset["id"] + ".json"
            checksum = freeze.seal(staging, name, annotation)
            cases.append({"id": asset["id"], "asset_id": asset["id"], "type": "released_dataset", "reference_path": snapshot + "/" + name, "reference_sha256": checksum, "reference_policy": "released-dataset", "settings": {"language": asset["language"], "quality": "Rapida", "voices": False, "hints": ""}})
        document = {"schema_version": 1, "reference_policy": "released-dataset", "local_human_review_required": False, "assets": assets, "cases": cases, "quality_certified": False}
        freeze.seal(staging, "catalog.json", document)
        freeze.seal(staging, "licenses.json", licenses)
        provision.rename_directory_exclusive(staging, destination)
    return document


def evaluate(root, catalog, licenses, case_id, result_path, speaker_turns_path=None):
    corpus.require(catalog.get("reference_policy") == "released-dataset", "dataset_policy_missing")
    case = next(item for item in catalog["cases"] if item["id"] == case_id)
    corpus.require(case.get("type") == "released_dataset" and case.get("reference_policy") == "released-dataset", "dataset_case_policy_invalid")
    asset = next(item for item in catalog["assets"] if item["id"] == case["asset_id"])
    legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
    corpus.verify_license(asset, legal, root, "download")
    reference_path = corpus.safe_target(root, case["reference_path"])
    corpus.require(manifest.sha256(reference_path) == case["reference_sha256"], "dataset_reference_changed")
    annotation = json.loads(reference_path.read_text())
    corpus.require(annotation == reference(root, asset), "dataset_reference_provenance_changed")
    result_file = corpus.safe_target(root, result_path)
    corpus.require(result_file.stat().st_size <= 64 * 1024 * 1024, "dataset_result_budget_exceeded")
    result = contracts.validate_result(json.loads(result_file.read_text()))
    corpus.require(result["language"] == asset["language"], "dataset_result_language_mismatch")
    hypothesis = " ".join(segment["text"] for segment in result["segments"])
    words = result.get("word_segments", [word for segment in result["segments"] for word in segment.get("words", [])])
    text = metrics.text_scores(annotation["transcript"], hypothesis, asset["language"])
    timestamps = metrics.timestamp_scores(annotation["words"], words) if annotation["words"] else {"status": "not_available", "reason": "dataset_word_times_not_supplied"}
    diarization = {"status": "not_measured", "reason": "speaker_turn_output_not_supplied"}
    if speaker_turns_path is not None:
        corpus.require(bool(annotation["turns"]), "dataset_speaker_reference_missing")
        turns_path = corpus.safe_target(root, speaker_turns_path)
        corpus.require(turns_path.stat().st_size <= 10 * 1024 * 1024, "dataset_turn_budget_exceeded")
        diarization = metrics.diarization_scores(annotation["turns"], json.loads(turns_path.read_text()), annotation["uem"])
    return {"schema_version": 1, "status": "released_dataset_reference_evaluated", "case_id": case_id, "reference_policy": "released-dataset", "reference_sha256": case["reference_sha256"], "audio_sha256": asset["sha256"], "result_sha256": manifest.sha256(result_file), "reference_origin": annotation["origin"], "language": asset["language"], "text": text, "dataset_timestamp_agreement": timestamps, "dataset_turn_agreement": diarization, "local_human_review_required": False, "local_human_review_performed": False, "independent_timestamp_accuracy_certified": False, "quality_certified": False, "policy": metrics.POLICY}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="Build source-verified released references in a new snapshot; no local review flags or reviewer identities are manufactured.")
    create.add_argument("--snapshot", required=True)
    score = commands.add_parser("evaluate", help="Score existing worker output against released text and available timing/speaker references; no new annotation work required.")
    score.add_argument("--case-id", required=True)
    score.add_argument("--result", required=True, help="Existing result.json path relative to evidence root, produced from the selected audio.")
    score.add_argument("--speaker-turns", help="Optional actual diarization turns JSON relative to evidence root; never inferred from dominant segment labels.")
    score.add_argument("--output", required=True, help="New score report path relative to evidence root; no overwrite.")
    args = parser.parse_args(argv)
    try:
        root = args.evidence_root.resolve(strict=True)
        catalog, licenses = json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text())
        if args.command == "prepare":
            report = prepare(root, args.snapshot, catalog, licenses)
            print(json.dumps({"status": "released_dataset_references_ready", "cases": len(report["cases"]), "local_human_review_required": False}))
        else:
            report = evaluate(root, catalog, licenses, args.case_id, args.result, args.speaker_turns)
            freeze.seal(root, args.output, report)
            print(json.dumps({"status": report["status"], "case_id": args.case_id, "primary_metric": report["text"]["primary_metric"], "local_human_review_required": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "reference_evaluation_failed", "local_human_review_required": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

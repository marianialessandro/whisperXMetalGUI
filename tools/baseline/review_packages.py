"""Prepare portable, licensed human annotation packages and validate completed reviews without certifying drafts."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, freeze, manifest, provision


def prepare(root, name, catalog, licenses, coverage):
    corpus.require(Path(name).name == name and name not in ("", ".", ".."), "package_name_invalid")
    destination = corpus.safe_target(root, name)
    corpus.require(not destination.exists(), "destination_exists")
    corpus.validate_coverage(coverage, Path(__file__).resolve().parents[2])
    audio = [item for item in catalog["assets"] if item.get("type") in ("parity_only", "natural_candidate") and item.get("duration", 0) > 0]
    corpus.require(audio and len({item["id"] for item in audio}) == len(audio), "review_audio_inventory_invalid")
    with tempfile.TemporaryDirectory(prefix=".review-prepare-", dir=root) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        entries = []
        for asset in audio:
            corpus.require(Path(asset["id"]).name == asset["id"] and asset["id"] not in ("", ".", ".."), "review_identity_invalid")
            legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
            corpus.verify_license(asset, legal, root, "share")
            source = corpus.safe_target(root, asset["path"])
            corpus.require(manifest.sha256(source) == asset["sha256"], "review_audio_changed")
            folder = staging / "recordings" / asset["id"]
            folder.mkdir(parents=True)
            audio_name = "audio" + source.suffix
            shutil.copyfile(source, folder / audio_name)
            shutil.copyfile(corpus.safe_target(root, legal["text_path"]), folder / "LICENSE.html")
            reference = {"status": "unreviewed_source_reference", "transcript": asset.get("transcript", ""), "provenance": "dataset_metadata_not_local_ground_truth", "asset_id": asset["id"], "audio_sha256": asset["sha256"], "license": legal}
            if asset.get("review_reference_path"):
                path = corpus.safe_target(root, asset["review_reference_path"])
                corpus.require(manifest.sha256(path) == asset.get("review_reference_sha256"), "review_reference_changed")
                reference["manual_dataset_reference"] = json.loads(path.read_text())
                corpus.require(reference["manual_dataset_reference"].get("audio_sha256") == asset["sha256"] and reference["manual_dataset_reference"].get("reviewed") is False, "review_reference_scope_invalid")
            annotation = {"audio_sha256": asset["sha256"], "provenance": "pending_independent_human", "reviewed": False, "reviewers": [], "uncertainties": ["Word boundaries, speaker turns and evaluation intervals require independent listening and two review passes."], "transcript": "", "words": [], "turns": [], "uem": []}
            assignment = {"recording_id": asset.get("recording_id", asset["id"]), "source_group": asset.get("source_group", ""), "speaker_ids": asset.get("speaker_ids", []), "split": "", "case_ids": [], "status": "pending_identity_and_split_adjudication"}
            freeze.seal(staging, str((folder / "source-reference.json").relative_to(staging)), reference)
            freeze.seal(staging, str((folder / "annotation.json").relative_to(staging)), annotation)
            freeze.seal(staging, str((folder / "assignment.json").relative_to(staging)), assignment)
            entries.append({"asset_id": asset["id"], "folder": str(folder.relative_to(staging)), "language": asset.get("language"), "duration": asset["duration"], "audio_sha256": asset["sha256"], "source_reference_sha256": manifest.sha256(folder / "source-reference.json"), "license_sha256": manifest.sha256(folder / "LICENSE.html"), "audio": audio_name, "quality_certified": False})
        policy = {"minimum_recordings_per_stratum_per_split": freeze.CALIBRATION_POLICY["minimum_recordings_per_stratum_per_split"], "minimum_replicates_per_recording": freeze.CALIBRATION_POLICY["minimum_replicates_per_recording"], "split_policy": "Recording, source group and stable person identities must be disjoint across calibration/evaluation; assign before oracle runs. Unknown identities remain unresolved.", "review_policy": "Independent listening and two distinguishable review passes; the same annotator may perform both passes. Use anonymous pass identifiers. Preserve overlap and adjudicate uncertainties. Source references are drafts, never model-derived certified labels."}
        index = {"schema_version": 1, "status": "pending_human_review", "recordings": entries, "required_natural_cases": [item["id"] for item in coverage["required_cases"] if item["requires_natural_quality"]], "policy": policy, "quality_certified": False, "full_baseline_gate_passed": False}
        freeze.seal(staging, "index.json", index)
        (staging / "README.md").write_text("# Pacchetti di revisione\n\nOgni cartella contiene audio, licenza, riferimento del dataset non revisionato, annotation.json e assignment.json da completare. Ascoltare l'audio, annotare parole e turni con tempi in secondi, conservare le sovrapposizioni e definire la UEM. Registrare due passate di revisione distinte e risolvere le incertezze prima di attestare provenance=independent_human e reviewed=true.\n\nAssegnare split e identita stabili prima delle esecuzioni dell'oracle. Non inventare speaker mancanti e non assegnare copie o ritagli della stessa registrazione a split diversi. Il riferimento del dataset e separato dalle annotazioni da certificare. index.json elenca le quantita minime; questi campioni iniziali non bastano automaticamente per la calibrazione.\n\nEseguire review_packages.py check con --evidence-root e --package. Il controllo non importa automaticamente le annotazioni, non attesta l'ascolto e non apre il gate completo.\n", encoding="utf-8")
        provision.rename_directory_exclusive(staging, destination)
    return index


def check(root, name):
    directory = corpus.safe_target(root, name)
    index = json.loads((directory / "index.json").read_text())
    completed, pending, cases = [], [], []
    for entry in index["recordings"]:
        try:
            folder = corpus.safe_target(directory, entry["folder"])
            for filename, expected in ((entry["audio"], entry["audio_sha256"]), ("LICENSE.html", entry["license_sha256"]), ("source-reference.json", entry["source_reference_sha256"])):
                corpus.require(manifest.sha256(corpus.safe_target(folder, filename)) == expected, "review_package_source_changed")
            annotation = json.loads((folder / "annotation.json").read_text())
            corpus.validate_annotation(annotation)
            corpus.require(annotation["audio_sha256"] == entry["audio_sha256"], "annotation_audio_mismatch")
            corpus.require(all(item["end"] <= entry["duration"] for key in ("words", "turns") for item in annotation[key]) and all(end <= entry["duration"] for _, end in annotation["uem"]), "annotation_outside_audio")
            assignment = json.loads((folder / "assignment.json").read_text())
            corpus.validate_splits([assignment])
            corpus.require(set(assignment["speaker_ids"]) == {item["speaker"] for item in annotation["turns"]}, "review_speaker_identity_mismatch")
            corpus.require(isinstance(assignment["case_ids"], list) and bool(assignment["case_ids"]) and all(item in index["required_natural_cases"] for item in assignment["case_ids"]), "review_case_assignment_missing")
            cases.append(assignment)
            completed.append(entry["asset_id"])
        except (ValueError, OSError, KeyError, TypeError):
            pending.append(entry["asset_id"])
    corpus.validate_splits(cases)
    return {"status": "review_structure_verified" if not pending else "pending_human_review", "completed": completed, "pending": pending, "quality_certified": False, "full_baseline_gate_passed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--package", required=True, help="New safe package basename for prepare; existing private package for check.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    for name in ("catalog", "licenses", "coverage"): create.add_argument("--" + name, type=Path, required=True)
    commands.add_parser("check", help="Verify immutable sources, completed annotations and disjoint split identities; drafts return exit 2.")
    args = parser.parse_args(argv)
    try:
        root = args.evidence_root.resolve(strict=True)
        if args.command == "prepare":
            report = prepare(root, args.package, json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text()), json.loads(args.coverage.read_text()))
            print(json.dumps({"status": report["status"], "recordings": len(report["recordings"]), "quality_certified": False}))
            return 0
        report = check(root, args.package)
        print(json.dumps(report))
        return 0 if not report["pending"] else 2
    except (ValueError, OSError, KeyError, TypeError, StopIteration):
        print(json.dumps({"status": "prerequisite_missing", "code": "review_package_invalid", "quality_certified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

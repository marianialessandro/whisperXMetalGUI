"""Licensed, bounded corpus acquisition and offline preflight; never reads app data."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import tarfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.manifest import BaselineError, contained, sha256


class CorpusError(ValueError):
    pass


CONTRACT_CASES = ("silence", "non_speech", "empty", "invalid_token", "model_failure", "cancelled", "interrupted", "unknown_chars", "out_of_range", "unicode_paths", "codec_mp3", "codec_m4a", "codec_mp4", "window_boundaries")
NATURAL_CASES = ("it_short", "en_short", "it_long", "en_long", "it_overlap", "en_overlap", "it_auto", "en_auto", "it_explicit", "en_explicit", "hints_on", "hints_off", "voices_off", "voices_auto", "voices_exact", "profile_base", "profile_small", "profile_large_v3")
REPRESENTATIVES = ("it", "en", "fr", "de", "ja", "zh", "ar", "uk", "no")
REQUIRED_IDS = CONTRACT_CASES + NATURAL_CASES + tuple("language_" + language for language in REPRESENTATIVES)
TYPES = ("synthetic_contract", "derived_audio", "natural_annotated", "parity_only")


def require(condition, code):
    if not condition:
        raise CorpusError(code)


def expected_coverage(root):
    tree = ast.parse((root / "whisperx/alignment.py").read_text())
    languages = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ("DEFAULT_ALIGN_MODELS_TORCH", "DEFAULT_ALIGN_MODELS_HF") for target in node.targets):
            mapping = ast.literal_eval(node.value)
            family = "torchaudio" if node.targets[0].id.endswith("TORCH") else "huggingface"
            languages.extend({"language": language, "model": model, "family": family, "representative": language in REPRESENTATIVES, "status": "untested"} for language, model in mapping.items())
    ui = (root / "macOS/Sources/WhisperXMac/ProjectDetail.swift").read_text()
    ui_languages = sorted(set(re.findall(r'\.tag\("(auto|[a-z]{2,3})"\)', ui)))
    return {"schema_version": 1, "required_cases": [{"id": case, "requires_natural_quality": case in NATURAL_CASES or case.startswith("language_")} for case in REQUIRED_IDS], "languages": sorted(languages, key=lambda row: row["language"]), "ui_languages": ui_languages}


def validate_coverage(coverage, root):
    expected = expected_coverage(root)
    require(coverage.get("required_cases") == expected["required_cases"], "coverage_matrix_changed")
    require(coverage.get("languages") == expected["languages"] and coverage.get("ui_languages") == expected["ui_languages"], "language_matrix_changed")


def safe_target(root, name):
    try:
        target = contained(root, name)
    except (OSError, BaselineError):
        raise CorpusError("path_escape") from None
    current, base = Path(root) / Path(name), Path(root)
    require(not base.is_symlink(), "path_symlink")
    while current != base:
        require(not current.is_symlink(), "path_symlink")
        current = current.parent
    return target


def verify_license(asset, license, license_root, action):
    require(license.get("verified") is True and asset.get("license_id") == license.get("id") and asset["id"] in license.get("asset_ids", []), "missing_license")
    require(license.get("rights", {}).get(action) is True, "license_permission_missing")
    require(isinstance(license.get("attribution"), str) and bool(license["attribution"]), "license_attribution_missing")
    require(urlsplit(license.get("source", "")).scheme == "https" and bool(license.get("revision")), "license_source_missing")
    path = safe_target(license_root, license.get("text_path", ""))
    require(path.is_file() and sha256(path) == license.get("text_sha256"), "license_text_missing")


def valid_digest(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def commit_file(temporary, target):
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise CorpusError("destination_exists") from None


def extract_selected(asset, license, evidence_root, license_root, selected, max_bytes):
    """Extract exact members from a hash-verified archive without tar path handling."""
    verify_license(asset, license, license_root, "derive")
    archive = safe_target(evidence_root, asset["path"])
    require(archive.is_file() and sha256(archive) == asset["sha256"], "asset_hash_mismatch")
    require(isinstance(max_bytes, int) and 0 < max_bytes <= 1024 * 1024 * 1024, "size_budget_invalid")
    require(isinstance(selected, dict) and bool(selected) and len(set(selected.values())) == len(selected), "archive_selection_invalid")
    for member, name in selected.items():
        require(not Path(member).is_absolute() and ".." not in Path(member).parts, "path_escape")
        require(not safe_target(evidence_root, name).exists(), "destination_exists")
    records, staged, published, total, scanned = [], {}, [], 0, 0
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=evidence_root) as staging:
        try:
            with tarfile.open(archive, "r|gz") as source:
                for member in source:
                    scanned += member.size
                    require(0 <= member.size and scanned <= 2 * 1024 * 1024 * 1024, "archive_expansion_budget_exceeded")
                    if member.name not in selected:
                        continue
                    require(member.isfile() and member.name not in staged, "archive_member_invalid")
                    total += member.size
                    require(total <= max_bytes, "size_budget_exceeded")
                    temporary = Path(staging) / str(len(staged))
                    with source.extractfile(member) as data, temporary.open("xb") as output:
                        remaining = member.size
                        while remaining:
                            chunk = data.read(min(1024 * 1024, remaining))
                            require(bool(chunk), "archive_member_truncated")
                            output.write(chunk)
                            remaining -= len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    staged[member.name] = temporary
                    if len(staged) == len(selected):
                        break
            require(len(staged) == len(selected), "archive_member_missing")
            for member, name in selected.items():
                target = safe_target(evidence_root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                commit_file(staged[member], target)
                published.append(target)
                records.append({"path": name, "sha256": sha256(target), "bytes": target.stat().st_size, "archive_sha256": asset["sha256"], "archive_member": member, "revision": asset["revision"]})
        except BaseException:
            for target in published:
                target.unlink()
            raise
    return records


def acquire(asset, license, evidence_root, license_root, transport=None):
    verify_license(asset, license, license_root, "download")
    target = safe_target(evidence_root, asset["path"])
    require(not target.exists(), "destination_exists")
    require(valid_digest(asset.get("sha256")), "asset_hash_missing")
    limit = asset.get("max_bytes")
    require(isinstance(limit, int) and not isinstance(limit, bool) and 0 < limit <= 1024 * 1024 * 1024, "size_budget_invalid")
    url = urlsplit(asset.get("url", ""))
    require(url.scheme == "https" and url.hostname in ("huggingface.co", "www.openslr.org", "groups.inf.ed.ac.uk") and not url.username and not url.password and not url.query, "source_url_invalid")
    require(bool(asset.get("revision")) and (asset["revision"] in url.path or asset.get("immutable_url") is True), "asset_revision_missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".acquire-", dir=target.parent)
    total, digest = 0, hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output, (transport(asset["url"]) if transport else urlopen(asset["url"], timeout=30)) as source:
            while chunk := source.read(min(1024 * 1024, limit + 1 - total)):
                total += len(chunk)
                require(total <= limit, "size_budget_exceeded")
                digest.update(chunk)
                output.write(chunk)
            require(digest.hexdigest() == asset["sha256"], "asset_hash_mismatch")
            output.flush()
            os.fsync(output.fileno())
        commit_file(temporary, target)
    finally:
        os.unlink(temporary)
    return target


def interval(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, (float, int)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0 for item in value) and value[0] < value[1]


def validate_annotation(annotation):
    require(valid_digest(annotation.get("audio_sha256")), "annotation_audio_hash_missing")
    require(annotation.get("provenance") == "independent_human" and annotation.get("reviewed") is True, "annotation_review_missing")
    reviewers = annotation.get("reviewers", [])
    require(isinstance(reviewers, list) and len(reviewers) >= 2 and all(isinstance(item, str) and item for item in reviewers), "annotation_second_review_missing")
    require(annotation.get("uncertainties") == [], "annotation_uncertainty_unresolved")
    require(isinstance(annotation.get("transcript"), str), "annotation_transcript_missing")
    uem = annotation.get("uem")
    require(isinstance(uem, list) and bool(uem) and all(interval(item) for item in uem), "annotation_uem_invalid")
    for key, label in (("words", "word"), ("turns", "speaker")):
        require(isinstance(annotation.get(key), list), "annotation_labels_missing")
        for item in annotation[key]:
            require(isinstance(item, dict) and isinstance(item.get(label), str) and interval([item.get("start"), item.get("end")]), "annotation_label_invalid")
            require(any(start <= item["start"] and item["end"] <= end for start, end in uem), "annotation_outside_uem")
    from tools.baseline.metrics import normalize
    expected_text = "".join(normalize(annotation["transcript"]).split())
    annotated_text = "".join("".join(normalize(word["word"]).split()) for word in annotation["words"])
    require(expected_text == annotated_text, "annotation_word_coverage_incomplete")
    require(not expected_text or bool(annotation["turns"]), "annotation_speaker_turns_missing")


def quality_eligible(case):
    if case.get("type") != "natural_annotated":
        return False
    try:
        validate_annotation(case.get("annotation", {}))
        return True
    except (ValueError, TypeError, KeyError):
        return False


def validate_splits(cases):
    groups = {}
    for case in cases:
        require(case.get("split") in ("calibration", "evaluation"), "split_missing")
        require(case.get("recording_id") and case.get("source_group") and case.get("speaker_ids"), "split_identity_missing")
        for field, identities in (("recording_id", [case["recording_id"]]), ("source_group", [case["source_group"]]), ("speaker", case["speaker_ids"])):
            for identity in identities:
                key = (field, identity)
                require(key not in groups or groups[key] == case["split"], "split_leakage")
                groups[key] = case["split"]


def validate_case_capabilities(case, asset, annotation):
    identifier = case["id"]
    settings = case.get("settings", {})
    from tools.baseline.contracts import validate_settings
    validate_settings(settings)
    if identifier.startswith(("it_", "en_")):
        require(asset.get("language") == identifier[:2], "case_language_mismatch")
    if identifier.startswith("language_"):
        require(asset.get("language") == identifier[len("language_"):], "case_language_mismatch")
    if identifier.endswith("_explicit"):
        require(settings["language"] == identifier[:2], "case_language_mode_mismatch")
    if identifier.endswith("_auto") and identifier != "voices_auto":
        require(settings["language"] == "auto", "case_language_mode_mismatch")
    qualities = {"profile_base": "Rapida", "profile_small": "Bilanciata", "profile_large_v3": "Massima"}
    if identifier in qualities:
        require(settings["quality"] == qualities[identifier], "case_profile_mismatch")
    if identifier in ("hints_on", "hints_off"):
        require(bool(settings.get("hints")) == (identifier == "hints_on"), "case_hints_mismatch")
    if identifier.startswith("voices_"):
        require(settings["voices"] == (identifier != "voices_off"), "case_voice_mode_mismatch")
    if identifier == "voices_auto":
        require("speaker_count" not in settings, "case_voice_count_mismatch")
    if identifier == "voices_exact":
        require(settings.get("speaker_count") == len({turn["speaker"] for turn in annotation["turns"]}), "case_voice_count_mismatch")
    if identifier.endswith("_overlap"):
        turns = annotation["turns"]
        overlap = any(left["speaker"] != right["speaker"] and max(left["start"], right["start"]) < min(left["end"], right["end"]) for index, left in enumerate(turns) for right in turns[index + 1:])
        require(overlap and settings["voices"], "case_natural_overlap_missing")
    if identifier.endswith(("_long", "_short")):
        duration = asset.get("duration")
        require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0, "case_duration_missing")
        require(duration >= 600 if identifier.endswith("_long") else duration <= 60, "case_duration_stratum_mismatch")


def preflight(catalog, coverage, licenses, evidence_root, license_root):
    validate_coverage(coverage, Path(__file__).resolve().parents[2])
    missing, passed = [], []
    assets = {asset["id"]: asset for asset in catalog.get("assets", [])}
    cases = {case["id"]: case for case in catalog.get("cases", [])}
    require(len(assets) == len(catalog.get("assets", [])) and len(cases) == len(catalog.get("cases", [])), "duplicate_identity")
    rights = {license["id"]: license for license in licenses.get("licenses", [])}
    try:
        validate_splits([case for case in cases.values() if case.get("type") == "natural_annotated"])
    except CorpusError as error:
        missing.append({"code": str(error)})
    for requirement in coverage["required_cases"]:
        case_id = requirement["id"]
        case = cases.get(case_id)
        try:
            require(case is not None, "case_missing")
            require(case.get("type") in TYPES, "case_type_invalid")
            asset = assets.get(case.get("asset_id"))
            require(asset is not None, "asset_missing")
            verify_license(asset, rights.get(asset.get("license_id"), {}), license_root, "download")
            path = safe_target(evidence_root, asset["path"])
            require(path.is_file() and sha256(path) == asset.get("sha256"), "asset_missing_or_corrupt")
            if requirement["requires_natural_quality"]:
                require(case["type"] == "natural_annotated", "natural_ground_truth_missing")
                annotation_path = safe_target(evidence_root, case.get("annotation_path", ""))
                require(annotation_path.is_file() and sha256(annotation_path) == case.get("annotation_sha256"), "annotation_missing_or_corrupt")
                annotation = json.loads(annotation_path.read_text())
                validate_annotation(annotation)
                require(annotation["audio_sha256"] == asset["sha256"], "annotation_audio_mismatch")
                validate_case_capabilities(case, asset, annotation)
            passed.append(case_id)
        except CorpusError as error:
            missing.append({"case_id": case_id, "code": str(error)})
        except (OSError, ValueError, KeyError, TypeError):
            missing.append({"case_id": case_id, "code": "case_metadata_invalid"})
    if any(item.get("case_id") == "it_overlap" for item in missing):
        missing.append({"code": "italian_natural_diarization_ground_truth"})
    return {"status": "prerequisite_missing" if missing else "verified", "passed": passed, "missing": missing, "languages": coverage["languages"]}


def annotation_import(source, evidence_root, name, expected_hash, audio_hash):
    require(Path(source).stat().st_size <= 10 * 1024 * 1024, "size_budget_exceeded")
    require(valid_digest(expected_hash) and sha256(source) == expected_hash, "annotation_hash_mismatch")
    payload = Path(source).read_bytes()
    require(len(payload) <= 10 * 1024 * 1024, "size_budget_exceeded")
    annotation = json.loads(payload)
    validate_annotation(annotation)
    require(annotation["audio_sha256"] == audio_hash, "annotation_audio_mismatch")
    target = safe_target(evidence_root, name)
    require(not target.exists(), "destination_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".annotation-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        commit_file(temporary, target)
    finally:
        os.unlink(temporary)
    return target


def derive_concat(assets, licenses, evidence_root, license_root, name, max_bytes):
    require(0 < max_bytes <= 1024 * 1024 * 1024, "size_budget_invalid")
    require(bool(assets), "derive_sources_missing")
    target = safe_target(evidence_root, name)
    require(not target.exists(), "destination_exists")
    sources = []
    for asset in assets:
        verify_license(asset, licenses[asset["license_id"]], license_root, "derive")
        source = safe_target(evidence_root, asset["path"])
        require(source.is_file() and sha256(source) == asset["sha256"], "asset_hash_mismatch")
        require(source != target, "derive_source_overwrite")
        sources.append(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".derive-", dir=target.parent)
    os.close(descriptor)
    parameters, total = None, 0
    try:
        with wave.open(temporary, "wb") as output:
            for source in sources:
                with wave.open(str(source), "rb") as audio:
                    current = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
                    require(current[3] == "NONE" and (parameters is None or current == parameters), "derive_format_mismatch")
                    if parameters is None:
                        parameters = current
                        output.setparams((current[0], current[1], current[2], 0, "NONE", "not compressed"))
                    while frames := audio.readframes(4096):
                        total += len(frames)
                        require(total + 44 <= max_bytes, "size_budget_exceeded")
                        output.writeframesraw(frames)
        commit_file(temporary, target)
    finally:
        os.unlink(temporary)
    return {"type": "derived_audio", "path": name, "sha256": sha256(target), "recipe": {"operation": "wav_concat_v1", "source_hashes": [asset["sha256"] for asset in assets]}, "quality_eligible": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--licenses", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path, help="Existing private corpus directory; no overwrites or personal media.")
    parser.add_argument("--license-root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Offline full gate; missing licenses/assets/independent reviews return exit 2.")
    acquisition = sub.add_parser("acquire", help="Explicit HTTPS acquisition of one licensed, hash-pinned asset.")
    acquisition.add_argument("--asset", required=True)
    extract = sub.add_parser("extract", help="Extract an exact member selection from a licensed hash-verified TAR.GZ; no links or overwrite.")
    extract.add_argument("--asset", required=True)
    extract.add_argument("--selection", type=Path, required=True, help="JSON object mapping exact archive members to relative output paths.")
    extract.add_argument("--max-bytes", type=int, required=True)
    annotation = sub.add_parser("annotate-import", help="Import two-pass independent human annotation; originals remain intact.")
    annotation.add_argument("--input", required=True, type=Path)
    annotation.add_argument("--name", required=True)
    annotation.add_argument("--sha256", required=True)
    annotation.add_argument("--audio-sha256", required=True)
    derive = sub.add_parser("derive", help="Concatenate licensed PCM WAV sources; always derived_audio, never natural quality evidence.")
    derive.add_argument("--assets", nargs="+", required=True)
    derive.add_argument("--name", required=True)
    derive.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        catalog, coverage, licenses = (json.loads(path.read_text()) for path in (args.corpus, args.coverage, args.licenses))
        assets = {item["id"]: item for item in catalog["assets"]}
        rights = {item["id"]: item for item in licenses["licenses"]}
        if args.command == "preflight":
            report = preflight(catalog, coverage, licenses, args.evidence_root, args.license_root)
            print(json.dumps(report, sort_keys=True))
            return 2 if report["missing"] else 0
        if args.command == "acquire":
            asset = assets[args.asset]
            acquire(asset, rights[asset["license_id"]], args.evidence_root, args.license_root)
        elif args.command == "extract":
            asset = assets[args.asset]
            print(json.dumps(extract_selected(asset, rights[asset["license_id"]], args.evidence_root, args.license_root, json.loads(args.selection.read_text()), args.max_bytes)))
        elif args.command == "annotate-import":
            annotation_import(args.input, args.evidence_root, args.name, args.sha256, args.audio_sha256)
        else:
            report = derive_concat([assets[item] for item in args.assets], rights, args.evidence_root, args.license_root, args.name, args.max_bytes)
            print(json.dumps(report, sort_keys=True))
        print(json.dumps({"status": "acquired"}))
        return 0
    except CorpusError as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "prerequisite_missing", "code": "corpus_operation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

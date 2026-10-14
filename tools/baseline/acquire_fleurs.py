"""Acquire selected FLEURS development audio from a fixed official revision without dataset packages."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import os
import re
from urllib.request import urlopen
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.corpus import CorpusError, acquire, extract_selected, require, safe_target
from tools.baseline.manifest import sha256

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CARD_HASH = "688f79f2a5c731af3796e9f683eb02f9b3f09d040decd8c5625d0f37098e71c6"
CONFIGS = {"it": "it_it", "en": "en_us", "fr": "fr_fr", "de": "de_de", "ja": "ja_jp", "zh": "cmn_hans_cn", "ar": "ar_eg", "uk": "uk_ua", "no": "nb_no"}


def read_public(url, limit=2 * 1024 * 1024):
    with urlopen(url, timeout=30) as source:
        result = source.read(limit + 1)
    require(len(result) <= limit, "metadata_budget_exceeded")
    return result


def keep_document(root, name, data):
    target = safe_target(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(sha256(target) == hashlib.sha256(data).hexdigest(), "existing_document_mismatch")
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=".document-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                require(sha256(target) == hashlib.sha256(data).hexdigest(), "existing_document_mismatch")
        finally:
            os.unlink(temporary)
    return target


def audio_info(path):
    probe = Path(__file__).resolve().parents[2] / "macOS/output/installer/WhisperX.app/Contents/Runtime/bin/ffprobe"
    result = subprocess.run([str(probe), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels,codec_name:format=duration", "-of", "json", str(path)], capture_output=True, check=True, timeout=20)
    info = json.loads(result.stdout)
    return {"sample_rate": int(info["streams"][0]["sample_rate"]), "channels": int(info["streams"][0]["channels"]), "duration": float(info["format"]["duration"]), "codec": info["streams"][0]["codec_name"]}


def legal_text(root, cached_sha256=None):
    if cached_sha256 is None:
        return read_public("https://groups.inf.ed.ac.uk/ami/corpus/license.shtml")
    require(isinstance(cached_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", cached_sha256), "license_cache_hash_invalid")
    path = safe_target(root, "licenses/CC-BY-4.0.html")
    require(path.is_file() and 0 < path.stat().st_size <= 2 * 1024 * 1024, "license_cache_invalid")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == cached_sha256 and b"Attribution 4.0" in data, "license_cache_mismatch")
    return data


def acquire_language(language, evidence_root, count=2, cached_license_sha256=None, raw_metadata=False, selection_offset=0):
    require(language in CONFIGS and 1 <= count <= 10, "selection_invalid")
    require(type(selection_offset) is int and 0 <= selection_offset <= 1000, "selection_offset_invalid")
    inventory_name = language + "-acquisition.json" if selection_offset == 0 else language + "-acquisition-offset-" + str(selection_offset) + "-count-" + str(count) + ".json"
    root = Path(evidence_root)
    require(root.is_dir() and not root.is_symlink(), "evidence_root_invalid")
    config = CONFIGS[language]
    base = "https://huggingface.co/datasets/google/fleurs/resolve/" + REVISION + "/"
    card = read_public("https://huggingface.co/datasets/google/fleurs/raw/" + REVISION + "/README.md")
    require(hashlib.sha256(card).hexdigest() == CARD_HASH and b"cc-by-4.0" in card, "license_card_mismatch")
    legal = legal_text(root, cached_license_sha256)
    require(b"Attribution 4.0" in legal, "license_text_invalid")
    keep_document(root, "licenses/fleurs-card.md", card)
    keep_document(root, "licenses/CC-BY-4.0.html", legal)
    archive_path = "data/" + config + "/audio/dev.tar.gz"
    rows = json.loads(read_public("https://huggingface.co/api/datasets/google/fleurs/tree/" + REVISION + "/data/" + config + "/audio"))
    entry = next((item for item in rows if item.get("path") == archive_path), None)
    require(entry is not None and entry.get("lfs", {}).get("oid"), "official_archive_identity_missing")
    asset = {"id": "fleurs-" + language + "-dev-archive", "path": "archives/" + language + "-dev.tar.gz", "url": base + archive_path, "revision": REVISION, "sha256": entry["lfs"]["oid"], "max_bytes": entry["size"], "license_id": "fleurs-cc-by-4.0"}
    license = {"id": asset["license_id"], "verified": True, "revision": REVISION, "source": "https://huggingface.co/datasets/google/fleurs/raw/" + REVISION + "/README.md", "text_path": "licenses/CC-BY-4.0.html", "text_sha256": hashlib.sha256(legal).hexdigest(), "legal_text_source": "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml", "attribution": "FLEURS: Conneau et al. (2022), Google, Few-shot Learning Evaluation of Universal Representations of Speech. CC BY 4.0. Selected unmodified development recordings.", "rights": {"download": True, "derive": True, "share": True}, "asset_ids": [asset["id"]], "dataset_card_sha256": CARD_HASH}
    if cached_license_sha256 is not None: license["legal_text_acquisition"] = "explicit_hash_verified_private_cache"
    archive = safe_target(root, asset["path"])
    if archive.exists():
        require(sha256(archive) == asset["sha256"], "existing_archive_mismatch")
    else:
        print(json.dumps({"stage": "acquisition", "language": language, "bytes": asset["max_bytes"]}), flush=True)
        acquire(asset, license, root, root)
    metadata_base = base.replace("/resolve/", "/raw/") if raw_metadata else base
    metadata = read_public(metadata_base + "data/" + config + "/dev.tsv")
    keep_document(root, "metadata/" + language + "-dev.tsv", metadata)
    candidates = sorted(csv.reader(io.StringIO(metadata.decode()), delimiter="\t"), key=lambda row: row[1])[selection_offset:selection_offset + count]
    require(len(candidates) == count, "source_selection_incomplete")
    require(all(len(row) >= 5 and row[1].endswith(".wav") and Path(row[1]).name == row[1] for row in candidates), "source_metadata_invalid")
    # The official archive uses dev/<filename>; no member path is inferred from a personal cache.
    selection = {"dev/" + row[1]: "audio/" + language + "/" + row[1] for row in candidates}
    present = [safe_target(root, name).exists() for name in selection.values()]
    require(not any(present) or all(present), "partial_extraction_requires_review")
    if all(present) and (root / inventory_name).exists():
        inventory = json.loads((root / inventory_name).read_text())
        require(all(sha256(safe_target(root, item["path"])) == item["sha256"] for item in inventory["audio"]), "existing_audio_mismatch")
        return inventory
    if all(present):
        # A prior interruption after extraction is recoverable only after re-deriving hashes from the verified archive.
        with tempfile.TemporaryDirectory(prefix=".recheck-", dir=root) as temporary:
            staging = Path(temporary)
            for name in (asset["path"], license["text_path"]):
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(root / name, target)
            extracted = extract_selected(asset, license, staging, staging, selection, 64 * 1024 * 1024)
            require(all(sha256(safe_target(root, item["path"])) == item["sha256"] for item in extracted), "existing_audio_mismatch")
    else:
        extracted = extract_selected(asset, license, root, root, selection, 64 * 1024 * 1024)
    for item, row in zip(extracted, candidates):
        item.update(audio_info(safe_target(root, item["path"])))
        item.update(id="fleurs-" + language + "-" + Path(row[1]).stem, language=language, license_id=license["id"], transcript=row[2], normalized_transcript=row[3], type="parity_only", annotation_status="word_times_and_speakers_unreviewed", speaker_identity_status="unavailable")
        license["asset_ids"].append(item["id"])
    inventory = {"schema_version": 1, "archive": asset, "license": license, "audio": extracted, "metadata_sha256": hashlib.sha256(metadata).hexdigest(), "status": "acquired_for_parity", "quality_certified": False}
    keep_document(root, inventory_name, (json.dumps(inventory, ensure_ascii=False, indent=2) + "\n").encode())
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True, help="Existing private directory; archives and samples never overwrite existing data.")
    parser.add_argument("--language", choices=CONFIGS, required=True)
    parser.add_argument("--count", type=int, default=2, help="1–10 deterministic clips; the first acquisition downloads one full official dev archive (IT 289 MB, EN 171 MB).")
    parser.add_argument("--cached-license-sha256", help="Optional exact SHA256 of previously acquired licenses/CC-BY-4.0.html in the evidence root; reuse that bounded, hash-verified legal text instead of fetching it again. The pinned dataset card is still fetched and verified.")
    parser.add_argument("--selection-offset", type=int, default=0, help="Optional deterministic offset in the sorted official TSV to acquire additional disjoint samples without replacing the initial inventory; each offset/count has its own immutable inventory.")
    parser.add_argument("--raw-metadata", action="store_true", help="Fetch the TSV from the official raw endpoint at the same fixed revision; optional alternative to the legacy resolve endpoint when it is rate limited. Audio archive URL/hash checks stay unchanged.")
    args = parser.parse_args(argv)
    try:
        result = acquire_language(args.language, args.evidence_root, args.count, args.cached_license_sha256, args.raw_metadata, args.selection_offset)
        print(json.dumps({"status": result["status"], "language": args.language, "clips": len(result["audio"]), "quality_certified": False}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration, wave.Error, subprocess.SubprocessError):
        print(json.dumps({"status": "prerequisite_missing", "code": "fleurs_acquisition_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

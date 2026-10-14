"""Acquire six licensed, speaker-disjoint AMI meetings and preserve manual references as unreviewed drafts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import acquire_fleurs, corpus, freeze, import_ami, manifest, provision

MEETINGS = {"calibration": ("ES2002a", "ES2007a", "ES2006a"), "evaluation": ("ES2004a", "ES2014a", "TS3003a")}
BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
LICENSE_SOURCE = "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml"


def plan(archive, archive_hash):
    corpus.require(corpus.valid_digest(archive_hash) and manifest.sha256(archive) == archive_hash, "ami_archive_identity_invalid")
    with zipfile.ZipFile(archive) as source:
        metadata, checksum = import_ami.xml_member(source, "corpusResources/meetings.xml")
    rows = []
    for split, meetings in MEETINGS.items():
        for meeting in meetings:
            matches = [item for item in metadata.findall("meeting") if item.get("observation") == meeting]
            corpus.require(len(matches) == 1, "ami_meeting_missing")
            people = [item.get("global_name") for item in matches[0].findall("speaker")]
            corpus.require(people and all(isinstance(person, str) and person for person in people) and len(set(people)) == len(people), "ami_speaker_identity_missing")
            rows.append({"meeting": meeting, "recording_id": "ami_" + meeting, "source_group": "ami_" + meeting[:-1], "speaker_ids": sorted("ami_person_" + hashlib.sha256(person.encode()).hexdigest()[:16] for person in people), "split": split})
    corpus.validate_splits(rows)
    corpus.require(len({person for row in rows for person in row["speaker_ids"]}) == sum(len(row["speaker_ids"]) for row in rows), "ami_speaker_reuse")
    return {"status": "source_split_plan_verified_not_quality", "recordings": rows, "archive_sha256": archive_hash, "meeting_metadata_sha256": checksum, "quality_certified": False}


def download(url, target, maximum, transport=None):
    with (transport(url) if transport else urlopen(url, timeout=30)) as source:
        location = urlsplit(source.geturl())
        corpus.require(location.scheme == "https" and location.hostname == "groups.inf.ed.ac.uk", "ami_redirect_unverified")
        declared = source.headers.get("Content-Length")
        corpus.require(isinstance(declared, str) and declared.isdigit(), "ami_response_size_missing")
        expected = int(declared)
        corpus.require(44 < expected <= maximum, "ami_audio_budget_exceeded")
        count = 0
        with target.open("xb") as output:
            while chunk := source.read(min(1024 * 1024, maximum + 1 - count)):
                count += len(chunk)
                corpus.require(count <= maximum, "ami_audio_budget_exceeded")
                output.write(chunk)
    corpus.require(count == expected, "ami_response_truncated")


def acquire(root, snapshot, archive, archive_hash, legal_path, legal_hash, reuse_root=None, reuse_inventory=None, download_workers=1, verified_audio_cache=None):
    corpus.require(Path(snapshot).name == snapshot and snapshot not in ("", ".", ".."), "snapshot_name_invalid")
    destination = corpus.safe_target(root, snapshot)
    corpus.require(not destination.exists(), "destination_exists")
    corpus.require(corpus.valid_digest(legal_hash) and 0 < legal_path.stat().st_size <= 2 * 1024 * 1024 and manifest.sha256(legal_path) == legal_hash, "ami_license_identity_invalid")
    legal_text = legal_path.read_bytes()
    corpus.require(b"The AMI corpus and its annotations are released" in legal_text and b"Attribution 4.0" in legal_text, "ami_license_scope_invalid")
    corpus.require(type(download_workers) is int and 1 <= download_workers <= 3, "ami_download_workers_invalid")
    selection = plan(archive, archive_hash)
    corpus.require((reuse_root is None) == (reuse_inventory is None), "ami_reuse_parameters_incomplete")
    if verified_audio_cache is not None: corpus.require(verified_audio_cache.is_dir() and not verified_audio_cache.is_symlink(), "ami_verified_cache_invalid")
    reusable = json.loads(reuse_inventory.read_text())["assets"] if reuse_inventory else []
    with tempfile.TemporaryDirectory(prefix=".ami-prepare-", dir=root) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        (staging / "LICENSE.html").write_bytes(legal_text)
        license = {"id": "ami-cc-by-4.0", "verified": True, "revision": "manual-1.6.2-sha256:" + archive_hash, "source": LICENSE_SOURCE, "text_path": snapshot + "/LICENSE.html", "text_sha256": legal_hash, "attribution": "AMI Meeting Corpus, AMI Consortium; speech transcription manual annotations version 1.6.2. CC BY 4.0.", "rights": {"download": True, "derive": True, "share": True}, "asset_ids": ["ami-" + row["meeting"] for row in selection["recordings"]]}
        def recording(row):
            meeting = row["meeting"]
            filename = meeting + ".Mix-Headset.wav"
            target = staging / filename
            asset = {"id": "ami-" + meeting, "license_id": license["id"]}
            corpus.verify_license(asset, dict(license, text_path="LICENSE.html"), staging, "download")
            prior = next((item for item in reusable if item.get("path") == filename), None)
            cached = corpus.safe_target(verified_audio_cache, meeting) if verified_audio_cache is not None else None
            if cached is not None and cached.exists():
                receipt = json.loads(corpus.safe_target(cached, "receipt.json").read_text())
                corpus.require(receipt.get("meeting") == meeting and receipt.get("url") == BASE + meeting + "/audio/" + filename and receipt.get("license_sha256") == legal_hash and receipt.get("archive_sha256") == archive_hash, "ami_cache_identity_changed")
                source = corpus.safe_target(cached, filename)
                corpus.require(manifest.sha256(source) == receipt.get("audio_sha256"), "ami_cache_audio_changed")
                shutil.copyfile(source, target)
                corpus.require(manifest.sha256(target) == receipt["audio_sha256"], "ami_cached_copy_changed")
            elif prior is not None:
                corpus.require(prior.get("license") == "CC-BY-4.0" and prior.get("license_source") == LICENSE_SOURCE and prior.get("license_sha256") == legal_hash, "ami_reuse_license_invalid")
                source = corpus.safe_target(reuse_root, prior["path"])
                corpus.require(manifest.sha256(source) == prior.get("sha256"), "ami_reuse_audio_changed")
                shutil.copyfile(source, target)
                corpus.require(manifest.sha256(target) == prior["sha256"], "ami_reuse_copy_changed")
            else:
                print(json.dumps({"stage": "acquisition", "meeting": meeting}), flush=True)
                download(BASE + meeting + "/audio/" + filename, target, 128 * 1024 * 1024)
            corpus.require(target.stat().st_size <= 128 * 1024 * 1024, "ami_audio_budget_exceeded")
            info = acquire_fleurs.audio_info(target)
            corpus.require(600 <= info["duration"] <= 3600 and info["sample_rate"] == 16000 and info["channels"] == 1 and info["codec"] == "pcm_s16le", "ami_audio_format_invalid")
            checksum = manifest.sha256(target)
            reference = import_ami.import_manual(archive, archive_hash, meeting, checksum, info["duration"])
            corpus.require(reference["speaker_ids"] == row["speaker_ids"] and reference["source_group"] == row["source_group"], "ami_reference_identity_changed")
            if cached is not None and not cached.exists():
                with tempfile.TemporaryDirectory(prefix=".ami-cache-", dir=verified_audio_cache) as cache_temporary:
                    publication = Path(cache_temporary) / "meeting"
                    publication.mkdir()
                    shutil.copyfile(target, publication / filename)
                    corpus.require(manifest.sha256(publication / filename) == checksum, "ami_cache_copy_changed")
                    freeze.seal(publication, "receipt.json", {"meeting": meeting, "url": BASE + meeting + "/audio/" + filename, "license_sha256": legal_hash, "archive_sha256": archive_hash, "audio_sha256": checksum, "bytes": target.stat().st_size, "quality_certified": False, "hash_provenance": "observed_official_https_acquisition_not_published_checksum"})
                    provision.rename_directory_exclusive(publication, cached)
            reference_name = meeting + ".manual-unreviewed.json"
            freeze.seal(staging, reference_name, reference)
            asset = {**asset, **info, "path": snapshot + "/" + filename, "sha256": checksum, "bytes": target.stat().st_size, "url": BASE + meeting + "/audio/" + filename, "type": "natural_candidate", "language": "en", "recording_id": row["recording_id"], "source_group": row["source_group"], "speaker_ids": row["speaker_ids"], "proposed_split": row["split"], "review_reference_path": snapshot + "/" + reference_name, "review_reference_sha256": manifest.sha256(staging / reference_name), "hash_provenance": "observed_official_https_acquisition_not_published_checksum", "quality_certified": False}
            print(json.dumps({"meeting": meeting, "status": "acquired_unreviewed", "proposed_split": row["split"]}), flush=True)
            return asset
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            assets = list(pool.map(recording, selection["recordings"]))
        document = {"status": "acquired_pending_independent_review", "source_plan": selection, "assets": assets, "cases": [], "licenses": [license], "quality_certified": False, "full_baseline_gate_passed": False}
        freeze.seal(staging, "inventory.json", document)
        provision.rename_directory_exclusive(staging, destination)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--snapshot")
    parser.add_argument("--license", type=Path)
    parser.add_argument("--license-sha256")
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--reuse-inventory", type=Path)
    parser.add_argument("--verified-audio-cache", type=Path, help="Optional existing private cache: atomically retain verified complete audio and receipts per meeting even if another acquisition fails; cached hashes/license/archive are rechecked before reuse.")
    parser.add_argument("--download-workers", type=int, choices=(1, 2, 3), default=1, help="Optional bounded concurrent meeting downloads; default remains sequential. Publication waits for every file and manual reference to verify.")
    parser.add_argument("--acquire", action="store_true", help="Explicitly download up to six official meetings, bounded to 128 MiB each, after verifying cached license and source split plan. Default is offline plan only.")
    parser.add_argument("--observe-official-hashes", action="store_true", help="Required with acquire: SHA256 is observed from the first official HTTPS acquisition, not an upstream published checksum. No quality attestation is created.")
    args = parser.parse_args(argv)
    if args.acquire and not (args.observe_official_hashes and args.evidence_root and args.snapshot and args.license and args.license_sha256): parser.error("acquire requires evidence-root, snapshot, license, license-sha256 and observe-official-hashes")
    try:
        if args.acquire:
            report = acquire(args.evidence_root.resolve(strict=True), args.snapshot, args.archive, args.archive_sha256, args.license, args.license_sha256, args.reuse_root, args.reuse_inventory, args.download_workers, args.verified_audio_cache)
            print(json.dumps({"status": report["status"], "recordings": len(report["assets"]), "quality_certified": False}))
        else: print(json.dumps(plan(args.archive, args.archive_sha256)))
        return 0
    except corpus.CorpusError as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error), "full_baseline_gate_passed": False}))
        return 2
    except (ValueError, OSError, KeyError, TypeError, StopIteration, zipfile.BadZipFile):
        print(json.dumps({"status": "prerequisite_missing", "code": "ami_acquisition_failed", "full_baseline_gate_passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

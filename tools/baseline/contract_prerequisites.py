"""Prepare and measure all synthetic contract prerequisites in a private, licensed evidence root."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts, corpus, freeze, manifest, provision, run


class PrerequisiteError(ValueError):
    pass


def require(condition, code):
    if not condition: raise PrerequisiteError(code)


def wav_file(path, samples):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(samples)


def prepare(root, name, catalog, licenses, source_id, runtime, encoder=None):
    require(Path(name).name == name and name not in ("", ".", ".."), "snapshot_name_invalid")
    target = corpus.safe_target(root, name)
    require(not target.exists(), "destination_exists")
    asset = next(item for item in catalog["assets"] if item["id"] == source_id)
    legal = next(item for item in licenses["licenses"] if item["id"] == asset["license_id"])
    corpus.verify_license(asset, legal, root, "derive")
    source = corpus.safe_target(root, asset["path"])
    require(manifest.sha256(source) == asset["sha256"], "source_hash_mismatch")
    ffmpeg = Path(runtime).parent / "ffmpeg"
    encoder = Path(encoder) if encoder is not None else ffmpeg
    decoded = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(source), "-t", "120", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=True).stdout
    require(32000 <= len(decoded) <= 120 * 32000 and len(decoded) % 2 == 0, "fixture_source_size_invalid")
    with tempfile.TemporaryDirectory(prefix=".contract-prepare-", dir=root) as temporary:
        staging = Path(temporary) / "snapshot"
        fixtures = staging / "fixtures"
        fixtures.mkdir(parents=True)
        base = decoded[:5 * 32000]
        wav_file(fixtures / "base.wav", base)
        wav_file(fixtures / "silence.wav", b"\0" * 64000)
        wav_file(fixtures / "empty.wav", b"")
        tone = b"".join(struct.pack("<h", round(1200 * math.sin(2 * math.pi * 440 * index / 16000))) for index in range(32000))
        wav_file(fixtures / "non_speech.wav", tone)
        wav_file(fixtures / "Voce è_日本.wav", base)
        wav_file(fixtures / "window_boundaries.wav", decoded * math.ceil(31 * 32000 / len(decoded)))
        for extension, codec in (("mp3", "libmp3lame"), ("m4a", "aac")):
            subprocess.run([str(encoder), "-v", "error", "-nostdin", "-n", "-i", str(fixtures / "base.wav"), "-c:a", codec, str(fixtures / ("codec." + extension))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=True)
        subprocess.run([str(encoder), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "color=c=black:s=160x90:r=5", "-i", str(fixtures / "base.wav"), "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(fixtures / "codec.mp4")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=True)
        assets = []
        for path in sorted(fixtures.iterdir()):
            require(path.stat().st_size <= 16 * 1024 * 1024, "fixture_size_invalid")
            assets.append({"id": "contract-" + path.name.replace(".", "-"), "path": name + "/fixtures/" + path.name, "sha256": manifest.sha256(path), "bytes": path.stat().st_size, "license_id": legal["id"], "type": "derived_audio", "language": asset.get("language", "it"), "recipe": {"operation": "synthetic_contract_fixture_v1", "source_hashes": [asset["sha256"]], "decoder_sha256": manifest.sha256(ffmpeg), "encoder_sha256": manifest.sha256(encoder), "not_independent_recording": True}, "quality_eligible": False})
        updated_licenses = copy.deepcopy(licenses)
        selected_license = next(item for item in updated_licenses["licenses"] if item["id"] == legal["id"])
        selected_license["asset_ids"] = sorted(set(selected_license["asset_ids"] + [item["id"] for item in assets]))
        filenames = {"silence": "silence.wav", "non_speech": "non_speech.wav", "empty": "empty.wav", "unicode_paths": "Voce è_日本.wav", "codec_mp3": "codec.mp3", "codec_m4a": "codec.m4a", "codec_mp4": "codec.mp4", "window_boundaries": "window_boundaries.wav"}
        cases = [{"id": identifier, "type": "synthetic_contract", "asset_id": "contract-" + filenames.get(identifier, "base.wav").replace(".", "-"), "execution_scope": "alignment_export_boundary" if identifier in ("unknown_chars", "out_of_range") else "startup_termination" if identifier in ("cancelled", "interrupted") else "original_worker", "settings": {"quality": "Rapida", "language": "it", "voices": identifier == "invalid_token", "hints": ""}} for identifier in corpus.CONTRACT_CASES]
        document = {"schema_version": 1, "snapshot": name, "source_asset_sha256": asset["sha256"], "catalog": catalog, "licenses": updated_licenses, "assets": assets, "cases": cases, "quality_certified": False}
        with (staging / "fixture-inventory.json").open("x") as output: json.dump(document, output, ensure_ascii=False, indent=2)
        (staging / "missing-model-cache").mkdir()
        provision.rename_directory_exclusive(staging, target)
    return document


def execute_pair(root, snapshot, case, asset, runtime, source_root, hf_cache, torch_cache):
    pair = corpus.safe_target(root, snapshot + "/pairs/" + case["id"])
    require(not pair.exists(), "pair_destination_exists")
    pair.mkdir(parents=True)
    worker = source_root / "macOS/Worker/transcribe_worker.py"
    outputs, audits, outcomes = [], [], []
    for instrumented in (False, True):
        output = pair / ("instrumented" if instrumented else "original")
        request = {**case["settings"], "audio": str(corpus.safe_target(root, asset["path"])), "output": str(output)}
        if case["id"] == "invalid_token": request["token"] = ""
        if case["execution_scope"] == "alignment_export_boundary":
            arguments = [Path(__file__).with_name("contract_entry.py"), "--source-root", source_root, "--kind", case["id"]]
            if instrumented: arguments += ["--instrumented", "--spans", pair / "spans.json"]
        elif instrumented:
            arguments = [Path(__file__).with_name("adapter.py"), "--worker", worker, "--source-root", source_root, "--spans", pair / "spans.json", "--seed", "1729", "--preserve-failure-message"]
        else:
            arguments = [Path(__file__).with_name("offline_worker.py"), "--worker", worker, "--source-root", source_root, "--seed", "1729"]
        cache = corpus.safe_target(root, snapshot + "/missing-model-cache") if case["id"] == "model_failure" else hf_cache
        audit, timer = {}, None
        try:
            if case["id"] == "cancelled":
                timer = threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT))
                timer.start()
            events = run.execute(runtime, arguments, request, 0.2 if case["id"] == "interrupted" else 180, cache, torch_home=torch_cache, allow_failure=True, process_audit=audit)
            outcome = events[-1]["event"]
            bundle = run.semantic_bundle(output, events, Path(request["audio"]).stem) if outcome == "completed" else {"events": events}
            outputs.append(contracts.normalize(bundle, str(output)))
        except run.RunError as error:
            expected = "job_cancelled" if case["id"] == "cancelled" else "job_interrupted" if case["id"] == "interrupted" else None
            require(str(error) == expected, "unexpected_worker_error")
            outcome = "cancelled" if case["id"] == "cancelled" else "interrupted"
            outputs.append({"outcome": outcome})
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
        require(audit.get("process_group_alive") is False and audit.get("stdin_closed") is True and audit.get("stdout_closed") is True, "worker_process_cleanup_missing")
        audits.append(audit)
        outcomes.append(outcome)
    expected = "failed" if case["id"] in ("invalid_token", "model_failure") else case["id"] if case["id"] in ("cancelled", "interrupted") else "completed"
    require(outcomes == [expected, expected], "contract_outcome_mismatch")
    require(outputs[0] == outputs[1], "contract_semantic_mismatch")
    if case["id"] == "empty": require(outputs[0]["artifacts"]["result.json"]["segments"] == [], "empty_contract_mismatch")
    return {"case_id": case["id"], "type": "synthetic_contract", "status": "contract_verified", "oracle_adapter_equivalent": True, "measured": True, "execution_scope": case["execution_scope"], "outcome": expected, "semantic_digest": freeze.digest(outputs[0]), "asset_sha256": asset["sha256"], "processes": audits, "termination_scope": "startup_deadline_not_all_pipeline_stages" if expected in ("cancelled", "interrupted") else None, "invalid_token_scope": "empty_credential_guard_not_http_authentication" if case["id"] == "invalid_token" else None, "asr_performed": case["execution_scope"] == "original_worker" and expected == "completed", "quality_certified": False, "gpu_placement_certified": False}


def measure(root, snapshot, runtime, source_root, specs):
    if corpus.safe_target(root, snapshot + "/contract-evidence.json").exists():
        from tools.baseline.contract_audit import audit
        audit(root, snapshot, runtime, source_root, specs)
        return json.loads(corpus.safe_target(root, snapshot + "/contract-evidence.json").read_text())["cases"]
    inventory = json.loads(corpus.safe_target(root, snapshot + "/fixture-inventory.json").read_text())
    require(len({asset["id"] for asset in inventory["assets"]}) == len(inventory["assets"]) and [case["id"] for case in inventory["cases"]] == list(corpus.CONTRACT_CASES), "fixture_inventory_invalid")
    models = [provision.inspect_spec(spec) for spec in specs]
    require(all(model["status"] == "files_verified" for model in models), "models_not_verified")
    hf_cache, torch_cache = run.model_cache_roots(specs)
    require(hf_cache is not None and torch_cache is not None, "model_cache_missing")
    source = manifest.source_identity(source_root, ["macOS/Worker/transcribe_worker.py"] + [str(path.relative_to(source_root)) for path in (source_root / "whisperx").rglob("*.py")])
    harness_root = Path(__file__).resolve().parents[2]
    harness = manifest.source_identity(harness_root, ["tools/baseline/contract_prerequisites.py", "tools/baseline/contract_entry.py", "tools/baseline/adapter.py", "tools/baseline/run.py"])
    identities = {"source": source, "harness": harness, "source_digest": freeze.digest(source), "harness_digest": freeze.digest(harness)}
    archived = corpus.safe_target(root, snapshot + "/execution-identities.json")
    if archived.exists():
        require(json.loads(archived.read_text()) == identities, "resume_identity_changed")
    else:
        for item in harness["files"]:
            destination = corpus.safe_target(root, snapshot + "/harness-source/" + item["name"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not destination.exists(), "harness_archive_exists")
            shutil.copyfile(harness_root / item["name"], destination)
        freeze.seal(root, snapshot + "/execution-identities.json", identities)
    identity = {"source_digest": freeze.digest(source), "runtime_digest": manifest.sha256(Path(runtime).resolve()), "model_digest": freeze.digest(models), "harness_digest": freeze.digest(harness)}
    rows = []
    for case in inventory["cases"]:
        asset = next(item for item in inventory["assets"] if item["id"] == case["asset_id"])
        legal = next(item for item in inventory["licenses"]["licenses"] if item["id"] == asset["license_id"])
        corpus.verify_license(asset, legal, root, "download")
        require(manifest.sha256(corpus.safe_target(root, asset["path"])) == asset["sha256"], "fixture_hash_mismatch")
        name = snapshot + "/results/" + case["id"] + ".json"
        destination = corpus.safe_target(root, name)
        if destination.exists():
            record = json.loads(destination.read_text())
            require(all(record.get(key) == value for key, value in identity.items()) and record.get("asset_sha256") == asset["sha256"] and record.get("status") == "contract_verified", "resume_identity_changed")
        else:
            record = {**execute_pair(root, snapshot, case, asset, runtime, source_root, hf_cache, torch_cache), **identity}
            freeze.seal(root, name, record)
        rows.append(record)
        print(json.dumps({"case_id": case["id"], "status": record["status"], "outcome": record["outcome"], "execution_scope": record["execution_scope"]}), flush=True)
    catalog = copy.deepcopy(inventory["catalog"])
    require(not set(corpus.CONTRACT_CASES) & {case["id"] for case in catalog["cases"]}, "contract_catalog_conflict")
    catalog["assets"].extend(inventory["assets"])
    catalog["cases"].extend(inventory["cases"])
    freeze.seal(root, snapshot + "/catalog.json", catalog)
    freeze.seal(root, snapshot + "/licenses.json", inventory["licenses"])
    freeze.seal(root, snapshot + "/contract-evidence.json", {"schema_version": 1, "cases": rows, "required_cases": list(corpus.CONTRACT_CASES), "quality_certified": False, "full_baseline_gate_passed": False})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True, help="New private snapshot name for prepare, existing snapshot for run; no overwrite.")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--catalog", type=Path, required=True)
    prepare_command.add_argument("--licenses", type=Path, required=True)
    prepare_command.add_argument("--source-asset", required=True)
    prepare_command.add_argument("--encoder-ffmpeg", type=Path, help="Optional existing FFmpeg with MP3/AAC/video encoders for fixture preparation only; measured worker decoding still uses the unchanged bundled runtime. Both binaries are fingerprinted.")
    execute = commands.add_parser("run")
    execute.add_argument("--model-specs", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.evidence_root.resolve(strict=True)
        if args.command == "prepare":
            document = prepare(root, args.snapshot, json.loads(args.catalog.read_text()), json.loads(args.licenses.read_text()), args.source_asset, args.runtime, args.encoder_ffmpeg)
            print(json.dumps({"status": "fixtures_prepared_unmeasured", "cases": len(document["cases"]), "quality_certified": False}))
        else:
            rows = measure(root, args.snapshot, args.runtime, args.source_root.resolve(strict=True), json.loads(args.model_specs.read_text()))
            print(json.dumps({"status": "contracts_verified", "cases": len(rows), "full_baseline_gate_passed": False}))
        return 0
    except (PrerequisiteError, corpus.CorpusError, provision.ProvisionError, run.RunError, freeze.FreezeError) as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error), "full_baseline_gate_passed": False}))
    except Exception:
        print(json.dumps({"status": "prerequisite_missing", "code": "contract_prerequisite_failed", "full_baseline_gate_passed": False}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

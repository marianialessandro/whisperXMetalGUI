"""Offline alignment selection/config inventory, with optional HF processor inspection and no checkpoint loading."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import manifest, provision
from tools.baseline.offline_worker import deny_python_network


class InventoryError(ValueError):
    pass


def alignment_defaults(source, variable="DEFAULT_ALIGN_MODELS_TORCH"):
    payload = Path(source).read_bytes()
    if len(payload) > 1024 * 1024: raise InventoryError("alignment_source_budget_exceeded")
    tree = ast.parse(payload)
    matches = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)]
    if len(matches) != 1: raise InventoryError("alignment_defaults_not_supported")
    try:
        defaults = ast.literal_eval(matches[0].value)
    except (ValueError, TypeError):
        raise InventoryError("alignment_defaults_not_supported") from None
    if not isinstance(defaults, dict) or not defaults or any(not isinstance(key, str) or not isinstance(value, str) for key, value in defaults.items()): raise InventoryError("alignment_defaults_not_supported")
    return defaults, hashlib.sha256(payload).hexdigest()


def collect_huggingface(source_root, specs, processor_module=None):
    selected = [spec for spec in specs if spec.get("transport", "huggingface") == "huggingface" and str(spec.get("role", "")).startswith("alignment_")]
    if not selected: return [], manifest.sha256(manifest.contained(source_root, "whisperx/alignment.py"))
    defaults, source_digest = alignment_defaults(manifest.contained(source_root, "whisperx/alignment.py"), "DEFAULT_ALIGN_MODELS_HF")
    torch_defaults, torch_digest = alignment_defaults(manifest.contained(source_root, "whisperx/alignment.py"))
    if torch_digest != source_digest: raise InventoryError("alignment_source_changed")
    if processor_module is None:
        import transformers
        processor_module = transformers
    records, roles = [], set()
    for spec in selected:
        role = spec["role"]
        language = role.removeprefix("alignment_")
        if role in roles or language in torch_defaults or defaults.get(language) != spec.get("repository"): raise InventoryError("alignment_default_repository_mismatch")
        roles.add(role)
        if spec.get("transformers_version") != processor_module.__version__: raise InventoryError("alignment_runtime_version_mismatch")
        revision = spec.get("revision", "")
        if not isinstance(revision, str) or len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision): raise InventoryError("alignment_revision_invalid")
        required = {"config.json", "preprocessor_config.json", "vocab.json"}
        metadata = required | ({"tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "tokenizer.json"} & set(spec.get("files", [])))
        if not required.issubset(spec.get("files", [])): raise InventoryError("alignment_metadata_missing")
        relative_snapshot = Path("models--" + spec["repository"].replace("/", "--")) / "snapshots" / revision
        snapshot = manifest.contained(Path(spec["cache_root"]), relative_snapshot)
        if any((snapshot / name).exists() and name not in spec["files"] for name in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "tokenizer.json")): raise InventoryError("alignment_metadata_undeclared")
        hashes, documents = {}, {}
        for name in sorted(metadata):
            path = manifest.contained(Path(spec["cache_root"]), relative_snapshot / name)
            if not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024 * 1024: raise InventoryError("alignment_metadata_invalid")
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != spec.get("sha256", {}).get(name): raise InventoryError("alignment_metadata_hash_mismatch")
            hashes[name], documents[name] = digest, json.loads(payload)
        config = documents["config.json"]
        if not isinstance(config, dict) or config.get("model_type") != "wav2vec2" or type(config.get("vocab_size")) is not int or not 0 < config["vocab_size"] <= 262144: raise InventoryError("alignment_model_config_not_supported")
        weights = [name for name in ("model.safetensors", "pytorch_model.bin") if name in spec["files"]]
        if len(weights) != 1 or weights[0] == "pytorch_model.bin" and (snapshot / "model.safetensors").exists(): raise InventoryError("alignment_checkpoint_selection_mismatch")
        with provision.quiet_download():
            processor = processor_module.Wav2Vec2Processor.from_pretrained(str(snapshot), local_files_only=True)
        vocabulary = processor.tokenizer.get_vocab()
        if not isinstance(vocabulary, dict) or not 0 < len(vocabulary) <= 262144 or any(not isinstance(token, str) or type(code) is not int or code < 0 for token, code in vocabulary.items()) or processor.feature_extractor.sampling_rate != 16000: raise InventoryError("alignment_processor_config_not_supported")
        dictionary = {token.lower(): code for token, code in vocabulary.items()}
        digest = hashlib.sha256(json.dumps(dictionary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        records.append({"role": role, "repository": spec["repository"], "revision": revision, "checkpoint": weights[0], "transformers_version": processor_module.__version__, "model_type": config["model_type"], "vocab_size": config["vocab_size"], "vocabulary_count": len(vocabulary), "dictionary_count": len(dictionary), "dictionary_sha256": digest, "out_of_range_vocabulary_ids": sorted({code for code in vocabulary.values() if code >= config["vocab_size"]}), "sample_rate": 16000, "metadata_sha256": hashes, "status": "default_selection_and_processor_config_verified", "checkpoint_loaded": False})
    return records, source_digest


def collect(source_root, specs, audio_module=None, include_hf=False, processor_module=None):
    source = manifest.contained(source_root, "whisperx/alignment.py")
    defaults, source_digest = alignment_defaults(source)
    if not isinstance(specs, list): raise InventoryError("alignment_specs_invalid")
    if audio_module is None:
        import torchaudio
        audio_module = torchaudio
    records = []
    roles = set()
    for spec in specs:
        if spec.get("transport") != "torch_hub": continue
        role = spec.get("role", "")
        if not role.startswith("alignment_") or role in roles: raise InventoryError("alignment_role_not_supported")
        roles.add(role)
        language = role.removeprefix("alignment_")
        name = defaults.get(language)
        if name is None or spec.get("repository") != "torchaudio/" + name or name not in audio_module.pipelines.__all__: raise InventoryError("alignment_default_bundle_mismatch")
        bundle = getattr(audio_module.pipelines, name)
        checkpoint = bundle._path
        if Path(checkpoint).name != checkpoint or spec.get("files") != [checkpoint] or spec.get("source") != "https://download.pytorch.org/torchaudio/models/" + checkpoint: raise InventoryError("alignment_checkpoint_selection_mismatch")
        if audio_module.__version__ != spec.get("torchaudio_version"): raise InventoryError("alignment_runtime_version_mismatch")
        if json.loads(json.dumps(bundle._params)) != spec.get("bundle_config") or list(bundle.get_labels()) != spec.get("labels"): raise InventoryError("alignment_bundle_configuration_mismatch")
        implementation = inspect.getsourcefile(type(bundle))
        if implementation is None: raise InventoryError("alignment_bundle_source_unavailable")
        records.append({"role": role, "repository": spec["repository"], "bundle": name, "checkpoint": checkpoint, "torchaudio_version": audio_module.__version__, "sample_rate": bundle.sample_rate, "model_type": bundle._model_type, "normalize_waveform": bundle._normalize_waveform, "remove_aux_axes": list(bundle._remove_aux_axis), "bundle_config": spec["bundle_config"], "labels": spec["labels"], "bundle_source_sha256": manifest.sha256(implementation), "status": "default_selection_and_config_verified", "checkpoint_loaded": False})
    if include_hf:
        hf_records, hf_source_digest = collect_huggingface(source_root, specs, processor_module)
        if hf_source_digest != source_digest: raise InventoryError("alignment_source_changed")
        records.extend(hf_records)
    if not records: raise InventoryError("alignment_torch_inventory_empty")
    return {"scope": "offline_alignment_selection_inventory", "status": "selection_inventory_verified", "alignment_source_sha256": source_digest, "models": records, "checkpoint_loaded": False, "loadability_verified": False, "gpu_placement_certified": False, "quality_certified": False, "full_baseline_gate_passed": False}


def inspect_runtime(runtime, source_root, specs_path, include_hf=False):
    try:
        with tempfile.TemporaryDirectory(prefix="whisperx-alignment-inventory-") as directory:
            output = Path(directory) / "inventory.json"
            completed = subprocess.run([str(Path(runtime).resolve()), "-I", "-B", str(Path(__file__).resolve()), *(["--include-hf"] if include_hf else []), "--source-root", str(Path(source_root).resolve()), "--model-specs", str(Path(specs_path).resolve()), "--out", str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, env=manifest.clean_environment(Path(runtime).resolve().parent.parent, {}))
            if completed.returncode != 0 or not output.is_file() or output.stat().st_size > 1024 * 1024: raise InventoryError("alignment_runtime_inventory_failed")
            report = json.loads(output.read_text())
            if report.get("status") != "selection_inventory_verified" or report.get("alignment_source_sha256") != manifest.sha256(manifest.contained(source_root, "whisperx/alignment.py")): raise InventoryError("alignment_runtime_inventory_failed")
            return report
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "prerequisite_missing", "code": "alignment_runtime_inventory_failed", "models": []}


def matches_spec(spec, inventory):
    if not inventory or inventory.get("status") != "selection_inventory_verified": return False
    records = [record for record in inventory.get("models", []) if record.get("role") == spec.get("role")]
    if len(records) != 1: return False
    record = records[0]
    if spec.get("transport", "huggingface") == "huggingface":
        required = {"config.json", "preprocessor_config.json", "vocab.json"}
        expected = required | ({"tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "tokenizer.json"} & set(spec.get("files", [])))
        metadata = record.get("metadata_sha256", {})
        return record.get("status") == "default_selection_and_processor_config_verified" and record.get("checkpoint_loaded") is False and not record.get("out_of_range_vocabulary_ids") and all(record.get(key) == spec.get(key) for key in ("repository", "revision", "transformers_version")) and record.get("checkpoint") in spec.get("files", []) and set(metadata) == expected and all(digest == spec.get("sha256", {}).get(name) for name, digest in metadata.items())
    return record.get("status") == "default_selection_and_config_verified" and record.get("checkpoint_loaded") is False and record.get("repository") == spec.get("repository") and [record.get("checkpoint")] == spec.get("files") and record.get("torchaudio_version") == spec.get("torchaudio_version") and record.get("bundle_config") == spec.get("bundle_config") and record.get("labels") == spec.get("labels")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Selected WhisperX import root; reads literal alignment defaults without importing it.")
    parser.add_argument("--model-specs", type=Path, required=True, help="Existing private pinned inventory; legacy mode checks Torch bundle selection/config, optional --include-hf also checks HF metadata hashes. Never loads neural checkpoints.")
    parser.add_argument("--out", type=Path, help="Optional new private inventory JSON; refuses overwrite.")
    parser.add_argument("--include-hf", action="store_true", help="Optional offline HF default/revision/config/hash/tokenizer inventory in the selected runtime. Loads bounded processor metadata only, never neural checkpoints or inference; legacy Torch-only inventory is unchanged.")
    args = parser.parse_args(argv)
    try:
        with deny_python_network():
            report = collect(args.source_root, json.loads(args.model_specs.read_text()), include_hf=args.include_hf)
        if args.out is not None:
            with args.out.open("x") as output: json.dump(report, output, indent=2)
        print(json.dumps({"status": report["status"], "models": len(report["models"]), "checkpoint_loaded": False, "full_baseline_gate_passed": False}))
        return 0
    except InventoryError as error:
        code = str(error)
    except Exception:
        code = "alignment_inventory_failed"
    print(json.dumps({"status": "prerequisite_missing", "code": code, "checkpoint_loaded": False, "full_baseline_gate_passed": False}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

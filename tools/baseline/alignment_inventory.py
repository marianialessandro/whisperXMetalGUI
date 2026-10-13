"""Offline Torch alignment selection/config inventory, without checkpoint loading or inference."""

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
from tools.baseline import manifest
from tools.baseline.offline_worker import deny_python_network


class InventoryError(ValueError):
    pass


def alignment_defaults(source):
    payload = Path(source).read_bytes()
    if len(payload) > 1024 * 1024: raise InventoryError("alignment_source_budget_exceeded")
    tree = ast.parse(payload)
    matches = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DEFAULT_ALIGN_MODELS_TORCH" for target in node.targets)]
    if len(matches) != 1: raise InventoryError("alignment_defaults_not_supported")
    try:
        defaults = ast.literal_eval(matches[0].value)
    except (ValueError, TypeError):
        raise InventoryError("alignment_defaults_not_supported") from None
    if not isinstance(defaults, dict) or not defaults or any(not isinstance(key, str) or not isinstance(value, str) for key, value in defaults.items()): raise InventoryError("alignment_defaults_not_supported")
    return defaults, hashlib.sha256(payload).hexdigest()


def collect(source_root, specs, audio_module=None):
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
    if not records: raise InventoryError("alignment_torch_inventory_empty")
    return {"scope": "offline_alignment_selection_inventory", "status": "selection_inventory_verified", "alignment_source_sha256": source_digest, "models": records, "checkpoint_loaded": False, "loadability_verified": False, "gpu_placement_certified": False, "quality_certified": False, "full_baseline_gate_passed": False}


def inspect_runtime(runtime, source_root, specs_path):
    try:
        with tempfile.TemporaryDirectory(prefix="whisperx-alignment-inventory-") as directory:
            output = Path(directory) / "inventory.json"
            completed = subprocess.run([str(Path(runtime).resolve()), "-I", "-B", str(Path(__file__).resolve()), "--source-root", str(Path(source_root).resolve()), "--model-specs", str(Path(specs_path).resolve()), "--out", str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, env=manifest.clean_environment(Path(runtime).resolve().parent.parent, {}))
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
    return record.get("status") == "default_selection_and_config_verified" and record.get("checkpoint_loaded") is False and record.get("repository") == spec.get("repository") and [record.get("checkpoint")] == spec.get("files") and record.get("torchaudio_version") == spec.get("torchaudio_version") and record.get("bundle_config") == spec.get("bundle_config") and record.get("labels") == spec.get("labels")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Selected WhisperX import root; reads its literal Torch alignment defaults without importing it.")
    parser.add_argument("--model-specs", type=Path, required=True, help="Existing private pinned model inventory; checks Torch bundle selection/config only, not hashes or loadability.")
    parser.add_argument("--out", type=Path, help="Optional new private inventory JSON; refuses overwrite.")
    args = parser.parse_args(argv)
    try:
        with deny_python_network():
            report = collect(args.source_root, json.loads(args.model_specs.read_text()))
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

"""Offline Silero default/cache/source inventory, without source execution or JIT model loading."""

import argparse
import ast
import hashlib
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


def tree_digest(spec):
    return hashlib.sha256(json.dumps(spec["sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collect(source_root, specs, torch_module=None):
    choices = [spec for spec in specs if spec.get("transport") == "torch_hub_source"]
    if len(choices) != 1: raise InventoryError("vad_source_inventory_invalid")
    spec = choices[0]
    identity = provision.inspect_torch_source(spec)
    if identity["status"] != "files_verified": raise InventoryError(identity["code"])
    source = manifest.contained(source_root, "whisperx/vads/silero.py")
    payload = source.read_bytes()
    if len(payload) > 1024 * 1024: raise InventoryError("vad_source_budget_exceeded")
    calls = [node for node in ast.walk(ast.parse(payload)) if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.hub.load"]
    if len(calls) != 1 or calls[0].args or any(keyword.arg is None for keyword in calls[0].keywords): raise InventoryError("vad_default_loader_mismatch")
    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in calls[0].keywords}
    if options != {"repo_or_dir": "snakers4/silero-vad", "model": "silero_vad", "force_reload": False, "onnx": False, "trust_repo": True}: raise InventoryError("vad_default_loader_mismatch")
    if torch_module is None:
        import torch
        torch_module = torch
    if torch_module.__version__ != spec.get("torch_version"): raise InventoryError("vad_runtime_version_mismatch")
    cache = manifest.contained(Path(spec["cache_root"]), "hub")
    if (cache / "snakers4_silero-vad_main").exists(): raise InventoryError("vad_default_cache_conflict")
    torch_module.hub.set_dir(str(cache))
    if torch_module.hub._parse_repo_info("snakers4/silero-vad") != ("snakers4", "silero-vad", "master"): raise InventoryError("vad_default_cache_selection_mismatch")
    checkpoint = "src/silero_vad/data/silero_vad.jit"
    record = {"role": "vad", "repository": spec["repository"], "revision": spec["revision"], "torch_version": torch_module.__version__, "cache_branch": "master", "checkpoint": checkpoint, "checkpoint_sha256": spec["sha256"][checkpoint], "source_tree_digest": tree_digest(spec), "hub_loader_sha256": manifest.sha256(torch_module.hub.__file__), "status": "default_cached_source_verified", "checkpoint_loaded": False}
    return {"scope": "offline_vad_selection_inventory", "status": "selection_inventory_verified", "vad_source_sha256": hashlib.sha256(payload).hexdigest(), "models": [record], "checkpoint_loaded": False, "loadability_verified": False, "quality_certified": False, "gpu_placement_certified": False, "full_baseline_gate_passed": False}


def matches_spec(spec, report):
    if not report or report.get("status") != "selection_inventory_verified": return False
    records = report.get("models", [])
    if len(records) != 1: return False
    record = records[0]
    return record.get("status") == "default_cached_source_verified" and record.get("checkpoint_loaded") is False and record.get("cache_branch") == "master" and all(record.get(key) == spec.get(key) for key in ("repository", "revision", "torch_version", "role")) and record.get("source_tree_digest") == tree_digest(spec) and record.get("checkpoint_sha256") == spec.get("sha256", {}).get("src/silero_vad/data/silero_vad.jit")


def inspect_runtime(runtime, source_root, specs_path):
    try:
        with tempfile.TemporaryDirectory(prefix="whisperx-vad-inventory-") as directory:
            output = Path(directory) / "inventory.json"
            child = subprocess.run([str(Path(runtime).resolve()), "-I", "-B", str(Path(__file__).resolve()), "--source-root", str(Path(source_root).resolve()), "--model-specs", str(Path(specs_path).resolve()), "--out", str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, env=manifest.clean_environment(Path(runtime).resolve().parent.parent, {}))
            if child.returncode != 0 or not output.is_file() or output.stat().st_size > 1024 * 1024: raise InventoryError("vad_runtime_inventory_failed")
            report = json.loads(output.read_text())
            if report.get("status") != "selection_inventory_verified" or report.get("vad_source_sha256") != manifest.sha256(manifest.contained(source_root, "whisperx/vads/silero.py")): raise InventoryError("vad_runtime_inventory_failed")
            return report
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "prerequisite_missing", "code": "vad_runtime_inventory_failed", "models": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-specs", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Optional new private report; never overwrite.")
    args = parser.parse_args(argv)
    try:
        with deny_python_network(): report = collect(args.source_root, json.loads(args.model_specs.read_text()))
        if args.out is not None:
            with args.out.open("x") as output: json.dump(report, output, indent=2)
        print(json.dumps({"status": report["status"], "checkpoint_loaded": False, "full_baseline_gate_passed": False}))
        return 0
    except InventoryError as error:
        code = str(error)
    except Exception:
        code = "vad_inventory_failed"
    print(json.dumps({"status": "prerequisite_missing", "code": code, "checkpoint_loaded": False, "full_baseline_gate_passed": False}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

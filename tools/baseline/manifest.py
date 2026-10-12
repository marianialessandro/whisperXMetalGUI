"""Development-only identity collection; no inference, network or credential access."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import uuid


class BaselineError(ValueError):
    """Only fixed error codes cross the persistence boundary."""


CACHE_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")
PACKAGES = ("torch", "torchaudio", "mlx", "mlx-whisper", "ctranslate2", "faster-whisper", "pyannote.audio", "pyannote.metrics", "numpy", "scipy", "transformers", "huggingface-hub")
SAFE_KEYS = frozenset("schema_version run_id status code source selection git_sha reference_sha digest files name sha256 bytes unique_bytes repository revision role format shape dtype hardware machine chip ram_bytes os sdk runtime executable lock runtime_manifest installed loaded packages version models prerequisites settings language quality voices speaker_count hints_sha256 hints_present imported_sources backend precision".split())
SAFE_KEYS |= frozenset("loadability_verified native_conversion_proven tensors harness".split())
SAFE_KEYS |= frozenset("seed deterministic_algorithms_forced".split())


def failure(code, error=None):
    # Exception messages may contain paths and secrets and are never retained.
    if not re.fullmatch(r"[a-z][a-z0-9_]*", code):
        raise BaselineError("invalid_error_code")
    return {"status": "prerequisite_missing", "code": code}


def sanitize(value, secrets=()):
    if isinstance(value, dict):
        return {key: sanitize(item, secrets) for key, item in value.items() if key in SAFE_KEYS or key in PACKAGES}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, secrets) for item in value]
    if isinstance(value, str):
        if any(secret and secret in value for secret in secrets) or re.search(r"hf_[A-Za-z0-9]+|/Users/|/home/|Bearer\s|://[^/\s]+@", value):
            return "[redacted]"
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and value == value and abs(value) != float("inf"):
        return value
    raise BaselineError("unsafe_metadata")


def clean_environment(runtime, inherited):
    environment = {key: inherited[key] for key in CACHE_KEYS if key in inherited}
    environment.update(PATH=f"{runtime}/bin:/usr/bin:/bin:/usr/sbin:/sbin", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return environment


def contained(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise BaselineError("path_escape")
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise BaselineError("path_escape")
    return target


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=20, env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_sha(root):
    top = command("/usr/bin/git", "-C", str(root), "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != Path(root).resolve():
        return None
    value = command("/usr/bin/git", "-C", str(root), "rev-parse", "HEAD")
    return value if value and re.fullmatch(r"[0-9a-f]{40}", value) else None


def source_identity(root, relative_files):
    files, missing = [], False
    for name in sorted(set(relative_files)):
        try:
            path = contained(root, name)
            files.append({"name": name, "sha256": sha256(path), "bytes": path.stat().st_size})
        except (OSError, BaselineError):
            files.append({"name": name, **failure("source_missing")})
            missing = True
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"status": "prerequisite_missing" if missing or not files else "identified", "git_sha": git_sha(root), "digest": digest, "files": files}


def model_identity(cache_root, repository, revision, required_files):
    identity = {"repository": repository, "revision": revision, "files": []}
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        return {**identity, **failure("model_repository_invalid")}
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return {**identity, **failure("model_revision_missing")}
    if not required_files:
        return {**identity, **failure("model_files_undeclared")}
    unique, total = set(), 0
    try:
        for name in required_files:
            prefix = Path("models--" + repository.replace("/", "--")) / "snapshots" / revision
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise BaselineError("path_escape")
            path = contained(cache_root, prefix / relative)
            stat = path.stat()
            if not path.is_file():
                raise BaselineError("model_file_invalid")
            identity["files"].append({"name": name, "sha256": sha256(path), "bytes": stat.st_size, "format": path.suffix.lstrip(".") or "unknown", "shape": None})
            inode = (stat.st_dev, stat.st_ino)
            if inode not in unique:
                total += stat.st_size
                unique.add(inode)
    except BaselineError as error:
        return {**identity, **failure(str(error))}
    except OSError:
        return {**identity, **failure("model_file_missing")}
    # Identified is deliberately distinct from verified loadability or model readiness.
    return {**identity, "unique_bytes": total, "status": "identified"}


def write_evidence(root, name, document, secrets=()):
    root = Path(root)
    if root.is_symlink():
        raise BaselineError("evidence_symlink")
    target = contained(root, name)
    current = root / Path(name)
    while current != root:
        if current.is_symlink():
            raise BaselineError("evidence_symlink")
        current = current.parent
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(sanitize(document, secrets), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".baseline-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise BaselineError("evidence_exists") from None
    finally:
        os.unlink(temporary)
    return target


def collect(selection, source_root, runtime, reference_root, model_specs=(), settings=None):
    source_root, runtime = Path(source_root), Path(runtime)
    if selection == "gui-checkout":
        worker, package = "macOS/Worker/transcribe_worker.py", "whisperx"
        required = [worker, "macOS/Packaging/requirements.lock", "macOS/Sources/WhisperXMac/TranscriptionEngine.swift", "macOS/Sources/WhisperXMac/ProjectModels.swift", "macOS/Sources/WhisperXMac/LibraryStore.swift"]
    elif selection == "gui-bundle":
        worker, package = "Contents/Resources/transcribe_worker.py", "Contents/Resources/whisperx"
        required = [worker, "Contents/Resources/RuntimeManifest.json"]
    else:
        raise BaselineError("oracle_selection_invalid")
    required += [str(path.relative_to(source_root)) for path in sorted((source_root / package).rglob("*.py"))]
    source = source_identity(source_root, required)
    prerequisites = []
    if not (source_root / package / "__init__.py").is_file():
        prerequisites.append(failure("source_package_missing"))
    if source["status"] != "identified": prerequisites.append(failure("source_missing"))
    reference_sha = git_sha(reference_root)
    if reference_sha is None: prerequisites.append(failure("reference_missing"))
    try:
        executable = {"sha256": sha256(runtime.resolve(strict=True)), "bytes": runtime.stat().st_size}
    except OSError:
        executable = failure("runtime_missing")
        prerequisites.append(executable)
    runtime_manifest = runtime.parent.parent / "../Resources/RuntimeManifest.json"
    metadata_script = "import importlib.metadata as m,json,sys; names=json.loads(sys.argv[1]); result={};\nfor name in names:\n try: result[name]=m.version(name)\n except m.PackageNotFoundError: result[name]=None\nprint(json.dumps(result))"
    installed = command(str(runtime), "-I", "-B", "-c", metadata_script, json.dumps(PACKAGES))
    versions = json.loads(installed) if installed else {name: None for name in PACKAGES}
    if not installed: prerequisites.append(failure("runtime_metadata_missing"))
    if not runtime_manifest.is_file(): prerequisites.append(failure("runtime_manifest_missing"))
    models = [model_identity(spec["cache_root"], spec["repository"], spec["revision"], spec["files"]) for spec in model_specs]
    if not models: prerequisites.append(failure("model_inventory_missing"))
    for model in models:
        if model["status"] != "identified": prerequisites.append(failure(model["code"]))
    ram = command("/usr/sbin/sysctl", "-n", "hw.memsize")
    hardware = {"machine": platform.machine(), "chip": command("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"), "ram_bytes": int(ram) if ram and ram.isdecimal() else None, "os": platform.mac_ver()[0], "sdk": command("/usr/bin/xcrun", "--show-sdk-version")}
    safe_settings = {}
    if settings is not None:
        from tools.baseline.contracts import validate_settings
        validate_settings(settings)
        safe_settings = {key: settings[key] for key in ("language", "quality", "voices", "speaker_count") if key in settings}
        safe_settings.update(hints_present=bool(settings.get("hints")), hints_sha256=hashlib.sha256(settings.get("hints", "").encode()).hexdigest())
    runtime_version = command(str(runtime), "-I", "-B", "-c", "import sys; print(sys.version.split()[0])")
    result = {"schema_version": 1, "run_id": str(uuid.uuid4()), "selection": selection, "source": source, "reference_sha": reference_sha, "hardware": hardware, "runtime": {"version": runtime_version, "executable": executable, "runtime_manifest": {"sha256": sha256(runtime_manifest)} if runtime_manifest.is_file() else failure("runtime_manifest_missing"), "installed": versions, "loaded": {}}, "settings": safe_settings, "models": models, "prerequisites": prerequisites, "status": "prerequisite_missing" if prerequisites else "identified"}
    return sanitize(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("gui-checkout", "gui-bundle"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path, help="Read-only whisper.cpp checkout.")
    parser.add_argument("--evidence-root", required=True, type=Path, help="Existing dedicated directory; existing reports are never overwritten.")
    parser.add_argument("--model-specs", type=Path, help="Local declarations with cache_root, repository, immutable revision and files; never credentials.")
    args = parser.parse_args(argv)
    try:
        specs = json.loads(args.model_specs.read_text()) if args.model_specs else []
        report = collect(args.source, args.source_root, args.runtime, args.reference_root, specs)
        write_evidence(args.evidence_root, report["run_id"] + ".json", report)
        print(json.dumps({"run_id": report["run_id"], "status": report["status"], "prerequisites": report["prerequisites"]}))
        return 2 if report["prerequisites"] else 0
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps(failure("manifest_collection_failed")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

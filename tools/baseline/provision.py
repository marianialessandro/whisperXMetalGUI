"""Explicit pinned model provisioning for development; credentials remain in private pipes."""

import argparse
import ast
import contextlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.manifest import BaselineError, contained, model_identity, sanitize, sha256


class ProvisionError(ValueError):
    pass


@contextlib.contextmanager
def quiet_download():
    # Native libraries and pre-existing log handlers can bypass Python's redirect_stderr.
    saved = [(descriptor, os.dup(descriptor)) for descriptor in (1, 2)]
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for descriptor, previous in saved: os.dup2(sink, descriptor)
        with open(os.devnull, "w") as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            yield
    finally:
        for descriptor, previous in saved:
            os.dup2(previous, descriptor)
            os.close(previous)
        os.close(sink)


def read_credential(bridge, launcher=subprocess.Popen):
    child = launcher([str(bridge)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"})
    try:
        data, _ = child.communicate(timeout=15)
        code = child.returncode
        if code != 0 or not 0 < len(data) <= 4096:
            raise ProvisionError("keychain_unavailable")
        token = data.decode("utf-8").strip()
        if not token or any(character.isspace() for character in token):
            raise ProvisionError("credential_invalid")
        return token
    except (OSError, ValueError, subprocess.SubprocessError):
        raise ProvisionError("keychain_unavailable") from None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def tensor_header(path):
    if path.suffix == ".npz":
        return npz_headers(path)
    if path.suffix != ".safetensors":
        return {"status": "conversion_unresolved", "format": path.suffix.lstrip(".")}
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ProvisionError("tensor_header_invalid")
        length = struct.unpack("<Q", prefix)[0]
        if length > 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ProvisionError("tensor_header_invalid")
        header = json.loads(source.read(length))
    tensors = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape) or not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(offset, int) for offset in offsets) or not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - 8 - length:
            raise ProvisionError("tensor_header_invalid")
        tensors.append({"name": name, "shape": shape, "dtype": tensor.get("dtype")})
    return {"status": "header_inspected", "format": "safetensors", "tensors": tensors, "native_conversion_proven": False}


def npz_headers(path):
    """Read bounded numeric NPY headers inside NPZ without loading tensors or pickle."""
    tensors = []
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > 10000 or len({entry.filename for entry in entries}) != len(entries) or sum(entry.file_size for entry in entries) > 16 * 1024 ** 3:
                raise ProvisionError("tensor_header_invalid")
            for entry in entries:
                if not entry.filename.endswith(".npy") or "/" in entry.filename or "\\" in entry.filename or entry.flag_bits & 1 or entry.file_size > max(entry.compress_size, 1) * 256:
                    raise ProvisionError("tensor_header_invalid")
                with archive.open(entry) as source:
                    prefix = source.read(8)
                    if prefix not in (b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00", b"\x93NUMPY\x03\x00"):
                        raise ProvisionError("tensor_header_invalid")
                    width = 2 if prefix[6] == 1 else 4
                    raw_length = source.read(width)
                    if len(raw_length) != width:
                        raise ProvisionError("tensor_header_invalid")
                    length = int.from_bytes(raw_length, "little")
                    if length > 65536 or 8 + width + length > entry.file_size:
                        raise ProvisionError("tensor_header_invalid")
                    header = ast.literal_eval(source.read(length).decode("utf-8" if prefix[6] == 3 else "latin-1"))
                shape, dtype = header.get("shape"), header.get("descr")
                if not isinstance(shape, tuple) or len(shape) > 32 or any(type(dimension) is not int or dimension < 0 for dimension in shape) or not isinstance(dtype, str) or not re.fullmatch(r"[<>=|][biufc](1|2|4|8|16)", dtype) or type(header.get("fortran_order")) is not bool:
                    raise ProvisionError("tensor_header_invalid")
                elements = 1
                for dimension in shape: elements *= dimension
                if elements * int(dtype[2:]) != entry.file_size - 8 - width - length:
                    raise ProvisionError("tensor_header_invalid")
                tensors.append({"name": entry.filename[:-4], "shape": list(shape), "dtype": dtype, "fortran_order": header["fortran_order"]})
    except (OSError, ValueError, SyntaxError, EOFError, AttributeError, zipfile.BadZipFile):
        raise ProvisionError("tensor_header_invalid") from None
    return {"status": "header_inspected", "format": "npz", "tensors": tensors, "native_conversion_proven": False}


def inspect_spec(spec):
    if spec.get("transport") == "torch_hub":
        return inspect_torch_spec(spec)
    identity = model_identity(Path(spec["cache_root"]), spec["repository"], spec["revision"], spec["files"])
    if identity["status"] != "identified":
        return identity
    expected = spec.get("sha256", {})
    for item in identity["files"]:
        if not re.fullmatch(r"[0-9a-f]{64}", expected.get(item["name"], "")):
            return {**identity, "status": "prerequisite_missing", "code": "model_expected_hash_missing"}
        if item["sha256"] != expected[item["name"]]:
            return {**identity, "status": "prerequisite_missing", "code": "model_hash_mismatch"}
    identity["status"] = "files_verified"
    identity["loadability_verified"] = False
    identity["native_conversion_proven"] = False
    return identity


def inspect_torch_spec(spec):
    result = {"repository": spec.get("repository"), "revision": spec.get("revision"), "files": [], "status": "prerequisite_missing", "loadability_verified": False, "native_conversion_proven": False}
    if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")) or not spec.get("files"):
        return {**result, "code": "model_revision_or_files_missing"}
    for name in spec["files"]:
        if Path(name).name != name or not name.endswith((".pt", ".pth")):
            return {**result, "code": "model_path_invalid"}
        try:
            path = contained(Path(spec["cache_root"]), Path("hub/checkpoints") / name)
            if not path.is_file(): return {**result, "code": "model_file_missing"}
            expected = spec.get("sha256", {}).get(name, "")
            if not re.fullmatch(r"[0-9a-f]{64}", expected): return {**result, "code": "model_expected_hash_missing"}
            digest = sha256(path)
            if digest != expected: return {**result, "code": "model_hash_mismatch"}
            result["files"].append({"name": name, "bytes": path.stat().st_size, "sha256": digest, "format": path.suffix.lstrip(".")})
        except (OSError, BaselineError):
            return {**result, "code": "model_file_missing"}
    result["status"] = "files_verified"
    return result


def provision_spec(spec, token=None, downloader=None):
    if spec.get("transport") == "torch_hub":
        raise ProvisionError("torch_hub_explicit_acquisition_required")
    if not spec.get("license_verified") or not spec.get("license_source") or not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
        raise ProvisionError("model_authorization_or_revision_missing")
    if spec.get("gated") and token is None:
        raise ProvisionError("keychain_unavailable")
    if not spec.get("files") or any(not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", {}).get(name, "")) for name in spec["files"]):
        raise ProvisionError("model_expected_hash_missing")
    for name in spec["files"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ProvisionError("model_path_invalid")
    # Third party diagnostics are discarded before they can enter any evidence stream.
    try:
        with quiet_download():
            constants = None
            previous_xet = None
            if downloader is None:
                from huggingface_hub import hf_hub_download
                from huggingface_hub import constants
                previous_xet = constants.HF_HUB_DISABLE_XET
                constants.HF_HUB_DISABLE_XET = True
                downloader = hf_hub_download
            try:
                for name in spec["files"]:
                    downloader(repo_id=spec["repository"], filename=name, revision=spec["revision"], cache_dir=spec["cache_root"], token=token if token else False, local_files_only=False)
            finally:
                if constants is not None: constants.HF_HUB_DISABLE_XET = previous_xet
    except Exception:
        raise ProvisionError("model_download_failed") from None
    result = inspect_spec(spec)
    if result["status"] != "files_verified":
        raise ProvisionError(result["code"])
    return result


def bind_default_reference(spec):
    if spec.get("transport") == "torch_hub": raise ProvisionError("torch_hub_has_no_huggingface_reference")
    if spec.get("license_verified") is not True or not spec.get("license_source", "").startswith("https://"): raise ProvisionError("model_authorization_or_revision_missing")
    result = inspect_spec(spec)
    if result["status"] != "files_verified": raise ProvisionError(result.get("code", "model_not_verified"))
    target = contained(Path(spec["cache_root"]), Path("models--" + spec["repository"].replace("/", "--")) / "refs/main")
    if target.exists():
        if target.read_text().strip() != spec["revision"]: raise ProvisionError("model_default_reference_conflict")
        return {**result, "default_reference_status": "already_pinned"}
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".baseline-ref-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(spec["revision"])
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
            status = "created_from_verified_snapshot"
        except FileExistsError:
            if target.read_text().strip() != spec["revision"]: raise ProvisionError("model_default_reference_conflict")
            status = "already_pinned"
    finally:
        os.unlink(temporary)
    return {**result, "default_reference_status": status}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True, help="Declared repositories, full revisions, required files, expected hashes and individually verified license/access records.")
    parser.add_argument("--bridge", type=Path, help="Development bridge compiled under the private evidence directory; used only for explicit gated downloads.")
    parser.add_argument("--bind-default-ref", action="store_true", help="Optional explicit cache metadata write after file/hash/license verification: create missing HF refs/main at the declared revision, never overwrite a conflicting ref. Torch Hub is unchanged; preflight remains offline.")
    parser.add_argument("command", choices=("preflight", "download"), help="preflight verifies HF/torch_hub files without model loading or Keychain/network; download provisions pinned HF snapshots only; torch_hub checkpoints require separate explicit acquisition.")
    args = parser.parse_args(argv)
    try:
        specs = json.loads(args.specs.read_text())
        token = read_credential(args.bridge) if args.command == "download" and any(spec.get("gated") for spec in specs) and args.bridge else None
        with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
            reports = [inspect_spec(spec) if args.command == "preflight" else provision_spec(spec, token) for spec in specs]
            if args.bind_default_ref:
                reports = [bind_default_reference(spec) if spec.get("transport") != "torch_hub" else report for spec, report in zip(specs, reports)]
        report = sanitize({"status": "files_verified" if reports and all(report["status"] == "files_verified" for report in reports) else "prerequisite_missing", "models": reports}, secrets=[token] if token else [])
        token = None
        print(json.dumps(report))
        return 0 if reports and all(report["status"] == "files_verified" for report in reports) else 2
    except ProvisionError as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "prerequisite_missing", "code": "model_provision_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

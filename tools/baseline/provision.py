"""Explicit pinned model provisioning for development; credentials remain in private pipes."""

import argparse
import ast
import contextlib
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.manifest import BaselineError, contained, model_identity, sanitize, sha256


class ProvisionError(ValueError):
    pass


TORCH_MAX_BYTES = 1024 ** 3


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
    if spec.get("transport") == "torch_hub_source":
        return inspect_torch_source(spec)
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
    if "components" in spec:
        try:
            identity.update(inspect_components(spec))
        except (ProvisionError, BaselineError, OSError, ValueError, TypeError, KeyError):
            return {**identity, "status": "prerequisite_missing", "code": "model_component_config_mismatch"}
    elif isinstance(spec.get("role"), str): identity["roles"] = [spec["role"]]
    return identity


def inspect_components(spec):
    expected = {"speaker_segmentation": ("segmentation", ["segmentation/pytorch_model.bin"]), "speaker_embedding": ("embedding", ["embedding/pytorch_model.bin"]), "speaker_plda": ("plda", ["plda/plda.npz", "plda/xvec_transform.npz"])}
    components = spec["components"]
    if spec["repository"] != "pyannote/speaker-diarization-community-1" or not isinstance(components, dict) or not components or set(components) - set(expected) or "config.yaml" not in spec["files"]: raise ProvisionError("model_component_config_mismatch")
    try:
        import yaml
    except ImportError:
        raise ProvisionError("model_component_config_mismatch") from None
    prefix = Path("models--" + spec["repository"].replace("/", "--")) / "snapshots" / spec["revision"]
    config = contained(Path(spec["cache_root"]), prefix / "config.yaml")
    if config.stat().st_size > 1024 * 1024: raise ProvisionError("model_component_config_mismatch")
    try:
        document = yaml.safe_load(config.read_text())
    except yaml.YAMLError:
        raise ProvisionError("model_component_config_mismatch") from None
    if not isinstance(document, dict) or not isinstance(document.get("pipeline"), dict) or not isinstance(document["pipeline"].get("params"), dict): raise ProvisionError("model_component_config_mismatch")
    pipeline = document["pipeline"]
    if pipeline["name"] != "pyannote.audio.pipelines.SpeakerDiarization": raise ProvisionError("model_component_config_mismatch")
    roles = {spec["role"]} if isinstance(spec.get("role"), str) else set()
    for role, files in components.items():
        name, required = expected[role]
        if files != required or set(files) - set(spec["files"]) or pipeline["params"].get(name) != "$model/" + name: raise ProvisionError("model_component_config_mismatch")
        roles.add(role)
    return {"roles": sorted(roles), "component_config_sha256": sha256(config), "components_verified": True}


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
    if isinstance(spec.get("role"), str): result["roles"] = [spec["role"]]
    return result


def provision_spec(spec, token=None, downloader=None):
    if spec.get("transport") == "torch_hub_source": raise ProvisionError("torch_hub_source_explicit_installation_required")
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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ProvisionError("model_download_redirect_rejected")


def torch_source_target(spec):
    required = {"LICENSE", "hubconf.py", "src/silero_vad/model.py", "src/silero_vad/utils_vad.py", "src/silero_vad/data/silero_vad.jit", ".baseline-source-revision"}
    files = spec.get("files")
    if spec.get("transport") != "torch_hub_source" or spec.get("repository") != "snakers4/silero-vad" or spec.get("role") != "vad" or not isinstance(spec.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]): raise ProvisionError("torch_source_spec_invalid")
    if not isinstance(files, list) or not 0 < len(files) <= 4096 or any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or not Path(name).parts for name in files) or len(set(files)) != len(files) or not required.issubset(files): raise ProvisionError("model_path_invalid")
    if not isinstance(spec.get("sha256"), dict) or any(not isinstance(spec["sha256"].get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"][name]) for name in files): raise ProvisionError("model_expected_hash_missing")
    return contained(Path(spec["cache_root"]), "hub/snakers4_silero-vad_master")


def inspect_torch_source(spec):
    result = {"repository": spec.get("repository"), "revision": spec.get("revision"), "status": "prerequisite_missing", "files": [], "loadability_verified": False, "native_conversion_proven": False}
    try:
        target = torch_source_target(spec)
        if not target.is_dir(): return {**result, "code": "model_file_missing"}
        if target.is_symlink(): raise ProvisionError("model_path_invalid")
        observed = set()
        entries = 0
        for path in target.rglob("*"):
            entries += 1
            if entries > 8192: raise ProvisionError("torch_source_tree_mismatch")
            if path.is_symlink(): raise ProvisionError("model_path_invalid")
            if path.is_file(): observed.add(path.relative_to(target).as_posix())
            if len(observed) > 4096: raise ProvisionError("torch_source_tree_mismatch")
        if observed != set(spec["files"]): raise ProvisionError("torch_source_tree_mismatch")
        total_bytes = 0
        for name in spec["files"]:
            path = contained(Path(spec["cache_root"]), Path("hub/snakers4_silero-vad_master") / name)
            if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024: raise ProvisionError("model_file_missing")
            total_bytes += path.stat().st_size
            if total_bytes > 128 * 1024 * 1024: raise ProvisionError("torch_source_tree_mismatch")
            digest = sha256(path)
            if digest != spec["sha256"][name]: raise ProvisionError("model_hash_mismatch")
            result["files"].append({"name": name, "sha256": digest, "bytes": path.stat().st_size})
        if (target / ".baseline-source-revision").read_text() != spec["revision"] + "\n": raise ProvisionError("torch_source_revision_mismatch")
    except ProvisionError as error:
        return {**result, "code": str(error)}
    except (OSError, BaselineError, ValueError, TypeError):
        return {**result, "code": "torch_source_inspection_failed"}
    return {**result, "status": "files_verified", "roles": ["vad"]}


def rename_directory_exclusive(source, target):
    if sys.platform != "darwin": raise ProvisionError("torch_source_atomic_install_platform_unsupported")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    rename = library.renamex_np
    rename.argtypes, rename.restype = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
    if rename(os.fsencode(source), os.fsencode(target), 0x00000004) != 0:
        raise ProvisionError("model_acquisition_conflict" if ctypes.get_errno() == errno.EEXIST else "torch_source_publication_failed")


def install_torch_source(spec, archive_path):
    target = torch_source_target(spec)
    if spec.get("license_verified") is not True or spec.get("license_source") != "https://raw.githubusercontent.com/snakers4/silero-vad/" + spec["revision"] + "/LICENSE": raise ProvisionError("model_authorization_or_revision_missing")
    if target.exists():
        report = inspect_torch_source(spec)
        if report["status"] != "files_verified": raise ProvisionError(report["code"])
        return {**report, "acquisition_status": "already_verified"}
    archive_path = Path(archive_path)
    if spec.get("source") != "https://codeload.github.com/snakers4/silero-vad/zip/" + spec["revision"] or archive_path.stat().st_size > 32 * 1024 * 1024 or sha256(archive_path) != spec.get("archive_sha256"): raise ProvisionError("torch_source_archive_mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".baseline-source-", dir=target.parent) as temporary, zipfile.ZipFile(archive_path) as archive:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            prefix = "silero-vad-" + spec["revision"] + "/"
            entries = archive.infolist()
            if len(entries) > 8192 or sum(entry.file_size for entry in entries) > 128 * 1024 * 1024: raise ProvisionError("torch_source_archive_invalid")
            files = {}
            for entry in entries:
                if not entry.filename.startswith(prefix) or entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > 32 * 1024 * 1024: raise ProvisionError("torch_source_archive_invalid")
                if entry.is_dir(): continue
                name = entry.filename.removeprefix(prefix)
                if name not in spec["files"] or name in files or entry.file_size > max(entry.compress_size, 1) * 256: raise ProvisionError("torch_source_archive_invalid")
                files[name] = entry
            if set(files) != set(spec["files"]) - {".baseline-source-revision"}: raise ProvisionError("torch_source_tree_mismatch")
            for name, entry in files.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(entry) as source, path.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if digest.hexdigest() != spec["sha256"][name]: raise ProvisionError("model_hash_mismatch")
            marker = (spec["revision"] + "\n").encode()
            if hashlib.sha256(marker).hexdigest() != spec["sha256"][".baseline-source-revision"]: raise ProvisionError("torch_source_revision_mismatch")
            with (staging / ".baseline-source-revision").open("xb") as output:
                output.write(marker)
                output.flush()
                os.fsync(output.fileno())
            rename_directory_exclusive(staging, target)
    except ProvisionError:
        raise
    except Exception:
        raise ProvisionError("torch_source_install_failed") from None
    report = inspect_torch_source(spec)
    if report["status"] != "files_verified": raise ProvisionError(report["code"])
    return {**report, "acquisition_status": "installed_verified_source_snapshot"}


def provision_torch_spec(spec, observe_hash=False, opener=None):
    """Acquire opaque official checkpoint bytes without deserialization or replacement."""
    files = spec.get("files")
    if spec.get("transport") != "torch_hub" or not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(pt|pth)", files[0]): raise ProvisionError("model_path_invalid")
    name = files[0]
    source = "https://download.pytorch.org/torchaudio/models/" + name
    repository, role = spec.get("repository"), spec.get("role")
    if spec.get("source") != source or not isinstance(repository, str) or not repository.startswith("torchaudio/") or not isinstance(role, str) or not role.startswith("alignment_"): raise ProvisionError("model_source_invalid")
    license_source, revision = spec.get("license_source"), spec.get("revision")
    if spec.get("license_verified") is not True or not isinstance(license_source, str) or not license_source.startswith("https://") or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision): raise ProvisionError("model_authorization_or_revision_missing")
    hashes = spec.get("sha256", {})
    if not isinstance(hashes, dict): raise ProvisionError("model_expected_hash_missing")
    expected = hashes.get(name, "")
    if observe_hash:
        if expected: raise ProvisionError("model_observation_requires_unpinned_spec")
    elif not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected): raise ProvisionError("model_expected_hash_missing")
    target = contained(Path(spec["cache_root"]), Path("hub/checkpoints") / name)
    if target.exists():
        if observe_hash: raise ProvisionError("model_observation_requires_new_acquisition")
        result = inspect_torch_spec(spec)
        if result["status"] != "files_verified": raise ProvisionError(result["code"])
        return {**result, "acquisition_status": "already_verified"}
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".baseline-torch-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if opener is None: opener = urllib.request.build_opener(NoRedirect()).open
            started = time.monotonic()
            with opener(urllib.request.Request(source, headers={"Accept-Encoding": "identity"}), timeout=30) as response:
                if response.geturl() != source or response.status != 200 or response.headers.get("Content-Encoding", "identity") != "identity": raise ProvisionError("model_download_response_invalid")
                raw_length = response.headers.get("Content-Length")
                length = int(raw_length) if raw_length is not None else None
                limit = TORCH_MAX_BYTES
                if length is not None and not 0 < length <= limit: raise ProvisionError("model_download_size_invalid")
                digest, size = hashlib.sha256(), 0
                while True:
                    if time.monotonic() - started > 600: raise ProvisionError("model_download_timeout")
                    chunk = response.read(1024 * 1024)
                    if not chunk: break
                    size += len(chunk)
                    if size > limit: raise ProvisionError("model_download_size_invalid")
                    digest.update(chunk)
                    output.write(chunk)
                if size == 0 or length is not None and size != length: raise ProvisionError("model_download_size_invalid")
                observed = digest.hexdigest()
                if not observe_hash and observed != expected: raise ProvisionError("model_hash_mismatch")
                output.flush()
                os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise ProvisionError("model_acquisition_conflict") from None
    except ProvisionError:
        raise
    except Exception:
        raise ProvisionError("model_download_failed") from None
    finally:
        os.unlink(temporary)
    if observe_hash:
        return {"repository": spec["repository"], "revision": spec["revision"], "status": "acquired_hash_observed_unpinned", "files": [{"name": name, "sha256": observed, "bytes": size}], "hash_provenance": "observed_official_https_first_acquisition_not_published_checksum", "source_revision_scope": "upstream_training_repository_not_download_content", "loadability_verified": False, "native_conversion_proven": False}
    result = inspect_torch_spec(spec)
    if result["status"] != "files_verified": raise ProvisionError(result["code"])
    return {**result, "acquisition_status": "downloaded_and_verified"}


def bind_default_reference(spec):
    if spec.get("transport") in ("torch_hub", "torch_hub_source"): raise ProvisionError("torch_hub_has_no_huggingface_reference")
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
    parser.add_argument("--download-torch", action="store_true", help="Explicitly allow official HTTPS Torch checkpoint acquisition during download, with pinned SHA256 verification and no checkpoint loading or overwrite.")
    parser.add_argument("--observe-torch-hash", action="store_true", help="With download --download-torch only: record the first official acquisition hash for an unpinned new checkpoint; exit 2 until the observed digest is explicitly pinned in the specs. Never a published checksum or verified role.")
    parser.add_argument("--install-torch-source", type=Path, help="Optional explicit download-mode installation of a hash-pinned official Silero GitHub ZIP into an absent Torch Hub master cache, using bounded verified extraction and macOS exclusive atomic directory publication. No source execution or checkpoint loading.")
    parser.add_argument("command", choices=("preflight", "download"), help="preflight verifies HF/torch_hub files without model loading or Keychain/network; download defaults to pinned HF snapshots only, with explicit --download-torch required for Torch acquisition.")
    args = parser.parse_args(argv)
    if args.download_torch and args.command != "download" or args.observe_torch_hash and not args.download_torch: parser.error("Torch acquisition options require download --download-torch")
    if args.install_torch_source and args.command != "download": parser.error("Torch source installation requires download")
    try:
        specs = json.loads(args.specs.read_text())
        token = read_credential(args.bridge) if args.command == "download" and any(spec.get("gated") for spec in specs) and args.bridge else None
        with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
            reports = [inspect_spec(spec) if args.command == "preflight" else install_torch_source(spec, args.install_torch_source) if spec.get("transport") == "torch_hub_source" and args.install_torch_source else provision_torch_spec(spec, args.observe_torch_hash) if spec.get("transport") == "torch_hub" and args.download_torch else provision_spec(spec, token) for spec in specs]
            if args.bind_default_ref:
                reports = [bind_default_reference(spec) if spec.get("transport") not in ("torch_hub", "torch_hub_source") else report for spec, report in zip(specs, reports)]
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

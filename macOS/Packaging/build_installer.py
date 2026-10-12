"""Build a self-contained Apple Silicon app and a single macOS Installer package."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile

from ffmpeg_build import prepare as prepare_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
MAGICS = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def run(*arguments, **kwargs):
    subprocess.run([str(value) for value in arguments], check=True, **kwargs)


def output(*arguments):
    return subprocess.check_output([str(value) for value in arguments], text=True).strip()


def macho(path):
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) in MAGICS


def dependencies(path):
    identifiers = set(output("otool", "-D", path).splitlines()[1:])
    return [dependency for line in output("otool", "-L", path).splitlines()[1:] if (dependency := line.strip().split(" (compatibility version", 1)[0]) not in identifiers]


def system_library(path):
    return path.startswith(("/usr/lib/", "/System/Library/"))


def bundle_native(runtime, ffmpeg):
    """Copy external dylibs recursively and rewrite absolute install names."""
    run("ditto", ffmpeg / "bin/ffmpeg", runtime / "bin/ffmpeg")
    run("ditto", ffmpeg / "bin/ffprobe", runtime / "bin/ffprobe")
    queue = [path for path in runtime.rglob("*") if macho(path)]
    names = {}
    for path in queue:
        names.setdefault(path.name, []).append(path)
    origins = {}
    # TorchCodec loads the versioned FFmpeg libraries by name at runtime.
    for source in sorted((ffmpeg / "lib").glob("*.dylib")):
        if source.name.startswith(("libav", "libsw", "libpostproc")):
            destination = runtime / "lib" / source.name
            shutil.copy2(source.resolve(), destination)
            origins[destination] = source.resolve()
            queue.append(destination)
            names.setdefault(destination.name, []).append(destination)
    visited = set()
    copied = {}
    while queue:
        target = queue.pop()
        if target in visited:
            continue
        visited.add(target)
        target.chmod(target.stat().st_mode | 0o200)
        for dependency in dependencies(target):
            if dependency.startswith("@rpath/"):
                name = dependency.removeprefix("@rpath/")
                local = target.parent / name
                matches = names.get(name, [])
                destination = local if local.is_file() else matches[0] if len(matches) == 1 else None
                if destination:
                    run("install_name_tool", "-change", dependency, "@loader_path/" + os.path.relpath(destination, target.parent), target)
                continue
            if not dependency.startswith("/") or system_library(dependency):
                continue
            source = Path(dependency).resolve(strict=True)
            if source.is_relative_to(runtime):
                destination = source
            else:
                destination = runtime / "lib" / Path(dependency).name
                previous = copied.get(destination)
                if previous and previous != source:
                    raise RuntimeError(f"Conflicting libraries: {previous} and {source}")
                copied[destination] = source
                if destination not in origins:
                    shutil.copy2(source, destination)
                    origins[destination] = source
                    queue.append(destination)
                    names.setdefault(destination.name, []).append(destination)
            relative = os.path.relpath(destination, target.parent)
            run("install_name_tool", "-change", dependency, "@loader_path/" + relative, target)
        if target.suffix == ".dylib":
            run("install_name_tool", "-id", "@loader_path/" + target.name, target)
    for target in visited:
        for dependency in dependencies(target):
            if dependency.startswith("/") and not system_library(dependency):
                raise RuntimeError(f"External library remains: {target}: {dependency}")
    return origins


def clean_environment(runtime):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "DYLD", "HF_", "HUGGING_FACE"))}
    environment.update(PATH=f"{runtime / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return environment


def verify(app):
    run("codesign", "--verify", "--deep", "--strict", app)
    # Run from a new path with no developer Python, Homebrew PATH, or working checkout.
    with tempfile.TemporaryDirectory(prefix="WhisperX relocation ") as temporary:
        relocated = Path(temporary) / "WhisperX.app"
        run("ditto", app, relocated)
        runtime = relocated / "Contents/Runtime"
        environment = clean_environment(runtime)
        run(runtime / "bin/python3", "-I", "-B", ROOT / "Packaging/smoke_runtime.py", relocated, env=environment, cwd=temporary)


def sign(app, identity):
    for path in sorted(app.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if macho(path):
            arguments = ["codesign", "--force", "--sign", identity]
            if identity != "-":
                arguments += ["--options", "runtime", "--timestamp"]
                if path.name.startswith("python"):
                    arguments += ["--entitlements", ROOT / "Packaging/worker-entitlements.plist"]
            run(*arguments, path)
    arguments = ["codesign", "--force", "--sign", identity]
    if identity != "-":
        arguments += ["--options", "runtime", "--timestamp"]
    run(*arguments, app)


def build(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This installer must be built on an Apple Silicon Mac.")
    if args.notary_profile and (not args.app_identity or not args.installer_identity):
        raise RuntimeError("Notarization requires Developer ID Application and Installer identities.")
    if args.app_identity and not args.app_identity.startswith("Developer ID Application:"):
        raise RuntimeError("Use a Developer ID Application identity for distribution.")
    if args.installer_identity and not args.installer_identity.startswith("Developer ID Installer:"):
        raise RuntimeError("Use a Developer ID Installer identity for distribution.")
    ffmpeg = args.ffmpeg_prefix.resolve() if args.ffmpeg_prefix else prepare_ffmpeg(ROOT / ".packaging/ffmpeg7", 8)
    if not (ffmpeg / "bin/ffmpeg").is_file():
        raise RuntimeError("A build-time FFmpeg 7 prefix is required; see Packaging/README.md.")
    if not output(ffmpeg / "bin/ffmpeg", "-version").startswith("ffmpeg version 7."):
        raise RuntimeError("TorchCodec 0.7 requires the tested FFmpeg 7 distribution.")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "WhisperX.app").exists():
        raise RuntimeError("Use a new --output directory; an app already exists at this destination.")
    with tempfile.TemporaryDirectory(prefix="installer-", dir=ROOT / ".packaging") as temporary:
        work = Path(temporary)
        lock = work / "requirements.lock"
        shutil.copy2(ROOT / "Packaging/requirements.lock", lock)
        app = work / "WhisperX.app"
        run("sh", ROOT / "build-app.sh", env={**os.environ, "WHISPERX_APP_OUTPUT": str(app)})
        runtime = app / "Contents/Runtime"
        python_store = ROOT / ".packaging/python"
        run("uv", "python", "install", "3.12.12", "--install-dir", python_store, "--no-bin")
        candidates = sorted({path.resolve() for path in python_store.glob("cpython-3.12*/bin/python3.12")})
        if len(candidates) != 1:
            raise RuntimeError("Could not uniquely locate standalone CPython 3.12.12.")
        run("ditto", candidates[0].parent.parent, runtime)
        run("uv", "pip", "install", "--system", "--break-system-packages", "--python", runtime / "bin/python3", "--requirements", lock, "--require-hashes", "--only-binary", ":all:", "--no-binary", "antlr4-python3-runtime", "--no-compile-bytecode")
        for native in (runtime / "lib/python3.12/site-packages/torchcodec").glob("*.dylib"):
            if native.stem.endswith(("4", "5", "6")):
                native.unlink()
        origins = bundle_native(runtime, ffmpeg)
        minimum = (14, 0)
        minimum_sources = []
        for binary in app.rglob("*"):
            if not macho(binary):
                continue
            if "arm64" not in output("lipo", "-archs", binary).split():
                raise RuntimeError(f"Missing Apple Silicon architecture: {binary}")
            commands = output("otool", "-l", binary)
            for value in re.findall(r"cmd LC_BUILD_VERSION\s+cmdsize \d+\s+platform \d+\s+minos (\d+\.\d+(?:\.\d+)?)", commands) + re.findall(r"cmd LC_VERSION_MIN_MACOSX\s+cmdsize \d+\s+version (\d+\.\d+(?:\.\d+)?)", commands):
                required = tuple(int(part) for part in value.split("."))
                if required > minimum:
                    minimum_sources = [str(binary.relative_to(app))]
                    minimum = required
                elif required == minimum:
                    minimum_sources.append(str(binary.relative_to(app)))
        minimum_macos = ".".join(str(part) for part in minimum)
        host_version = tuple(int(part) for part in platform.mac_ver()[0].split("."))
        if minimum > host_version + (0,) * (len(minimum) - len(host_version)):
            raise RuntimeError(f"Runtime requires macOS {minimum_macos}, newer than this build Mac: {minimum_sources}")
        info_path = app / "Contents/Info.plist"
        info = plistlib.loads(info_path.read_bytes())
        info["LSMinimumSystemVersion"] = minimum_macos
        info_path.write_bytes(plistlib.dumps(info))
        notices = app / "Contents/Resources/Licenses"
        notices.mkdir()
        shutil.copy2(ROOT.parent / "LICENSE", notices / "WhisperX-BSD-2-Clause.txt")
        licenses = ffmpeg / "share/doc"
        if licenses.exists():
            run("ditto", licenses, notices / "FFmpeg")
        # Homebrew's installed formula records the exact build options and source URLs.
        native_sources = []
        for source in sorted(set(origins.values())):
            formula = next((parent for parent in source.parents if (parent / "INSTALL_RECEIPT.json").exists()), None)
            if formula:
                destination = notices / "Native" / formula.parent.name
                if not destination.exists():
                    run("ditto", formula / ".brew", destination / "Formula")
                    for license_file in formula.iterdir():
                        if license_file.is_file() and license_file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                            shutil.copy2(license_file, destination / license_file.name)
                native_sources.append(str(formula))
        manifest = {"architecture": "arm64", "minimumMacOS": minimum_macos, "minimumRequiredBy": minimum_sources, "python": "3.12.12", "requirementsSHA256": hashlib.sha256(lock.read_bytes()).hexdigest(), "ffmpeg": output(ffmpeg / "bin/ffmpeg", "-version").splitlines()[0], "nativeBuildSources": sorted(set(native_sources)), "publicDistribution": bool(args.notary_profile)}
        (app / "Contents/Resources/RuntimeManifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copy2(lock, notices / "requirements.lock")
        sign(app, args.app_identity or "-")
        verify(app)
        version = plistlib.loads((app / "Contents/Info.plist").read_bytes())["CFBundleShortVersionString"]
        payload = work / "payload"
        payload.mkdir()
        app = app.rename(payload / app.name)
        component = work / "component.pkg"
        run("pkgbuild", "--root", payload, "--install-location", "/Applications", "--identifier", "it.whisperx.mac", "--version", version, "--ownership", "recommended", "--component-plist", ROOT / "Packaging/component.plist", component)
        distribution = work / "Distribution.xml"
        distribution.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
<title>WhisperX</title>
<options customize="never" require-scripts="false" hostArchitectures="arm64"/>
<domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
<volume-check><allowed-os-versions><os-version min="{minimum_macos}"/></allowed-os-versions></volume-check>
<welcome file="Welcome.html"/>
<conclusion file="Conclusion.html"/>
<choices-outline><line choice="default"/></choices-outline>
<choice id="default" visible="false"><pkg-ref id="it.whisperx.mac"/></choice>
<pkg-ref id="it.whisperx.mac" version="{version}" onConclusion="none">component.pkg</pkg-ref>
</installer-gui-script>
''')
        package = work / f"WhisperX-{version}-arm64.pkg"
        resources = work / "InstallerResources"
        shutil.copytree(ROOT / "Packaging/Resources", resources)
        welcome = resources / "Welcome.html"
        welcome.write_text(welcome.read_text().replace("macOS 14", "macOS " + minimum_macos))
        arguments = ["productbuild", "--distribution", distribution, "--package-path", work, "--resources", resources]
        if args.installer_identity:
            arguments += ["--sign", args.installer_identity, "--timestamp"]
        run(*arguments, package)
        if args.notary_profile:
            run("xcrun", "notarytool", "submit", package, "--keychain-profile", args.notary_profile, "--wait")
            run("xcrun", "stapler", "staple", package)
            run("xcrun", "stapler", "validate", package)
            run("spctl", "--assess", "--type", "install", "--verbose", package)
        final_package = args.output / package.name
        shutil.copy2(package, final_package)
        (args.output / (package.name + ".sha256")).write_text(hashlib.sha256(package.read_bytes()).hexdigest() + "  " + package.name + "\n")
        app_destination = args.output / "WhisperX.app"
        if app_destination.exists():
            raise RuntimeError(f"App output already exists: {app_destination}; use a new --output directory.")
        run("ditto", app, app_destination)
        print(f"Installer ready: {final_package}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/installer", help="Output directory for app, pkg and SHA-256.")
    parser.add_argument("--ffmpeg-prefix", type=Path, help="Optional FFmpeg 7 prefix; by default build the pinned private FFmpeg distribution.")
    parser.add_argument("--app-identity", help="Developer ID Application certificate; defaults to local ad hoc signing.")
    parser.add_argument("--installer-identity", help="Developer ID Installer certificate for signing the pkg.")
    parser.add_argument("--notary-profile", help="Existing notarytool Keychain profile; notarize, staple and assess the pkg.")
    args = parser.parse_args()
    args.output = args.output.resolve()
    (ROOT / ".packaging").mkdir(exist_ok=True)
    build(args)


if __name__ == "__main__":
    main()

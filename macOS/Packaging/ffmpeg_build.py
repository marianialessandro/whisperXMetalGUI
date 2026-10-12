"""Prepare a pinned FFmpeg build without Homebrew or optional external libraries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

VERSION = "7.1.5"
RECIPE_VERSION = 2
SHA256 = "de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f"
URL = f"https://ffmpeg.org/releases/ffmpeg-{VERSION}.tar.xz"


def prepare(prefix, jobs):
    prefix = prefix.resolve()
    stamp = prefix / "Build.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("sourceSHA256") == SHA256 and json.loads(stamp.read_text()).get("recipeVersion") == RECIPE_VERSION:
        return prefix
    archive = prefix.parent / f"ffmpeg-{VERSION}.tar.xz"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("FFmpeg source SHA-256 does not match the pinned release.")
    source = prefix.parent / f"ffmpeg-{VERSION}"
    if not source.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(prefix.parent, filter="data")
    arguments = ["./configure", f"--prefix={prefix}", "--cc=/usr/bin/clang", "--disable-autodetect", "--disable-doc", "--disable-debug", "--enable-shared", "--disable-static", "--disable-network", "--disable-devices", "--disable-postproc", "--disable-hwaccels", "--disable-encoders", "--enable-encoder=pcm_s16le,aac", "--disable-muxers", "--enable-muxer=wav,ipod,null,pcm_s16le", "--disable-filters", "--enable-filter=aresample,aformat,anull,atrim,sine", "--extra-cflags=-mmacosx-version-min=14.0", "--extra-ldflags=-mmacosx-version-min=14.0"]
    environment = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": "14.0", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    subprocess.run(arguments, cwd=source, env=environment, check=True)
    subprocess.run(["make", f"-j{jobs}"], cwd=source, env=environment, check=True)
    subprocess.run(["make", "install"], cwd=source, env=environment, check=True)
    notices = prefix / "share/doc/ffmpeg"
    notices.mkdir(parents=True, exist_ok=True)
    for name in ["COPYING.LGPLv2.1", "LICENSE.md"]:
        shutil.copy2(source / name, notices / name)
    shutil.copy2(archive, notices / archive.name)
    manifest = {"version": VERSION, "recipeVersion": RECIPE_VERSION, "sourceURL": URL, "sourceSHA256": SHA256, "configure": arguments, "minimumMacOS": "14.0", "compiler": subprocess.check_output(["/usr/bin/clang", "--version"], text=True)}
    (notices / "Build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(json.dumps(manifest, indent=2) + "\n")
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / ".packaging/ffmpeg7", help="Private build prefix.")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8), help="Maximum parallel compiler jobs.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(prepare(args.output, args.jobs))


if __name__ == "__main__":
    main()

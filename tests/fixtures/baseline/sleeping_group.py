"""Bounded process-group lifecycle fixture with observable parent and grandchild termination."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--grandchild", action="store_true")
    parser.add_argument("--exit-parent", action="store_true")
    args = parser.parse_args()
    name = "grandchild" if args.grandchild else "parent"
    child = None
    def terminated(signum, frame):
        (args.markers / (name + "-stopped")).write_text(str(os.getpid()))
        if child is not None:
            try: child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminated)
    if not args.grandchild:
        os.setpgrp()
        json.load(sys.stdin)
        child = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--markers", str(args.markers), "--grandchild"], stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + 5
        while not (args.markers / "grandchild-ready").exists():
            if time.monotonic() > deadline: child.kill(); child.wait(); return 2
            time.sleep(0.01)
    (args.markers / (name + "-ready")).write_text(str(os.getpid()))
    if not args.grandchild:
        print(json.dumps({"event": "stage", "message": "fixture_ready"}), flush=True)
        if args.exit_parent: return 0
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline: time.sleep(0.05)
    terminated(signal.SIGTERM, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Private attach-ready entrypoint for a separate diagnostic pipeline trace."""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import adapter, contracts
from tools.evaluation import voice_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--turns", type=Path)
    parser.add_argument("--ready-delay", type=float, default=5, help="Attach delay outside diagnostic phase timings; 0 to 10 seconds.")
    args = parser.parse_args(argv)
    if not 0 <= args.ready_delay <= 10: parser.error("ready-delay must be between 0 and 10")
    request = json.load(sys.stdin)
    contracts.validate_settings(request)
    os.setpgrp()
    print(json.dumps({"trace_ready": True, "pid": os.getpid(), "run_id": args.run_id}), flush=True)
    time.sleep(args.ready_delay)
    arguments = ["--worker", str(args.source_root / "macOS/Worker/transcribe_worker.py"), "--source-root", str(args.source_root), "--spans", str(args.spans), "--run-id", args.run_id, "--synchronized", "--seed", "1729"]
    original = sys.stdin
    private_input = io.StringIO(json.dumps(request))
    request.clear()
    try:
        sys.stdin = private_input
        if args.turns is not None: return voice_adapter.main(["--speaker-turns", str(args.turns), *arguments])
        return adapter.main(arguments)
    finally:
        sys.stdin = original
        private_input.seek(0)
        private_input.truncate(0)
        private_input.close()


if __name__ == "__main__":
    raise SystemExit(main())

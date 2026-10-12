"""Run the original GUI entrypoint with Python networking denied; native networking still requires physical offline validation."""

import argparse
import contextlib
import os
from pathlib import Path
import runpy
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.reproducibility import seed_process


class OfflineNetworkError(OSError):
    pass


@contextlib.contextmanager
def deny_python_network():
    originals = []
    def denied(*args, **kwargs):
        raise OfflineNetworkError("offline_network_denied")
    targets = [(socket, "create_connection"), (socket, "getaddrinfo"), (socket.socket, "connect"), (socket.socket, "connect_ex"), (socket.socket, "sendto"), (socket.socket, "sendmsg")]
    try:
        for owner, name in targets:
            if hasattr(owner, name):
                originals.append((owner, name, getattr(owner, name)))
                setattr(owner, name, denied)
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Optional recorded diagnostic RNG initialization; no default change to the selected original worker or decoder options.")
    args = parser.parse_args(argv)
    sys.argv = [str(args.worker), "--source-root", str(args.source_root)]
    with deny_python_network():
        if args.seed is not None:
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink): seed_process(args.seed)
        runpy.run_path(str(args.worker), run_name="__main__")


if __name__ == "__main__":
    main()

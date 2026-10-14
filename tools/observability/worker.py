"""Optional transfer audit around the unchanged diagnostic adapter; stdin remains private."""

import argparse
import contextlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import adapter, freeze
from tools.observability.transfers import TransferAudit, phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " Worker/source/spans/seed arguments are forwarded to the original adapter.")
    parser.add_argument("--transfer-report", type=Path, required=True, help="Exclusive private metadata report; diagnostic timing is not a benchmark replica.")
    parser.add_argument("--python-readback-report", type=Path, help="Optional separate original Tensor.cpu/numpy method metadata; never added to ATen byte counts.")
    args, remainder = parser.parse_known_args(argv)
    if args.transfer_report.exists(): raise ValueError("transfer_destination_exists")
    if args.python_readback_report is not None and (args.python_readback_report.exists() or args.python_readback_report.resolve() == args.transfer_report.resolve()): raise ValueError("python_readback_destination_exists")
    audit = TransferAudit()
    readbacks = None
    if args.python_readback_report is not None:
        from tools.observability.python_readbacks import PythonReadbacks
        readbacks = PythonReadbacks(audit.stage)
    with phases(adapter, audit), audit, readbacks.observe() if readbacks is not None else contextlib.nullcontext():
        status = adapter.main(remainder)
    if status != 0: return status
    report = {"schema_version": 1, "status": "transfer_audit_completed", "pid": os.getpid(), **audit.report()}
    freeze.seal(args.transfer_report.parent.resolve(strict=True), args.transfer_report.name, report)
    if readbacks is not None:
        freeze.seal(args.python_readback_report.parent.resolve(strict=True), args.python_readback_report.name, {"schema_version": 1, "status": "python_readback_audit_completed", "pid": os.getpid(), **readbacks.report()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

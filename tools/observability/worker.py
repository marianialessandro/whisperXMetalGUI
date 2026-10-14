"""Optional transfer audit around the unchanged diagnostic adapter; stdin remains private."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import adapter, freeze
from tools.observability.transfers import TransferAudit, phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " Worker/source/spans/seed arguments are forwarded to the original adapter.")
    parser.add_argument("--transfer-report", type=Path, required=True, help="Exclusive private metadata report; diagnostic timing is not a benchmark replica.")
    args, remainder = parser.parse_known_args(argv)
    if args.transfer_report.exists(): raise ValueError("transfer_destination_exists")
    audit = TransferAudit()
    with phases(adapter, audit), audit:
        status = adapter.main(remainder)
    if status != 0: return status
    report = {"schema_version": 1, "status": "transfer_audit_completed", "pid": os.getpid(), **audit.report()}
    freeze.seal(args.transfer_report.parent.resolve(strict=True), args.transfer_report.name, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

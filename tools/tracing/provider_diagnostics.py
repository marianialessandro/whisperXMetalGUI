"""Read-only Metal provider diagnostics from an externally pinned trace archive."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, freeze, manifest, profiling


SCHEMAS = ("metal-gpu-info", "metal-command-buffer-completed", "metal-gpu-submission-to-command-buffer-id", "metal-gpu-intervals")


def export_table(trace, schema, output, log):
    query = '/trace-toc/run[@number="1"]/data/table[@schema="' + schema + '"]'
    with log.open("xb") as stream:
        result = subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", query, "--output", str(output)], stdout=stream, stderr=subprocess.STDOUT, timeout=45)
    return result.returncode


def count_rows(path, schema):
    corpus.require(path.stat().st_size <= 64 * 1024 * 1024, "provider_export_budget_exceeded")
    rows, names = 0, []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "schema":
            names.append(element.get("name"))
        elif element.tag == "row":
            rows += 1
        element.clear()
    corpus.require(names == [schema], "provider_export_schema_missing_or_invalid")
    return rows


def diagnose(root, snapshot, checksum, destination, exporter=export_table):
    corpus.require(isinstance(snapshot, str) and snapshot not in ("", ".", "..") and Path(snapshot).name == snapshot, "provider_snapshot_invalid")
    corpus.require(corpus.valid_digest(checksum), "provider_checksum_invalid")
    snapshot_path = corpus.safe_target(root, snapshot)
    output = corpus.safe_target(root, destination)
    corpus.require(not output.is_relative_to(snapshot_path) and not snapshot_path.is_relative_to(output), "provider_output_overlaps_snapshot")
    trace_name = snapshot + "/pipeline.trace"
    before = profiling.artifact_identity(root, trace_name)
    corpus.require(before == checksum, "provider_trace_hash_mismatch")
    output.mkdir()
    tables = []
    for schema in SCHEMAS:
        path = output / (schema + ".xml")
        row = {"schema": schema, "status": "export_failed", "rows_all_processes": None}
        try:
            row["exit_code"] = exporter(snapshot_path / "pipeline.trace", schema, path, output / (schema + ".log"))
            if row["exit_code"] == 0:
                row["rows_all_processes"] = count_rows(path, schema)
                row["export_sha256"] = manifest.sha256(path)
                row["status"] = "exported"
        except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired):
            row["status"] = "export_failed"
        tables.append(row)
    try:
        unchanged = profiling.artifact_identity(root, trace_name) == before
    except (OSError, ValueError):
        unchanged = False
    present = unchanged and all(row["status"] == "exported" and row["rows_all_processes"] > 0 for row in tables)
    report = {"schema_version": 1, "scope": "historical_trace_provider_tables_only", "snapshot": snapshot, "trace_sha256": before, "original_trace_unchanged": unchanged, "status": "provider_tables_present" if present else "incomplete", "tables": tables, "run_number": 1, "process_attribution_verified": False, "gpu_dispatch_certified": False, "full_pipeline_gpu_certified": False, "quality_certified": False, "promotion_allowed": False, "new_capture_performed": False}
    freeze.seal(output, "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True, help="Existing trace snapshot directory name; pipeline.trace is read without recording or inference.")
    parser.add_argument("--trace-sha256", required=True, help="Externally pinned tree identity from tools.baseline.profiling.artifact_identity; checked before exports.")
    parser.add_argument("--output-directory", required=True, help="New private directory relative to evidence root, outside the snapshot. Existing paths are never overwritten.")
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.evidence_root, args.snapshot, args.trace_sha256, args.output_directory)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "provider_tables_present" else 2
    except (OSError, ValueError):
        print(json.dumps({"status": "provider_diagnostics_invalid", "promotion_allowed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

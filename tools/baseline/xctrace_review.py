"""Review command-line Metal System Trace exports by exact target PID; generic compute intervals never certify individual operators."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.profiling import artifact_identity


def parse_xml(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("trace_xml_budget_exceeded")
    data = path.read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("trace_xml_entities_forbidden")
    return ET.fromstring(data)


def review(toc, gpu, report):
    pid = report.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("trace_target_identity_unverified")
    target = toc.find("./run/info/target/process")
    if target is None or target.get("pid") != str(pid) or target.get("return-exit-status") != "0" or report.get("status") != "primitive_parity_passed":
        raise ValueError("trace_target_identity_unverified")
    schema = gpu.find(".//schema")
    if schema is None or schema.get("name") != "metal-gpu-intervals":
        raise ValueError("trace_schema_invalid")
    columns = [column.findtext("mnemonic") for column in schema.findall("col")]
    if len(columns) != len(set(columns)) or not {"start", "duration", "process", "channel-name", "state"}.issubset(columns):
        raise ValueError("trace_schema_invalid")
    ids = {}
    for element in gpu.iter():
        if element.get("id"):
            if element.get("id") in ids: raise ValueError("trace_reference_invalid")
            ids[element.get("id")] = element
    def resolve(element):
        seen = set()
        while element is not None and element.get("ref"):
            reference = element.get("ref")
            if reference in seen or reference not in ids: raise ValueError("trace_reference_invalid")
            seen.add(reference)
            element = ids[reference]
        if element is None: raise ValueError("trace_reference_invalid")
        return element
    intervals, excluded = [], 0
    for row in gpu.findall(".//row"):
        if len(row) != len(columns): raise ValueError("trace_row_invalid")
        values = dict(zip(columns, list(row)))
        owner = resolve(values["process"])
        owner_pid = resolve(owner.find("pid"))
        if owner_pid.text != str(pid):
            excluded += 1
            continue
        channel = resolve(values["channel-name"]).text
        state = resolve(values["state"]).text
        if channel != "Compute" or state != "Active": continue
        start, duration = int(resolve(values["start"]).text), int(resolve(values["duration"]).text)
        if start < 0 or duration <= 0: raise ValueError("trace_interval_invalid")
        intervals.append({"start_ns": start, "duration_ns": duration})
    return {"run_id": report["run_id"], "pid": pid, "status": "target_gpu_compute_observed" if intervals else "target_gpu_work_missing", "scope": "ctc_primitive_process_only", "target_active_compute_intervals": intervals, "other_process_rows_excluded": excluded, "gpu_work_observed": bool(intervals), "individual_operator_dispatch_verified": False, "gpu_placement_certified": False, "quality_certified": False, "full_baseline_gate_passed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toc", type=Path, required=True)
    parser.add_argument("--gpu-intervals", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private review; unrelated process rows/labels are never copied into the report.")
    args = parser.parse_args(argv)
    try:
        if args.report.stat().st_size > 1024 * 1024 or args.output.exists(): raise ValueError("trace_review_inputs_invalid")
        report = json.loads(args.report.read_text())
        result = review(parse_xml(args.toc), parse_xml(args.gpu_intervals), report)
        result["artifacts"] = {"trace_sha256": artifact_identity(args.trace.parent.resolve(), args.trace.name), "toc_sha256": hashlib.sha256(args.toc.read_bytes()).hexdigest(), "gpu_intervals_sha256": hashlib.sha256(args.gpu_intervals.read_bytes()).hexdigest(), "primitive_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output: json.dump(result, output, indent=2)
        print(json.dumps({"status": result["status"], "run_id": result["run_id"], "pid": result["pid"], "target_compute_intervals": len(result["target_active_compute_intervals"]), "individual_operator_dispatch_verified": False, "full_baseline_gate_passed": False}))
        return 0 if result["gpu_work_observed"] else 2
    except (OSError, ValueError, KeyError, ET.ParseError):
        print(json.dumps({"status": "trace_review_failed", "code": "trace_identity_or_export_invalid", "full_baseline_gate_passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

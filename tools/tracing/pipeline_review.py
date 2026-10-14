"""Process-level trace review; generic GPU activity never certifies individual operators."""

from tools.baseline import corpus, profiling


def review(toc, gpu, spans, controller):
    pid = spans.get("pid")
    corpus.require(type(pid) is int and pid > 0 and controller.get("pid") == pid and controller.get("child_returncode") == 0 and controller.get("run_id") == spans.get("run_id") and controller.get("status") == "diagnostic_pipeline_completed", "pipeline_trace_identity_invalid")
    target = toc.find("./run/info/target/process")
    corpus.require(target is not None and target.get("pid") == str(pid) and target.get("return-exit-status") in (None, "0"), "pipeline_trace_target_invalid")
    corpus.require(spans.get("timing_kind") == "synchronized_completion", "pipeline_trace_completion_missing")
    stages = {span["stage"] for span in spans["spans"] if span.get("status") == "synchronized"}
    required = {"decode", "asr_mlx", "alignment", "export_write"}
    if controller.get("voices"): required.update(("diarization", "speaker_assignment"))
    corpus.require(required.issubset(stages), "pipeline_trace_phase_missing")
    schema = gpu.find(".//schema")
    corpus.require(schema is not None and schema.get("name") == "metal-gpu-intervals", "pipeline_trace_schema_invalid")
    columns = [column.findtext("mnemonic") for column in schema.findall("col")]
    corpus.require(len(columns) == len(set(columns)) and {"start", "duration", "process", "channel-name", "state"}.issubset(columns), "pipeline_trace_schema_invalid")
    ids = {}
    for element in gpu.iter():
        if element.get("id"):
            corpus.require(element.get("id") not in ids, "pipeline_trace_reference_invalid")
            ids[element.get("id")] = element
    def resolve(element):
        seen = set()
        while element is not None and element.get("ref"):
            reference = element.get("ref")
            corpus.require(reference not in seen and reference in ids, "pipeline_trace_reference_invalid")
            seen.add(reference)
            element = ids[reference]
        corpus.require(element is not None, "pipeline_trace_reference_invalid")
        return element
    intervals, excluded, unattributed = [], 0, 0
    for row in gpu.findall(".//row"):
        corpus.require(len(row) == len(columns), "pipeline_trace_row_invalid")
        values = dict(zip(columns, list(row)))
        owner = resolve(values["process"])
        if owner.tag == "sentinel" and owner.find("pid") is None:
            unattributed += 1
            continue
        corpus.require(owner.tag == "process", "pipeline_trace_process_invalid")
        if resolve(owner.find("pid")).text != str(pid):
            excluded += 1
            continue
        if resolve(values["channel-name"]).text != "Compute" or resolve(values["state"]).text != "Active": continue
        start, duration = int(resolve(values["start"]).text), int(resolve(values["duration"]).text)
        corpus.require(start >= 0 and duration > 0, "pipeline_trace_interval_invalid")
        intervals.append({"start_ns": start, "duration_ns": duration})
    return {"status": "pipeline_gpu_activity_observed" if intervals else "pipeline_gpu_activity_missing", "scope": "selected_pipeline_process_only", "run_id": spans["run_id"], "pid": pid, "observed_api_phases": sorted(stages), "target_compute_intervals": intervals, "foreign_process_rows_excluded": excluded, "unattributed_process_rows_excluded": unattributed, "gpu_activity_observed": bool(intervals), "individual_operator_dispatch_verified": False, "phase_interval_attribution_verified": False, "benchmark_timing_eligible": False, "gpu_placement_certified": False, "quality_certified": False, "operator_ledger": [{"id": item["id"], "status": "unverified", "required": True} for item in profiling.expected_operators()]}

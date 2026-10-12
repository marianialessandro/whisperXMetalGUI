"""Small synthetic MLX/MPS runtime probe, separate from corpus inference and release certification."""

import argparse
import json
import os
from pathlib import Path
import sys
import uuid
import importlib.metadata
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.profiling import RuntimeProbe, measure, artifact_identity
from tools.baseline.corpus import safe_target


def run_probe(root, capture=False):
    run_id = str(uuid.uuid4())
    report = {"run_id": run_id, "pid": os.getpid(), "kind": "synthetic_backend_probe", "quality_certified": False, "gpu_placement_certified": False, "spans": [], "trace_status": "not_requested"}
    probe = RuntimeProbe()
    report["installed_versions"] = {name: importlib.metadata.version(name) for name in ("mlx", "torch")}
    report["capture_tool"] = "mlx.core.metal.start_capture"
    expected = math.fsum(((127.5 * 256 + column) / (256 * 256)) ** 2 for column in range(256))
    unit_roundoff = 2 ** -24
    operations = 256 * 256 + 256 + 2
    error_bound = expected * operations * unit_roundoff / (1 - operations * unit_roundoff)
    report["analytic_reference"] = {"value": expected, "fp32_forward_error_bound": error_bound, "scope": "nonnegative_synthetic_matrix_reduction_only"}
    with probe.track_streams():
        if probe.mx:
            mx = probe.mx
            stream = mx.new_stream(mx.gpu)
            tracing = False
            trace_path = safe_target(root, run_id + ".gputrace")
            if capture:
                try:
                    mx.metal.start_capture(str(trace_path))
                    tracing = True
                    report["trace_status"] = "capturing"
                except Exception:
                    report["trace_status"] = "capture_unavailable"
            try:
                def mlx_call():
                    with mx.stream(stream):
                        matrix = mx.arange(256 * 256, dtype=mx.float32).reshape(256, 256) / (256 * 256)
                        return mx.mean(matrix @ matrix.T)
                output, span = probe.profile("synthetic_mlx", mlx_call)
                span.update(scalar_result=float(output.item()), logical_readback_bytes=4, backend_requested="mlx_gpu")
                span["analytic_check_passed"] = abs(span["scalar_result"] - expected) <= error_bound
                if not span["analytic_check_passed"]: raise ValueError("synthetic_numerical_check_failed")
                report["spans"].append(span)
            finally:
                if tracing:
                    try:
                        mx.metal.stop_capture()
                        report["trace_status"] = "captured_unreviewed"
                        report["trace"] = {"path": trace_path.name, "sha256": artifact_identity(root, trace_path.name), "run_id": run_id, "pid": os.getpid(), "phases": ["synthetic_mlx"], "kind": "diagnostic", "reviewed": False}
                    except Exception:
                        report["trace_status"] = "capture_incomplete"
        if probe.torch:
            torch = probe.torch
            def mps_call():
                matrix = torch.arange(256 * 256, dtype=torch.float32, device="mps").reshape(256, 256) / (256 * 256)
                return (matrix @ matrix.T).mean()
            output, span = probe.profile("synthetic_mps", mps_call)
            span.update(scalar_result=float(output.item()), logical_readback_bytes=4, backend_requested="torch_mps")
            span["analytic_check_passed"] = abs(span["scalar_result"] - expected) <= error_bound
            if not span["analytic_check_passed"]: raise ValueError("synthetic_numerical_check_failed")
            report["spans"].append(span)
    report["status"] = "synchronized_probe_completed" if report["spans"] else "backend_unavailable"
    with safe_target(root, run_id + ".json").open("x") as output:
        json.dump(report, output, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True, help="Existing private directory; synthetic traces never contribute natural accuracy or full operator coverage.")
    parser.add_argument("--capture", action="store_true", help="Attempt MLX Metal capture with MTL_CAPTURE_ENABLED=1; unavailable capture stays explicit.")
    args = parser.parse_args(argv)
    try:
        report = run_probe(args.evidence_root, args.capture)
        print(json.dumps({"run_id": report["run_id"], "status": report["status"], "trace_status": report["trace_status"], "spans": len(report["spans"]), "gpu_placement_certified": False}))
        return 0 if report["spans"] else 2
    except Exception:
        print(json.dumps({"status": "prerequisite_missing", "code": "synthetic_probe_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

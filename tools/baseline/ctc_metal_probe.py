"""Development-only CTC Metal feasibility probe, not a production backend or certified alignment replacement."""

from functools import lru_cache


PREFIX_SOURCE = r"""
int frames = emission_shape[0];
int vocabulary = emission_shape[1];
float high = 0;
float low = 0;
prefix[0] = 0;
for (int t = 0; t < frames; ++t) {
    float value = emission[t * vocabulary + blank[0]];
    if (metal::isinf(high) || metal::isinf(value)) {
        high += value;
        low = 0;
    } else {
        float sum = high + value;
        if (metal::isinf(sum)) {
            high = sum;
            low = 0;
        } else {
            float virtual_value = sum - high;
            float error = (high - (sum - virtual_value)) + (value - virtual_value);
            float correction = low + error;
            float next = sum + correction;
            low = correction - (next - sum);
            high = next;
        }
    }
    prefix[t + 1] = high;
}
"""


TRELLIS_SOURCE = r"""
uint lane = thread_position_in_threadgroup.x;
uint width = threads_per_threadgroup.x;
int frames = emission_shape[0];
int vocabulary = emission_shape[1];
int count = tokens_shape[0];
int stride = count + 1;
for (int t = lane; t <= frames; t += width) {
    trellis[t * stride] = t >= frames + 1 - count ? INFINITY : prefix[t];
}
for (int j = lane + 1; j <= count; j += width) {
    trellis[j] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_device);
for (int t = 0; t < frames; ++t) {
    for (int j = lane + 1; j <= count; j += width) {
        float stayed = trellis[t * stride + j] + emission[t * vocabulary + blank[0]];
        float changed = trellis[t * stride + j - 1] + emission[t * vocabulary + tokens[j - 1]];
        trellis[(t + 1) * stride + j] = metal::isnan(stayed) || metal::isnan(changed) ? NAN : metal::max(stayed, changed);
    }
    threadgroup_barrier(mem_flags::mem_device);
}
"""


BACKTRACK_SOURCE = r"""
int frames = emission_shape[0];
int vocabulary = emission_shape[1];
int count = tokens_shape[0];
int stride = count + 1;
for (int i = 0; i < frames; ++i) {
    path_data[1 + i * 3] = -1;
    path_data[1 + i * 3 + 1] = -1;
    path_data[1 + i * 3 + 2] = 0;
}
int start = 0;
float best = trellis[count];
for (int t = 1; t <= frames; ++t) {
    float score = trellis[t * stride + count];
    if (!metal::isnan(best) && (metal::isnan(score) || score > best)) { best = score; start = t; }
}
int j = count;
int length = 0;
for (int t = start; t > 0; --t) {
    float stayed = trellis[(t - 1) * stride + j] + emission[(t - 1) * vocabulary + blank[0]];
    float changed = trellis[(t - 1) * stride + j - 1] + emission[(t - 1) * vocabulary + tokens[j - 1]];
    bool advance = changed > stayed;
    path_data[1 + length * 3] = j - 1;
    path_data[1 + length * 3 + 1] = t - 1;
    path_data[1 + length * 3 + 2] = metal::exp(emission[(t - 1) * vocabulary + (advance ? tokens[j - 1] : blank[0])]);
    ++length;
    if (advance) { --j; if (j == 0) break; }
}
path_data[0] = j == 0 ? length : -1;
"""


@lru_cache(maxsize=1)
def kernels():
    import mlx.core as mx
    if not mx.metal.is_available():
        raise ValueError("metal_unavailable")
    prefix = mx.fast.metal_kernel(name="whisperx_ctc_prefix_probe", input_names=["emission", "blank"], output_names=["prefix"], source=PREFIX_SOURCE, compile_options={"math_mode": "safe"})
    trellis = mx.fast.metal_kernel(name="whisperx_ctc_trellis_probe", input_names=["emission", "tokens", "blank", "prefix"], output_names=["trellis"], source=TRELLIS_SOURCE, compile_options={"math_mode": "safe"})
    backtrack = mx.fast.metal_kernel(name="whisperx_ctc_backtrack_probe", input_names=["trellis", "emission", "tokens", "blank"], output_names=["path_data"], source=BACKTRACK_SOURCE, compile_options={"math_mode": "safe"})
    return prefix, trellis, backtrack


def compute(emission, tokens, blank_id=0, max_cells=16 * 1024 * 1024):
    """Return GPU trellis and GPU path arrays; host reads only validation flags, never DP decisions."""
    import mlx.core as mx
    if emission.ndim != 2 or emission.dtype != mx.float32 or tokens.ndim != 1 or tokens.dtype != mx.int32:
        raise ValueError("ctc_probe_input_format_invalid")
    frames, vocabulary = emission.shape
    count = tokens.size
    if not 0 < count <= frames <= 32768 or not 0 <= blank_id < vocabulary or not 0 < max_cells <= 16 * 1024 * 1024 or (frames + 1) * (count + 1) > max_cells:
        raise ValueError("ctc_probe_shape_or_budget_invalid")
    with mx.stream(mx.gpu):
        if not bool(mx.all((tokens >= 0) & (tokens < vocabulary)).item()) or not bool(mx.all((emission <= 0) & ~mx.isnan(emission)).item()):
            raise ValueError("ctc_probe_values_invalid")
        blank = mx.array([blank_id], dtype=mx.int32)
        prefix_kernel, trellis_kernel, path_kernel = kernels()
        prefix = prefix_kernel(inputs=[emission, blank], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(frames + 1,)], output_dtypes=[mx.float32])[0]
        trellis = trellis_kernel(inputs=[emission, tokens, blank, prefix], grid=(256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(frames + 1, count + 1)], output_dtypes=[mx.float32])[0]
        path_data = path_kernel(inputs=[trellis, emission, tokens, blank], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1 + frames * 3,)], output_dtypes=[mx.float32])[0]
    return {"prefix": prefix, "trellis": trellis, "path_data": path_data, "scope": "development_ctc_primitive", "production_backend": False, "quality_certified": False, "validation_logical_readback_bytes": 2, "physical_copy_bytes": None}


def materialize_path(result):
    """Read the final bounded path once; no host loop controls GPU backtracking."""
    import mlx.core as mx
    mx.eval(result["path_data"])
    length = int(result["path_data"][0].item())
    if length < 0:
        return None
    path = result["path_data"][1:1 + length * 3].reshape(length, 3).tolist()
    return [(int(token), int(frame), score) for token, frame, score in reversed(path)]


def load_inputs(path, expected_sha256):
    """Read only hash-verified bounded numeric emission/token arrays, with pickle disabled."""
    import hashlib
    from pathlib import Path
    import numpy as np
    from tools.baseline.provision import tensor_header
    path = Path(path)
    if path.suffix != ".npz" or path.stat().st_size > 64 * 1024 * 1024 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("ctc_input_identity_invalid")
    headers = {item["name"]: item for item in tensor_header(path)["tensors"]}
    if set(headers) != {"emission", "tokens", "blank_id"} or headers["emission"]["dtype"] != "<f4" or headers["tokens"]["dtype"] != "<i4" or headers["blank_id"]["dtype"] != "<i4" or len(headers["emission"]["shape"]) != 2 or len(headers["tokens"]["shape"]) != 1 or headers["blank_id"]["shape"]:
        raise ValueError("ctc_input_header_invalid")
    frames, vocabulary = headers["emission"]["shape"]
    count = headers["tokens"]["shape"][0]
    if not 0 < count <= frames <= 32768 or vocabulary < 1 or (frames + 1) * (count + 1) > 16 * 1024 * 1024 or frames * vocabulary * 4 > 64 * 1024 * 1024:
        raise ValueError("ctc_input_budget_invalid")
    with np.load(path, allow_pickle=False) as data:
        return data["emission"], data["tokens"], int(data["blank_id"])


def main(argv=None):
    import argparse
    import json
    import hashlib
    import importlib.metadata
    import math
    import os
    from pathlib import Path
    import sys
    import time
    import uuid
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import mlx.core as mx
    import numpy as np
    import torch
    from whisperx.alignment import get_trellis, backtrack
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=512, help="Seeded synthetic log-probability frames; not speech ground truth.")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--emissions", type=Path, help="Optional real-model diagnostic NPZ containing numeric emission, tokens and blank_id; requires --input-sha256 and never supplies independent ground truth.")
    parser.add_argument("--input-sha256", help="Expected SHA-256 of the private numeric NPZ, verified before loading.")
    parser.add_argument("--replicas", type=int, default=3, help="Warm kernel replicas, separate from the first JIT dispatch.")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic JSON, never overwritten.")
    parser.add_argument("--capture", action="store_true", help="Separate warm diagnostic Metal capture with MTL_CAPTURE_ENABLED=1, never used as a timing replica or reviewed placement certificate.")
    parser.add_argument("--trace-ready-delay", type=float, default=0, help="Optional 0–10 s delay outside all timings for an external command-line tracer to attach before short-lived GPU work.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.trace_ready_delay) or not 0 <= args.trace_ready_delay <= 10:
        parser.error("trace-ready-delay must be finite and between 0 and 10 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture_path = args.output.with_suffix(".gputrace")
    if args.capture and (os.environ.get("MTL_CAPTURE_ENABLED") != "1" or capture_path.exists()):
        parser.error("capture requires MTL_CAPTURE_ENABLED=1 and a new trace destination")
    if not 0 < args.tokens <= args.frames <= 32768 or (args.frames + 1) * (args.tokens + 1) > 16 * 1024 * 1024 or not 1 <= args.replicas <= 20 or args.output.exists():
        parser.error("invalid dimensions, replicas, budget or existing output")
    blank_id = 0
    if args.emissions:
        if not args.input_sha256: parser.error("real emissions require --input-sha256")
        data, token_data, blank_id = load_inputs(args.emissions, args.input_sha256)
        emission, tokens = torch.from_numpy(data), token_data.tolist()
        args.frames, args.tokens = emission.shape[0], len(tokens)
    else:
        generator = np.random.default_rng(args.seed)
        logits = generator.normal(size=(args.frames, 32)).astype(np.float32)
        emission = torch.log_softmax(torch.from_numpy(logits), dim=-1)
        tokens = generator.integers(1, 32, size=args.tokens).tolist()
    gpu_emission, gpu_tokens = mx.array(emission.numpy()), mx.array(tokens, dtype=mx.int32)
    mx.eval(gpu_emission, gpu_tokens)
    mx.synchronize()
    if args.trace_ready_delay: time.sleep(args.trace_ready_delay)
    started = time.perf_counter_ns()
    result = compute(gpu_emission, gpu_tokens, blank_id)
    path = materialize_path(result)
    mx.synchronize()
    cold_ns = time.perf_counter_ns() - started
    warm_ns, cpu_ns = [], []
    for _ in range(args.replicas):
        mx.synchronize()
        started = time.perf_counter_ns()
        result = compute(gpu_emission, gpu_tokens, blank_id)
        path = materialize_path(result)
        mx.synchronize()
        warm_ns.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        reference_trellis = get_trellis(emission, tokens, blank_id)
        reference_path = backtrack(reference_trellis, emission, tokens, blank_id)
        cpu_ns.append(time.perf_counter_ns() - started)
    reference = None if reference_path is None else [(point.token_index, point.time_index, point.score) for point in reference_path]
    indices_equal = (path is None and reference is None) or (path is not None and reference is not None and [(token, frame) for token, frame, score in path] == [(token, frame) for token, frame, score in reference])
    score_error = max((abs(left[2] - right[2]) for left, right in zip(path or [], reference or [])), default=0)
    matched = indices_equal and score_error <= 16 * np.finfo(np.float32).eps
    report = {"run_id": str(uuid.uuid4()), "scope": "synthetic_ctc_primitive", "frames": args.frames, "tokens": args.tokens, "seed": args.seed, "metal_path_indices_equal": indices_equal, "maximum_probability_absolute_error": score_error, "probability_roundoff_bound": float(16 * np.finfo(np.float32).eps), "status": "primitive_parity_passed" if matched else "primitive_parity_failed", "cold_jit_completion_ns": cold_ns, "warm_gpu_completion_and_path_readback_ns": warm_ns, "cpu_original_trellis_and_backtrack_ns": cpu_ns, "logical_final_path_readback_bytes": 4 + (0 if path is None else 12 * len(path)), "validation_logical_readback_bytes": 2, "physical_copy_bytes": None, "quality_certified": False, "production_backend": False, "full_baseline_gate_passed": False}
    report.update({"probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "oracle_source_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "whisperx/alignment.py").read_bytes()).hexdigest(), "installed_versions": {name: importlib.metadata.version(name) for name in ("mlx", "torch")}, "torch_cpu_threads": torch.get_num_threads(), "trellis_logical_bytes": (args.frames + 1) * (args.tokens + 1) * 4, "gpu_dispatch_observed": False, "timing_scope": "primitive_only_excludes_input_creation_model_inference_and_trace_capture"})
    report.update({"pid": os.getpid(), "trace_ready_delay_s": args.trace_ready_delay})
    if args.emissions: report.update({"scope": "real_model_emission_ctc_primitive", "input_sha256": args.input_sha256, "seed": None, "independent_ground_truth": False})
    if args.capture:
        from tools.baseline.profiling import artifact_identity
        mx.synchronize()
        mx.metal.start_capture(str(capture_path))
        try:
            captured = compute(gpu_emission, gpu_tokens, blank_id)
            mx.eval(captured["path_data"])
            mx.synchronize()
        finally:
            mx.metal.stop_capture()
        report["trace"] = {"path": capture_path.name, "sha256": artifact_identity(args.output.parent, capture_path.name), "run_id": report["run_id"], "pid": os.getpid(), "phases": ["ctc_prefix_probe", "ctc_trellis_probe", "ctc_backtrack_probe"], "kind": "diagnostic", "reviewed": False, "status": "captured_unreviewed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output: json.dump(report, output, indent=2)
    print(json.dumps(report))
    return 0 if matched else 2


if __name__ == "__main__":
    raise SystemExit(main())

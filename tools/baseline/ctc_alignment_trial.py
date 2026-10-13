"""Compare original WhisperX alignment with scoped experimental Metal CTC using a direct CLI script; no UI or production changes."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import manifest, provision, ctc_metal_probe
from tools.baseline.offline_worker import deny_python_network
from tools.baseline.native_ctc import NativeCTC


class MetalTrellis:
    def __init__(self, result, frames, count):
        self.result = result
        self.shape = (frames + 1, count + 1)

    def size(self, dimension):
        return self.shape[dimension]


class NativeCharacterPath:
    def __init__(self, segments):
        self.segments = segments


@contextlib.contextmanager
def metal_ctc(alignment, spans):
    import mlx.core as mx
    original_trellis, original_backtrack = alignment.get_trellis, alignment.backtrack
    def trellis(emission, tokens, blank_id=0):
        if emission.device.type != "cpu":
            raise ValueError("trial_expected_original_cpu_emission_boundary")
        started = time.perf_counter_ns()
        result = ctc_metal_probe.compute(mx.array(emission.detach().numpy()), mx.array(tokens, dtype=mx.int32), blank_id)
        return MetalTrellis(result, emission.shape[0], len(tokens)), started
    def wrapped_trellis(emission, tokens, blank_id=0):
        result, started = trellis(emission, tokens, blank_id)
        result.started = started
        return result
    def backtrack(trellis, emission, tokens, blank_id=0):
        path = ctc_metal_probe.materialize_path(trellis.result)
        mx.synchronize()
        spans.append({"frames": emission.shape[0], "tokens": len(tokens), "completion_and_bridge_wall_ns": time.perf_counter_ns() - trellis.started, "logical_cpu_to_mlx_emission_bytes": emission.numel() * emission.element_size(), "logical_final_path_readback_bytes": 4 + (0 if path is None else 12 * len(path)), "physical_copy_bytes": None})
        return None if path is None else [alignment.Point(token, frame, score) for token, frame, score in path]
    alignment.get_trellis, alignment.backtrack = wrapped_trellis, backtrack
    try:
        yield
    finally:
        alignment.get_trellis, alignment.backtrack = original_trellis, original_backtrack


@contextlib.contextmanager
def native_metal_ctc(alignment, spans, binary, binary_sha256):
    import numpy as np
    if manifest.sha256(binary) != binary_sha256:
        raise ValueError("native_binary_hash_mismatch")
    kernels = Path(__file__).with_name("native_ctc.metal")
    kernel_sha256 = manifest.sha256(kernels)
    original_trellis, original_backtrack = alignment.get_trellis, alignment.backtrack
    def trellis(emission, tokens, blank_id=0):
        if emission.device.type != "cpu" or str(emission.dtype) != "torch.float32":
            raise ValueError("trial_expected_original_cpu_float32_boundary")
        started = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="whisperx-native-ctc-") as directory:
            source = Path(directory) / "emissions.npz"
            np.savez(source, emission=emission.detach().numpy(), tokens=np.asarray(tokens, dtype=np.int32), blank_id=np.asarray(blank_id, dtype=np.int32))
            os.chmod(source, 0o600)
            process = subprocess.run([str(binary), "--input", str(source), "--sha256", manifest.sha256(source), "--kernels", str(kernels), "--replicas", "1", "--compact", "--batched"], capture_output=True, text=True, timeout=30, env=manifest.clean_environment(Path(sys.executable).parent.parent, {}))
            if process.returncode != 0:
                raise ValueError("native_ctc_process_failed")
            report = json.loads(process.stdout)
        if report.get("status") != "native_metal_ctc_executed" or report.get("shader_sha256") != kernel_sha256 or report.get("frames") != emission.shape[0] or report.get("tokens") != len(tokens):
            raise ValueError("native_ctc_result_identity_mismatch")
        spans.append({"frames": emission.shape[0], "tokens": len(tokens), "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "logical_cpu_to_native_emission_bytes": emission.numel() * emission.element_size(), "physical_copy_bytes": None, "transport": "private_npz_subprocess_per_segment", "native_runs": report["runs"], "command_buffers_per_run": report["command_buffers_per_run"]})
        return MetalTrellis(report["path"], emission.shape[0], len(tokens))
    def backtrack(trellis, emission, tokens, blank_id=0):
        return None if trellis.result is None else [alignment.Point(token, frame, score) for token, frame, score in trellis.result]
    alignment.get_trellis, alignment.backtrack = trellis, backtrack
    try:
        yield
    finally:
        alignment.get_trellis, alignment.backtrack = original_trellis, original_backtrack


@contextlib.contextmanager
def persistent_metal_ctc(alignment, spans, context, character_segments=False):
    import numpy as np
    from tools.baseline.native_ctc_fused import DeferredWildcard
    original_trellis, original_backtrack, original_merge = alignment.get_trellis, alignment.backtrack, alignment.merge_repeats
    def trellis(emission, tokens, blank_id=0):
        if isinstance(emission, DeferredWildcard):
            if emission.kind != "expanded" or blank_id != emission.blank: raise ValueError("fused_wildcard_metadata_invalid")
            path, record = context.compute_segments(emission.base_emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id, wildcard=True) if character_segments else context.compute_wildcard(emission.base_emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id)
            spans.append(record)
            return MetalTrellis(path, emission.shape[0], len(tokens))
        if emission.device.type != "cpu" or str(emission.dtype) != "torch.float32":
            raise ValueError("trial_expected_original_cpu_float32_boundary")
        path, record = context.compute_segments(emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id) if character_segments else context.compute(emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id)
        spans.append(record)
        return MetalTrellis(path, emission.shape[0], len(tokens))
    def backtrack(trellis, emission, tokens, blank_id=0):
        if character_segments: return None if trellis.result is None else NativeCharacterPath(trellis.result)
        return None if trellis.result is None else [alignment.Point(token, frame, score) for token, frame, score in trellis.result]
    def merge(path, transcript):
        if not isinstance(path, NativeCharacterPath) or len(path.segments) != len(transcript): raise ValueError("native_character_segment_metadata_invalid")
        return [alignment.Segment(transcript[i], start, end, score) for i, (start, end, score) in enumerate(path.segments)]
    alignment.get_trellis, alignment.backtrack = trellis, backtrack
    if character_segments: alignment.merge_repeats = merge
    try:
        yield
    finally:
        alignment.get_trellis, alignment.backtrack = original_trellis, original_backtrack
        if character_segments: alignment.merge_repeats = original_merge


@contextlib.contextmanager
def native_wildcard_max(context, spans):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    class WildcardMax(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten.max.dim:
                tensor = args[0]
                dimension = args[1] if len(args) > 1 else kwargs.get("dim")
                keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
                if tensor.device.type == "cpu" and tensor.dtype == torch.float32 and tensor.ndim == 2 and not tensor.requires_grad and dimension == 1 and not keepdim:
                    values, indices, record = context.row_max(tensor.detach().numpy())
                    record["intercepted_operator"] = "aten.max.dim"
                    spans.append(record)
                    return torch.from_numpy(values), torch.from_numpy(indices)
            return func(*args, **kwargs)
    with WildcardMax():
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--segments", type=Path, required=True, help="Private hash-verified ASR original JSON; never independent word/speaker ground truth.")
    parser.add_argument("--segments-sha256", required=True)
    parser.add_argument("--language", choices=("it", "en"), required=True)
    parser.add_argument("--torch-specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New private output directory for both aligned results and explicit partial comparison.")
    parser.add_argument("--replicas", type=int, default=1, help="Optional 1–10 same-loaded-model replicas with alternating original/candidate order; no cold-process benchmark claim.")
    parser.add_argument("--warmup", action="store_true", help="Run both paths once outside timings before the recorded replicas.")
    parser.add_argument("--return-char-alignments", action="store_true", help="Include original character-level JSON in both outputs and exact comparison; default preserves the word-only trial.")
    parser.add_argument("--ctc-backend", choices=("mlx", "native", "native-persistent"), default="mlx", help="Optional experimental CTC backend; native uses a private NPZ subprocess per segment; native-persistent uses one in-memory C ABI context for all segments, with setup reported separately.")
    parser.add_argument("--native-binary", type=Path, help="Explicit prebuilt native_ctc_probe executable, required with native backend.")
    parser.add_argument("--native-binary-sha256", help="Expected SHA256 of the native executable, required with native backend.")
    parser.add_argument("--native-library", type=Path, help="Explicit prebuilt native CTC dylib, required with native-persistent backend.")
    parser.add_argument("--native-library-sha256", help="Expected SHA256 of the native dylib, required with native-persistent backend.")
    parser.add_argument("--native-wildcard-max", action="store_true", help="Optional trial-only interception of CPU FP32 max(dim=1) using native Metal; requires native-persistent. Mask/gather/concatenation and max readback still use the original CPU boundary.")
    parser.add_argument("--native-fused-wildcard", action="store_true", help="Optional trial-only deferred wildcard preparation; native Metal expands the source emission and runs CTC without intermediate wildcard readback. Requires native-persistent; excludes --native-wildcard-max.")
    parser.add_argument("--native-character-segments", action="store_true", help="Optional GPU character bounds/means and segment-only readback; requires native-persistent and pinned CPython 3.12 sum semantics. Word reductions/timestamp scaling remain original.")
    parser.add_argument("--native-timestamp-bounds", action="store_true", help="Optional Metal binary64 timestamp min/max for word/sentence character subsets; requires native-persistent. Means, timestamp scaling and interpolation remain original; negative zero fails explicitly.")
    parser.add_argument("--native-timestamp-bounds-batched", action="store_true", help="Optional shared-input Metal timestamp min/max batch per character frame; requires native-persistent and excludes --native-timestamp-bounds. Uses original tokenizer spans and word metadata; means/interpolation remain original.")
    parser.add_argument("--native-timestamp-fast-frames", action="store_true", help="Optional Pandas 3.0.6 manager constructors avoiding subclass reconstruction in batched timestamp mode; requires --native-timestamp-bounds-batched. Preserves snapshot checks and copy-on-write semantics.")
    parser.add_argument("--native-word-means", action="store_true", help="Optional Metal binary64 word score means with pinned Pandas 3.0.6/NumPy 2.2.6 summation and buffer policy; requires --native-timestamp-bounds-batched. Adds one score batch per character frame; rounding remains original.")
    args = parser.parse_args(argv)
    if not 1 <= args.replicas <= 10: parser.error("replicas must be between 1 and 10")
    if args.ctc_backend == "native" and (args.native_binary is None or args.native_binary_sha256 is None): parser.error("native backend requires binary and its expected SHA256")
    if args.ctc_backend != "native" and (args.native_binary is not None or args.native_binary_sha256 is not None): parser.error("native binary options require --ctc-backend native")
    if args.ctc_backend == "native-persistent" and (args.native_library is None or args.native_library_sha256 is None): parser.error("native-persistent backend requires library and its expected SHA256")
    if args.ctc_backend != "native-persistent" and (args.native_library is not None or args.native_library_sha256 is not None): parser.error("native library options require --ctc-backend native-persistent")
    if args.native_wildcard_max and args.ctc_backend != "native-persistent": parser.error("--native-wildcard-max requires --ctc-backend native-persistent")
    if args.native_fused_wildcard and args.ctc_backend != "native-persistent": parser.error("--native-fused-wildcard requires --ctc-backend native-persistent")
    if args.native_fused_wildcard and args.native_wildcard_max: parser.error("wildcard modes are mutually exclusive")
    if args.native_word_means and not args.native_timestamp_bounds_batched: parser.error("--native-word-means requires --native-timestamp-bounds-batched")
    if args.native_timestamp_fast_frames and not args.native_timestamp_bounds_batched: parser.error("--native-timestamp-fast-frames requires --native-timestamp-bounds-batched")
    if args.native_timestamp_bounds_batched and args.ctc_backend != "native-persistent": parser.error("--native-timestamp-bounds-batched requires --ctc-backend native-persistent")
    if args.native_timestamp_bounds_batched and args.native_timestamp_bounds: parser.error("timestamp bounds modes are mutually exclusive")
    if args.native_timestamp_bounds and args.ctc_backend != "native-persistent": parser.error("--native-timestamp-bounds requires --ctc-backend native-persistent")
    if args.native_character_segments and args.ctc_backend != "native-persistent": parser.error("--native-character-segments requires --ctc-backend native-persistent")
    try:
        if args.output_dir.exists() or manifest.sha256(args.audio) != args.audio_sha256 or manifest.sha256(args.segments) != args.segments_sha256:
            raise ValueError("trial_inputs_or_destination_invalid")
        specs = json.loads(args.torch_specs.read_text())
        spec = next(item for item in specs if item.get("role") == "alignment_" + args.language)
        if spec.get("license_verified") is not True or provision.inspect_spec(spec)["status"] != "files_verified":
            raise ValueError("trial_alignment_checkpoint_not_verified")
        original = json.loads(args.segments.read_text())
        args.output_dir.mkdir(parents=True)
        os.environ.update(TORCH_HOME=spec["cache_root"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        os.environ["PATH"] = str(Path(sys.executable).parent) + ":/usr/bin:/bin"
        spans = []
        wildcard_spans = []
        timestamp_spans = []
        mean_spans = []
        native_identity = None
        if args.ctc_backend == "native":
            args.native_binary = args.native_binary.resolve(strict=True)
            if manifest.sha256(args.native_binary) != args.native_binary_sha256: raise ValueError("native_binary_hash_mismatch")
            native_identity = {"binary_sha256": args.native_binary_sha256, "shader_sha256": manifest.sha256(Path(__file__).with_name("native_ctc.metal"))}
        @contextlib.contextmanager
        def candidate_context(records, maximum_records, bounds_records, word_records):
            with contextlib.ExitStack() as scopes:
                if args.native_timestamp_bounds_batched:
                    from tools.baseline.native_timestamp_bounds import native_timestamp_bounds_batched
                    scopes.enter_context(native_timestamp_bounds_batched(alignment, persistent, bounds_records, fast_frames=args.native_timestamp_fast_frames, word_means=args.native_word_means, mean_records=word_records))
                if args.native_timestamp_bounds:
                    from tools.baseline.native_timestamp_bounds import native_timestamp_bounds
                    scopes.enter_context(native_timestamp_bounds(alignment, persistent, bounds_records))
                if args.native_fused_wildcard:
                    from tools.baseline.native_ctc_fused import fused_wildcard_preparation
                    scopes.enter_context(fused_wildcard_preparation(max(metadata["dictionary"].values()) + 1))
                if args.native_wildcard_max: scopes.enter_context(native_wildcard_max(persistent, maximum_records))
                if args.ctc_backend == "native-persistent": scopes.enter_context(persistent_metal_ctc(alignment, records, persistent, args.native_character_segments))
                else: scopes.enter_context(native_metal_ctc(alignment, records, args.native_binary, args.native_binary_sha256) if args.ctc_backend == "native" else metal_ctc(alignment, records))
                yield
        native_setup_ns = None
        with contextlib.ExitStack() as resources, provision.quiet_download(), deny_python_network():
            if args.ctc_backend == "native-persistent":
                persistent = resources.enter_context(NativeCTC(args.native_library, args.native_library_sha256))
                native_identity, native_setup_ns = persistent.identity, persistent.initialization_wall_ns
            import whisperx
            import whisperx.alignment as alignment
            import torch
            if not torch.backends.mps.is_available(): raise ValueError("mps_required")
            audio = whisperx.load_audio(str(args.audio))
            model, metadata = whisperx.load_align_model(args.language, "mps")
            if args.warmup:
                alignment.align(original["segments"], model, metadata, audio, "mps", return_char_alignments=args.return_char_alignments)
                with candidate_context([], [], [], []): alignment.align(original["segments"], model, metadata, audio, "mps", return_char_alignments=args.return_char_alignments)
                torch.mps.synchronize()
            baseline_times, candidate_times, matches, orders = [], [], [], []
            for replica in range(args.replicas):
                order = ("original", "candidate") if replica % 2 == 0 else ("candidate", "original")
                orders.append(list(order))
                results = {}
                for backend in order:
                    torch.mps.synchronize()
                    started = time.perf_counter_ns()
                    with candidate_context(spans, wildcard_spans, timestamp_spans, mean_spans) if backend == "candidate" else contextlib.nullcontext(): results[backend] = alignment.align(original["segments"], model, metadata, audio, "mps", return_char_alignments=args.return_char_alignments)
                    torch.mps.synchronize()
                    elapsed = time.perf_counter_ns() - started
                    (baseline_times if backend == "original" else candidate_times).append(elapsed)
                baseline, candidate = results["original"], results["candidate"]
                matches.append(baseline == candidate)
            baseline_ns, candidate_ns = baseline_times[-1], candidate_times[-1]
        for name, result in (("original-aligned.json", baseline), ("metal-ctc-aligned.json", candidate)):
            with (args.output_dir / name).open("x") as output: json.dump(result, output, ensure_ascii=False, indent=2)
        report = {"scope": "experimental_ctc_alignment_only", "status": "aligned_exactly_equal" if all(matches) else "aligned_mismatch", "source_audio_sha256": args.audio_sha256, "asr_result_sha256": args.segments_sha256, "original_completion_ns": baseline_ns, "metal_trial_completion_ns": candidate_ns, "timing_scope": "one_original_then_one_candidate_same_loaded_model_not_speedup_benchmark" if args.replicas == 1 else "alternating_same_loaded_model_alignment_only", "original_replicas_ns": baseline_times, "candidate_replicas_ns": candidate_times, "orders": orders, "all_replicas_exactly_equal": all(matches), "warmup_performed": args.warmup, "ctc_spans": spans, "residuals": ["original_mps_to_cpu_emission_boundary", "cpu_to_mlx_probe_bridge", "cpu_word_char_timestamp_reductions", "cpu_wildcard_column_reduction"], "quality_certified": False, "production_backend": False, "full_baseline_gate_passed": False}
        report["ctc_backend"], report["native_identity"] = args.ctc_backend, native_identity
        report["native_context_initialization_wall_ns"] = native_setup_ns
        report["native_wildcard_max_enabled"], report["wildcard_spans"] = args.native_wildcard_max, wildcard_spans
        report["native_wildcard_max_dispatch_count"] = len(wildcard_spans)
        report["native_fused_wildcard_enabled"] = args.native_fused_wildcard
        report["native_fused_wildcard_dispatch_count"] = sum(record.get("wildcard_fused") is True for record in spans)
        report["native_character_segments_enabled"] = args.native_character_segments
        report["return_char_alignments"] = args.return_char_alignments
        report["native_character_segments_dispatch_count"] = sum(record.get("character_segments_gpu") is True for record in spans)
        report["native_timestamp_bounds_enabled"], report["timestamp_bounds_spans"] = args.native_timestamp_bounds, timestamp_spans
        report["native_timestamp_bounds_dispatch_count"] = len(timestamp_spans)
        report["native_word_means_enabled"], report["word_mean_spans"] = args.native_word_means, mean_spans
        report["native_word_means_dispatch_count"] = len(mean_spans)
        report["native_word_means_queries_consumed"] = sum(record["queries_consumed"] for record in mean_spans)
        report["native_timestamp_fast_frames_enabled"] = args.native_timestamp_fast_frames
        report["native_timestamp_bounds_batched_enabled"] = args.native_timestamp_bounds_batched
        report["native_timestamp_bounds_query_count"] = sum(record.get("queries", 1) for record in timestamp_spans)
        report["native_timestamp_bounds_queries_consumed"] = sum(record.get("queries_consumed", 1) for record in timestamp_spans)
        if args.native_timestamp_bounds or args.native_timestamp_bounds_batched:
            import numpy as np
            report["timestamp_bounds_reference_runtime"] = {"pandas": alignment.pd.__version__, "numpy": np.__version__, "policy": "binary64_min_max_skipna_negative_zero_rejected"}
        if args.ctc_backend == "native": report["residuals"][1] = "private_npz_and_native_process_startup_per_segment"
        if args.ctc_backend == "native-persistent": report["residuals"][1] = "cpu_to_native_shared_buffer_copy_per_segment"
        if args.native_wildcard_max: report["residuals"][-1] = "cpu_wildcard_mask_gather_concat_and_max_readback"
        if args.native_fused_wildcard: report["residuals"][-1] = "cpu_wildcard_mask_and_token_metadata"
        if args.native_character_segments: report["residuals"][2] = "cpu_word_reductions_and_character_timestamp_scaling"
        if args.native_timestamp_bounds or args.native_timestamp_bounds_batched: report["residuals"][2] = "cpu_word_means_character_timestamp_scaling_interpolation_dataframe_selection_and_snapshot_validation"
        if args.native_word_means: report["residuals"][2] = "cpu_character_timestamp_scaling_rounding_interpolation_dataframe_selection_and_snapshot_validation"
        with (args.output_dir / "comparison.json").open("x") as output: json.dump(report, output, indent=2)
        print(json.dumps({key: value for key, value in report.items() if key not in ("ctc_spans", "wildcard_spans", "timestamp_bounds_spans", "word_mean_spans")}))
        return 0 if all(matches) else 2
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, AttributeError, subprocess.TimeoutExpired):
        print(json.dumps({"status": "trial_failed", "code": "ctc_alignment_trial_failed", "full_baseline_gate_passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

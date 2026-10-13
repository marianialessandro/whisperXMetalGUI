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
def persistent_metal_ctc(alignment, spans, context, character_segments=False, character_rounding=False, rounding_records=None, resident_character_rounding=False):
    import numpy as np
    from tools.baseline.native_ctc_fused import DeferredWildcard
    if character_rounding and (not character_segments or not isinstance(rounding_records, list)): raise ValueError("native_character_rounding_requires_segments_and_records")
    if resident_character_rounding and not character_rounding: raise ValueError("native_resident_rounding_requires_character_rounding")
    segment_compute = context.prepare_segments if resident_character_rounding else context.compute_segments
    original_trellis, original_backtrack, original_merge = alignment.get_trellis, alignment.backtrack, alignment.merge_repeats
    def trellis(emission, tokens, blank_id=0):
        if isinstance(emission, DeferredWildcard):
            if emission.kind != "expanded" or blank_id != emission.blank: raise ValueError("fused_wildcard_metadata_invalid")
            path, record = segment_compute(emission.base_emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id, wildcard=True) if character_segments else context.compute_wildcard(emission.base_emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id)
            spans.append(record)
            return MetalTrellis(path, emission.shape[0], len(tokens))
        if emission.device.type != "cpu" or str(emission.dtype) != "torch.float32":
            raise ValueError("trial_expected_original_cpu_float32_boundary")
        path, record = segment_compute(emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id) if character_segments else context.compute(emission.detach().numpy(), np.asarray(tokens, dtype=np.int32), blank_id)
        spans.append(record)
        return MetalTrellis(path, emission.shape[0], len(tokens))
    def backtrack(trellis, emission, tokens, blank_id=0):
        if character_segments: return None if trellis.result is None else NativeCharacterPath(trellis.result)
        return None if trellis.result is None else [alignment.Point(token, frame, score) for token, frame, score in trellis.result]
    def merge(path, transcript):
        if not isinstance(path, NativeCharacterPath) or len(path.segments) != len(transcript): raise ValueError("native_character_segment_metadata_invalid")
        if character_rounding:
            from tools.baseline.native_character_rounding import rounded_character_segments
            return rounded_character_segments(alignment, path.segments, transcript, context, rounding_records)
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
    parser.add_argument("--ctc-backend", choices=("mlx", "native", "native-persistent", "native-mps"), default="mlx", help="Optional experimental CTC backend; native uses a private NPZ subprocess per segment; native-persistent uses one in-memory C ABI context; native-mps borrows pending MPS emissions directly. Setup is reported separately.")
    parser.add_argument("--native-binary", type=Path, help="Explicit prebuilt native_ctc_probe executable, required with native backend.")
    parser.add_argument("--native-binary-sha256", help="Expected SHA256 of the native executable, required with native backend.")
    parser.add_argument("--native-library", type=Path, help="Explicit prebuilt native CTC dylib, required with native-persistent backend.")
    parser.add_argument("--native-library-sha256", help="Expected SHA256 of the native dylib, required with native-persistent backend.")
    parser.add_argument("--mps-library", type=Path, help="Optional pinned Torch interop dylib built by mps_storage_probe.py; required with native-mps backend.")
    parser.add_argument("--mps-library-sha256", help="Expected SHA256 of the Torch interop dylib, required with native-mps backend.")
    parser.add_argument("--mps-resident-statistics", action="store_true", help="Optional native-mps-only character grouping, resident timestamp scaling/rounding and fused timestamp/word statistics with GPU word rounding; requires a rebuilt Torch extension.")
    parser.add_argument("--mps-nearest-interpolation", action="store_true", help="Optional native-mps-only nearest interpolation in the same context, with GPU missing/filled counters. Supports --mps-interpolation-batched; requires a rebuilt Torch extension.")
    parser.add_argument("--mps-interpolation-batched", action="store_true", help="Optional paired start/end interpolation with the native-mps context; requires --mps-nearest-interpolation.")
    parser.add_argument("--native-wildcard-max", action="store_true", help="Optional trial-only interception of CPU FP32 max(dim=1) using native Metal; requires native-persistent. Mask/gather/concatenation and max readback still use the original CPU boundary.")
    parser.add_argument("--native-fused-wildcard", action="store_true", help="Optional trial-only deferred wildcard preparation; native Metal expands the source emission and runs CTC without intermediate wildcard readback. Requires native-persistent; excludes --native-wildcard-max.")
    parser.add_argument("--native-character-segments", action="store_true", help="Optional GPU character bounds/means and segment-only readback; requires native-persistent and pinned CPython 3.12 sum semantics. Word reductions/timestamp scaling remain original.")
    parser.add_argument("--native-timestamp-bounds", action="store_true", help="Optional Metal binary64 timestamp min/max for word/sentence character subsets; requires native-persistent. Means, timestamp scaling and interpolation remain original; negative zero fails explicitly.")
    parser.add_argument("--native-timestamp-bounds-batched", action="store_true", help="Optional shared-input Metal timestamp min/max batch per character frame; requires native-persistent and excludes --native-timestamp-bounds. Uses original tokenizer spans and word metadata; means/interpolation remain original.")
    parser.add_argument("--native-timestamp-fast-frames", action="store_true", help="Optional Pandas 3.0.6 manager constructors avoiding subclass reconstruction in batched timestamp mode; requires --native-timestamp-bounds-batched. Preserves snapshot checks and copy-on-write semantics.")
    parser.add_argument("--native-word-means", action="store_true", help="Optional Metal binary64 word score means with pinned Pandas 3.0.6/NumPy 2.2.6 summation and buffer policy; requires --native-timestamp-bounds-batched. Adds one score batch per character frame; rounding remains original.")
    parser.add_argument("--native-fused-reductions", action="store_true", help="Optional shared row queries and one Metal kernel for timestamp minima/maxima and word means; requires --native-word-means. Previous separate-batch mode remains available.")
    parser.add_argument("--native-compact-reduction-plan", action="store_true", help="Optional clipped sentence ranges, shared word row tuples and one snapshot gather per fused subset; requires --native-fused-reductions. Preserves snapshot mutation validation.")
    parser.add_argument("--native-character-rounding", action="store_true", help="Optional exact CPython 3.12.12 round(value,3) Metal batch for character timestamps and scores; requires --native-character-segments. Original CPU scaling is preserved; adds one round batch per segment.")
    parser.add_argument("--native-resident-character-rounding", action="store_true", help="Optional retained Metal character bounds/means, exact bounded binary64 scaling and rounding without intermediate character readback/reupload; requires --native-character-rounding. Original ratio calculation remains CPU.")
    parser.add_argument("--native-resident-reductions", action="store_true", help="Optional shared retained rounded character triples for fused statistics; verifies original row values and maps NaN positions without uploading character values. Requires --native-fused-reductions and --native-resident-character-rounding.")
    parser.add_argument("--native-word-rounding", action="store_true", help="Optional pinned NumPy three-decimal word mean rounding on Metal in the resident fused-stat command group; preserves unrounded numerical means. Requires --native-resident-reductions.")
    parser.add_argument("--native-nearest-interpolation", action="store_true", help="Optional default nearest timestamp interpolation and end filling on Metal for pinned Pandas/SciPy and contiguous zero-based Series; requires native-persistent. Adds one command group per numerical Series.")
    parser.add_argument("--native-interpolation-batched", action="store_true", help="Optional paired start/end subsegment nearest interpolation in one Metal completion with snapshot validation; requires --native-nearest-interpolation. Word Series remain standalone.")
    parser.add_argument("--interpolation-gap-control", action="store_true", help="Diagnostic synthetic sentence-span gaps applied identically to reference and candidate over the unchanged audio/CTC text; verifies real filling, never natural accuracy. Defaults off.")
    args = parser.parse_args(argv)
    if not 1 <= args.replicas <= 10: parser.error("replicas must be between 1 and 10")
    if args.ctc_backend == "native" and (args.native_binary is None or args.native_binary_sha256 is None): parser.error("native backend requires binary and its expected SHA256")
    if args.ctc_backend != "native" and (args.native_binary is not None or args.native_binary_sha256 is not None): parser.error("native binary options require --ctc-backend native")
    if args.ctc_backend == "native-persistent" and (args.native_library is None or args.native_library_sha256 is None): parser.error("native-persistent backend requires library and its expected SHA256")
    if args.ctc_backend != "native-persistent" and (args.native_library is not None or args.native_library_sha256 is not None): parser.error("native library options require --ctc-backend native-persistent")
    if args.ctc_backend == "native-mps" and (args.mps_library is None or args.mps_library_sha256 is None): parser.error("native-mps backend requires interop library and its expected SHA256")
    if args.ctc_backend != "native-mps" and (args.mps_library is not None or args.mps_library_sha256 is not None): parser.error("MPS library options require --ctc-backend native-mps")
    if args.mps_resident_statistics and args.ctc_backend != "native-mps": parser.error("MPS resident statistics require --ctc-backend native-mps")
    if args.mps_nearest_interpolation and args.ctc_backend != "native-mps": parser.error("MPS nearest interpolation requires --ctc-backend native-mps")
    if args.mps_interpolation_batched and not args.mps_nearest_interpolation: parser.error("MPS paired interpolation requires --mps-nearest-interpolation")
    if args.native_wildcard_max and args.ctc_backend != "native-persistent": parser.error("--native-wildcard-max requires --ctc-backend native-persistent")
    if args.native_fused_wildcard and args.ctc_backend != "native-persistent": parser.error("--native-fused-wildcard requires --ctc-backend native-persistent")
    if args.native_fused_wildcard and args.native_wildcard_max: parser.error("wildcard modes are mutually exclusive")
    if args.native_interpolation_batched and not args.native_nearest_interpolation: parser.error("batched interpolation requires native nearest interpolation")
    if args.native_nearest_interpolation and args.ctc_backend != "native-persistent": parser.error("native nearest interpolation requires native-persistent")
    if args.native_word_rounding and not args.native_resident_reductions: parser.error("native word rounding requires resident reductions")
    if args.native_resident_reductions and (not args.native_fused_reductions or not args.native_resident_character_rounding): parser.error("native resident reductions require fused reductions and resident character rounding")
    if args.native_resident_character_rounding and not args.native_character_rounding: parser.error("native resident character rounding requires native character rounding")
    if args.native_character_rounding and not args.native_character_segments: parser.error("--native-character-rounding requires --native-character-segments")
    if args.native_compact_reduction_plan and not args.native_fused_reductions: parser.error("--native-compact-reduction-plan requires --native-fused-reductions")
    if args.native_fused_reductions and not args.native_word_means: parser.error("--native-fused-reductions requires --native-word-means")
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
        rounding_spans, interpolation_spans = [], []
        native_identity = None
        if args.ctc_backend == "native":
            args.native_binary = args.native_binary.resolve(strict=True)
            if manifest.sha256(args.native_binary) != args.native_binary_sha256: raise ValueError("native_binary_hash_mismatch")
            native_identity = {"binary_sha256": args.native_binary_sha256, "shader_sha256": manifest.sha256(Path(__file__).with_name("native_ctc.metal"))}
        @contextlib.contextmanager
        def candidate_context(records, maximum_records, bounds_records, word_records, round_records, interpolate_records):
            with contextlib.ExitStack() as scopes:
                if args.mps_resident_statistics:
                    from tools.baseline.native_timestamp_bounds import native_timestamp_bounds_batched
                    scopes.enter_context(native_timestamp_bounds_batched(alignment, mps_context, bounds_records, fast_frames=True, word_means=True, mean_records=word_records, fused_reductions=True, compact_plan=True, resident_reductions=True, word_rounding=True))
                if args.mps_nearest_interpolation:
                    from tools.baseline.native_interpolation import native_nearest_interpolation
                    scopes.enter_context(native_nearest_interpolation(alignment, mps_context, interpolate_records, batched_frames=args.mps_interpolation_batched))
                if args.native_timestamp_bounds_batched:
                    from tools.baseline.native_timestamp_bounds import native_timestamp_bounds_batched
                    scopes.enter_context(native_timestamp_bounds_batched(alignment, persistent, bounds_records, fast_frames=args.native_timestamp_fast_frames, word_means=args.native_word_means, mean_records=word_records, fused_reductions=args.native_fused_reductions, compact_plan=args.native_compact_reduction_plan, resident_reductions=args.native_resident_reductions, word_rounding=args.native_word_rounding))
                if args.native_timestamp_bounds:
                    from tools.baseline.native_timestamp_bounds import native_timestamp_bounds
                    scopes.enter_context(native_timestamp_bounds(alignment, persistent, bounds_records))
                if args.native_nearest_interpolation:
                    from tools.baseline.native_interpolation import native_nearest_interpolation
                    scopes.enter_context(native_nearest_interpolation(alignment, persistent, interpolate_records, batched_frames=args.native_interpolation_batched))
                if args.native_fused_wildcard:
                    from tools.baseline.native_ctc_fused import fused_wildcard_preparation
                    scopes.enter_context(fused_wildcard_preparation(max(metadata["dictionary"].values()) + 1))
                if args.native_wildcard_max: scopes.enter_context(native_wildcard_max(persistent, maximum_records))
                if args.ctc_backend == "native-mps":
                    from tools.baseline.native_mps_ctc import native_mps_ctc
                    scopes.enter_context(native_mps_ctc(alignment, mps_context, records, resident_statistics=args.mps_resident_statistics, rounding_records=round_records))
                elif args.ctc_backend == "native-persistent": scopes.enter_context(persistent_metal_ctc(alignment, records, persistent, args.native_character_segments, args.native_character_rounding, round_records, args.native_resident_character_rounding))
                else: scopes.enter_context(native_metal_ctc(alignment, records, args.native_binary, args.native_binary_sha256) if args.ctc_backend == "native" else metal_ctc(alignment, records))
                yield
        native_setup_ns = None
        with contextlib.ExitStack() as resources, provision.quiet_download(), deny_python_network():
            if args.ctc_backend == "native-persistent":
                persistent = resources.enter_context(NativeCTC(args.native_library, args.native_library_sha256))
                native_identity, native_setup_ns = persistent.identity, persistent.initialization_wall_ns
            if args.ctc_backend == "native-mps":
                from tools.baseline.mps_storage_probe import MPSCTC
                kernels = Path(__file__).with_name("native_ctc.metal")
                mps_context = resources.enter_context(MPSCTC(args.mps_library, args.mps_library_sha256, kernels, manifest.sha256(kernels)))
                native_identity, native_setup_ns = mps_context.identity, mps_context.initialization_wall_ns
            import whisperx
            import whisperx.alignment as alignment
            import torch
            if not torch.backends.mps.is_available(): raise ValueError("mps_required")
            if args.interpolation_gap_control:
                from tools.baseline.native_interpolation import synthetic_sentence_gaps
                resources.enter_context(synthetic_sentence_gaps(alignment))
            audio = whisperx.load_audio(str(args.audio))
            model, metadata = whisperx.load_align_model(args.language, "mps")
            if args.warmup:
                alignment.align(original["segments"], model, metadata, audio, "mps", return_char_alignments=args.return_char_alignments)
                with candidate_context([], [], [], [], [], []): alignment.align(original["segments"], model, metadata, audio, "mps", return_char_alignments=args.return_char_alignments)
                torch.mps.synchronize()
            baseline_times, candidate_times, matches, orders = [], [], [], []
            for replica in range(args.replicas):
                order = ("original", "candidate") if replica % 2 == 0 else ("candidate", "original")
                orders.append(list(order))
                results = {}
                for backend in order:
                    torch.mps.synchronize()
                    started = time.perf_counter_ns()
                    with candidate_context(spans, wildcard_spans, timestamp_spans, mean_spans, rounding_spans, interpolation_spans) if backend == "candidate" else contextlib.nullcontext(): results[backend] = alignment.align(original["segments"], model, metadata, audio, "mps", return_char_alignments=args.return_char_alignments)
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
        report["native_fused_wildcard_enabled"] = args.native_fused_wildcard or args.ctc_backend == "native-mps"
        report["native_fused_wildcard_dispatch_count"] = sum(record.get("wildcard_fused") is True for record in spans)
        report["native_character_rounding_enabled"], report["character_rounding_spans"] = args.native_character_rounding, rounding_spans
        report["native_character_rounding_dispatch_count"] = len(rounding_spans)
        report["synthetic_sentence_gap_control"] = args.interpolation_gap_control
        report["native_interpolation_batched_enabled"] = args.native_interpolation_batched or args.mps_interpolation_batched
        report["native_nearest_interpolation_enabled"], report["interpolation_spans"] = args.native_nearest_interpolation or args.mps_nearest_interpolation, interpolation_spans
        report["native_nearest_interpolation_dispatch_count"] = sum(record["dispatch_completed"] for record in interpolation_spans)
        report["native_word_rounding_enabled"] = args.native_word_rounding
        report["native_resident_reductions_enabled"] = args.native_resident_reductions
        report["native_resident_character_rounding_enabled"] = args.native_resident_character_rounding
        report["native_character_segments_enabled"] = args.native_character_segments
        report["return_char_alignments"] = args.return_char_alignments
        report["native_character_segments_dispatch_count"] = sum(record.get("character_segments_gpu") is True for record in spans)
        report["native_timestamp_bounds_enabled"], report["timestamp_bounds_spans"] = args.native_timestamp_bounds, timestamp_spans
        report["native_timestamp_bounds_dispatch_count"] = len(timestamp_spans)
        report["native_word_means_enabled"], report["word_mean_spans"] = args.native_word_means, mean_spans
        report["native_word_means_dispatch_count"] = sum(record.get("separate_dispatch", True) for record in mean_spans)
        report["native_word_means_completed_group_count"] = len(mean_spans)
        report["native_compact_reduction_plan_enabled"] = args.native_compact_reduction_plan
        report["native_fused_reductions_enabled"] = args.native_fused_reductions
        report["native_fused_reductions_command_groups"] = sum(record.get("fused_reductions", False) for record in timestamp_spans)
        report["native_fused_reductions_subset_count"] = sum(record.get("fused_subset_count", 0) for record in timestamp_spans)
        report["native_word_means_queries_consumed"] = sum(record["queries_consumed"] for record in mean_spans)
        report["native_timestamp_fast_frames_enabled"] = args.native_timestamp_fast_frames
        report["native_timestamp_bounds_batched_enabled"] = args.native_timestamp_bounds_batched
        report["native_timestamp_bounds_query_count"] = sum(record.get("queries", 1) for record in timestamp_spans)
        report["native_timestamp_bounds_queries_consumed"] = sum(record.get("queries_consumed", 1) for record in timestamp_spans)
        if args.mps_resident_statistics:
            for field in ("native_character_rounding_enabled", "native_word_rounding_enabled", "native_resident_reductions_enabled", "native_resident_character_rounding_enabled", "native_character_segments_enabled", "native_word_means_enabled", "native_compact_reduction_plan_enabled", "native_fused_reductions_enabled", "native_timestamp_fast_frames_enabled", "native_timestamp_bounds_batched_enabled"): report[field] = True
        if args.native_timestamp_bounds or args.native_timestamp_bounds_batched or args.mps_resident_statistics:
            import numpy as np
            report["timestamp_bounds_reference_runtime"] = {"pandas": alignment.pd.__version__, "numpy": np.__version__, "policy": "binary64_min_max_skipna_negative_zero_rejected"}
        if args.ctc_backend == "native": report["residuals"][1] = "private_npz_and_native_process_startup_per_segment"
        if args.ctc_backend == "native-persistent": report["residuals"][1] = "cpu_to_native_shared_buffer_copy_per_segment"
        report["native_mps_emission_handoff_enabled"] = args.ctc_backend == "native-mps"
        report["native_mps_resident_statistics_enabled"] = args.mps_resident_statistics
        report["native_mps_nearest_interpolation_enabled"] = args.mps_nearest_interpolation
        if args.ctc_backend == "native-mps": report["residuals"] = ["mps_producer_wait_and_two_native_ctc_completions", "cpu_final_ctc_path_character_word_reductions_scaling_rounding_interpolation", "cpu_mask_and_token_metadata", "cpu_final_path_and_result_readback"]
        if args.mps_resident_statistics: report["residuals"] = ["mps_producer_wait_and_native_completion_groups", "cpu_ratio_and_metadata_selection_provenance_snapshots_and_csr_mapping", "cpu_final_rounded_character_readback_and_dataframe_construction", "cpu_original_nearest_interpolation"]
        if args.native_wildcard_max: report["residuals"][-1] = "cpu_wildcard_mask_gather_concat_and_max_readback"
        if args.native_fused_wildcard: report["residuals"][-1] = "cpu_wildcard_mask_and_token_metadata"
        if args.native_character_segments: report["residuals"][2] = "cpu_word_reductions_and_character_timestamp_scaling"
        if args.native_timestamp_bounds or args.native_timestamp_bounds_batched: report["residuals"][2] = "cpu_word_means_character_timestamp_scaling_interpolation_dataframe_selection_and_snapshot_validation"
        if args.native_word_means: report["residuals"][2] = "cpu_character_timestamp_scaling_rounding_interpolation_dataframe_selection_and_snapshot_validation"
        if args.native_character_rounding:
            report["residuals"][2] = "cpu_character_scaling_word_mean_rounding_interpolation_selection_and_snapshot_validation" if args.native_word_means else "cpu_word_reductions_and_character_scaling"
            report["residuals"].append("cpu_character_round_batch_upload_and_readback")
        if args.native_resident_character_rounding:
            report["residuals"] = [item for item in report["residuals"] if item != "cpu_character_round_batch_upload_and_readback"]
            report["residuals"][2] = "cpu_ratio_calculation_word_mean_rounding_interpolation_selection_and_snapshot_validation" if args.native_word_means else "cpu_word_reductions_and_ratio_calculation"
            report["residuals"].append("cpu_final_rounded_character_readback_and_dataframe_upload")
        if args.native_resident_reductions: report["residuals"][-1] = "cpu_final_character_readback_dataframe_fingerprints_and_row_mapping_no_character_reupload"
        if args.native_word_rounding: report["residuals"][2] = "cpu_ratio_calculation_interpolation_selection_and_snapshot_validation"
        if args.native_nearest_interpolation:
            report["residuals"][2] = report["residuals"][2].replace("_interpolation", "")
            report["residuals"].append("cpu_interpolation_series_metadata_and_upload_readback")
            if args.native_word_rounding: report["residuals"][2] = "cpu_ratio_calculation_selection_snapshot_validation_and_series_metadata"
        if args.mps_nearest_interpolation:
            report["residuals"] = [entry for entry in report["residuals"] if entry != "cpu_original_nearest_interpolation"]
            if not args.mps_resident_statistics: report["residuals"][1] = "cpu_final_ctc_path_character_word_reductions_scaling_rounding"
            report["residuals"].append("cpu_interpolation_series_metadata_and_final_timestamp_upload_readback")
            if args.mps_interpolation_batched: report["residuals"].append("cpu_interpolation_pair_packing_fingerprints_and_output_series_copy")
        with (args.output_dir / "comparison.json").open("x") as output: json.dump(report, output, indent=2)
        print(json.dumps({key: value for key, value in report.items() if key not in ("ctc_spans", "wildcard_spans", "timestamp_bounds_spans", "word_mean_spans", "character_rounding_spans", "interpolation_spans")}))
        return 0 if all(matches) else 2
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration, AttributeError, subprocess.TimeoutExpired):
        print(json.dumps({"status": "trial_failed", "code": "ctc_alignment_trial_failed", "full_baseline_gate_passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

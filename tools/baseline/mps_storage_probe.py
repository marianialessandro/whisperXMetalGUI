"""CLI-only pinned Torch MPS storage interop diagnostic; no production backend hooks."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import os
import threading
import time
import struct
import numpy as np

import torch

_LOAD_LOCK = threading.RLock()


def runtime_check():
    if torch.__version__ != "2.8.0" or not torch.backends.mps.is_available(): raise ValueError("mps_probe_runtime_not_supported")


def build_library(output):
    runtime_check()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink(): raise ValueError("mps_probe_output_exists")
    if not output.parent.is_dir(): raise ValueError("mps_probe_output_parent_missing")
    package = Path(torch.__file__).resolve().parent
    source = Path(__file__).with_suffix(".mm")
    with tempfile.TemporaryDirectory(prefix="mps-probe-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "probe.dylib"
        command = ["xcrun", "clang++", "-std=c++17", "-O1", "-fobjc-arc", "-dynamiclib", "-Wl,-headerpad_max_install_names", "-Wno-deprecated-declarations", str(source), "-I", str(package / "include"), "-I", str(package / "include/torch/csrc/api/include"), "-L", str(package / "lib"), "-ltorch_cpu", "-lc10", "-Wl,-rpath," + str(package / "lib"), "-framework", "Foundation", "-framework", "Metal", "-o", str(staged)]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        subprocess.run(["xcrun", "install_name_tool", "-change", "@loader_path/libtorch_cpu.dylib", str(package / "lib/libtorch_cpu.dylib"), "-change", "@loader_path/libc10.dylib", str(package / "lib/libc10.dylib"), str(staged)], check=True, capture_output=True, timeout=20)
        subprocess.run(["codesign", "--force", "--sign", "-", str(staged)], check=True, capture_output=True, timeout=20)
        os.link(staged, output)
    return {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "core_sha256": hashlib.sha256(source.with_name("native_ctc_core.hpp").read_bytes()).hexdigest(), "torch_version": torch.__version__, "scope": "development_only_local_torch_linkage"}


class MPSStorageProbe:
    def __init__(self, library, expected_sha256):
        runtime_check()
        library = Path(library).resolve(strict=True)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or hashlib.sha256(library.read_bytes()).hexdigest() != expected_sha256: raise ValueError("mps_probe_library_hash_mismatch")
        with _LOAD_LOCK:
            loaded = getattr(torch, "_whisperx_mps_probe_library_identity", None)
            if loaded is not None and loaded != (str(library), expected_sha256): raise ValueError("mps_probe_different_library_already_loaded")
            if loaded is None:
                torch.ops.load_library(str(library))
                torch._whisperx_mps_probe_library_identity = (str(library), expected_sha256)
        self.library_sha256 = expected_sha256

    def row_max(self, tensor):
        runtime_check()
        if not isinstance(tensor, torch.Tensor) or tensor.is_neg() or tensor.is_conj(): raise ValueError("mps_probe_tensor_not_supported")
        maxima, indices, metrics = torch.ops.whisperx_mps_probe.row_max(tensor)
        wait, wall, gpu, offset, source_bytes, readback, setup = metrics.tolist()
        return maxima, indices, {"scope": "development_mps_storage_row_max", "torch_version": torch.__version__, "library_sha256": self.library_sha256, "producer_completion_wait_seconds": wait, "completion_wall_seconds": wall, "computation_gpu_seconds": gpu, "pipeline_setup_included": bool(setup), "source_storage_offset_bytes": int(offset), "source_logical_bytes": int(source_bytes), "logical_emission_host_readback_bytes": 0, "logical_emission_reupload_bytes": 0, "logical_final_result_readback_bytes": int(readback), "physical_copy_bytes": None, "command_buffers_per_run": 1, "synchronization": "mps_stream_commit_and_wait_then_private_metal_queue_wait", "input_lifetime": "owning_tensor_and_strong_buffer_until_completion", "quality_certified": False, "production_backend": False}


class MPSCTC:
    def __init__(self, library, expected_sha256, kernels, kernels_sha256, retain_timestamps=False):
        if type(retain_timestamps) is not bool: raise ValueError("mps_ctc_timestamp_option_invalid")
        probe = MPSStorageProbe(library, expected_sha256)
        kernels = Path(kernels).resolve(strict=True)
        if kernels.stat().st_size > 1024 * 1024 or hashlib.sha256(kernels.read_bytes()).hexdigest() != kernels_sha256: raise ValueError("mps_ctc_shader_identity_invalid")
        self.lock = threading.RLock()
        self._resident_rounded_snapshot = None
        self._statistics_snapshot = None
        self.retain_timestamps = retain_timestamps
        started = time.perf_counter_ns()
        self.context = torch.classes.whisperx_mps_probe.CTC(kernels.read_text())
        self.initialization_wall_ns = time.perf_counter_ns() - started
        self.identity = {"library_sha256": probe.library_sha256, "shader_sha256": kernels_sha256, "torch_version": torch.__version__}

    def __enter__(self): return self

    def __exit__(self, *args): self.close()

    def close(self):
        with self.lock:
            if self.context is not None:
                self.context.close()
                self.context = None
                self._resident_rounded_snapshot = None
                self._statistics_snapshot = None

    def compute(self, emission, tokens, blank=0, wildcard=False):
        return self._compute(emission, tokens, blank, wildcard, False)

    def prepare_segments(self, emission, tokens, blank=0, wildcard=False):
        if sys.version_info[:3] != (3, 12, 12): raise ValueError("native_resident_round_runtime_not_supported")
        return self._compute(emission, tokens, blank, wildcard, True)

    def _compute(self, emission, tokens, blank, wildcard, resident):
        with self.lock:
            runtime_check()
            if self.context is None: raise ValueError("mps_ctc_context_closed")
            if not isinstance(tokens, (list, tuple)) or not 0 < len(tokens) <= 32768 or any(type(token) is not int or not -(2**31) <= token < 2**31 for token in tokens) or type(blank) is not int or not -(2**31) <= blank < 2**31 or type(wildcard) is not bool: raise ValueError("mps_ctc_token_metadata_invalid")
            started = time.perf_counter_ns()
            self._resident_rounded_snapshot = None
            self._statistics_snapshot = None
            native_tokens = torch.tensor(tokens, dtype=torch.int32, device="cpu")
            path, metrics = (self.context.prepare if resident else self.context.compute)(emission, native_tokens, blank, wildcard)
            found, wait, wall, validation_gpu, compute_gpu, dp_bytes, allocated, offset, readback = metrics.tolist()
            if resident:
                from tools.baseline.native_ctc import ResidentCharacterSegments
                generation, count = path.tolist()
                if (generation == 0) == bool(found) or count != len(tokens): raise ValueError("mps_ctc_resident_result_invalid")
                points = ResidentCharacterSegments(self, generation, count) if found else None
            else: points = [(int(row[0]), int(row[1]), row[2]) for row in path.tolist()] if found else None
            record = {"scope": "experimental_native_mps_ctc", "frames": emission.shape[0], "vocabulary": emission.shape[1], "tokens": len(tokens), "wildcard_fused": wildcard, "producer_completion_wait_seconds": wait, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_completion_wall_ns": int(wall * 1e9), "validation_gpu_seconds": validation_gpu, "computation_gpu_seconds": compute_gpu, "command_buffers_per_run": 2, "dp_logical_storage_bytes": int(dp_bytes), "native_owned_buffer_bytes": int(allocated), "borrowed_source_storage_offset_bytes": int(offset), "logical_emission_host_readback_bytes": 0, "logical_cpu_to_native_emission_bytes": 0, "logical_token_upload_bytes": len(tokens) * 4, "logical_final_path_and_control_readback_bytes": int(readback), "physical_copy_bytes": None, "source_lifetime": "owning_tensor_and_strong_metal_reference_until_completion", "synchronization": "mps_producer_wait_then_two_native_ctc_completions", "quality_certified": False, "production_backend": False}
            record.update(character_segments_gpu=resident, character_segments_resident=resident, logical_character_segment_readback_bytes=0, logical_resident_generation_metadata_bytes=16 if resident else 0)
            return points, record

    def scale_round_segments(self, segments, ratio, offset):
        from tools.baseline.native_ctc import ResidentCharacterSegments
        if not isinstance(segments, ResidentCharacterSegments) or segments.owner is not self or type(segments.generation) is not int or type(segments.count) is not int or not 0 < segments.generation < 2**63 or not 0 < segments.count <= 32768: raise ValueError("native_resident_handle_invalid")
        if sys.version_info[:3] != (3, 12, 12): raise ValueError("native_resident_round_runtime_not_supported")
        if type(ratio) is not float or type(offset) not in (int, float) or (type(offset) is int and abs(offset) > 2**53): raise ValueError("native_resident_scale_type_invalid")
        bits = np.frombuffer(struct.pack("dd", ratio, float(offset)), dtype=np.uint64).copy()
        with self.lock:
            if self.context is None: raise ValueError("mps_ctc_context_closed")
            started = time.perf_counter_ns()
            self._resident_rounded_snapshot = None
            self._statistics_snapshot = None
            result, metrics = self.context.scale(segments.generation, segments.count, torch.from_numpy(bits))
            output = result.numpy()
            wall, gpu, allocated = metrics.tolist()
            self._resident_rounded_snapshot = (segments.generation, segments.count, int(bits[0]), int(bits[1]), output.tobytes())
            return output, {"scope": "experimental_mps_resident_character_scale_round", "elements": output.size, "characters": segments.count, "dispatch_completed": True, "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": int(wall), "computation_gpu_seconds": gpu, "logical_cpu_to_native_bytes": 16, "logical_character_reupload_bytes": 0, "cpu_resident_snapshot_bytes": output.nbytes, "logical_output_readback_bytes": output.nbytes, "logical_invalid_control_readback_bytes": 4, "physical_copy_bytes": None, "allocated_shared_buffer_bytes": int(allocated)}

    def interpolate_nearest(self, values):
        return self._interpolate_nearest(values, 1)

    def interpolate_nearest_pair(self, values):
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 2 or not 0 < len(values) <= 32768: raise ValueError("native_nearest_pair_format_invalid")
        output, record = self._interpolate_nearest(np.ascontiguousarray(values.T).reshape(-1), 2)
        return output.reshape(2, -1).T, record

    def _interpolate_nearest(self, values, columns):
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 1 or not 0 < values.size <= 32768 * columns or values.size % columns: raise ValueError("native_nearest_format_invalid")
        with self.lock:
            if self.context is None: raise ValueError("mps_ctc_context_closed")
            started = time.perf_counter_ns()
            packed = np.ascontiguousarray(values)
            readonly_copy_bytes = 0
            if not packed.flags.writeable:
                packed = packed.copy()
                readonly_copy_bytes = packed.nbytes
            result, counts, metrics = self.context.nearest(torch.from_numpy(packed), columns)
            output = result.numpy()
            wall, gpu, allocated = metrics.tolist()
            completed_counts = counts.tolist()
            return output, {"scope": "experimental_mps_nearest_interpolation", "elements": values.size, "columns": columns, "input_missing_counts": [count[0] for count in completed_counts], "filled_missing_counts": [count[1] for count in completed_counts], "logical_control_readback_bytes": columns * 8, "dispatch_completed": True, "command_buffers_per_run": 1, "gpu_kernels": ["ctc_nearest_neighbors", "ctc_nearest_fill"], "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": int(wall), "computation_gpu_seconds": gpu, "logical_cpu_to_native_bytes": packed.nbytes, "logical_output_readback_bytes": output.nbytes, "cpu_readonly_input_copy_bytes": readonly_copy_bytes, "native_host_result_staging_bytes": 0, "physical_copy_bytes": None, "allocated_shared_buffer_bytes": int(allocated)}

    def alignment_stats_resident(self, values, indices, offsets, modes):
        statistics, _, record = self._alignment_stats_resident(values, indices, offsets, modes, False)
        return statistics, record

    def alignment_stats_resident_rounded(self, values, indices, offsets, modes):
        return self._alignment_stats_resident(values, indices, offsets, modes, True)

    def _alignment_stats_resident(self, values, indices, offsets, modes, rounded):
        from tools.baseline.resident_statistics import resident_statistics_plan
        with self.lock:
            if self.context is None: raise ValueError("mps_ctc_context_closed")
            started = time.perf_counter_ns()
            generation, count, ratio, offset, reference, selected, mapping, mapped, offsets, modes = resident_statistics_plan(values, indices, offsets, modes, self._resident_rounded_snapshot)
            ready = time.perf_counter_ns()
            arguments = (generation, count, torch.from_numpy(np.array([ratio, offset], dtype=np.uint64)), torch.from_numpy(mapped), torch.from_numpy(offsets), torch.from_numpy(modes), rounded)
            self._statistics_snapshot = None
            if self.retain_timestamps:
                output, metrics, handle = self.context.stats_retained(*arguments)
                stats_generation, queries = handle.tolist()
                if stats_generation <= 0 or queries != modes.size: raise ValueError("mps_ctc_statistics_handle_invalid")
            else: output, metrics = self.context.stats(*arguments)
            result = output.numpy()
            wall, gpu, allocated = metrics.tolist()
            record = {"scope": "experimental_mps_resident_fused_stats", "rows": len(values), "resident_rows": count + 1, "queries": modes.size, "index_elements": mapped.size, "dispatch_completed": True, "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": int(wall), "computation_gpu_seconds": gpu, "logical_cpu_to_native_bytes": mapped.nbytes + offsets.nbytes + modes.nbytes, "logical_character_values_upload_bytes": 0, "logical_output_readback_bytes": result.nbytes + 4, "word_rounding_gpu": rounded, "cpu_resident_snapshot_bytes": len(reference), "cpu_resident_snapshot_gather_bytes": selected.nbytes, "cpu_resident_mapping_bytes": mapping.nbytes, "cpu_resident_mapping_wall_ns": ready - started, "physical_copy_bytes": None, "allocated_shared_buffer_bytes": int(allocated)}
            statistics = result[:modes.size * 3].reshape(-1, 3)
            if self.retain_timestamps:
                from tools.baseline.resident_timestamps import ResidentStatistics
                self._statistics_snapshot = stats_generation, modes.size, statistics.tobytes()
                statistics = ResidentStatistics(statistics, self, stats_generation)
                record.update(statistics_retained_gpu=True, logical_statistics_handle_metadata_bytes=16, cpu_statistics_snapshot_bytes=statistics.nbytes)
            return statistics, result[modes.size * 3:] if rounded else None, record

    def interpolate_retained_pair(self, data):
        from tools.baseline.resident_timestamps import ResidentTimestamp
        if not isinstance(data, list) or not 0 < len(data) <= 32768: raise ValueError("mps_ctc_retained_pair_shape_invalid")
        with self.lock:
            if self.context is None: raise ValueError("mps_ctc_context_closed")
            if self._statistics_snapshot is None: raise ValueError("mps_ctc_statistics_snapshot_missing")
            started = time.perf_counter_ns()
            generation, queries, reference = self._statistics_snapshot
            mapping = []
            for column, name in enumerate(("start", "end")):
                for row in data:
                    if not isinstance(row, dict): raise ValueError("mps_ctc_retained_timestamp_metadata_invalid")
                    value = row.get(name)
                    if not isinstance(value, ResidentTimestamp) or value.owner is not self or value.generation != generation or not 0 <= value.position < queries * 3 or value.position % 3 != column: raise ValueError("mps_ctc_retained_timestamp_metadata_invalid")
                    if value.tobytes() != reference[value.position * 8:(value.position + 1) * 8]: raise ValueError("mps_ctc_retained_timestamp_changed")
                    mapping.append(value.position)
            packed = np.asarray(mapping, dtype=np.uint32)
            output, counts, metrics = self.context.nearest_retained(generation, torch.from_numpy(packed))
            result = output.numpy().reshape(2, -1).T
            wall, gpu, allocated = metrics.tolist()
            completed_counts = counts.tolist()
            return result, {"scope": "experimental_mps_retained_nearest", "elements": result.size, "columns": 2, "input_missing_counts": [count[0] for count in completed_counts], "filled_missing_counts": [count[1] for count in completed_counts], "logical_control_readback_bytes": 16, "logical_invalid_control_readback_bytes": 4, "dispatch_completed": True, "command_buffers_per_run": 1, "gpu_kernels": ["ctc_resident_nearest_gather", "ctc_nearest_neighbors", "ctc_nearest_fill"], "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "computation_wall_ns": int(wall), "computation_gpu_seconds": gpu, "logical_cpu_to_native_bytes": packed.nbytes, "logical_timestamp_values_upload_bytes": 0, "logical_output_readback_bytes": result.nbytes, "cpu_retained_timestamp_snapshot_bytes": len(reference), "physical_copy_bytes": None, "allocated_shared_buffer_bytes": int(allocated)}


def probe_npz(args):
    source = args.emissions.resolve(strict=True)
    if args.out.exists() or args.out.is_symlink(): raise ValueError("mps_probe_output_exists")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.baseline.ctc_metal_probe import load_inputs
    values, _, _ = load_inputs(source, args.emissions_sha256)
    if values.shape[1] > 4096: raise ValueError("mps_probe_npz_format_invalid")
    tensor = torch.from_numpy(values.copy()).to("mps")
    probe = MPSStorageProbe(args.library, args.library_sha256)
    results = []
    for _ in range(args.replicas):
        actual, indices, record = probe.row_max(tensor)
        expected, expected_indices = tensor.cpu().max(dim=1)
        if actual.numpy().tobytes() != expected.numpy().tobytes() or not torch.equal(indices.to(torch.int64), expected_indices): raise ValueError("mps_probe_reference_mismatch")
        results.append(record)
    report = {"status": "mps_storage_row_max_exactly_equal", "emissions_sha256": args.emissions_sha256, "input_scope": "previously_captured_neural_emissions_reuploaded_to_mps_not_live_neural_pipeline", "oracle_emission_readback_separate_from_candidate": True, "runs": results, "full_baseline_gate_passed": False}
    with args.out.open("x") as handle: json.dump(report, handle, indent=2, allow_nan=False)
    return report


def probe_live(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.baseline import manifest, provision
    from tools.baseline.offline_worker import deny_python_network
    runtime_check()
    if args.out.exists() or args.out.is_symlink() or manifest.sha256(args.audio) != args.audio_sha256: raise ValueError("mps_probe_live_inputs_invalid")
    specs = json.loads(args.torch_specs.read_text())
    spec = next((item for item in specs if item.get("role") == "alignment_" + args.language), None)
    if spec is None or spec.get("license_verified") is not True or provision.inspect_spec(spec)["status"] != "files_verified": raise ValueError("mps_probe_live_checkpoint_not_verified")
    probe = MPSStorageProbe(args.library, args.library_sha256)
    changes = {"TORCH_HOME": spec["cache_root"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin"}
    previous = {key: os.environ.get(key) for key in changes}
    try:
        os.environ.update(changes)
        with provision.quiet_download(), deny_python_network():
            import whisperx
            audio = whisperx.load_audio(str(args.audio))
            first, last = int(args.start * 16000), int((args.start + args.duration) * 16000)
            if first < 0 or last > len(audio) or last <= first: raise ValueError("mps_probe_live_audio_span_invalid")
            model, metadata = whisperx.load_align_model(args.language, "mps")
            if metadata["type"] != "torchaudio": raise ValueError("mps_probe_live_model_type_not_supported")
            waveform = torch.from_numpy(audio[first:last]).unsqueeze(0).to("mps")
            results = []
            for _ in range(args.replicas):
                with torch.inference_mode():
                    logits, _ = model(waveform)
                    emission = torch.log_softmax(logits, dim=-1)[0].detach()
                    actual, indices, record = probe.row_max(emission)
                    expected, expected_indices = emission.cpu().max(dim=1)
                if actual.numpy().tobytes() != expected.numpy().tobytes() or not torch.equal(indices.to(torch.int64), expected_indices): raise ValueError("mps_probe_reference_mismatch")
                record.update(frames=emission.shape[0], vocabulary=emission.shape[1], oracle_maxima_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest())
                results.append(record)
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    report = {"status": "live_mps_storage_row_max_exactly_equal", "input_scope": "live_torchaudio_neural_logits_and_log_softmax_on_mps", "audio_sha256": args.audio_sha256, "checkpoint_sha256": spec["sha256"], "language": args.language, "start_seconds": args.start, "duration_seconds": args.duration, "oracle_emission_readback_separate_from_candidate": True, "network_guard_scope": "python_not_os_isolation", "runs": results, "full_baseline_gate_passed": False}
    with args.out.open("x") as handle: json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Compile a new local development dylib against the pinned installed Torch headers/libraries.")
    build.add_argument("--out", required=True, type=Path)
    probe = commands.add_parser("probe", help="Read captured NPZ emissions and compare direct MPS-buffer row maxima with a separate CPU oracle.")
    probe.add_argument("--library", required=True, type=Path)
    probe.add_argument("--library-sha256", required=True)
    probe.add_argument("--emissions", required=True, type=Path)
    probe.add_argument("--emissions-sha256", required=True)
    probe.add_argument("--replicas", type=int, default=6)
    probe.add_argument("--out", required=True, type=Path)
    live = commands.add_parser("live", help="Run a verified local IT/EN torchaudio aligner and consume its pending MPS log emissions directly from Metal.")
    live.add_argument("--library", required=True, type=Path)
    live.add_argument("--library-sha256", required=True)
    live.add_argument("--audio", required=True, type=Path)
    live.add_argument("--audio-sha256", required=True)
    live.add_argument("--torch-specs", required=True, type=Path)
    live.add_argument("--language", required=True, choices=("it", "en"))
    live.add_argument("--start", type=float, default=0.)
    live.add_argument("--duration", type=float, default=6.)
    live.add_argument("--replicas", type=int, default=6)
    live.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command != "build" and not 1 <= args.replicas <= 100: parser.error("replicas must be in 1..100")
    if args.command == "live" and (not 0 <= args.start <= 86400 or not .5 <= args.duration <= 30): parser.error("live start must be in 0..86400 seconds and duration in .5..30 seconds")
    try:
        result = build_library(args.out) if args.command == "build" else probe_live(args) if args.command == "live" else probe_npz(args)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError):
        print(json.dumps({"status": "mps_storage_probe_failed"}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())

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
    return {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "torch_version": torch.__version__, "scope": "development_only_local_torch_linkage"}


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

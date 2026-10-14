"""Source-checkout CLI experiment: original WhisperX pipeline with native Metal CTC, never a CPU fallback or a certified production backend."""

import argparse
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@contextlib.contextmanager
def native_alignment(task, context, records, resident_statistics=False):
    import whisperx.alignment as alignment
    from tools.baseline.native_mps_ctc import native_mps_ctc
    from tools.baseline.native_ctc_fused import fused_wildcard_preparation

    original = task.align

    def align(*args, **kwargs):
        arguments = inspect.signature(original).bind(*args, **kwargs)
        arguments.apply_defaults()
        metadata = arguments.arguments["align_model_metadata"]
        if str(arguments.arguments["device"]) != "mps":
            raise ValueError("experimental_ctc_requires_mps")
        if metadata["language"] not in ("it", "en"):
            raise ValueError("experimental_ctc_language_not_supported")
        if resident_statistics and arguments.arguments["interpolate_method"] != "nearest":
            raise ValueError("experimental_resident_statistics_require_nearest")
        with contextlib.ExitStack() as scopes:
            if resident_statistics:
                from tools.baseline.native_timestamp_bounds import native_timestamp_bounds_batched
                from tools.baseline.native_interpolation import native_nearest_interpolation
                scopes.enter_context(native_timestamp_bounds_batched(alignment, context, records["bounds"], fast_frames=True, word_means=True, mean_records=records["means"], fused_reductions=True, compact_plan=True, resident_reductions=True, word_rounding=True))
                scopes.enter_context(native_nearest_interpolation(alignment, context, records["interpolation"], batched_frames=True, retained_frames=True))
            scopes.enter_context(fused_wildcard_preparation(max(metadata["dictionary"].values()) + 1))
            scopes.enter_context(native_mps_ctc(alignment, context, records["ctc"], resident_statistics=resident_statistics, rounding_records=records["rounding"]))
            return original(*args, **kwargs)

    task.align = align
    try:
        yield
    finally:
        task.align = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, epilog="Remaining arguments are forwarded to python -m whisperx. Supply audio paths, --asr_backend mlx, --device mps, --language it or en and --output_dir. See python -m whisperx --help for the unchanged pipeline options. Requires the source checkout and Torch 2.8.0; resident statistics also require CPython 3.12.12.")
    parser.add_argument("--mps-library", type=Path, required=True, help="Prebuilt local Torch MPS interop dylib; build with tools/baseline/mps_storage_probe.py build --out PATH.")
    parser.add_argument("--mps-library-sha256", required=True, help="Expected SHA256 of that exact dylib; checked before loading native code.")
    parser.add_argument("--native-report", type=Path, required=True, help="New private JSON report; existing files are never overwritten. No CLI arguments, credentials, audio or transcript values are stored.")
    parser.add_argument("--resident-statistics", action="store_true", help="Additionally keep character statistics resident and run timestamp/word reductions, rounding and paired nearest interpolation on Metal; nearest interpolation only.")
    parser.add_argument("--seed", type=int, help="Optional reproducible diagnostic RNG initialization; omitted by default, with no decoder policy change.")
    args, forwarded = parser.parse_known_args(argv)
    if not forwarded:
        parser.error("WhisperX audio paths and pipeline options are required.")
    if "--no_align" in forwarded:
        parser.error("The native CTC experiment requires alignment.")
    report = {"schema_version": 1, "scope": "source_checkout_native_mps_ctc", "status": "failed", "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False, "resident_statistics": args.resident_statistics}
    records = {key: [] for key in ("ctc", "bounds", "means", "rounding", "interpolation")}
    with args.native_report.open("x", encoding="utf-8") as output:
        try:
            import whisperx.transcribe as task
            from whisperx.__main__ import cli
            from tools.baseline.mps_storage_probe import MPSCTC

            if args.seed is not None:
                from tools.baseline.reproducibility import seed_process
                seed_process(args.seed)
                report["seed"] = args.seed

            kernels = Path(__file__).parent / "baseline/native_ctc.metal"
            shader_hash = hashlib.sha256(kernels.read_bytes()).hexdigest()
            original_argv = sys.argv
            try:
                with MPSCTC(args.mps_library, args.mps_library_sha256, kernels, shader_hash, retain_timestamps=args.resident_statistics) as context:
                    report["native_identity"] = context.identity
                    report["initialization_wall_ns"] = context.initialization_wall_ns
                    with native_alignment(task, context, records, args.resident_statistics):
                        sys.argv = ["whisperx", *forwarded]
                        cli()
                    report["status"] = "completed_with_native_ctc" if records["ctc"] else "completed_without_native_ctc"
            finally:
                sys.argv = original_argv
        finally:
            report["operations"] = records
            report["native_ctc_calls"] = len(records["ctc"])
            report["logical_emission_host_readback_bytes"] = sum(item["logical_emission_host_readback_bytes"] for item in records["ctc"])
            report["logical_cpu_to_native_emission_bytes"] = sum(item["logical_cpu_to_native_emission_bytes"] for item in records["ctc"])
            json.dump(report, output, allow_nan=False, indent=2)
            output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    parser.add_argument("--speaker-library", type=Path, help="Optional prebuilt native Metal speaker-assignment dylib; requires --diarize and --speaker-library-sha256.")
    parser.add_argument("--speaker-library-sha256", help="Expected SHA256 of the optional speaker-assignment dylib; no default backend change.")
    parser.add_argument("--embedding-library", type=Path, help="Optional native Metal VBx training-embedding normalization dylib; requires --diarize and --embedding-library-sha256, with pinned pyannote 4.0.4.")
    parser.add_argument("--embedding-library-sha256", help="Expected SHA256 of the optional embedding-normalization dylib; clustering defaults remain unchanged.")
    parser.add_argument("--plda-normalization-library", type=Path, help="Optional native binary64 L2 normalization inside PLDA preprocessing; requires --diarize and its paired SHA256 flag. Matrix products remain original unless --plda-matrix-products is selected; model setup remains original.")
    parser.add_argument("--plda-normalization-library-sha256", help="Expected SHA256 of the binary64-normalization dylib; requires pyannote 4.0.4 and NumPy 2.2.6.")
    parser.add_argument("--plda-matrix-products", action="store_true", help="Additionally run LDA/PLDA products on Metal with sequential binary64 multiply/add; requires --plda-normalization-library with matrix ABI. This policy is not bitwise BLAS equivalence.")
    parser.add_argument("--plda-centering-scaling", action="store_true", help="Additionally run all three PLDA centering and both dimension-scaling steps on Metal, including scalar square roots; requires --plda-matrix-products and elementwise ABI.")
    parser.add_argument("--vbx-matrix-products", action="store_true", help="Run the three recurring VBx matrix/vector products on native Metal; requires --plda-normalization-library with matrix ABI. Other VBx arithmetic and stopping remain original.")
    parser.add_argument("--vbx-precomputed-terms", action="store_true", help="Additionally prepare G, sqrt(Phi) and rho on Metal; requires --vbx-matrix-products and precompute ABI. NumPy reduction order and pinned log(2*pi) constant are preserved.")
    parser.add_argument("--vbx-statistics", action="store_true", help="Additionally reduce gamma and update invL/alpha on Metal; requires --vbx-matrix-products and statistics ABI. Log/exp, posteriors and ELBO remain original.")
    parser.add_argument("--vbx-score-arithmetic", action="store_true", help="Additionally compute invL+alpha squared and combine frame scores on Metal; requires --vbx-matrix-products and score ABI. Log/exp, posterior/prior and ELBO remain original.")
    parser.add_argument("--vbx-prior-updates", action="store_true", help="Additionally reduce posterior responsibilities and normalize speaker priors on Metal; requires --vbx-matrix-products and prior ABI. Exponentials, logs and ELBO remain original.")
    parser.add_argument("--vbx-posterior-arithmetic", action="store_true", help="Additionally add log priors to scores and subtract frame normalizers on Metal; requires --vbx-matrix-products and posterior ABI. Log/exp/logsumexp remain original.")
    parser.add_argument("--vbx-elbo-arithmetic", action="store_true", help="Additionally reduce likelihood and compute ELBO regularization on Metal; requires --vbx-matrix-products and ELBO ABI. Inverse-precision log and stopping remain original.")
    parser.add_argument("--vbx-transcendentals", action="store_true", help="Additionally run bounded OpenLibm-policy binary64 log/exp on Metal; requires --vbx-matrix-products and transcendental ABI. Rounding can differ from NumPy; logsumexp remains original.")
    parser.add_argument("--vbx-logsumexp", action="store_true", help="Additionally run bounded finite unweighted logsumexp on Metal; requires --vbx-matrix-products, logsumexp ABI and pinned SciPy 1.15.3. Uses OpenLibm-policy math; may differ from SciPy rounding.")
    parser.add_argument("--vbx-initialization", action="store_true", help="Additionally initialize uniform VBx priors and normalize generated gamma on Metal; requires --vbx-matrix-products and initialization ABI. NumPy random sampling remains unchanged on CPU.")
    parser.add_argument("--vbx-scalar-arithmetic", action="store_true", help="Additionally add prior epsilon and subtract ELBO scalars on Metal; requires --vbx-matrix-products and posterior ABI. Original comparisons and stopping control are preserved.")
    args, forwarded = parser.parse_known_args(argv)
    if not forwarded:
        parser.error("WhisperX audio paths and pipeline options are required.")
    if "--no_align" in forwarded:
        parser.error("The native CTC experiment requires alignment.")
    if bool(args.speaker_library) != bool(args.speaker_library_sha256):
        parser.error("--speaker-library and --speaker-library-sha256 must be supplied together.")
    if args.speaker_library and "--diarize" not in forwarded:
        parser.error("--speaker-library requires --diarize.")
    if bool(args.embedding_library) != bool(args.embedding_library_sha256):
        parser.error("--embedding-library and --embedding-library-sha256 must be supplied together.")
    if args.embedding_library and "--diarize" not in forwarded:
        parser.error("--embedding-library requires --diarize.")
    if bool(args.plda_normalization_library) != bool(args.plda_normalization_library_sha256):
        parser.error("--plda-normalization-library and --plda-normalization-library-sha256 must be supplied together.")
    if args.plda_normalization_library and "--diarize" not in forwarded:
        parser.error("--plda-normalization-library requires --diarize.")
    if args.plda_matrix_products and not args.plda_normalization_library:
        parser.error("--plda-matrix-products requires --plda-normalization-library and its SHA256.")
    if args.plda_centering_scaling and not args.plda_matrix_products:
        parser.error("--plda-centering-scaling requires --plda-matrix-products.")
    if args.vbx_matrix_products and not args.plda_normalization_library:
        parser.error("--vbx-matrix-products requires --plda-normalization-library and its SHA256.")
    if args.vbx_initialization and not args.vbx_matrix_products:
        parser.error("--vbx-initialization requires --vbx-matrix-products.")
    if args.vbx_scalar_arithmetic and not args.vbx_matrix_products:
        parser.error("--vbx-scalar-arithmetic requires --vbx-matrix-products.")
    if args.vbx_logsumexp and not args.vbx_matrix_products:
        parser.error("--vbx-logsumexp requires --vbx-matrix-products.")
    if args.vbx_transcendentals and not args.vbx_matrix_products:
        parser.error("--vbx-transcendentals requires --vbx-matrix-products.")
    if args.vbx_elbo_arithmetic and not args.vbx_matrix_products:
        parser.error("--vbx-elbo-arithmetic requires --vbx-matrix-products.")
    if args.vbx_posterior_arithmetic and not args.vbx_matrix_products:
        parser.error("--vbx-posterior-arithmetic requires --vbx-matrix-products.")
    if args.vbx_prior_updates and not args.vbx_matrix_products:
        parser.error("--vbx-prior-updates requires --vbx-matrix-products.")
    if args.vbx_score_arithmetic and not args.vbx_matrix_products:
        parser.error("--vbx-score-arithmetic requires --vbx-matrix-products.")
    if args.vbx_statistics and not args.vbx_matrix_products:
        parser.error("--vbx-statistics requires --vbx-matrix-products.")
    if args.vbx_precomputed_terms and not args.vbx_matrix_products:
        parser.error("--vbx-precomputed-terms requires --vbx-matrix-products.")
    report = {"schema_version": 1, "scope": "source_checkout_native_mps_ctc", "status": "failed", "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False, "resident_statistics": args.resident_statistics, "embedding_normalization_requested": bool(args.embedding_library), "plda_normalization_requested": bool(args.plda_normalization_library), "plda_matrix_products_requested": args.plda_matrix_products, "plda_centering_scaling_requested": args.plda_centering_scaling, "vbx_matrix_products_requested": args.vbx_matrix_products, "vbx_initialization_requested": args.vbx_initialization, "vbx_scalar_arithmetic_requested": args.vbx_scalar_arithmetic, "vbx_logsumexp_requested": args.vbx_logsumexp, "vbx_transcendentals_requested": args.vbx_transcendentals, "vbx_elbo_arithmetic_requested": args.vbx_elbo_arithmetic, "vbx_posterior_arithmetic_requested": args.vbx_posterior_arithmetic, "vbx_prior_updates_requested": args.vbx_prior_updates, "vbx_score_arithmetic_requested": args.vbx_score_arithmetic, "vbx_statistics_requested": args.vbx_statistics, "vbx_precomputed_terms_requested": args.vbx_precomputed_terms}
    records = {key: [] for key in ("ctc", "bounds", "means", "rounding", "interpolation", "speaker_assignment", "embedding_normalization", "plda_normalization", "plda_matrix_products", "plda_centering_scaling", "vbx_matrix_products", "vbx_precomputed_terms", "vbx_statistics", "vbx_score_arithmetic", "vbx_prior_updates", "vbx_posterior_arithmetic", "vbx_elbo_arithmetic", "vbx_transcendentals", "vbx_logsumexp", "vbx_scalar_arithmetic", "vbx_initialization")}
    vbx_calls = []
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
                    with contextlib.ExitStack() as scopes:
                        if args.plda_normalization_library:
                            from tools.baseline.native_double import NativeDoubleNormalization
                            from tools.baseline.native_plda_adapter import native_plda_normalization
                            plda_context = scopes.enter_context(NativeDoubleNormalization(args.plda_normalization_library, args.plda_normalization_library_sha256))
                            report["plda_normalization_native_identity"] = plda_context.identity
                            report["plda_normalization_initialization_wall_ns"] = plda_context.initialization_wall_ns
                            scopes.enter_context(native_plda_normalization(plda_context, records["plda_normalization"]))
                            if args.plda_matrix_products:
                                from tools.baseline.native_plda_matrix import native_plda_matrix_products
                                scopes.enter_context(native_plda_matrix_products(plda_context, records["plda_matrix_products"], centering_scaling=args.plda_centering_scaling, elementwise_records=records["plda_centering_scaling"]))
                            if args.vbx_matrix_products:
                                from tools.baseline.native_vbx_matrix import native_vbx_matrix_products
                                scopes.enter_context(native_vbx_matrix_products(plda_context, records["vbx_matrix_products"], vbx_calls, precompute=args.vbx_precomputed_terms, precompute_records=records["vbx_precomputed_terms"], statistics=args.vbx_statistics, statistics_records=records["vbx_statistics"], scores=args.vbx_score_arithmetic, score_records=records["vbx_score_arithmetic"], prior=args.vbx_prior_updates, prior_records=records["vbx_prior_updates"], posterior=args.vbx_posterior_arithmetic, posterior_records=records["vbx_posterior_arithmetic"], elbo=args.vbx_elbo_arithmetic, elbo_records=records["vbx_elbo_arithmetic"], transcendentals=args.vbx_transcendentals, transcendental_records=records["vbx_transcendentals"], logsumexp=args.vbx_logsumexp, logsumexp_records=records["vbx_logsumexp"], scalars=args.vbx_scalar_arithmetic, scalar_records=records["vbx_scalar_arithmetic"], initialization=args.vbx_initialization, initialization_records=records["vbx_initialization"]))
                        if args.embedding_library:
                            from tools.baseline.native_embedding import NativeEmbeddingNormalization
                            from tools.baseline.native_embedding_adapter import native_embedding_normalization
                            embedding_context = scopes.enter_context(NativeEmbeddingNormalization(args.embedding_library, args.embedding_library_sha256))
                            report["embedding_native_identity"] = embedding_context.identity
                            report["embedding_initialization_wall_ns"] = embedding_context.initialization_wall_ns
                            scopes.enter_context(native_embedding_normalization(embedding_context, records["embedding_normalization"]))
                        if args.speaker_library:
                            from tools.baseline.native_speaker import NativeSpeakerAssignment, native_speaker_assignment
                            speaker_context = scopes.enter_context(NativeSpeakerAssignment(args.speaker_library, args.speaker_library_sha256))
                            report["speaker_native_identity"] = speaker_context.identity
                            report["speaker_initialization_wall_ns"] = speaker_context.initialization_wall_ns
                            scopes.enter_context(native_speaker_assignment(task, speaker_context, records["speaker_assignment"]))
                        scopes.enter_context(native_alignment(task, context, records, args.resident_statistics))
                        sys.argv = ["whisperx", *forwarded]
                        cli()
                    report["status"] = "completed_with_native_ctc" if records["ctc"] else "completed_without_native_ctc"
            finally:
                sys.argv = original_argv
        finally:
            report["operations"] = records
            report["native_ctc_calls"] = len(records["ctc"])
            report["native_vbx_initialization_calls"] = len(records["vbx_initialization"])
            report["vbx_initialization_observed"] = bool(records["vbx_initialization"])
            report["native_vbx_scalar_calls"] = len(records["vbx_scalar_arithmetic"])
            report["vbx_scalar_arithmetic_observed"] = bool(records["vbx_scalar_arithmetic"])
            report["native_vbx_logsumexp_calls"] = len(records["vbx_logsumexp"])
            report["vbx_logsumexp_observed"] = bool(records["vbx_logsumexp"])
            report["native_vbx_transcendental_calls"] = len(records["vbx_transcendentals"])
            report["vbx_transcendentals_observed"] = bool(records["vbx_transcendentals"])
            report["native_vbx_elbo_calls"] = len(records["vbx_elbo_arithmetic"])
            report["vbx_elbo_arithmetic_observed"] = bool(records["vbx_elbo_arithmetic"])
            report["native_vbx_posterior_calls"] = len(records["vbx_posterior_arithmetic"])
            report["vbx_posterior_arithmetic_observed"] = bool(records["vbx_posterior_arithmetic"])
            report["native_vbx_prior_calls"] = len(records["vbx_prior_updates"])
            report["vbx_prior_updates_observed"] = bool(records["vbx_prior_updates"])
            report["native_vbx_score_calls"] = len(records["vbx_score_arithmetic"])
            report["vbx_score_arithmetic_observed"] = bool(records["vbx_score_arithmetic"])
            report["native_vbx_statistics_calls"] = len(records["vbx_statistics"])
            report["vbx_statistics_observed"] = bool(records["vbx_statistics"])
            report["vbx_calls"] = vbx_calls
            report["native_vbx_precompute_calls"] = len(records["vbx_precomputed_terms"])
            report["vbx_precomputed_terms_observed"] = bool(records["vbx_precomputed_terms"])
            report["native_vbx_matrix_product_calls"] = len(records["vbx_matrix_products"])
            report["vbx_matrix_products_observed"] = bool(records["vbx_matrix_products"])
            report["native_plda_centering_scaling_calls"] = len(records["plda_centering_scaling"])
            report["plda_centering_scaling_observed"] = bool(records["plda_centering_scaling"])
            report["native_plda_matrix_product_calls"] = len(records["plda_matrix_products"])
            report["plda_matrix_products_observed"] = bool(records["plda_matrix_products"])
            report["native_plda_normalization_calls"] = len(records["plda_normalization"])
            report["plda_normalization_observed"] = bool(records["plda_normalization"])
            report["native_embedding_normalization_calls"] = len(records["embedding_normalization"])
            report["embedding_normalization_observed"] = bool(records["embedding_normalization"])
            report["native_speaker_assignment_calls"] = len(records["speaker_assignment"])
            report["logical_emission_host_readback_bytes"] = sum(item["logical_emission_host_readback_bytes"] for item in records["ctc"])
            report["logical_cpu_to_native_emission_bytes"] = sum(item["logical_cpu_to_native_emission_bytes"] for item in records["ctc"])
            json.dump(report, output, allow_nan=False, indent=2)
            output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

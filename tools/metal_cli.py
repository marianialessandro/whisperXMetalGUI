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
    parser.add_argument("--ahc-distance-cut", action="store_true", help="Cut bounded valid AHC trees on Metal with the original SciPy distance criterion and label traversal; requires PLDA library/hash and fcluster ABI. Includes zero-based label offset, while dendrogram construction remains original.")
    parser.add_argument("--ahc-label-reindex", action="store_true", help="Reindex bounded nonnegative AHC integer labels on Metal in ascending distinct-class order; requires PLDA library/hash and reindex ABI. Replaces only np.unique(..., return_inverse=True) after fcluster.")
    parser.add_argument("--ahc-euclidean-distances", action="store_true", help="Compute bounded Euclidean condensed distances on Metal before original SciPy centroid linkage; requires PLDA library/hash and pdist ABI. Keeps dendrogram construction and fcluster on CPU.")
    parser.add_argument("--vbx-resident-assignment", action="store_true", help="Keep inactive-speaker masking, global NaN cleanup and all constrained assignments in one Metal command buffer, returning original public masked scores and hard labels; requires PLDA library/hash and resident ABI. Select instead of standalone inactive/cleanup/assignment flags.")
    parser.add_argument("--vbx-batched-assignment", action="store_true", help="Batch original constrained NaN cleanup, rectangular assignments and int8 label scatter in one native Metal command buffer; requires PLDA library/hash and batch ABI. Select instead of standalone --vbx-constrained-nan-cleanup/--vbx-constrained-assignment.")
    parser.add_argument("--vbx-constrained-assignment", action="store_true", help="Maximize each bounded finite float64 constrained speaker score matrix on Metal using the pinned SciPy rectangular assignment policy; requires --vbx-constrained-nan-cleanup and assignment ABI. Keeps label scatter and branch decisions original.")
    parser.add_argument("--kmeans-query-scaling", action="store_true", help="Scale original float64 KMeans++ uniform draws by the original float32 potential on native Metal binary64 arithmetic; requires --kmeans-feature-mean and matrix ABI. Keeps the RNG call and draw sequence original.")
    parser.add_argument("--kmeans-best-candidate", action="store_true", help="Choose the dense float32 KMeans++ candidate with minimum original potential on Metal, preserving first ties; requires --kmeans-feature-mean and best-candidate ABI. Keeps weighted potential values and RNG original.")
    parser.add_argument("--kmeans-candidate-search", action="store_true", help="Search the dense float32 KMeans++ cumulative distribution using original float64 queries and clip candidate indices on Metal; requires --kmeans-feature-mean and search ABI. Keeps RNG and weighted potentials original.")
    parser.add_argument("--kmeans-seed-probabilities", action="store_true", help="Compute dense float32 KMeans++ first-seed normalized weights on Metal; requires --kmeans-feature-mean and seed-probability ABI. Keeps RandomState.choice and RNG state handling original.")
    parser.add_argument("--kmeans-candidate-minimum", action="store_true", help="Compute dense float32 KMeans++ in-place broadcast candidate minima on Metal; requires --kmeans-feature-mean and candidate-minimum ABI. Keeps RNG, distances, potentials and candidate decisions original.")
    parser.add_argument("--kmeans-weighted-prefix", action="store_true", help="Compute dense float32 KMeans++ weighted cumulative candidate distributions on Metal; requires --kmeans-feature-mean and weighted-prefix ABI. Keeps RNG, distances, potentials and candidate decisions original.")
    parser.add_argument("--kmeans-squared-norms", action="store_true", help="Compute dense float32 KMeans precomputed squared row norms on Metal using the pinned ARM NumPy einsum order; requires --kmeans-feature-mean and squared-norm ABI. Initialization internal norms and Lloyd/Elkan remain original.")
    parser.add_argument("--kmeans-tolerance", action="store_true", help="Compute the original dense float32 KMeans variance-based stopping tolerance on Metal; requires --kmeans-feature-mean and tolerance ABI. Keeps configured tol, zero-tolerance control and sparse behavior unchanged.")
    parser.add_argument("--kmeans-centering", action="store_true", help="Run dense KMeans feature/initializer centering, input restoration and final-center translation on Metal with in-place host copyback; requires --kmeans-feature-mean and float-shift ABI. Lloyd/Elkan remain original CPU code.")
    parser.add_argument("--kmeans-feature-mean", action="store_true", help="Compute the initial dense float32 KMeans feature mean on Metal; requires --plda-normalization-library with KMeans-centroid ABI. Preserves the original fit decorator, subtraction, initialization, Lloyd/Elkan and restoration.")
    parser.add_argument("--vbx-kmeans-centroids", action="store_true", help="Compute float32 centroids after forced KMeans on Metal; requires --plda-normalization-library and KMeans-centroid ABI. Original KMeans remains CPU; at least two features and nonempty clusters are required.")
    parser.add_argument("--clustering-cpu-audit", action="store_true", help="Observe original AHC, KMeans and constrained-solver API calls on the current thread without replacing them; requires --diarize. Adds profiling overhead; reports inclusive wall time, not exclusive CPU time or GPU placement.")
    parser.add_argument("--vbx-constrained-nan-cleanup", action="store_true", help="Replace NaNs with the global finite minimum on Metal before constrained assignment; requires --plda-normalization-library and cleanup ABI. The original solver and all-NaN warning/error behavior remain.")
    parser.add_argument("--vbx-inactive-speaker-mask", action="store_true", help="Compute the global score minimum and inactive-speaker mask on Metal; requires --plda-normalization-library and inactive-mask ABI. Binary segmentation only; constrained solver remains original.")
    parser.add_argument("--vbx-resident-initialization", action="store_true", help="Keep the initial one-hot matrix resident through softmax; requires --vbx-initial-one-hot and --vbx-initial-softmax with fused ABI. Negative smoothing keeps the original one-hot bypass.")
    parser.add_argument("--vbx-resident-argmax", action="store_true", help="Keep scores on Metal through unconstrained argmax in the same command buffer; requires --vbx-soft-scores and --vbx-unconstrained-argmax with fused ABI. Scores and compact labels are both read back; constrained branch remains original.")
    parser.add_argument("--vbx-unconstrained-argmax", action="store_true", help="Run the original unconstrained speaker argmax on Metal; requires --plda-normalization-library and argmax ABI. Keeps first-NaN/first-maximum behavior; constrained assignment remains original.")
    parser.add_argument("--vbx-soft-scores", action="store_true", help="Additionally compute 2 - cosine_distance on Metal in the final cosine kernel; requires --vbx-cosine-distances and cosine-score ABI. Only scores are read back; both original subtractions remain separately rounded.")
    parser.add_argument("--vbx-cosine-distances", action="store_true", help="Run embedding-to-centroid cosine distances on Metal; requires --plda-normalization-library and cosine ABI. Preserves undefined NaN pairs; only cosine metric is supported.")
    parser.add_argument("--vbx-centroids", action="store_true", help="Run weighted centroids after VBx on Metal; requires --plda-normalization-library and centroid ABI. Uses sequential binary64 products; BLAS rounding may differ. Optional K-means remains original.")
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
    parser.add_argument("--vbx-initial-one-hot", action="store_true", help="Additionally construct the cluster one-hot matrix on Metal; requires --vbx-matrix-products and one-hot ABI. Label maximum remains host shape metadata; composes with --vbx-initial-softmax.")
    parser.add_argument("--vbx-initial-softmax", action="store_true", help="Additionally run the one-hot cluster initialization softmax on Metal; requires --vbx-matrix-products and qinit ABI. Uses bounded OpenLibm-policy exp; may differ from SciPy rounding.")
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
    if args.kmeans_query_scaling and not args.kmeans_feature_mean:
        parser.error("--kmeans-query-scaling requires --kmeans-feature-mean.")
    if args.kmeans_best_candidate and not args.kmeans_feature_mean:
        parser.error("--kmeans-best-candidate requires --kmeans-feature-mean.")
    if args.kmeans_candidate_search and not args.kmeans_feature_mean:
        parser.error("--kmeans-candidate-search requires --kmeans-feature-mean.")
    if args.kmeans_seed_probabilities and not args.kmeans_feature_mean:
        parser.error("--kmeans-seed-probabilities requires --kmeans-feature-mean.")
    if args.kmeans_candidate_minimum and not args.kmeans_feature_mean:
        parser.error("--kmeans-candidate-minimum requires --kmeans-feature-mean.")
    if args.kmeans_weighted_prefix and not args.kmeans_feature_mean:
        parser.error("--kmeans-weighted-prefix requires --kmeans-feature-mean.")
    if args.kmeans_squared_norms and not args.kmeans_feature_mean:
        parser.error("--kmeans-squared-norms requires --kmeans-feature-mean.")
    if args.kmeans_tolerance and not args.kmeans_feature_mean:
        parser.error("--kmeans-tolerance requires --kmeans-feature-mean.")
    if args.kmeans_centering and not args.kmeans_feature_mean:
        parser.error("--kmeans-centering requires --kmeans-feature-mean.")
    if args.kmeans_feature_mean and not args.plda_normalization_library:
        parser.error("--kmeans-feature-mean requires --plda-normalization-library and its SHA256.")
    if args.vbx_kmeans_centroids and not args.plda_normalization_library:
        parser.error("--vbx-kmeans-centroids requires --plda-normalization-library and its SHA256.")
    if args.clustering_cpu_audit and "--diarize" not in forwarded:
        parser.error("--clustering-cpu-audit requires --diarize.")
    if args.ahc_distance_cut and not args.plda_normalization_library:
        parser.error("--ahc-distance-cut requires --plda-normalization-library and its SHA256.")
    if args.ahc_label_reindex and not args.plda_normalization_library:
        parser.error("--ahc-label-reindex requires --plda-normalization-library and its SHA256.")
    if args.ahc_euclidean_distances and not args.plda_normalization_library:
        parser.error("--ahc-euclidean-distances requires --plda-normalization-library and its SHA256.")
    if args.vbx_resident_assignment and (args.vbx_inactive_speaker_mask or args.vbx_constrained_nan_cleanup or args.vbx_constrained_assignment or args.vbx_batched_assignment):
        parser.error("--vbx-resident-assignment is mutually exclusive with standalone inactive/cleanup/assignment flags.")
    if args.vbx_resident_assignment and not args.plda_normalization_library:
        parser.error("--vbx-resident-assignment requires --plda-normalization-library and its SHA256.")
    if args.vbx_batched_assignment and (args.vbx_constrained_nan_cleanup or args.vbx_constrained_assignment):
        parser.error("--vbx-batched-assignment is mutually exclusive with standalone constrained cleanup/assignment flags.")
    if args.vbx_batched_assignment and not args.plda_normalization_library:
        parser.error("--vbx-batched-assignment requires --plda-normalization-library and its SHA256.")
    if args.vbx_constrained_assignment and not args.vbx_constrained_nan_cleanup:
        parser.error("--vbx-constrained-assignment requires --vbx-constrained-nan-cleanup.")
    if args.vbx_constrained_nan_cleanup and not args.plda_normalization_library:
        parser.error("--vbx-constrained-nan-cleanup requires --plda-normalization-library and its SHA256.")
    if args.vbx_inactive_speaker_mask and not args.plda_normalization_library:
        parser.error("--vbx-inactive-speaker-mask requires --plda-normalization-library and its SHA256.")
    if args.vbx_resident_initialization and not (args.vbx_initial_one_hot and args.vbx_initial_softmax):
        parser.error("--vbx-resident-initialization requires --vbx-initial-one-hot and --vbx-initial-softmax.")
    if args.vbx_resident_argmax and not (args.vbx_soft_scores and args.vbx_unconstrained_argmax):
        parser.error("--vbx-resident-argmax requires --vbx-soft-scores and --vbx-unconstrained-argmax.")
    if args.vbx_unconstrained_argmax and not args.plda_normalization_library:
        parser.error("--vbx-unconstrained-argmax requires --plda-normalization-library and its SHA256.")
    if args.vbx_soft_scores and not args.vbx_cosine_distances:
        parser.error("--vbx-soft-scores requires --vbx-cosine-distances.")
    if args.vbx_cosine_distances and not args.plda_normalization_library:
        parser.error("--vbx-cosine-distances requires --plda-normalization-library and its SHA256.")
    if args.vbx_centroids and not args.plda_normalization_library:
        parser.error("--vbx-centroids requires --plda-normalization-library and its SHA256.")
    if args.plda_matrix_products and not args.plda_normalization_library:
        parser.error("--plda-matrix-products requires --plda-normalization-library and its SHA256.")
    if args.plda_centering_scaling and not args.plda_matrix_products:
        parser.error("--plda-centering-scaling requires --plda-matrix-products.")
    if args.vbx_matrix_products and not args.plda_normalization_library:
        parser.error("--vbx-matrix-products requires --plda-normalization-library and its SHA256.")
    if args.vbx_initial_one_hot and not args.vbx_matrix_products:
        parser.error("--vbx-initial-one-hot requires --vbx-matrix-products.")
    if args.vbx_initial_softmax and not args.vbx_matrix_products:
        parser.error("--vbx-initial-softmax requires --vbx-matrix-products.")
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
    report = {"schema_version": 1, "scope": "source_checkout_native_mps_ctc", "status": "failed", "quality_certified": False, "full_pipeline_gpu_certified": False, "production_backend": False, "resident_statistics": args.resident_statistics, "embedding_normalization_requested": bool(args.embedding_library), "plda_normalization_requested": bool(args.plda_normalization_library), "plda_matrix_products_requested": args.plda_matrix_products, "plda_centering_scaling_requested": args.plda_centering_scaling, "vbx_constrained_nan_cleanup_requested": args.vbx_constrained_nan_cleanup, "vbx_inactive_speaker_mask_requested": args.vbx_inactive_speaker_mask, "vbx_resident_initialization_requested": args.vbx_resident_initialization, "vbx_resident_argmax_requested": args.vbx_resident_argmax, "vbx_unconstrained_argmax_requested": args.vbx_unconstrained_argmax, "vbx_soft_scores_requested": args.vbx_soft_scores, "vbx_cosine_distances_requested": args.vbx_cosine_distances, "vbx_centroids_requested": args.vbx_centroids, "vbx_matrix_products_requested": args.vbx_matrix_products, "vbx_initial_one_hot_requested": args.vbx_initial_one_hot, "vbx_initial_softmax_requested": args.vbx_initial_softmax, "vbx_initialization_requested": args.vbx_initialization, "vbx_scalar_arithmetic_requested": args.vbx_scalar_arithmetic, "vbx_logsumexp_requested": args.vbx_logsumexp, "vbx_transcendentals_requested": args.vbx_transcendentals, "vbx_elbo_arithmetic_requested": args.vbx_elbo_arithmetic, "vbx_posterior_arithmetic_requested": args.vbx_posterior_arithmetic, "vbx_prior_updates_requested": args.vbx_prior_updates, "vbx_score_arithmetic_requested": args.vbx_score_arithmetic, "vbx_statistics_requested": args.vbx_statistics, "vbx_precomputed_terms_requested": args.vbx_precomputed_terms}
    records = {key: [] for key in ("ctc", "bounds", "means", "rounding", "interpolation", "speaker_assignment", "embedding_normalization", "plda_normalization", "plda_matrix_products", "plda_centering_scaling", "vbx_matrix_products", "vbx_precomputed_terms", "vbx_statistics", "vbx_score_arithmetic", "vbx_prior_updates", "vbx_posterior_arithmetic", "vbx_elbo_arithmetic", "vbx_transcendentals", "vbx_logsumexp", "vbx_scalar_arithmetic", "vbx_initialization", "vbx_initial_softmax", "vbx_centroids", "vbx_initial_one_hot", "vbx_cosine_distances", "vbx_unconstrained_argmax", "vbx_inactive_speaker_mask", "vbx_constrained_nan_cleanup", "vbx_kmeans_centroids", "kmeans_feature_mean", "kmeans_centering", "kmeans_tolerance", "kmeans_squared_norms", "kmeans_weighted_prefix", "kmeans_candidate_minimum", "kmeans_seed_probabilities", "kmeans_candidate_search", "kmeans_best_candidate", "kmeans_query_scaling", "vbx_constrained_assignment", "vbx_batched_assignment", "vbx_resident_assignment", "ahc_euclidean_distances", "ahc_label_reindex", "ahc_distance_cut")}
    report["ahc_distance_cut_requested"] = args.ahc_distance_cut
    report["ahc_label_reindex_requested"] = args.ahc_label_reindex
    report["ahc_euclidean_distances_requested"] = args.ahc_euclidean_distances
    report["vbx_resident_assignment_requested"] = args.vbx_resident_assignment
    report["vbx_batched_assignment_requested"] = args.vbx_batched_assignment
    report["vbx_constrained_assignment_requested"] = args.vbx_constrained_assignment
    report["kmeans_query_scaling_requested"] = args.kmeans_query_scaling
    report["kmeans_best_candidate_requested"] = args.kmeans_best_candidate
    report["kmeans_candidate_search_requested"] = args.kmeans_candidate_search
    report["kmeans_seed_probabilities_requested"] = args.kmeans_seed_probabilities
    report["kmeans_candidate_minimum_requested"] = args.kmeans_candidate_minimum
    report["kmeans_weighted_prefix_requested"] = args.kmeans_weighted_prefix
    report["kmeans_squared_norms_requested"] = args.kmeans_squared_norms
    report["kmeans_tolerance_requested"] = args.kmeans_tolerance
    report["kmeans_centering_requested"] = args.kmeans_centering
    report["kmeans_feature_mean_requested"] = args.kmeans_feature_mean
    report["vbx_kmeans_centroids_requested"] = args.vbx_kmeans_centroids
    report["clustering_cpu_audit_requested"] = args.clustering_cpu_audit
    cpu_audit_records = []
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
                        if args.clustering_cpu_audit:
                            from tools.baseline.clustering_cpu_audit import clustering_cpu_audit
                            scopes.enter_context(clustering_cpu_audit(cpu_audit_records, include_pairwise=args.ahc_euclidean_distances))
                        if args.plda_normalization_library:
                            from tools.baseline.native_double import NativeDoubleNormalization
                            from tools.baseline.native_plda_adapter import native_plda_normalization
                            plda_context = scopes.enter_context(NativeDoubleNormalization(args.plda_normalization_library, args.plda_normalization_library_sha256))
                            report["plda_normalization_native_identity"] = plda_context.identity
                            report["plda_normalization_initialization_wall_ns"] = plda_context.initialization_wall_ns
                            scopes.enter_context(native_plda_normalization(plda_context, records["plda_normalization"]))
                            if args.vbx_batched_assignment:
                                from tools.baseline.native_assignment import native_batched_assignment
                                scopes.enter_context(native_batched_assignment(plda_context, records["vbx_batched_assignment"]))
                            elif args.vbx_constrained_nan_cleanup:
                                from tools.baseline.native_constrained_cleanup import native_constrained_nan_cleanup
                                scopes.enter_context(native_constrained_nan_cleanup(plda_context, records["vbx_constrained_nan_cleanup"], assignment_records=records["vbx_constrained_assignment"] if args.vbx_constrained_assignment else None))
                            if args.plda_matrix_products:
                                from tools.baseline.native_plda_matrix import native_plda_matrix_products
                                scopes.enter_context(native_plda_matrix_products(plda_context, records["plda_matrix_products"], centering_scaling=args.plda_centering_scaling, elementwise_records=records["plda_centering_scaling"]))
                            if args.vbx_initial_softmax or args.vbx_initial_one_hot:
                                from tools.baseline.native_vbx_qinit import native_vbx_qinit
                                scopes.enter_context(native_vbx_qinit(plda_context, records["vbx_initial_softmax"], one_hot=args.vbx_initial_one_hot, one_hot_records=records["vbx_initial_one_hot"], softmax=args.vbx_initial_softmax, resident_initialization=args.vbx_resident_initialization))
                            if args.vbx_matrix_products:
                                from tools.baseline.native_vbx_matrix import native_vbx_matrix_products
                                scopes.enter_context(native_vbx_matrix_products(plda_context, records["vbx_matrix_products"], vbx_calls, precompute=args.vbx_precomputed_terms, precompute_records=records["vbx_precomputed_terms"], statistics=args.vbx_statistics, statistics_records=records["vbx_statistics"], scores=args.vbx_score_arithmetic, score_records=records["vbx_score_arithmetic"], prior=args.vbx_prior_updates, prior_records=records["vbx_prior_updates"], posterior=args.vbx_posterior_arithmetic, posterior_records=records["vbx_posterior_arithmetic"], elbo=args.vbx_elbo_arithmetic, elbo_records=records["vbx_elbo_arithmetic"], transcendentals=args.vbx_transcendentals, transcendental_records=records["vbx_transcendentals"], logsumexp=args.vbx_logsumexp, logsumexp_records=records["vbx_logsumexp"], scalars=args.vbx_scalar_arithmetic, scalar_records=records["vbx_scalar_arithmetic"], initialization=args.vbx_initialization, initialization_records=records["vbx_initialization"]))
                        if args.kmeans_feature_mean:
                            from tools.baseline.native_kmeans_mean import native_kmeans_feature_mean
                            scopes.enter_context(native_kmeans_feature_mean(plda_context, records["kmeans_feature_mean"], shift_records=records["kmeans_centering"] if args.kmeans_centering else None, tolerance_records=records["kmeans_tolerance"] if args.kmeans_tolerance else None, norm_records=records["kmeans_squared_norms"] if args.kmeans_squared_norms else None, prefix_records=records["kmeans_weighted_prefix"] if args.kmeans_weighted_prefix else None, minimum_records=records["kmeans_candidate_minimum"] if args.kmeans_candidate_minimum else None, probability_records=records["kmeans_seed_probabilities"] if args.kmeans_seed_probabilities else None, search_records=records["kmeans_candidate_search"] if args.kmeans_candidate_search else None, best_records=records["kmeans_best_candidate"] if args.kmeans_best_candidate else None, query_records=records["kmeans_query_scaling"] if args.kmeans_query_scaling else None))
                        embedding_context = None
                        if args.embedding_library:
                            from tools.baseline.native_embedding import NativeEmbeddingNormalization
                            embedding_context = scopes.enter_context(NativeEmbeddingNormalization(args.embedding_library, args.embedding_library_sha256))
                            report["embedding_native_identity"] = embedding_context.identity
                            report["embedding_initialization_wall_ns"] = embedding_context.initialization_wall_ns
                        if args.embedding_library or args.vbx_centroids or args.vbx_cosine_distances or args.vbx_unconstrained_argmax or args.vbx_inactive_speaker_mask or args.vbx_kmeans_centroids or args.vbx_resident_assignment or args.ahc_euclidean_distances or args.ahc_label_reindex or args.ahc_distance_cut:
                            from tools.baseline.native_embedding_adapter import native_embedding_normalization
                            scopes.enter_context(native_embedding_normalization(embedding_context, records["embedding_normalization"], centroid_context=plda_context if args.vbx_centroids else None, centroid_records=records["vbx_centroids"], distance_context=plda_context if args.vbx_cosine_distances else None, distance_records=records["vbx_cosine_distances"], soft_scores=args.vbx_soft_scores, argmax_context=plda_context if args.vbx_unconstrained_argmax else None, argmax_records=records["vbx_unconstrained_argmax"], resident_argmax=args.vbx_resident_argmax, inactive_context=plda_context if args.vbx_inactive_speaker_mask else None, inactive_records=records["vbx_inactive_speaker_mask"], kmeans_context=plda_context if args.vbx_kmeans_centroids else None, kmeans_records=records["vbx_kmeans_centroids"], resident_assignment_context=plda_context if args.vbx_resident_assignment else None, resident_assignment_records=records["vbx_resident_assignment"], ahc_context=plda_context if args.ahc_euclidean_distances else None, ahc_records=records["ahc_euclidean_distances"], reindex_context=plda_context if args.ahc_label_reindex else None, reindex_records=records["ahc_label_reindex"], cut_context=plda_context if args.ahc_distance_cut else None, cut_records=records["ahc_distance_cut"]))
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
            report["clustering_cpu_api_observations"] = cpu_audit_records
            report["clustering_cpu_api_calls"] = sum(item["calls_started"] for item in cpu_audit_records)
            report["clustering_cpu_audit_observed"] = bool(report["clustering_cpu_api_calls"])
            report["clustering_cpu_audit_complete"] = len(cpu_audit_records) == 4 + int(args.ahc_euclidean_distances) and all(item["observation_complete"] for item in cpu_audit_records)
            report["native_kmeans_query_scaling_calls"] = len(records["kmeans_query_scaling"])
            report["kmeans_query_scaling_observed"] = bool(records["kmeans_query_scaling"])
            report["native_kmeans_best_candidate_calls"] = len(records["kmeans_best_candidate"])
            report["kmeans_best_candidate_observed"] = bool(records["kmeans_best_candidate"])
            report["native_kmeans_candidate_search_calls"] = len(records["kmeans_candidate_search"])
            report["kmeans_candidate_search_observed"] = bool(records["kmeans_candidate_search"])
            report["native_kmeans_seed_probabilities_calls"] = len(records["kmeans_seed_probabilities"])
            report["kmeans_seed_probabilities_observed"] = bool(records["kmeans_seed_probabilities"])
            report["native_kmeans_candidate_minimum_calls"] = len(records["kmeans_candidate_minimum"])
            report["kmeans_candidate_minimum_observed"] = bool(records["kmeans_candidate_minimum"])
            report["native_kmeans_weighted_prefix_calls"] = len(records["kmeans_weighted_prefix"])
            report["kmeans_weighted_prefix_observed"] = bool(records["kmeans_weighted_prefix"])
            report["native_kmeans_squared_norms_calls"] = len(records["kmeans_squared_norms"])
            report["kmeans_squared_norms_observed"] = bool(records["kmeans_squared_norms"])
            report["native_kmeans_tolerance_calls"] = len(records["kmeans_tolerance"])
            report["kmeans_tolerance_observed"] = bool(records["kmeans_tolerance"])
            report["native_kmeans_centering_calls"] = len(records["kmeans_centering"])
            report["kmeans_centering_observed"] = bool(records["kmeans_centering"])
            report["native_kmeans_feature_mean_calls"] = len(records["kmeans_feature_mean"])
            report["kmeans_feature_mean_observed"] = bool(records["kmeans_feature_mean"])
            report["native_vbx_kmeans_centroid_calls"] = len(records["vbx_kmeans_centroids"])
            report["vbx_kmeans_centroids_observed"] = bool(records["vbx_kmeans_centroids"])
            report["operations"] = records
            report["native_ctc_calls"] = len(records["ctc"])
            report["native_ahc_distance_cut_calls"] = len(records["ahc_distance_cut"])
            report["ahc_distance_cut_observed"] = bool(records["ahc_distance_cut"])
            report["native_ahc_label_reindex_calls"] = len(records["ahc_label_reindex"])
            report["ahc_label_reindex_observed"] = bool(records["ahc_label_reindex"])
            report["native_ahc_euclidean_distance_calls"] = len(records["ahc_euclidean_distances"])
            report["ahc_euclidean_distances_observed"] = bool(records["ahc_euclidean_distances"])
            report["native_vbx_resident_assignment_calls"] = len(records["vbx_resident_assignment"])
            report["vbx_resident_assignment_observed"] = bool(records["vbx_resident_assignment"])
            report["vbx_resident_assignment_matrices"] = sum(row["logical_assignment_matrices"] for row in records["vbx_resident_assignment"])
            report["native_vbx_batched_assignment_calls"] = len(records["vbx_batched_assignment"])
            report["vbx_batched_assignment_observed"] = bool(records["vbx_batched_assignment"])
            report["vbx_batched_assignment_matrices"] = sum(row["logical_assignment_matrices"] for row in records["vbx_batched_assignment"])
            report["native_vbx_constrained_assignment_calls"] = len(records["vbx_constrained_assignment"])
            report["vbx_constrained_assignment_observed"] = bool(records["vbx_constrained_assignment"])
            report["native_vbx_constrained_cleanup_calls"] = len(records["vbx_constrained_nan_cleanup"])
            report["vbx_constrained_nan_cleanup_observed"] = bool(records["vbx_constrained_nan_cleanup"])
            report["native_vbx_inactive_mask_calls"] = len(records["vbx_inactive_speaker_mask"])
            report["vbx_inactive_speaker_mask_observed"] = bool(records["vbx_inactive_speaker_mask"])
            report["native_vbx_resident_initialization_calls"] = sum(item.get("resident_initialization", False) for item in records["vbx_initial_softmax"])
            report["vbx_resident_initialization_observed"] = bool(report["native_vbx_resident_initialization_calls"])
            report["native_vbx_initial_one_hot_separate_dispatches"] = sum(item["command_buffers_per_run"] for item in records["vbx_initial_one_hot"])
            report["native_vbx_resident_argmax_calls"] = sum(item.get("resident_argmax", False) for item in records["vbx_cosine_distances"])
            report["vbx_resident_argmax_observed"] = bool(report["native_vbx_resident_argmax_calls"])
            report["native_vbx_argmax_separate_dispatches"] = sum(item["command_buffers_per_run"] for item in records["vbx_unconstrained_argmax"])
            report["native_vbx_argmax_calls"] = len(records["vbx_unconstrained_argmax"])
            report["vbx_unconstrained_argmax_observed"] = bool(records["vbx_unconstrained_argmax"])
            report["native_vbx_soft_score_calls"] = sum(item.get("result_kind") == "soft_scores" for item in records["vbx_cosine_distances"])
            report["vbx_soft_scores_observed"] = bool(report["native_vbx_soft_score_calls"])
            report["native_vbx_cosine_calls"] = len(records["vbx_cosine_distances"])
            report["vbx_cosine_distances_observed"] = bool(records["vbx_cosine_distances"])
            report["native_vbx_initial_one_hot_calls"] = len(records["vbx_initial_one_hot"])
            report["vbx_initial_one_hot_observed"] = bool(records["vbx_initial_one_hot"])
            report["native_vbx_centroid_calls"] = len(records["vbx_centroids"])
            report["vbx_centroids_observed"] = bool(records["vbx_centroids"])
            report["native_vbx_initial_softmax_calls"] = len(records["vbx_initial_softmax"])
            report["vbx_initial_softmax_observed"] = bool(records["vbx_initial_softmax"])
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

# CLI development with native Metal CTC

Use the checkout's independent `.venv`, CPython 3.12.12 and Torch 2.8.0. No app bundle, packaged runtime or GUI is involved. The original `python -m whisperx` CLI remains available; this separate experiment uses its actual ASR, alignment, diarization and exporters with scoped native CTC hooks.

Build the pinned Torch interop library once:

```bash
mkdir -p .cache/native
.venv/bin/python tools/baseline/mps_storage_probe.py build --out .cache/native/mps-ctc.dylib
```

The command prints the library SHA256. Supply that exact value to a new run:

```bash
.venv/bin/python tools/metal_cli.py --mps-library .cache/native/mps-ctc.dylib --mps-library-sha256 SHA256 --native-report outputs/native-report.json --resident-statistics recording.wav --asr_backend mlx --model small --device mps --language it --output_dir outputs/recording
```

Create `outputs/` first. The report path must be new; it is never overwritten. Other pipeline flags, including `--diarize`, are forwarded to the existing CLI. `--help` describes the experimental options; `python -m whisperx --help` lists the unchanged pipeline options. Use the cache and FFmpeg environment described in the root README for offline runs. Models and native libraries stay local.

This source-only path supports IT/EN alignment on MPS and fails if the native handoff is unavailable. It runs native Metal trellis/backtracking with fused wildcard preparation directly from MPS emission storage. Optional `--resident-statistics` adds resident character statistics, timestamp/word reductions, exact three-decimal rounding and paired nearest interpolation; it requires the `nearest` interpolation method. Original alignability decisions and export formats remain in the existing code. Optional `--seed N` records the diagnostic RNG seed and does not change decoder policy.

The private report records actual native calls, completion timings, shader/library hashes and logical boundary bytes. Timestamp means can share a completion group with bounds; their GPU durations must not be added twice. Zero emission host readback refers to this native CTC handoff only, not all pipeline traffic. The process still uses Python orchestration, CPU metadata/grouping, codec and export work, and CPU portions of community-1. This experiment does not certify full GPU placement, end-to-end quality, speedup or a production backend. A failed run retains a report marked `failed`, with no token, CLI argument list, audio or transcript values.

## Optional native speaker assignment

Build the additional independent C ABI library:

```bash
.venv/bin/python tools/baseline/native_speaker.py --build .cache/native/speaker.dylib
```

To the `metal_cli.py` command add `--diarize --speaker-library .cache/native/speaker.dylib --speaker-library-sha256 SPEAKER_SHA256`, using the hash printed by that build. The existing speaker-assignment implementation remains the default when these flags are absent.

This path performs stable turn ordering, timestamp intersections, per-speaker overlap accumulation, winner selection and optional nearest-turn selection on Metal. Binary64 arithmetic is emulated with integer operations using the existing CTC helpers; it preserves stable tie order and sequential sums without converting timestamps or totals to float32. Only integer speaker choices return to Python. String labels, transcript dictionaries and optional embedding references are retained by the host adapter. Zero-duration intersections are excluded, words without a start remain untouched, and no-match words keep their previous label unless nearest assignment is requested.

The experimental domain is at most 4096 turns and 256 speakers per call, with batches of at most 4096 queries and one million queries per transcript. Timestamps must be finite, nonnegative, ordered within each interval, and at most 2^32 seconds; negative zero and nonzero values smaller than 2^-900 are rejected. Query timestamps support Python binary64-compatible scalars, including the existing native alignment scalar subclasses; float32 query scalars are rejected because their original midpoint policy can differ. No unsupported case falls back to CPU assignment. Stable ranking currently uses quadratic work in the number of turns, bounded by the explicit turn limit; this is a correctness implementation with no speedup claim.

`operations.speaker_assignment` in the private run report records both kernels, actual command-buffer GPU duration, retained buffer capacity, logical uploads and compact label readback bytes. Segmentation, embeddings, clustering and other community-1 operations keep their existing backends; this flag does not move the entire diarization pipeline to GPU.

## Native embedding-normalization primitive

The separate `tools/baseline/native_embedding.py` experiment builds a C ABI library for float32 row normalization, the operation preceding community-1 centroid linkage:

```bash
.venv/bin/python tools/baseline/native_embedding.py --build .cache/native/embedding.dylib
```

Use `NativeEmbeddingNormalization(library_path, library_sha256)` as a context manager and call `normalize(contiguous_float32_matrix)`; it returns a new array and a metadata-only dispatch report. The matrix remains unchanged, including readonly arrays. The primitive can also be connected to the experimental CLI with the optional adapter described below.

The kernel computes squared components, the contiguous NumPy pairwise reduction order, square root and division in one Metal command buffer. Fast math is disabled. The experimental domain is 1–4096 rows and 1–1024 components; finite nonzero components must have magnitude between 2^-32 and 2^32. Zero components and signed zero are supported, but an entirely zero row is rejected because the current primitive has no NaN-output contract. Unsupported dtypes, strided arrays and values outside this domain fail explicitly without CPU normalization. Input validation runs on the host; normalization arithmetic runs on Metal.

Input and normalized output cross the host boundary as float32 arrays. Two retained shared buffers use at most 32 MiB total; initialization, wall time, actual GPU completion time, logical upload/readback sizes and retained capacity are reported separately. Logical bytes are not physical-copy measurements. Tests compare float32 bits with NumPy over reduction boundaries, realistic dimensions, signed zero, extreme supported magnitudes and repeated/threaded calls. The C ABI is also exercised under AddressSanitizer and UndefinedBehaviorSanitizer. Identical results on those cases establish bounded parity evidence; complete numerical-domain certification and end-to-end quality/performance remain pending.


## Optional VBx training-embedding normalization

Add `--embedding-library .cache/native/embedding.dylib --embedding-library-sha256 EMBEDDING_SHA256` to a `metal_cli.py` run with `--diarize`. These flags are independent of the optional speaker-assignment library and require pyannote-audio 4.0.4. The library hash comes from the build above. The existing CLI and its default clustering implementation are preserved.

The scoped adapter checks the hash of the original `VBxClustering.__call__` source and its normalization expression before replacing exactly that assignment in a private compiled function. It retains the original signature, filtering, linkage, PLDA, VBx, centroid formation, optional K-means, distances and constrained assignment. NumPy itself and the installed dependency files are unchanged. A version/source mismatch or unsupported native input raises an error, with no CPU normalization fallback for the adapted call. The original method is restored on success or error; nested adapter scopes are rejected. Calls from unrelated threads use the original method, since the experiment applies only to the owning CLI thread.

The original fewer-than-two-training-embeddings branch does not perform normalization and is preserved. Reports distinguish `embedding_normalization_requested`, `embedding_normalization_observed` and `native_embedding_normalization_calls`; requested flags alone are not dispatch evidence. Each entry in `operations.embedding_normalization` records actual Metal completion, logical boundary sizes and the pinned original source hash. This moves only the pre-linkage row normalization onto Metal; PLDA/VBx, linkage, centroid and distance computation retain their existing backends, and normalized embeddings still return to the CPU.

## Optional native binary64 normalization in PLDA

Build the independent binary64-normalization library:

```bash
.venv/bin/python tools/baseline/native_double.py --build .cache/native/double.dylib
```

Add `--plda-normalization-library .cache/native/double.dylib --plda-normalization-library-sha256 DOUBLE_SHA256` to a `metal_cli.py` run with `--diarize`. These flags are independent of the embedding-normalization and speaker-assignment libraries. They require pyannote-audio 4.0.4 and NumPy 2.2.6, plus verified hashes for the original `l2_norm` and `vbx_setup` functions. The hook is scoped to the owning CLI thread, rejects nested activation and restores the original function on success or error. Unsupported input raises an error without CPU normalization fallback in the adapted call. Original positional and keyword calls are preserved.

PLDA preprocessing performs two float64 L2 normalizations. This kernel represents binary64 values as integers and performs rounded multiplication, addition, square root and division on Metal, without converting the values to float32. It preserves NumPy's pairwise sum for row-contiguous input and its sequential sum for column-contiguous input. The latter layout requires an explicit host staging copy before upload and a host layout copy after readback; the adapter preserves the original result layout so later matrix products retain their original inputs and BLAS layout. Other strided layouts and one-dimensional inputs are outside the adapter's explicit domain.

The bounded primitive supports 1–4096 rows and 1–1024 components. Components must be finite and either zero or have magnitude between 2^-200 and 2^200. Signed zero is preserved, while entirely zero rows are rejected. Two retained shared buffers use at most 64 MiB. `NativeDoubleNormalization.normalize` accepts a C-contiguous float64 matrix and an explicit `pairwise` or `sequential` reduction order; the PLDA adapter chooses the order from the original array layout.

Reports distinguish requested flags from observed native calls, with per-operation arithmetic policy, source hashes, layout staging bytes, logical upload/readback sizes and actual GPU completion duration. Both normalizations still return their full outputs to the CPU. Centering, scaling, LDA/PLDA matrix products, model setup inverse/eigendecomposition and the VBx iteration remain on their original backends. This is a port of two numerical operations inside PLDA, not the complete PLDA/VBx engine or a performance/quality certification. Tests compare bits against NumPy over wide exponents and reduction boundaries, preserve both layouts and exercise the C ABI under ASan/UBSan.

## Optional native LDA/PLDA matrix products

Build a new version of the binary64 library at a new path with `native_double.py --build`, then add `--plda-matrix-products` alongside `--plda-normalization-library` and its library hash. A library without the additive matrix-product ABI is rejected before running the pipeline. The legacy normalization ABI and its arguments remain available; normalization contexts also accept the earlier shader without a matrix kernel.

The native product multiplies and adds binary64 values through integer emulation, in increasing inner-index order, with a separate nearest-even rounding at each multiplication and addition. BLAS can use different accumulation orders and fused instructions. Therefore this flag explicitly selects an experimental numerical policy, and `reference_blas_bitwise_certified` remains false. Primitive tests compare bits to an independent sequential NumPy multiply/add reference; the observed agreement does not imply bitwise equality to BLAS. Tests of complete PLDA transformations use a provisional numerical comparison guard, which is not a frozen quality tolerance.

The scoped adapter validates the original PLDA call and setup source hashes, the identity of the two original transform closure code objects, parameter names/dtypes/shapes and output dimension. It extracts the actual prepared LDA/PLDA matrices from those closures, preserves original centering/scaling/normalization order and truncation, and takes a private parameter snapshot for each invocation, hashing the same arrays used in both products. Parameters mutated between calls produce new recorded hashes. Both matrix products execute on Metal; centering/scaling and model preparation inverse/eigendecomposition remain on their original backends. The result arrays and normalization intermediates still return to the host.

The bounded matrix primitive accepts C-contiguous float64 operands, 1–4096 output rows, inner/output dimensions 1–1024 and at most 32 million multiply/add pairs per call. Finite nonzero input magnitudes must be between 2^-100 and 2^100; zeros, signed zeros and cancellation are supported. In the two-product PLDA adapter, at most 1024 training embeddings are supported because they become the first product's output columns; the same work limit also applies. No unsupported case falls back to CPU multiplication. Three retained buffers can total at most 72 MiB across normalization and matrix calls. Reports count actual product dispatches, source and parameter hashes, explicit layout/dtype staging copies, parameter-copy and parameter-hash snapshot bytes, logical device-boundary bytes and GPU completion time. They distinguish `plda_matrix_products_requested`, `plda_matrix_products_observed` and `native_plda_matrix_product_calls`.

The first bounded real-checkpoint experiment on the acquired 42x256 embeddings measured a maximum final-PLDA difference of about 2.93e-14 from the original BLAS path over six repetitions. This is a numerical diagnostic on those same inputs, not complete model coverage, quality certification or a speedup result.

## Optional native PLDA centering and scaling

Rebuild the binary64 library at a new path, then add `--plda-centering-scaling` together with `--plda-matrix-products` and the library/hash options. The new flag requires the additive `wxdouble_elementwise` ABI; missing capability fails before inference. Existing flags, normalization/matrix ABIs and older shaders remain usable without this option.

This moves all three per-column centering steps and both dimension-scaling steps in the PLDA transform onto Metal. Integer binary64 arithmetic preserves a separately rounded subtraction or multiplication for each value. The scalar square roots of the integer embedding dimensions are also computed on Metal. The adapter preserves the original C/F array layouts so the following L2 normalization retains the same NumPy reduction order. Host staging still widens float32 embeddings/parameters exactly to float64, packs arrays and restores column-contiguous outputs; these copies/conversions are reported explicitly.

The standalone methods `center(values, mean)` and `scale_dimension(values, dimension)` accept C-contiguous float64 arrays. Matrices have 1–4096 rows and 1–1024 columns; the mean is a matching one-dimensional vector and the dimension is an integer from 1 to 1024. Inputs must be finite and either zero or have magnitude between 2^-450 and 2^450. Signed zeros and cancellation are preserved. The existing PLDA adapter embedding/work limits also apply. Shared buffers and the 72 MiB retained-capacity bound are reused, with one command buffer per operation.

Reports distinguish the requested flag from observed calls in `operations.plda_centering_scaling`. A complete transform produces five entries: `center_input`, `scale_before_lda`, `center_after_lda`, `scale_before_plda` and `center_before_plda`. Each entry records actual GPU completion, parameter/source hashes, layout/dtype staging and logical device-boundary bytes. Together with the other options, the transform uses two native normalizations, two matrix products and these five operations. Model preparation inverse/eigendecomposition, host staging and Python orchestration remain; intermediate results are still uploaded/read back between steps, so this is not a resident or fused pipeline. Adapter-local remainder lists describe each isolated adapter; use observed operation records when assessing the combined configuration.

Centering/scaling tests compare binary64 bits against NumPy, including broad exponents, cancellation, signed zero and non-square dimensions. Complete-transform tests compare against the previous sequential matrix policy and reject CPU square-root execution. The BLAS difference from the earlier experimental matrix policy remains explicitly separate from this bitwise centering/scaling check; no final quality/performance certification follows from either comparison.

## Optional native products inside VBx iterations

Add `--vbx-matrix-products` with the binary64 library/hash options and `--diarize`. It requires `--plda-normalization-library` with the matrix-product ABI, and is independent of the optional PLDA matrix/centering flags. No new native build is needed if the supplied library already provides that ABI.

The adapter checks pyannote-audio 4.0.4, NumPy 2.2.6 and the exact original `VBx` source hash. A private compiled copy replaces exactly three expression nodes: `gamma.T.dot(rho)` for speaker statistics, `rho.dot(alpha.T)` for frame scores, and `(invL + alpha**2).dot(Phi)` for the speaker penalty. Matrix/vector shape, original signature/defaults, initialization, posterior/prior updates, warnings, ELBO, stopping and return structures are retained. NumPy and the installed files are unchanged. The hook applies to the owning CLI thread and is restored on error or success; nested adapter scopes are rejected.

The three products use the existing sequential, separately rounded binary64 multiply/add kernel. This remains a different accumulation policy from BLAS, and does not certify bitwise BLAS equivalence. Matrix/vector staging is recorded explicitly and each result returns to the CPU. Initialization and precomputed terms, reductions, elementwise statistics/score arithmetic, log/exp/logsumexp, posterior/prior updates and ELBO/stopping still use the original implementations. This is a port of the repeated products, not a complete GPU VBx engine or a speedup result.

Operands must be plain float64 NumPy arrays. The existing native matrix bounds apply: inner/output dimensions at most 1024, at most 32 million multiply/add pairs per product, finite nonzero magnitudes in 2^-100..2^100, and zeros allowed. In VBx, the statistics product makes the number of feature frames an inner dimension, so at most 1024 frames are supported by this experiment. Tiny nonzero probabilities outside the domain fail explicitly; they are neither clipped nor silently sent to CPU multiplication. The original CLI remains unchanged without the flag.

`operations.vbx_matrix_products` identifies the operation, VBx call and zero-based iteration, source hash, actual GPU duration and logical staging/boundary bytes. Top-level `vbx_calls` records completion or failure, completed iteration count and native product count. An ordinary iteration executes three products; an initial iteration with supplied `alpha` and `invL` skips the statistics product, as before. Zero iterations produces zero native products. Requested and observed reporting are separate, and `full_vbx_gpu_certified` remains false.

Tests compare the complete iteration result against an independent sequential multiply/add reference, check the original BLAS result and iteration count with a provisional numerical guard, exercise supplied/random model initialization and zero iterations, and verify failure restoration, source/AST/runtime drift, per-call metadata and unrelated threads. Numerical guards in these tests are development checks, not frozen quality tolerances.

## Optional native VBx precomputed terms

Rebuild the binary64 library at a new path, then add `--vbx-precomputed-terms` with `--vbx-matrix-products` and the library/hash flags. The new option requires the additive `wxdouble_vbx_precompute` ABI. Older flags, APIs and shaders retain their existing behavior when this option is absent.

A single private tuple assignment replaces the three original assignments for `G`, `V` and `rho`, after verifying each original AST expression and the source pin. Two kernels share one command-buffer completion: the first computes row sums of squared features and `G`, plus `sqrt(Phi)`; the second consumes those square roots to compute `rho`. Binary64 integer arithmetic preserves the original separately rounded operations. The constant for NumPy 2.2.6 `log(2*pi)` is pinned as `0x3ffd67f1c864beb4`, and its multiplication by the dimension is performed on Metal. No float32 log approximation is introduced.

The primitive `precompute_vbx(features, phi, reduction_order)` returns `(G, V, rho)` and dispatch metadata. It accepts C-contiguous float64 features with 1–4096 rows and 1–1024 columns, and a matching covariance vector. Features must be finite, either zero or have magnitude in 2^-200..2^200; covariance must be finite/nonnegative and either zero or in that range. Signed covariance zero is preserved through the square root and product. The adapter also accepts Fortran-contiguous features, explicitly stages them and restores the original `rho` layout. It selects pairwise or sequential reduction according to the original layout. Other feature layouts are rejected without CPU fallback.

The existing three buffers are reused and a dedicated `(rows + columns)` term buffer adds at most 40 KiB; total retained capacity is bounded by the earlier 72 MiB plus this optional 40 KiB. All operations report the complete retained capacity. Inputs and all three outputs still cross the host boundary, so this does not establish full residency or a speedup. The model/iteration limits from the other adapters also apply.

`operations.vbx_precomputed_terms` records actual kernels, completion, layout staging, pinned constant and logical bytes. Requested/observed fields and `native_vbx_precompute_calls` are separate. Call summaries include the precompute count; even `maxIters=0` still computes the original pre-loop terms once. Initialization, other statistics reductions/elementwise updates, log/exp/logsumexp, posterior/prior updates and ELBO/stopping remain on their original backends. Tests compare the precomputed terms and the complete iteration output bitwise to the preceding native-product policy, including both layouts, invalid covariance, signed zeros, old shaders and error restoration.

## Optional native VBx statistical updates

Rebuild the binary64 library at a new path and add `--vbx-statistics` with `--vbx-matrix-products` and the library/hash flags. This requires the additive `wxdouble_vbx_statistics` ABI; `--vbx-precomputed-terms` can be enabled independently. Existing flags and ABI exports retain their behavior when the new option is absent.

The source-pinned adapter replaces the exact `invL` and `alpha` assignments in the original iteration. Metal reduces `gamma` over frames, computes `Fa / Fb`, forms the inverse precision `1 / (1 + (Fa / Fb * sum(gamma)) * Phi)`, and scales the existing native statistics product as `(Fa / Fb * invL) * statistics`. Every division, multiplication and addition rounds separately in binary64 integer emulation. The three existing product records per ordinary iteration are retained. C-layout gamma uses the original sequential axis-zero sum; F-layout gamma and a single contiguous speaker column use the NumPy pairwise order. Gamma is staged as contiguous speaker rows before upload. The reduction and inverse precision use two kernels in one command buffer; alpha uses one kernel in another buffer submission after the statistics product returns.

The standalone primitive supports 1..1024 speakers, 1..4096 frames and 1..1024 features; the existing matrix-product inner-dimension bound still limits the integrated experiment to 1024 frames. Gamma must be finite in [0, 1], with nonzero values at least 2^-160. Input inverse precision must be strictly positive in [2^-160, 1]. Phi must be finite and nonnegative, zero or in 2^-100..2^100. Signed statistics are zero or have magnitude in 2^-100..2^100. Fa and Fb must be positive finite Python scalars in 2^-20..2^20. Unsupported inputs fail explicitly without clipping or CPU fallback. The existing four shared buffers are reused, including the speaker-sum scratch space; no additional unbounded allocation is introduced.

`operations.vbx_statistics` distinguishes `invL` and `alpha`, recording the call and iteration, reduction order, staging bytes, logical boundary bytes, kernels and actual GPU completion duration. Requested/observed fields, `native_vbx_statistics_calls` and the per-call `native_statistics_calls` distinguish intent from execution. Ordinary iterations add two statistical calls. Supplied alpha/invL skip the first update as in the original function; zero iterations execute no statistical updates. Failure restores the original hook and leaves a failed call summary.

The tested primitive results and complete VBx return values match the preceding native-product policy bitwise, including C/F inputs and signed zero. The existing native product policy still differs from BLAS accumulation. Score elementwise arithmetic, log/exp/logsumexp, posterior/prior updates, ELBO and stopping remain on their original backends. Results still return to the host between operations; full GPU residency, speedup and quality certification are not claimed.

## Optional native VBx score arithmetic

Rebuild the binary64 library at a new path and add `--vbx-score-arithmetic` with `--vbx-matrix-products` and the library/hash flags. This requires the additive `wxdouble_vbx_scores` ABI. It works independently of `--vbx-precomputed-terms` and `--vbx-statistics`; all previous flags and ABI exports retain their behavior when the option is absent.

The pinned original `log_p_` assignment is compiled privately with two native score calls. The first computes `invL + alpha**2` before the existing speaker-penalty product; the second computes `Fa * (frame_product - 0.5 * penalty_product + G)`. Square, addition, multiplication by 0.5, subtraction, addition of G and multiplication by Fa each round separately in binary64 integer emulation. The three existing products retain their original order and metadata. No host score arithmetic or CPU fallback is used for these two replaced expressions. Each score call executes one Metal kernel and returns its result to the host.

The penalty primitive accepts 1..1024 speakers and 1..1024 features; score composition accepts 1..4096 frames and 1..1024 speakers. All operands must be plain contiguous float64 arrays after explicit adapter staging. Inverse precision must be nonnegative. Every operand is finite, zero or has magnitude in 2^-200..2^200; Fa is a positive finite Python scalar in 2^-20..2^20. The penalty primitive can produce values up to roughly 2^400, but the existing matrix-product input domain still applies to the subsequent product and can reject those values. These limits are checked without clipping. The original four buffers are reused; the retained shared-buffer bound remains 72 MiB plus 40 KiB across supported operations. For score composition, the speaker penalty and per-frame G vectors are packed into one bounded input buffer, with the host packing bytes recorded separately.

Reports distinguish `vbx_score_arithmetic_requested`, `vbx_score_arithmetic_observed` and `native_vbx_score_calls`. Entries in `operations.vbx_score_arithmetic` identify `penalty_terms` or `frame_scores`, the call and iteration, source hash, staging/packing bytes and actual GPU completion duration. Per-call `native_score_calls` counts two score calls per completed ordinary iteration. Supplying initial alpha/invL still computes both score calls while skipping the first statistics update; zero iterations perform no score calls. Errors restore the hook and preserve the failed-call evidence.

Tests compare exact bits with NumPy for separate rounding, cancellation, signed zero and bounded exponents, then compare complete VBx results against the preceding native policy across all precompute/statistics combinations and both feature layouts. Transcendental functions, posterior/prior updates, ELBO and stopping remain on their original backends. This step does not establish full GPU residency, quality certification or an end-to-end speedup.

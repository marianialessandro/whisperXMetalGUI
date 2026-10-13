# Baseline and experimental Metal CTC

These tools run directly from the command line. They do not launch or test the macOS UI. Experimental CTC candidates do not change the production WhisperX backend and do not constitute the full quality/GPU certification gate.

## Build the native tools on Apple silicon

Use the installed Xcode command-line compiler and system Foundation/Metal frameworks. Generated binaries belong under the ignored `macOS/output/` directory.

```sh
mkdir -p macOS/output/baseline/tools
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Wno-unused-function -Wno-deprecated-declarations tools/baseline/native_ctc_probe.mm -framework Foundation -framework Metal -o macOS/output/baseline/tools/native_ctc_probe
xcrun clang++ -std=c++20 -O2 -fobjc-arc -dynamiclib -Wall -Wextra -Wno-unused-function -Wno-deprecated-declarations tools/baseline/native_ctc_bridge.mm -framework Foundation -framework Metal -o macOS/output/baseline/tools/libnative_ctc.dylib
shasum -a 256 macOS/output/baseline/tools/libnative_ctc.dylib
```

`native_ctc_probe --help` describes the standalone NPZ-based primitive diagnostic, including optional `--compact` and `--batched` modes. Its defaults and existing arguments remain supported.

## Run alignment trials

```sh
macOS/output/installer/WhisperX.app/Contents/Runtime/bin/python3 -I -B tools/baseline/ctc_alignment_trial.py --help
```

Supply an existing licensed audio asset and its expected SHA256, original ASR JSON and its expected SHA256, the verified private torchaudio model specification, language, and a new output directory. The ASR JSON is an oracle input, not independent word/speaker ground truth. The script refuses overwrite and missing checkpoint/license prerequisites.

The default `--ctc-backend mlx` preserves the first experimental trial. `--ctc-backend native` uses the standalone binary and private temporary NPZ files per segment; it requires `--native-binary` and `--native-binary-sha256`. Startup and compilation remain part of that candidate's timings.

`--ctc-backend native-persistent` requires `--native-library` and `--native-library-sha256`. It creates one native context for all segments and replicas, compiling the shaders once and reusing growable Metal buffers. Input and final path cross a synchronous in-memory C ABI. Native setup time is reported separately; input copies, completion and final path materialization remain part of the candidate alignment timing. Optional `--warmup --replicas 6` runs warm alternating original/candidate replicas. Run timing trials separately from GPU tests or other benchmark jobs.

Optional `--native-wildcard-max` requires the persistent backend and the `wxctc_row_max` ABI extension. A scoped Torch dispatch mode redirects CPU FP32 `max(dim=1)` without gradients/keepdim to the native Metal row reduction. Other operators follow their original implementations. The kernel returns exact maximum values and first indices on ties, supports nonpositive log scores including negative infinity, and rejects NaN/positive values. The additional pipeline is created lazily on first use and cached; warmup includes its setup. No production source or default is changed.

The original blank mask/gather and concatenation remain on CPU. Reduced values and int64 indices return to CPU before the original code constructs the CTC emission matrix. This extra readback and the dispatch-mode overhead are included in candidate alignment timings. The report records `wildcard_spans` and the actual dispatch count, which can be zero when no wildcard is needed. It does not certify a speedup caused by moving this reduction alone.

Alternatively, `--native-fused-wildcard` uses the persistent backend and the `wxctc_compute_wildcard` ABI extension; it cannot be combined with `--native-wildcard-max`. Trial-only wrapper tensors defer the original masked selection, maximum and concatenation without allocating or returning placeholder numerical values. The CTC boundary receives the original unextended FP32 emission and tokens; token index equal to the source vocabulary selects the virtual wildcard. Metal validates input, excludes the blank from the row maximum, extends the emission on GPU, and runs prefix/trellis/backtracking in the same second command buffer. There is no intermediate wildcard readback or host gather/concatenation of the score matrix. Two command-buffer completions remain per CTC call, plus the final path readback.

The deferred adapter supports the exact preparation operations used by the current alignment oracle on the pinned Torch runtime. It checks mask, vocabulary, blank and source identity; requests to materialize deferred values or use unsupported operations fail explicitly. It is a feasibility adapter, not a general tensor implementation or production backend. Mask/token metadata, original MPS-to-CPU emission and the copy to native shared buffers still use CPU. The report counts actual fused dispatches in `native_fused_wildcard_dispatch_count` and records logical intermediate readback bytes and retained buffer capacity. GPU extension needs an additional retained emission buffer, so fewer transfers do not imply lower total memory use.

Optional `--native-character-segments` requires the persistent backend and pinned CPython 3.12 float-sum semantics. After GPU backtracking, `ctc_merge_repeats` computes frame bounds and the mean score per token in the same command buffer. Binary64 addition/subtraction/division and the CPython 3.12 Neumaier correction are emulated with 64-bit integer operations for this bounded nonnegative FP32 probability domain. This preserves the sum policy without switching character means to FP32. It is not a general IEEE floating-point library.

The `wxctc_compute_segments` ABI extension returns 16-byte character segments plus path-length control, without reading back all frame points. Scoped trial hooks use those segments with the original transcript labels. GPU grouping/means do not change the original CPU timestamp scaling, word/sentence reductions, missing values or interpolation. `--return-char-alignments` optionally includes the original detailed character JSON in both outputs and their exact comparison. Both new flags default to false. Output payload can be larger than a path when token/frame counts are similar; the report exposes actual logical sizes and does not claim total memory/driver transfer measurements.

Results contain exact aligned JSON comparison, source input identities, native library/shader identity, setup time, command completion times and explicit CPU residuals. Buffer lengths describe retained Metal shared-buffer capacity; they do not measure total physical process/GPU memory. Neural inference follows the original alignment implementation; the optional flags select which character, timestamp, word mean and wildcard stages use native Metal. Unselected reductions use the original implementation. MPS-to-CPU and CPU-to-native copies remain; this is not a zero-copy pipeline.

## Experimental word/sentence timestamp bounds

Add `--native-timestamp-bounds` to a `native-persistent` alignment trial to run timestamp minima/maxima on Metal. It can be combined with fused wildcard and character segments. A temporary proxy replaces only the alignment module's Pandas reference; character-frame subsets retain a scoped Series subclass whose `start`/`end` min/max call the native context. Other Pandas users, means, labels, selection, timestamp scaling, rounding, interpolation and final sentence grouping keep the original behavior. The proxy restores even after an error.

The optional `wxctc_timestamp_bounds` ABI extension compares binary64 bit patterns using ordered integer keys, without FP32 conversion or double arithmetic. One 256-lane GPU group skips NaNs and emits both extrema plus a validation word. Empty/all-missing inputs return NaN; finite signed values and infinities are supported. Negative zero is rejected explicitly because NumPy's zero tie behavior can depend on its reduction path. Unsupported reducer options/dtypes or inputs over 32768 values fail without a numerical CPU fallback. Object columns containing only `None` are converted to float64 NaNs before transport.

The report includes dispatch counts, per-call completion/GPU times, reference Pandas/NumPy versions and copies: eight input bytes per value and 24 logical output bytes, excluding ABI metadata and driver copies. This feasibility adapter performs a separate synchronous dispatch for each reduction. Small word subsets may cost more in CPU/GPU synchronization than their arithmetic; timing must be checked before promotion. Batching and keeping character timestamps resident are subsequent work. No production default changes or full quality certification follow from these trials.

## Batched timestamp bounds

Use `--native-timestamp-bounds-batched` instead of `--native-timestamp-bounds` with the persistent backend to group extrema for all words/sentences of each character frame. It preserves the existing flags/defaults and is mutually exclusive with the individual timestamp mode. The scoped NLTK loader proxy records the spans emitted by the original tokenizer, then the Pandas factory builds the exact inclusive sentence selections and non-space word selections used by the oracle. Labels, spaces and word IDs are CPU metadata; no sentence splitting is reimplemented.

The `wxctc_timestamp_bounds_batch` ABI v1 extension receives one binary64 timestamp array and CSR query indices/offsets. Metal validates offsets and indices before reading selected values, then runs one 256-lane group per subset in one command buffer. Limits are 65536 input values, 32768 queries and 4194304 index entries. Empty subsets and NaNs follow the individual mode; selected negative zero fails explicitly. Validation failure leaves caller output untouched. Existing ABI exports/layouts are unchanged.

Each character frame uploads its start/end columns once for the batch. The scoped Series retrieves the requested extrema from the completed GPU results, checking subset identity and timestamp snapshot bytes; mutation or an unregistered subset fails instead of using stale data or falling back to CPU min/max. The Pandas/NLTK references restore on exit and error. This adapter belongs to a standalone CLI process and does not support concurrent original alignment calls while its hooks are installed.

Reports retain `native_timestamp_bounds_dispatch_count` for actual batch dispatches and add query/consumption counts. Logical transfer includes timestamp bytes plus CSR metadata; readback is 16 bytes per unique subset plus a four-byte validation word, excluding ABI metadata and physical copies. This does not eliminate character readback, CPU scaling/rounding, snapshot checks or the upload from CPU; retaining timestamps across the complete character/word pipeline remains subsequent work. Word means remain original unless `--native-word-means` is selected; interpolation and production defaults remain original.

## Optional Pandas constructor optimization

Add `--native-timestamp-fast-frames` with `--native-timestamp-bounds-batched` to avoid Pandas subclass reconstruction through temporary base Series/DataFrame objects. The adapter constructs its scoped subclasses directly from the managers already supplied by Pandas, then leaves column names, metadata finalization and copy-on-write behavior to the original caller. The option supports the inspected Pandas 3.0.6 runtime only; a different version fails explicitly before installing hooks because these manager methods are private APIs. The existing batched path remains available without the option.

This reduces CPU adapter overhead for the existing Metal kernels; it does not move another numerical operator to GPU, change transfers, remove snapshot validation or keep timestamps resident between pipeline stages. Reports record the flag globally and on each batch. Regression tests cover numeric/string dtypes, column names, selection and projection, missing values, original means, cache invalidation, parent isolation after modifying a derived frame, hook restoration and the runtime guard. All runs remain direct CLI processes without UI.

## Experimental word score means on Metal

Add `--native-word-means` to the batched timestamp trial, optionally with the fast frame constructors. It moves the selected non-space word score means to `ctc_word_means_batch`, through the additive ABI v1 export `wxctc_word_means_batch`. Each character frame uploads its rounded scores and CSR word selections once and completes one additional command buffer for all its means. Timestamp and score batches are currently separate; the previous modes/defaults remain available.

The pinned reference is Pandas 3.0.6 with NumPy 2.2.6 and `np.getbufsize() == 8192`; unsupported settings fail before installing the mean adapter. Pandas replaces NaNs with positive zero for the sum and excludes them from the count. Metal retains their positions and reproduces the [NumPy pairwise order](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_utils.h.src): eight accumulators, balanced leaf combination, tail additions and recursive splits at multiples of eight for blocks above 128. A bounded explicit stack implements the split tree. The outer sum combines 8192-element blocks in order, matching the pinned reduction buffering verified against the actual runtime. Binary64 addition and division use integer emulation already present in the character prototype; no FP32 narrowing or CPU numerical sum/mean is performed.

Supported score values are positive zero, finite values in 0.001..1 and NaN, matching the rounded character probability domain. Selected negative zero, negative values, infinities and other magnitudes are rejected. Limits: 32768 source scores and elements per word query, 32768 queries, 4194304 CSR indices. GPU validation checks metadata and score domain before summing; ABI output is untouched on failure. Empty/all-missing queries yield NaN. The scoped score Series returns a NumPy float64 scalar from the completed GPU batch so original `round(..., 3)` semantics are preserved. Unsupported subsets/options/dtypes and changed snapshots fail without a CPU mean fallback.

Reports separate `word_mean_spans`, real mean dispatch count and consumed word requests from timestamp counts. Per mean batch, logical readback is eight bytes per unique word plus four validation bytes; input includes scores and CSR metadata. CPU character scaling/rounding, final mean rounding, labels, selection, snapshots, interpolation, character readback and uploads remain. This is another numerical GPU feasibility step, not production promotion, a zero-copy claim or independent quality certification.

## Fused timestamp and word reductions

Add `--native-fused-reductions` with `--native-word-means` to use one interleaved binary64 `(start, end, score)` input, one shared CSR row query set and one Metal kernel per character frame. Modes bits 1/2/4 request minimum start, maximum end and mean score. A word's row indices serve all three operations; sentence selections still follow the original inclusive spans and literal-space exclusion. Previous separate-batch modes and defaults remain available.

The additive `wxctc_alignment_stats_batch` ABI v1 export validates row counts, lengths, capacities, mode bits, CSR metadata and selected numerical domains. Unrequested output fields are NaN and the scoped adapter rejects requests for operations it did not register. The binary64 extrema and NumPy score-sum policies are unchanged; the score helper reads interleaved data at stride three without altering its order. Limits: 32768 rows/queries/elements per subset and 4194304 CSR indices. Outputs are caller-owned triples, copied only after successful GPU completion/validation.

The fused core writes directly from its Metal shared output buffer into the caller's output memory while holding the context lock. It creates no intermediate C++ result vector for this export. The record exposes zero native host result staging bytes and one native result memcpy; this does not mean zero physical copies, zero Python snapshot allocations or zero-copy GPU transport. Previously exported functions preserve their behavior.

`timestamp_bounds_spans` carries the fused group's actual input/readback bytes, completion time and one command buffer. `word_mean_spans` is an explicitly marked component of that same completion group: `separate_dispatch=false`, zero additional command buffers and zero additional transfer bytes. Its GPU interval is shared, not a second independently measured kernel interval; do not sum duplicated component times. `native_word_means_dispatch_count` excludes these components; completed mean group/consumed request counts remain available. The report separately counts fused groups and unique row subsets.

Add `--native-compact-reduction-plan` to optionally use clipped integer sentence ranges, shared word row tuples and one row gather per fused subset for snapshot construction. Original snapshot bytes and mutation checks are preserved. A whole row gather can copy unused columns on CPU; the report distinguishes requested snapshot bytes from temporary gathered bytes and counts gather calls. This changes host preparation only, not numerical GPU placement or GPU transfer sizes. Plan/cache wall times are partial local intervals, not a partition of complete alignment time. Compare complete alignment timing before drawing performance conclusions.

All extensions remain standalone CLI feasibility work. Character readback, CPU timestamp scaling/rounding, final score rounding, interpolation, metadata, snapshots and uploads remain. The production backend and full quality gate are separate pending work.

## Lifecycle and verification

The development ABI in `native_ctc_bridge.h` returns zero on success and nonzero with a fixed error code on failure. Input pointers and declared lengths must remain valid until a synchronous call returns. Output capacity must cover `frames`. Calls on one context serialize; finish all calls before destroying it and never race destruction with a new call. The Python wrapper's context manager and lock enforce compute/close ordering. No CPU fallback is invoked when native validation fails.

```sh
macOS/output/installer/WhisperX.app/Contents/Runtime/bin/python3 -I -B -m unittest discover -s tests/baseline -p 'test_*.py'
```

Native tests require Apple Metal hardware and Xcode. They compile temporary binaries, execute real GPU kernels, and use an ASan/UBSan C++ ABI harness to check capacities, context lifetime and growing/reused buffers. All verification uses CLI scripts and binaries. Complete corpus, independent annotations, calibrated quality limits and the production ABI/helper remain separate pending work.

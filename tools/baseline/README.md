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

Results contain exact aligned JSON comparison, source input identities, native library/shader identity, setup time, command completion times and explicit CPU residuals. Buffer lengths describe retained Metal shared-buffer capacity; they do not measure total physical process/GPU memory. Neural inference and word/character reductions still follow the original alignment implementation; the wildcard reduction follows it when the optional flag is absent. MPS-to-CPU and CPU-to-native copies remain; this is not a zero-copy pipeline.

## Lifecycle and verification

The development ABI in `native_ctc_bridge.h` returns zero on success and nonzero with a fixed error code on failure. Input pointers and declared lengths must remain valid until a synchronous call returns. Output capacity must cover `frames`. Calls on one context serialize; finish all calls before destroying it and never race destruction with a new call. The Python wrapper's context manager and lock enforce compute/close ordering. No CPU fallback is invoked when native validation fails.

```sh
macOS/output/installer/WhisperX.app/Contents/Runtime/bin/python3 -I -B -m unittest discover -s tests/baseline -p 'test_*.py'
```

Native tests require Apple Metal hardware and Xcode. They compile temporary binaries, execute real GPU kernels, and use an ASan/UBSan C++ ABI harness to check capacities, context lifetime and growing/reused buffers. All verification uses CLI scripts and binaries. Complete corpus, independent annotations, calibrated quality limits and the production ABI/helper remain separate pending work.

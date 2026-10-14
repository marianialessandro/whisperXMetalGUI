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

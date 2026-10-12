# Technology Stack

**Analysis Date:** 2026-10-12

## Languages

**Primary:** Swift con tools-version 6.0 (`macOS/Package.swift`); Python >=3.10,<3.14 (`pyproject.toml`).
**Secondary:** Bash (`macOS/build-app.sh`); C/C++/Objective-C/Metal nelle dipendenze native incluse, non come motore applicativo proprio.

## Runtime

**Environment:** app macOS SwiftUI; runtime installer CPython 3.12.12 (`macOS/Packaging/build_installer.py`).
**Package Manager:** SwiftPM (`macOS/Package.swift`), uv/setuptools (`pyproject.toml`, `macOS/Packaging/build_installer.py`).
**Lockfile:** `macOS/Packaging/requirements.lock` con hash per il runtime; vincoli più ampi nel pacchetto Python.

## Frameworks

**Core:** SwiftUI/AppKit/Observation/Foundation; AVKit/AVFoundation per player e media (`PlaybackController.swift`, `ProjectDetail.swift`); Security per Portachiavi (`HuggingFaceConnection.swift`).
**Testing:** Swift Testing (`macOS/Tests/WhisperXMacTests`); pytest e unittest.mock (`tests/test_word_timestamp_interpolation.py`).
**Build/Dev:** SwiftPM, codesign/pkgbuild/productbuild/notarytool, uv, FFmpeg 7 privato (`macOS/build-app.sh`, `macOS/Packaging/build_installer.py`, `ffmpeg_build.py`).

## Key Dependencies

**Critical:**
- MLX e mlx-whisper: ASR GPU GUI (`macOS/Worker/transcribe_worker.py`, `macOS/Packaging/requirements.lock`).
- Torch ~2.8.0, torchaudio ~2.8.0, torchvision ~0.23.0: modelli e MPS (`pyproject.toml`).
- pyannote-audio >=4.0.0: community-1, segmentazione, embedding, clustering (`whisperx/diarize.py`, `pyproject.toml`).
- transformers >=4.48.0: Wav2Vec2 (`whisperx/alignment.py`, `pyproject.toml`).
- faster-whisper >=1.2.0, CTranslate2 >=4.5.0: API/CLI e fallback CPU (`whisperx/asr.py`, `pyproject.toml`).
- NumPy, pandas, SciPy transitivo: allineamento e clustering CPU (`whisperx/alignment.py`, dipendenze nel runtime installer).

**Infrastructure:** Hugging Face Hub e Torch Hub/cache (`whisperx/alignment.py`, `macOS/Sources/WhisperXMac/StorageInventory.swift`); FFmpeg e TorchCodec (`whisperx/audio.py`, `macOS/Packaging/smoke_runtime.py`).

## Configuration

**Environment:** `macOS/Sources/WhisperXMac/WorkerRuntime.swift` filtra PYTHON/DYLD/HF, imposta PATH privato e trasmette cache selezionate; token tramite stdin.
**Build:** `macOS/Package.swift`, `macOS/build-app.sh`, `macOS/Packaging/build_installer.py`; manifesto generato `RuntimeManifest.json`.

## Platform Requirements

**Development:** macOS con toolchain Swift 6 e strumenti Apple; packaging richiede Darwin arm64, uv e librerie FFmpeg preparate (`build_installer.py`).
**Production:** Swift package macOS 14; minimo installer calcolato tramite Mach-O delle dipendenze, quindi non dedurlo dal solo `Package.swift`. Doppio click e nessuna installazione manuale secondo `AGENTS.md`.

---
*Stack analysis: 2026-10-12. Versioni dichiarate/artefatti letti; nessuna prova di accelerazione eseguita.*

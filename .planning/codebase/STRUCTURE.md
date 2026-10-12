# Codebase Structure

**Analysis Date:** 2026-10-12

## Directory Layout

```text
whisperXGUI/
├── whisperx/        API Python, CLI, allineamento, diarizzazione, export
├── tests/           Regressioni Python
├── macOS/
│   ├── Sources/WhisperXMac/    UI, persistenza e controllo worker
│   ├── Tests/WhisperXMacTests/ Test Swift
│   ├── Worker/                Worker privato
│   ├── Packaging/             Runtime, lock, installer e smoke
│   └── output/                Artefatti generati
├── docs/            Progetto prodotto
├── .github/workflows/ CI Python upstream
└── .planning/       Pianificazione GSD
```

## Directory Purposes

- `whisperx/`: mantenere la superficie API legacy; `__init__.py` usa import lazy, `__main__.py` configura la CLI.
- `macOS/Sources/WhisperXMac/`: componenti Swift con nomi PascalCase; `ProjectModels.swift` contiene il catalogo, `TranscriptionEngine.swift` il processo.
- `macOS/Worker/`: solo protocollo e orchestrazione privata; `transcribe_worker.py` viene copiato nel bundle.
- `macOS/Packaging/`: `requirements.in`, `requirements.lock`, `build_installer.py`, `ffmpeg_build.py`, `smoke_runtime.py`.

## Key File Locations

**Entry Points:** `macOS/Sources/WhisperXMac/WhisperXApp.swift`, `whisperx/__main__.py`, `macOS/Worker/transcribe_worker.py`.
**Configuration:** `pyproject.toml`, `macOS/Package.swift`, `macOS/Packaging/requirements.lock`.
**Core Logic:** `whisperx/alignment.py`, `whisperx/diarize.py`, `whisperx/asr.py`, `macOS/Sources/WhisperXMac/LibraryStore.swift`.
**Testing:** `tests/test_word_timestamp_interpolation.py`, `macOS/Tests/WhisperXMacTests/LibraryTests.swift`, `WorkerRuntimeTests.swift`, `LocalizationTests.swift`.

## Naming Conventions

- Swift: file e tipi PascalCase, funzioni camelCase (`LibraryStore.swift`).
- Python: moduli e funzioni snake_case (`whisperx/alignment.py`), classi PascalCase.
- Output contrattuali: `original.json`, `aligned.json`, `result.json` definiti in `transcribe_worker.py`.

## Where to Add New Code

- UI e adapter dell'app: `macOS/Sources/WhisperXMac/`, test in `macOS/Tests/WhisperXMacTests/`.
- Regressioni delle API Python: `tests/`; non sostituire i contratti di `whisperx/schema.py`.
- Runtime e distribuzione: `macOS/Packaging/`; evitare dipendenze da percorsi personali nello script `build_installer.py`.
- Il repository non contiene un motore C++/Metal integrato: directory/target nativi devono essere definiti nei piani, non trattati come componenti esistenti.

## Special Directories

- `macOS/output/installer/WhisperX.app/Contents/Runtime/`: artefatto generato con dipendenze Python consultabili; non modificarne i sorgenti come soluzione permanente.
- `macOS/.packaging/`, `macOS/.build/`: cache/build generate dagli script.
- `/Users/marianialessandro/repository/whisper.cpp`: repository esterno di riferimento; non parte dell'app e non modificato.
- `.planning/codebase/`: documenti di analisi; nessun commit consentito da `AGENTS.md`.

---
*Structure analysis: 2026-10-12. Lettura del checkout, nessuna implementazione.*

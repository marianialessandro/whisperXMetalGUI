<!-- refreshed: 2026-10-12 -->
# Architecture

**Analysis Date:** 2026-10-12

## System Overview

```text
SwiftUI / AppKit: macOS/Sources/WhisperXMac
  ├─ LibraryStore → Library.json / Media / Projects
  ├─ HuggingFaceConnection → URLSession / Portachiavi
  └─ TranscriptionEngine → stdin JSON / stdout NDJSON
       └─ macOS/Worker/transcribe_worker.py
            ├─ FFmpeg → audio NumPy
            ├─ MLX Whisper / fallback CTranslate2 CPU
            ├─ Wav2Vec2 MPS / CTC CPU
            ├─ pyannote MPS / clustering NumPy-SciPy
            └─ original.json / aligned.json / result.json / TXT-SRT-VTT
```

## Component Responsibilities

| Componente | Responsabilità | File |
|---|---|---|
| App | Menu, scena, impostazioni | `macOS/Sources/WhisperXMac/WhisperXApp.swift` |
| Libreria | Importazione, catalogo, trasferimento, eliminazione | `macOS/Sources/WhisperXMac/LibraryStore.swift` |
| Motore UI | Avvio, annullamento, eventi e risultati | `macOS/Sources/WhisperXMac/TranscriptionEngine.swift` |
| Worker | Orchestrazione sequenziale e output | `macOS/Worker/transcribe_worker.py` |
| Allineamento | Modello, CTC, parole e caratteri | `whisperx/alignment.py` |
| Diarizzazione | Community-1 e assegnazione voci | `whisperx/diarize.py` |
| Schema | Contratti Python | `whisperx/schema.py` |

## Pattern Overview

**Overall:** Applicazione nativa con motore separato tramite processo.

- Usare `@MainActor @Observable` per lo stato della UI, seguendo `TranscriptionEngine.swift` e `LibraryStore.swift`.
- Mantenere separate API pubbliche Python (`whisperx/__init__.py`) e protocollo privato del worker (`macOS/Worker/transcribe_worker.py`).
- L'ASR della GUI usa MLX; la CLI (`whisperx/transcribe.py`) usa faster-whisper: non confondere i due percorsi.

## Layers

**Presentazione:** `macOS/Sources/WhisperXMac/ProjectDetail.swift`, `LibraryView.swift`, `AppSettingsView.swift`; legge modelli e stato osservabile.

**Persistenza:** `ProjectModels.swift`, `LibraryStore.swift`; `LibraryManifest.version = 1`, UUID stabili, URL del media, registrazioni, nomi delle voci per trascrizione; JSON atomico, nessun database SQLite rilevato.

**Calcolo:** `macOS/Worker/transcribe_worker.py`, `whisperx/alignment.py`, `whisperx/diarize.py`, `whisperx/utils.py`; dipende da runtime e cache modelli.

## Data Flow

### Primary Request Path

1. `ProjectDetail.swift` richiede una trascrizione e individua una directory risultato.
2. `TranscriptionEngine.swift` serializza audio/output/lingua/qualità/voci/hints e token opzionale su stdin.
3. `transcribe_worker.py` esegue audio → ASR → allineamento → diarizzazione → export.
4. Il worker emette `stage`, `failed`, `completed`; il completamento contiene percorsi e avvisi.
5. `TranscriptionEngine.swift` legge `result.json`, decodifica i segmenti e invoca il callback di persistenza.
6. `LibraryStore.swift` salva `TranscriptionRecord`; `ProjectDetail.swift` ricarica il risultato.

### Trasferimento libreria

`LibraryStore.swift:relocateLibrary` sposta `Media` e `Projects`, aggiorna URL gestiti e risultati, scrive/verifica il manifest, rimuove il catalogo precedente; su errore ripristina gli spostamenti. I media esterni mantengono i loro percorsi.

**State Management:** `ProjectModels.swift` conserva configurazioni e registrazioni; `TranscriptionEngine.swift` gestisce un solo processo attivo.

## Key Abstractions

- `MediaItem`, `TranscriptionRecord`, `LibraryManifest`: `ProjectModels.swift`; preservare i dati persistiti.
- `Transcript`: `TranscriptionEngine.swift`; la UI decodifica start/end/text/speaker, mentre il JSON completo può includere parole.
- `SingleAlignedSegment`, `AlignedTranscriptionResult`: `whisperx/schema.py`; preservare parole, caratteri opzionali e punteggi.

## Entry Points

- App: `macOS/Sources/WhisperXMac/WhisperXApp.swift`.
- Worker privato: `macOS/Worker/transcribe_worker.py:main`.
- CLI legacy: `whisperx/__main__.py`, entrypoint in `pyproject.toml`.
- Packaging: `macOS/build-app.sh`, `macOS/Packaging/build_installer.py`.

## Architectural Constraints

- **Threading:** stato UI MainActor; lettura libreria e trasferimenti usano task utility in `LibraryStore.swift`.
- **Global state:** `mlx_whisper.transcribe.ModelHolder` nel runtime distribuito; il worker lo svuota dopo ASR.
- **GPU:** MLX/Metal e Torch/MPS sono backend distinti; `transcribe_worker.py` non condivide esplicitamente buffer fra essi.
- **Distribuzione:** `build_installer.py` richiede Darwin arm64; `Package.swift` dichiara macOS 14, ma il packaging calcola il minimo dalle librerie native.
- **Prodotto:** seguire `AGENTS.md`; Hugging Face solo nelle Impostazioni, player nella toolbar, nessun token nei log/argv/progetto.

## Anti-Patterns

### Confondere dispositivo del modello con dispositivo della pipeline

`whisperx/alignment.py` usa inferenza sul dispositivo richiesto ma copia le emissioni alla CPU; `whisperx/diarize.py` delega a una pipeline con postprocessing NumPy. Verificare ogni operatore e confine di buffer prima di dichiarare copertura GPU.

### Interpretare un avviso come verifica della qualità

`transcribe_worker.py` può completare dopo fallimenti di allineamento o diarizzazione. Tracciare capacità effettive e fallimenti di fase: `completed` non prova diarizzazione riuscita.

## Error Handling

`transcribe_worker.py` conserva ASR in `original.json`, salva `aligned.json` e degrada tramite avvisi; errore globale emette `failed` con redazione del token. `TranscriptionEngine.swift` scarta stderr e ignora eventi non riconosciuti; preservare compatibilità del protocollo e aggiungere diagnostica strutturata sanitizzata.

## Cross-Cutting Concerns

**Logging:** eventi NDJSON nel worker; `whisperx/log_utils.py` per le API Python.
**Validation:** file sorgente risolto con strict=True nel worker; validazione destinazioni in `LibraryStore.swift`.
**Authentication:** `HuggingFaceConnection.swift:TokenVault` usa Security/Portachiavi; il worker riceve il token solo su stdin.

---
*Architecture analysis: 2026-10-12. Evidenze da lettura sorgenti; nessun runtime o benchmark eseguito in questa mappatura.*

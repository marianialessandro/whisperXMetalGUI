# Coding Conventions

**Analysis Date:** 2026-10-12

## Naming Patterns

**Files:** PascalCase per Swift (`LibraryStore.swift`), snake_case per Python (`transcribe_worker.py`).
**Functions:** camelCase Swift (`relocateLibrary`); snake_case Python (`assign_word_speakers`).
**Variables:** UUID per identità persistita in `ProjectModels.swift`; parole Italiane per etichette, identificatori inglesi nel codice.
**Types:** PascalCase (`MediaItem`, `DiarizationPipeline`); TypedDict per contratti Python (`whisperx/schema.py`).

## Code Style

**Formatting:** nessun formatter specifico rilevato nei manifest esaminati. Seguire `AGENTS.md`: non spezzare arbitrariamente contenuti fra parentesi e andare a capo nei commenti solo a frase conclusa. Conservare lo stile del file quando non entra in conflitto.
**Linting:** non rilevato un gate specifico del motore macOS in `macOS/Package.swift`.
**Project skills:** `.codex/skills/` e `.agents/skills/` non presenti nel checkout esaminato.

## Import Organization

1. Framework di sistema Swift (`TranscriptionEngine.swift`).
2. Standard library Python, librerie esterne, moduli locali (`whisperx/diarize.py`).
3. Import costosi lazy dentro il worker e wrapper API (`transcribe_worker.py`, `whisperx/__init__.py`).

**Path Aliases:** non rilevati; evitare percorsi personali per distribuzione, seguire le risorse Bundle (`TranscriptionEngine.swift`).

## Error Handling

- Swift throws per filesystem/Portachiavi; stato errore localizzato per UI (`LibraryStore.swift`, `HuggingFaceConnection.swift`).
- Worker: `failed` per errore globale, warnings per degradazione di fasi (`transcribe_worker.py`).
- Preservare i risultati originali e distinguere successo ASR da successo delle voci.

## Logging

**Framework:** `whisperx/log_utils.py`; stdout NDJSON per worker.
**Patterns:** non scrivere testo libero sul canale eventi; `emit` usa sys.__stdout__, librerie redirette a stderr (`transcribe_worker.py`). Non propagare token: `AGENTS.md`, `WorkerRuntime.swift`.

## Comments

**When to Comment:** spiegare invarianti, wildcard, cache e condizioni di errore, seguendo `alignment.py` e `WorkerRuntime.swift`.
**JSDoc/TSDoc:** non applicabile; docstring Python e commenti Swift.

## Function Design

**Size:** separare parsing e core: `transcribe_worker.py:main/run`.
**Parameters:** mantenere posizionali/flag legacy; nuove capacità opzionali, `AGENTS.md` e `whisperx/__init__.py`.
**Return Values:** strutture stabili in `whisperx/schema.py`; non restringere implicitamente output diarizzazione/embedding di `whisperx/diarize.py`.

## Module Design

**Exports:** wrapper lazy dell'API in `whisperx/__init__.py`; executable target Swift in `macOS/Package.swift`.
**Barrel Files:** `whisperx/__init__.py` mantiene i wrapper pubblici; nessun modulo C ABI del motore rilevato.
**Product constraints:** `AGENTS.md` prevale su sezioni storiche contrastanti in `docs/PROGETTO_APP_MACOS.md`; Impostazioni Hugging Face/Modelli/Libreria e nessuna configurazione HF nella finestra principale.

---
*Convention analysis: 2026-10-12. Analisi statica; nessuna modifica sorgente o commit.*

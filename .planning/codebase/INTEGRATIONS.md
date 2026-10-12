# External Integrations

**Analysis Date:** 2026-10-12

## APIs & External Services

**Modelli:** Hugging Face per MLX Whisper, modelli di allineamento e pyannote community-1 (`macOS/Worker/transcribe_worker.py`, `whisperx/alignment.py`, `whisperx/diarize.py`). Client Python Hub/transformers/pyannote; configurazione Swift tramite URLSession (`HuggingFaceConnection.swift`).

**Accesso:** `HuggingFaceConnection.swift` verifica identità e accesso al config community-1 tramite header Authorization. Non presupporre hosting proprietario o token condivisi: `AGENTS.md`.

## Data Storage

**Databases:** nessun database rilevato per la GUI; `LibraryStore.swift` salva `Library.json` tramite Codable e scrittura atomica.
**File Storage:** locale; `Media` per file gestiti e `Projects/<UUID>` per risultati (`LibraryStore.swift`). Media esterni restano esterni.
**Caching:** cache Hugging Face/Torch censite in `StorageInventory.swift` e condivise con il worker tramite `WorkerRuntime.swift`.

## Authentication & Identity

**Auth Provider:** Hugging Face per modelli gated.
**Implementation:** token personale nel Portachiavi (`HuggingFaceConnection.swift:TokenVault`); consegna al processo su stdin (`TranscriptionEngine.swift`). Vietare argv, log e salvataggio nei risultati.

## Monitoring & Observability

**Error Tracking:** servizio remoto non rilevato nei componenti esaminati.
**Logs:** eventi NDJSON nel worker, logger Python (`whisperx/log_utils.py`); stderr del processo scartato da `TranscriptionEngine.swift`. Non sono presenti misure Metal o tempi per fase nel protocollo attuale.

## CI/CD & Deployment

**Hosting:** non richiesto; inferenza locale, download modelli online (`docs/PROGETTO_APP_MACOS.md`).
**CI Pipeline:** GitHub Actions per test Python Linux e compatibilità (`.github/workflows/tests.yml`, `python-compatibility.yml`); nessun gate macOS GPU rilevato.
**Installer:** `macOS/Packaging/build_installer.py` costruisce PKG, firma ad hoc oppure identità Developer ID esplicite e notarizzazione opzionale. L'esistenza dello script non prova pubblicazione o notarizzazione.

## Environment Configuration

**Required env vars:** nessun token ambientale richiesto dalla GUI; PATH privato e HF_HUB_DISABLE_IMPLICIT_TOKEN generati in `WorkerRuntime.swift`. Cache opzionali HF_HOME/HF_HUB_CACHE/TORCH_HOME/XDG_CACHE_HOME.
**Secrets location:** Portachiavi macOS; non leggere archivi di credenziali. `TokenVault` è codice sorgente dell'accesso, non un file contenente token.

## Webhooks & Callbacks

**Incoming:** nessun webhook rilevato; JSON stdin nel processo locale (`transcribe_worker.py`).
**Outgoing:** callback locale di completamento e stdout NDJSON (`TranscriptionEngine.swift`).

---
*Integration audit: 2026-10-12. Solo lettura sorgenti; nessuna chiamata autenticata o download.*

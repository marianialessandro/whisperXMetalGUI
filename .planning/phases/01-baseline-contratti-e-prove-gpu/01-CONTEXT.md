# Phase 1: Baseline, contratti e prove GPU — Context

**Gathered:** 2026-10-12
**Status:** Ready for planning
**Source:** PRD Express Path da PROJECT.md, REQUIREMENTS.md e decisioni già espresse nella conversazione. Non è una nuova richiesta di approvazione dell'architettura.

<domain>
## Phase Boundary

Implementare in seguito una baseline eseguibile del motore attuale e un framework riproducibile di confronto di qualità, risorse e placement. Questa fase consegna BASE-01..04: manifest, corpus/oracle, profili sincronizzati ed evaluation contract congelabile. Non implementa il core nativo o i porting GPU delle fasi successive e non dichiara prestazioni del nuovo motore.

La consegna attuale è soltanto il piano. Nessun task viene eseguito automaticamente dopo la revisione.
</domain>

<decisions>
## Implementation Decisions

### Obiettivo bloccato dall'utente
- Spostare sulla GPU tutta la computazione numerica portabile; C++, Objective-C++, Metal e riscritture estese sono autorizzati come direzione del piano.
- Conservare diarizzazione community-1, numero voci automatico/esatto, overlap e attribuzione segmenti/parole.
- Conservare profili base/small/large-v3 completo e qualità; un confronto con modello diverso non costituisce parità.
- whisper.cpp nel repository personale è read-only reference, mai dipendenza di release dal percorso personale.

### Baseline e contratti
- Registrare SHA completo, versioni di runtime e modelli/revision/hash, impostazioni reali e hardware in manifest sanitizzati.
- Distinguere GUI MLX da CLI CTranslate2 e da whisper.cpp reference; non eseguire un altro checkout pensando sia la baseline GUI.
- Congelare result/schema/events/export/language/word/speaker e scenari library/Keychain/errore/cancellazione; non modificare catalogo o media personali per ottenere fixture.
- Usare fixture sintetiche/versionabili e corpus autorizzato con licenze esplicite. I test sintetici non certificano WER o DER naturale; senza annotazione un oracle dà parità, non accuratezza.
- Test nativi, modelli convertiti e framework futuri non sono prerequisiti della fase 1. Tool baseline può usare Python del runtime esistente in sviluppo, senza installazioni globali o modifica della produzione.

### Misure e qualità
- Profiling deve separare enqueue, completamento GPU, CPU compute, copie/readback, attese, cold/warm e download.
- Disponibilità Metal/MPS non è evidenza d'inferenza; trace/backend osservato è collegato a run ID e fase.
- Casi mancanti e fallback sono visibili; il ledger deve rilevare un caso intenzionalmente non coperto.
- Definire metriche e metodo di calibrazione/freeze prima di promuovere nuovi backend: WER/CER, timestamp coverage/start/end, DER/JER e speaker count, policy overlap/collar/permutazione label.
- Non inventare soglie o speedup sulla base della sola lettura sorgenti. Pianificare raccolta e congelamento delle soglie con manifest/artefatti e divieto di retuning per far passare un candidato.
- Token gated assente o modello non disponibile è una precondizione/esito esplicito, non una prova simulata di qualità o GPU; niente token argv/env/file. Preferire uso del percorso Impostazioni corrente e dei modelli locali autorizzati.

### Git e workflow
- NO commit/push/tag/branch/worktree automatici, anche se il workflow standard li prescrive.
- Non sovrascrivere AGENTS.md, istruzioni, source work preesistente o il checkout whisper.cpp.
- Pianificare ogni test/command con tool/runtime reali o infrastruttura Wave 0 esplicitamente prevista; non usare test che specchiano soltanto l'implementazione.

### Discrezione dell'agente
- Organizzazione di tools/tests/fixture e numero di piani da 2–3 task ciascuno; granularity fine.
- Metriche e normalizzazioni esatte, sampling della trace e selezione corpus rappresentativo documentati nel contratto, con soglie ancora da congelare dopo misure.
- La conversione produttiva senza Python appartiene alla fase 2. In fase 1 si inventariano formati/config e si identifica un proof-of-conversion minimo solo se non confonde la responsabilità o richiede download estesi.
</decisions>

<canonical_refs>
## Canonical References

Downstream agents devono leggere queste fonti prima di pianificare/implementare:
- `AGENTS.md` — istruzioni vincolanti e no commit.
- `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/GPU-PLACEMENT.md` — scope e criteri.
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/TESTING.md`, `.planning/codebase/CONCERNS.md` — stato brownfield.
- `.planning/research/SUMMARY.md`, `.planning/research/ARCHITECTURE.md`, `.planning/research/PITFALLS.md` — vincoli di parità, algoritmi e placement.
- `macOS/Worker/transcribe_worker.py`, `whisperx/alignment.py`, `whisperx/diarize.py`, `whisperx/schema.py`, `whisperx/utils.py` — oracle semantico.
- `macOS/Sources/WhisperXMac/TranscriptionEngine.swift`, `WorkerRuntime.swift`, `ProjectModels.swift`, `LibraryStore.swift` nella stessa directory — protocollo e prodotto.
- `macOS/Packaging/requirements.lock`, `macOS/Packaging/smoke_runtime.py`, `macOS/Package.swift`, `tests/test_word_timestamp_interpolation.py` — runtime/tooling e verifiche esistenti.
- `docs/PROGETTO_APP_MACOS.md` — prodotto; sezioni recenti e AGENTS.md prevalgono su indicazioni storiche.
- `/Users/marianialessandro/repository/whisper.cpp/src/whisper.cpp` — reference esterno read-only, revisione da registrare senza modifiche.
</canonical_refs>

<specifics>
## Specific Ideas

Mettere a confronto risultati intermedi quando possibile: feature, emissioni, decisioni CTC, embedding, cluster e word-speaker assignment. Il harness deve poter rilevare drift numerico e placement incompleto anche se il testo finale appare plausibile. Eventi di diagnostica non contengono token, testo personale o path sensibili non necessari.
</specifics>

<deferred>
## Deferred Ideas

Core nativo e helper (fase 3), porting feature/modelli/kernel (fasi 4–9), wiring app e installer (10–11). Nessun residuo numerico è rinviato a v2.
</deferred>

---
*Phase: 01-baseline-contratti-e-prove-gpu. Context raccolto dalla direzione già approvata, non da nuove scelte implicite dell'utente.*

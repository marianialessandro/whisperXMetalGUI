# Codebase Concerns

**Analysis Date:** 2026-10-12

## Tech Debt

**Motori separati e orchestrazione per processo:**
- Issue: MLX nella GUI, CTranslate2 nella CLI, Torch per altre fasi; modelli caricati per lavoro, cache MLX esplicitamente liberata.
- Files: `macOS/Worker/transcribe_worker.py`, `whisperx/asr.py`, `macOS/Sources/WhisperXMac/TranscriptionEngine.swift`.
- Impact: cold start, buffering separato e difficile misurazione del costo fra backend.
- Fix approach: contratto motore esplicito e lifecycle nativo; mantenere API/CLI legacy, misurare riuso memoria e modelli.

## Known Bugs

Nessun nuovo bug runtime riprodotto in questa analisi. Le criticità seguenti derivano dal codice e non sono misure osservate. `macOS/Worker/transcribe_worker.py` può completare senza allineamento o diarizzazione con avvisi: è comportamento deliberato, non prova di output equivalente.

## Security Considerations

**Token e diagnostica:**
- Risk: eventuale nuovo logger/processo può esporre credenziali.
- Files: `HuggingFaceConnection.swift`, `TranscriptionEngine.swift`, `WorkerRuntime.swift`, `transcribe_worker.py` sotto `macOS`.
- Current mitigation: Portachiavi, stdin, filtro env, token redatto nell'errore globale, stderr scartato.
- Recommendations: mantenere redazione strutturata e test di esclusione da argv/log/file; la migrazione non autorizza token condivisi o hosting proprio (`AGENTS.md`).

## Performance Bottlenecks

**Allineamento CPU dopo MPS:**
- Problem: inferenza per segmento, `.cpu().detach()` delle emissioni, trellis e backtracking CPU.
- Files: `whisperx/alignment.py:align`, `get_trellis`, `backtrack`.
- Cause: cicli Python nel tempo, letture scalari, wildcard/reduzioni e dataframe.
- Improvement path: batch Wav2Vec2 con padding/lunghezze corrette; kernel Metal CTC, wildcard, percorso e aggregazione; preservare blank, inizializzazione, tie-breaking e precisione punteggi.

**Diarizzazione ibrida:**
- Problem: `.to(device)` non comprende clustering e tutte le feature.
- Files: `whisperx/diarize.py`; dipendenze in `macOS/output/installer/WhisperX.app/Contents/Runtime/lib/python3.12/site-packages/pyannote/audio/pipelines/speaker_diarization.py`, `clustering.py`, `speaker_verification.py`.
- Cause: embeddings riportati a NumPy; clustering SciPy linkage; VBx e PLDA numerici fuori dal grafo Torch; batch configurabili, non impostati esplicitamente dal worker.
- Improvement path: mappare esattamente modelli/config/checkpoint scaricati, poi portare feature, segmentazione, embedding, PLDA/VBx, distanze/fusioni e ricostruzione su GPU. I default di classe non provano configurazione del modello effettivo.

**ASR a finestre sequenziali:**
- Problem: loop `seek` nella dipendenza MLX, assenza di batch ASR controllato dal worker.
- Files: `macOS/Worker/transcribe_worker.py`; `macOS/output/installer/WhisperX.app/Contents/Runtime/lib/python3.12/site-packages/mlx_whisper/transcribe.py`.
- Cause: progressione sequenziale delle finestre e decoder autoregressivo.
- Improvement path: confrontare motore GGML/Metal preservando qualità, prompt, lingua e confini; batching quando semanticamente valido. C++ da solo non dimostra accelerazione.

**Audio intero e assegnazione voci:**
- Problem: FFmpeg produce array completo; query IntervalTree filtra tutti i candidati anteriori e find_nearest scorre tutti i midpoint.
- Files: `whisperx/audio.py`, `whisperx/diarize.py:IntervalTree`.
- Cause: memoria proporzionale alla durata; query non garantisce costo O(log n) nonostante la docstring.
- Improvement path: blocchi audio e kernel/riduzioni di overlap; conservare policy di assegnazione corrente, non dedurre accelerazioni dalla docstring.

## Fragile Areas

**Contratti risultato e libreria:**
- Files: `whisperx/schema.py`, `macOS/Sources/WhisperXMac/ProjectModels.swift`, `LibraryStore.swift`, `ProjectDetail.swift`, `whisperx/utils.py`.
- Why fragile: UUID/URL/catalogo versione 1, segmenti/parole, etichette speaker, nomi personalizzati e file esportati hanno consumatori distinti.
- Safe modification: test contratti e round-trip; migrazioni esplicite se necessarie, preservare file esterni e rollback del trasferimento.
- Test coverage: libreria e wildcard coperti da test sorgente; equivalenza export/modello nativo non rilevata.

**Diagnostica perdita di dettaglio:** `TranscriptionEngine.swift` scarta stderr, `transcribe_worker.py` cattura Exception genericamente per fasi opzionali. Aggiungere eventi tecnici sanitizzati con fase/backend/errore; nessuna telemetria GPU attuale rilevata.

## Scaling Limits

**Memoria unificata:** nessun budget o batch adattivo nel worker; audio, logits e trellis crescono con durata/segmento (`transcribe_worker.py`, `alignment.py`). Capacità massima non misurata; introdurre limiti espliciti, chunk e prove OOM/cancellazione.
**Concorrenza:** `TranscriptionEngine.swift` ammette un processo attivo; non assumere che più modelli simultanei aumentino throughput GPU senza misure.

## Dependencies at Risk

**Runtime pesante e requisiti minimi:** `macOS/Packaging/build_installer.py` include Python/Torch/MLX/FFmpeg e calcola deployment target nativo. Il nuovo motore richiede metalli compilati, dylib rilocabili, licenze e firma; aggiornare packaging senza perdere installazione autonoma.
**whisper.cpp esterno:** `/Users/marianialessandro/repository/whisper.cpp/CMakeLists.txt` dichiara versione 1.8.3; è riferimento non incorporato. Pin/versione/licenza/conversione modelli devono essere gestiti dal progetto, senza dipendere dal checkout personale.

## Missing Critical Features

**Motore nativo GPU integrale:** nessun target C++/Objective-C++/Metal proprio in `macOS/Package.swift`; manca copertura Wav2Vec2/CTC/community-1 nativa.
**Baseline misurata:** mancano corpus e prove reali per decisioni prestazioni/qualità (`tests/`, `macOS/Packaging/smoke_runtime.py`); non promettere fattori di accelerazione.

## Test Coverage Gaps

- **Alta:** equivalenza modelli convertiti, CTC edge case/tie-breaking, speaker automatici/noti e overlap (`alignment.py`, `diarize.py`).
- **Alta:** tracce Metal e contabilità dei fallback; smoke import non prova inferenza (`smoke_runtime.py`).
- **Alta:** end-to-end e cancellazione/OOM/sleep in motore nativo (`TranscriptionEngine.swift`).
- **Media:** export TXT/SRT/VTT e schema raw/nomi personalizzati dopo migrazione (`whisperx/utils.py`, `ProjectDetail.swift`).
- **Alta:** installazione su Mac pulito e deployment target reale (`build_installer.py`); report di verifica non sostituisce un nuovo gate candidato.

---
*Concerns audit: 2026-10-12. Lettura di sorgenti e dipendenze presenti; nessun modello scaricato, benchmark o test eseguito.*

---
gsd_state_version: '1.0'
status: ready_to_execute
progress:
  total_phases: 11
  completed_phases: 0
  total_plans: 5
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-12)

**Core value:** GPU Apple per tutta la computazione numerica portabile, mantenendo qualità, tempi parola e diarizzazione completa.
**Current focus:** Fase 1 pianificata e verificata; esecuzione non avviata.

## Current Position

Phase: 1 of 11 (Baseline, contratti e prove GPU)
Plan: 0 of 5 in current phase
Status: Ready to execute — pianificazione soltanto, nessuna implementazione avviata
Last activity: 2026-10-12 — 11 fasi, 39/39 requisiti mappati; fase 1 con 5 piani, 11 task e 4 wave; checker PASS dopo una revisione, 0 blocker e 0 warning.

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: N/A
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 0 | 5 | N/A |
| 2–11 | 0 | TBD | N/A |

**Recent Trend:** Nessuna esecuzione; nessun benchmark o speedup dimostrato.

## Accumulated Context

### Decisions

Vedi PROJECT.md, Key Decisions, e GPU-PLACEMENT.md.
- Core C++/Objective-C++/Metal con ABI C e helper isolato; SwiftUI e prodotto attuali preservati.
- whisper.cpp read-only a 21411d81, SHA completo da congelare; niente aggiornamento upstream automatico.
- Diarizzazione community-1 completa: AHC centroid, PLDA, VBx locale, K-means/vincoli e attribuzione parole GPU.
- Python solo oracle di sviluppo, ritiro dal bundle dopo tutte le lingue/modelli e gate; nessun fallback numerico nascosto.
- Pianificazione autorizzata; auto_advance disattivato; nessun commit/push/tag/branch o source edit.

### Pending Todos

- Prossima azione: `$gsd-execute-phase 1`, soltanto quando richiesta dall'utente, nella directory `01-baseline-contratti-e-prove-gpu`.
- Congelare corpus, annotazioni, metriche/tolleranze, hardware, budget e contratti prima di promozione backend.

### Blockers/Concerns

- Fattibilità ancora da provare: checkpoint reali, licenze e conversione/download senza Python produttivo; gate iniziale fase 2 prima dei porting estesi.
- SDK/macOS minimo, operatori neurali e interop Metal/GGML/MPSGraph da verificare; disponibilità backend non prova GPU completa.
- Numerica FP64 reference contro FP32 GPU, AHC a memoria limitata e segmenti CTC grandi richiedono prove sui casi instabili.
- IT/EN non completano ALIGN-02: tutte le famiglie/mapping correnti devono passare prima della release.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | Nessun calcolo numerico v1 rinviato | - | - |

## Session Continuity

Last session: 2026-10-12
Stopped at: Pianificazione della fase 1 verificata, rapporto `01-PLAN-CHECK.md`; tutti i requisiti Pending e nessun test/inferenza del prodotto eseguiti.
Resume file: None

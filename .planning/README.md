# Pianificazione GSD — motore GPU macOS

**Stato al 12 ottobre 2026:** roadmap di 11 fasi, 39 requisiti mappati, prima fase pianificata con 5 piani/11 task/4 wave. Il checker indipendente ha dato PASS dopo una revisione, senza blocker o warning residui. Nessuna fase implementata; nessun benchmark o nuova inferenza eseguiti.

## Documenti principali

- [Progetto e decisioni](PROJECT.md)
- [Roadmap completa](ROADMAP.md)
- [Requisiti e traceability](REQUIREMENTS.md)
- [Contratto di placement GPU](GPU-PLACEMENT.md)
- [Stato e prossima azione](STATE.md)
- [Sintesi della ricerca](research/SUMMARY.md)
- [Architettura nativa proposta](research/ARCHITECTURE.md)
- [Rischi tecnici e gate](research/PITFALLS.md)
- [Mappa del codice esistente](codebase/ARCHITECTURE.md)

## Prima fase: baseline, contratti e prove GPU

| Piano | Wave | Consegna prevista |
|-------|------|------------------|
| [01-01](phases/01-baseline-contratti-e-prove-gpu/01-01-PLAN.md) | 1 | Identità, manifest sanitizzati e contratti del motore corrente |
| [01-02](phases/01-baseline-contratti-e-prove-gpu/01-02-PLAN.md) | 1 | Corpus autorizzato, annotazioni e matrice di copertura |
| [01-03](phases/01-baseline-contratti-e-prove-gpu/01-03-PLAN.md) | 2 | Provisioning privato e confronto worker originale/adapter |
| [01-04](phases/01-baseline-contratti-e-prove-gpu/01-04-PLAN.md) | 3 | Profiling sincronizzato, trace e ledger del placement |
| [01-05](phases/01-baseline-contratti-e-prove-gpu/01-05-PLAN.md) | 4 | Metriche, calibrazione reale e freeze verificabile |

Riferimenti: [contesto](phases/01-baseline-contratti-e-prove-gpu/01-CONTEXT.md), [ricerca implementativa](phases/01-baseline-contratti-e-prove-gpu/01-RESEARCH.md), [pattern di riuso](phases/01-baseline-contratti-e-prove-gpu/01-PATTERNS.md), [validazione prevista](phases/01-baseline-contratti-e-prove-gpu/01-VALIDATION.md), [revisione PASS](phases/01-baseline-contratti-e-prove-gpu/01-PLAN-CHECK.md).

## Confini della consegna

Il piano comprende la diarizzazione completa su GPU, inclusi feature, modelli, AHC/PLDA/VBx, assegnazione vincolata e attribuzione alle parole. Le fasi successive avranno task dettagliati dopo la baseline; non sono dichiarate già pianificate a livello esecutivo.

Modelli gated, corpus naturale annotato, trace e soglie misurate sono precondizioni/task di esecuzione, attualmente pending. Il PASS del checker riguarda il piano, non prova disponibilità, qualità o prestazioni. Python rimane oracle separato nello sviluppo e non viene ritirato dal bundle prima della copertura completa.

Prossimo comando GSD: `$gsd-execute-phase 1`. Questa sessione autorizza soltanto pianificazione; non è stato eseguito. Nessun commit/push/tag o cambio di branch è consentito, anche durante i workflow futuri. `AGENTS.md` prevale sui passi Git standard di GSD e `.planning/config.json` mantiene `commit_docs: false` e `auto_advance: false`.

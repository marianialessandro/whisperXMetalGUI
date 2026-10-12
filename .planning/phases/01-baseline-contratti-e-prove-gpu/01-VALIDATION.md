---
phase: 01
slug: baseline-contratti-e-prove-gpu
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-10-12
---

# Phase 01 — Validation Strategy (draft)

Piano di validazione, non evidenza di esecuzione. BASE-01..04 restano Pending. Task IDs, file e sampling sono mappati qui; la coerenza della mappa pianificata non equivale a Nyquist eseguito. Nessun test, inferenza, download o trace eseguito durante ricerca o pianificazione. nyquist_compliant e wave_0_complete restano false.

## Test Infrastructure

| Property | Value |
|---|---|
| Framework | unittest stdlib Python 3.12 bundled; Swift Testing esistente |
| Config | nuovi test in tests/baseline/ da creare in Wave 0 |
| Quick run | `macOS/output/installer/WhisperX.app/Contents/Runtime/bin/python3 -I -B -m unittest discover -s tests/baseline -p 'test_manifest.py' -v` |
| Full baseline suite | `macOS/output/installer/WhisperX.app/Contents/Runtime/bin/python3 -I -B -m unittest discover -s tests/baseline -p 'test_*.py' -v` |
| App contracts | `swift test --package-path macOS` |
| Expected runtime | quick unit suite progettata sotto 30s; GPU/corpus e Swift full non promettono questo limite |

Con -I i test devono aggiungere il repo root calcolato da __file__ per importare tools/baseline. Niente uv, download automatici o setup globale. Disponibilità del percorso runtime non attesta import/inferenza riusciti.

## Sampling Rate

- Per task: suite rapida pertinente; nessun commit, come richiesto da AGENTS.md.
- Per wave: suite baseline completa e contratti Swift pertinenti.
- Gate finale: corpus reale, oracle non instrumented vs adapter, metriche/freeze e trace manuali; tutte le precondizioni soddisfatte.
- Non usare fake clock/backend, sinusoid o mock come prova WER/DER o inferenza GPU.

## Per-Requirement Validation Map

| Req | Test Type | Planned Command / Evidence | File Exists | Status |
|---|---|---|---|---|
| BASE-01 | unit | unittest discover con test_manifest.py: fake token leakage, dirty digest, runtime identity | no — Wave 0 | pending |
| BASE-02 | unit/contract | test_corpus.py + test_contracts.py: missing license/case blocks, IPC/schema/export | no — Wave 0 | pending |
| BASE-03 | unit | test_profiling.py: drain ordering/error, required operator senza trace non passa | no — Wave 0 | pending |
| BASE-04 | unit | test_metrics.py + test_freeze.py: edits analitici, permutation/overlap, coverage denominator e freeze mutation | no — Wave 0 | pending |
| BASE-01..04 | integration | tools/baseline/run.py preflight/run/evaluate/freeze, CLI definitiva nel piano | no — Wave 0 | pending |
| BASE-03 | manual | Instruments Metal System Trace / capture MLX-MPS associata a run ID e fase | artifact missing | pending |

## Per-Task Validation Map

Prefisso comando U: `macOS/output/installer/WhisperX.app/Contents/Runtime/bin/python3 -I -B -m unittest discover -s tests/baseline`. Ogni comando completo aggiunge `-p 'FILE' -v`; ogni test calcola il repo root da `__file__` per il bootstrap isolato. Tutti i file nuovi indicati sono da creare durante esecuzione; nessun modello è necessario ai test unit.

| Task ID | Plan / Wave | Req | Automated sampling | Actual integration / manual evidence | Status |
|---|---|---|---|---|---|
| 01-01-01 | 01 / 1 | BASE-01 | U -p 'test_manifest.py' -v | Selected source/dirty digest, loaded runtime and actual model/hash inventory | pending |
| 01-01-02 | 01 / 1 | BASE-02 | U -p 'test_contracts.py' -v | Actual worker result/events/export; isolated existing Swift library/Keychain/cancel scenarios | pending |
| 01-02-01 | 02 / 1 | BASE-02 | U -p 'test_corpus.py' -v | Independent expected case/language-family ledger and license-gated acquisition | pending |
| 01-02-02 | 02 / 1 | BASE-02 | U -p 'test_corpus.py' -v | Actual licensed natural files, independent IT/EN turns/words/overlap/UEM, adjudication and split provenance | pending |
| 01-03-01 | 03 / 2 | BASE-01/02 | U -p 'test_provision.py' -v | Private current Keychain pipe, actual authorized local revisions/config/formats, separate provision | pending |
| 01-03-02 | 03 / 2 | BASE-01/02 | U -p 'test_adapter.py' -v | Original selected GUI worker vs instrumented adapter on every mandatory case; equality and offline preflight | pending |
| 01-04-01 | 04 / 3 | BASE-03 | U -p 'test_profiling.py' -v | Actual lazy-eval/MLX stream/MPS drains, phase timing/domain memory/copy states | pending |
| 01-04-02 | 04 / 3 | BASE-03 | U -p 'test_profiling.py' -v | Local actual trace review/PID/run/phase/operator hashes, separate diagnostic dumps and cold/warm replicas | pending |
| 01-05-01 | 05 / 4 | BASE-04 | U -p 'test_metrics.py' -v | Natural quality vs independent labels under fixed normalization/UEM/overlap/mapping | pending |
| 01-05-02 | 05 / 4 | BASE-01..04 | U -p 'test_freeze.py' -v | Missing mandatory case/operator/trace negative gates, immutable hashes and measured calibration state | pending |
| 01-05-03 | 05 / 4 | BASE-01..04 | U -p 'test_freeze.py' -v | Actual calibration/evaluation runs, concrete numeric thresholds/budgets, freeze/verify-freeze/full-gate | pending |

Sampling: each task runs its listed fast suite, each wave runs U -p 'test_*.py' -v, and final execution additionally uses the real offline full-gate. Each small unit suite targets <60s; full real corpus/calibration and trace capture have no artificial fast-time assertion. No uv, install or network setup. Commands are planned, not executed.

Dependency/file ownership: 01 and 02 share no modified files and run in wave 1; 03 consumes their APIs/assets in wave 2; 04 changes adapter only after 03 in wave 3; 05 changes run.py and docs/BASELINE_GPU.md only after 03/04 in wave 4. Required schemas are defined by their owning prior plans, without false parallel writes.

## Wave 0 Requirements

- [ ] tools/baseline/run.py, adapter nonproduction, manifest/contracts/corpus/metrics/profiling/freeze/provision e credential_bridge.swift, creati dai task mappati.
- [ ] tests/baseline/ e bootstrap runtime; fixture sintetiche/versionabili prive di media personali.
- [ ] evaluation contract e coverage manifest con case IDs obbligatori indipendenti dai casi raccolti.
- [ ] Provisioning esplicito di modelli autorizzati mancanti, licenze corpus e annotazioni naturali IT/EN; stato prerequisite_missing fail-closed.
- [ ] Credenziale gated dal Portachiavi tramite bridge privato di sviluppo/stdin, mai argv/env/file o token request raw.

Wave 0 indica infrastruttura iniziale da costruire nel RED dei task proprietari, non una wave già completata: 01-01-01/02 creano manifest/contracts tests; 01-02-01 corpus tests; 01-03-01/02 provision/adapter tests; 01-04-01 profiling tests; 01-05-01/02 metrics/freeze tests. Non avviare task, provisioning, misure o test dalla sola pianificazione.

## Mandatory Multi-Source Coverage Audit

CONTEXT non contiene decisioni D-NN: si riportano i gruppi e le singole capacità bloccate sotto i loro titoli canonici, senza inventare ID decisionali. COVERED significa task previsto, mai requisito eseguito.

| Source | Item / constraint | Plan / task | Planned status |
|---|---|---|---|
| GOAL | Baseline riproducibile e criteri congelati per confrontare risultati/costi futuri | 01..05 | COVERED |
| REQ | BASE-01 versioni/sorgenti/modelli/hardware/settings senza segreti | 01-01-01, 01-03-01/02 | COVERED |
| REQ | BASE-02 IT/EN breve/lungo/silenzio/overlap/auto-esatto/famiglie e corpus senza dati personali | 01-01-02, 01-02-01/02, 01-03-02 | COVERED |
| REQ | BASE-03 sincronizzazione/memoria/copie/placement e download/cold/warm distinti | 01-04-01/02 | COVERED |
| REQ | BASE-04 WER/CER/timestamp/DER/JER/count/mapping e calibrazione/freeze reale | 01-05-01/02/03 | COVERED |
| RESEARCH | GUI checkout/bundle espliciti, hash dirty/importati e SHA whisper.cpp completo read-only | 01-01-01, 01-03-02 | COVERED |
| RESEARCH | No nuovi package; bundled stdlib unittest -I bootstrap e FFmpeg reale | Tutti verify; 01-01-01, 01-03-02 | COVERED |
| RESEARCH | Formati/config/pesi/PLDA e proof locale autorizzato, conversione produttiva fase 2 | 01-03-01 | COVERED |
| RESEARCH | FLEURS mirato, AMI subordinato a licenza, IT naturale annotato e adjudication indipendente | 01-02-01/02 | COVERED |
| RESEARCH | Required cases indipendenti, lingue/mapping con represented/untested, derivati distinti | 01-02-01/02 | COVERED |
| RESEARCH | Provision separato, modelli/accesso non provati, prerequisite_missing fail-closed | 01-03-01/02, 01-05-03 | COVERED |
| RESEARCH | Hook scoped, original worker equality, dump intermedi opt-in/intermediate_unavailable | 01-03-02, 01-04-01/02 | COVERED |
| RESEARCH | MLX lazy/nondefault drain, MPS drain, memory domains, logical/physical bytes, trace locale | 01-04-01/02 | COVERED |
| RESEARCH | Cold-process/warm-cache non warm-model, diagnostic run separata e controlli thermal/power/seed | 01-04-02 | COVERED |
| RESEARCH | Strata/source-split, recording bootstrap, finite valori reali e nuove identità dopo revisione | 01-05-01/02/03 | COVERED |
| CONTEXT | GPU portabile obiettivo; residui CPU e operatore scoperto impediscono complete-GPU claim | 01-04-01/02, 01-05-02 | COVERED |
| CONTEXT | community-1, overlap, voci auto/esatte, profili completi e qualità modello identico | 01-01-02, 01-02-01, 01-03-01/02 | COVERED |
| CONTEXT | result/schema/events/export/language/word/speaker/library/Keychain/errore/cancel preservati | 01-01-02, 01-03-01/02 | COVERED |
| CONTEXT | Synthetic non accuratezza, oracle non ground truth, no media/catalogo personali | 01-02-01/02, 01-05-01/03 | COVERED |
| CONTEXT | Token solo Keychain/private stdin, no argv/env/raw payload/log/fixture | 01-01-01, 01-03-01/02, 01-05-02 | COVERED |
| CONTEXT | Freeze prima promozione; niente soglie/speedup inventati o retuning candidato | 01-05-01/02/03 | COVERED |
| CONTEXT | No commit/push/tag/branch/worktree/source overwrite, no production/native core in fase 1 | Tutti execution_context/output e file ownership | COVERED |

Esclusioni deliberate di scope: core/helper nativi, porting feature/modelli/kernel, wiring app/installer e conversione produttiva sono delle fasi 2..11; non sono gap della Fase 1. No redesign o nuova approvazione architetturale. Mancanze operative di corpus/modelli/annotazioni/trace restano pending e bloccano full-gate, anche se l'audit di pianificazione è coperto.

## Manual-Only Validations

| Behavior | Why Manual | Instructions |
|---|---|---|
| GPU placement reale per ASR/align/diar | API available non dimostra dispatch | correlare PID/run ID/spans con trace; annotare CPU/readback/unknown e hash artefatto |
| Ground truth tempi/turns/overlap | oracle automatico non è annotazione | revisionare corpus licenziato, annotazioni e UEM con adjudication documentata |
| Token/errore/cancellazione e libreria | preserve flussi prodotto | solo fixture temporanee e credenziali via percorso corrente, non catalogo/media personali |

## Validation Sign-Off

- [ ] Tutti task mappati ad automated/manual evidence e Wave 0 concreta.
- [ ] Pesi, accesso, licenze e annotazioni realmente disponibili; nessun missing case mascherato.
- [ ] Download, cold-process, warm-cache e diagnostic trace distinti.
- [ ] Margini misurati/calibrati, corpus/policies congelati e freeze hash verificato.
- [ ] Nyquist compliant soltanto dopo esecuzione/evidenze; oggi draft/pending.

**Approval:** pending; nessuna dichiarazione di qualità, prestazione o placement certificato.

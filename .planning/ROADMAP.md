# Roadmap: WhisperX macOS — motore nativo GPU

**Milestone:** v1 — migrazione numerica GPU Apple. **Data:** 2026-10-12. **Granularity:** fine. **Stato:** bozza di pianificazione, nessuna implementazione completata.

## Overview

L'utente continua a trascrivere, allineare parole e distinguere voci nell'app attuale, mentre ogni stadio numerico portabile passa a un motore C++/Objective-C++/Metal. La consegna finale conserva community-1, tutti i profili ASR e tutte le lingue supportate, senza Python produttivo, downgrade di qualità o fallback numerici CPU nascosti.

Riferimenti canonici: [PROJECT.md](PROJECT.md), [REQUIREMENTS.md](REQUIREMENTS.md), [GPU-PLACEMENT.md](GPU-PLACEMENT.md), [sintesi ricerca](research/SUMMARY.md), [architettura](research/ARCHITECTURE.md), [stack](research/STACK.md), [rischi](research/PITFALLS.md), `AGENTS.md` e `docs/PROGETTO_APP_MACOS.md`. In caso di conflitto prevalgono le istruzioni recenti dell'utente e del repository. Il checkout `/Users/marianialessandro/repository/whisper.cpp` a `21411d81` è un riferimento read-only: fissare SHA completo in fase 1, nessun aggiornamento upstream implicito né dipendenza dal percorso personale nelle build.

## Phases

- [ ] **Phase 1: Baseline, contratti e prove GPU** - Corpus, oracle e gate congelati per confronti riproducibili.
- [ ] **Phase 2: Modelli e accesso Hugging Face nativi** - Download e conversione individuali fattibili senza Python produttivo.
- [ ] **Phase 3: Core Metal e helper nativo** - Job isolati, risorse a budget e backend realmente osservabile.
- [ ] **Phase 4: Audio e feature GPU a memoria limitata** - DSP equivalente su GPU per registrazioni brevi e lunghe.
- [ ] **Phase 5: Trascrizione Whisper e selezione token GPU** - Profili completi e decoding numerico GPU.
- [ ] **Phase 6: Tempi parola GPU in italiano e inglese** - Wav2Vec2, CTC, backtracking e riduzioni senza emissioni host.
- [ ] **Phase 7: Allineamento multilingue completo** - Tutte le famiglie e lingue della baseline restano utilizzabili.
- [ ] **Phase 8: Segmentazione ed embedding delle voci GPU** - Modelli community-1 equivalenti senza feature CPU.
- [ ] **Phase 9: Diarizzazione e attribuzione delle voci GPU** - AHC, PLDA, VBx, vincoli e attribuzione parole completi.
- [ ] **Phase 10: App e libreria compatibili con il motore nativo** - Flussi, dati, player e formati mantenuti.
- [ ] **Phase 11: Installer autonomo e certificazione finale** - Release rilocabile/offline senza Python, con prove complete.

## Gate comuni

- Prima di promuovere un backend, passare contratti, placement, qualità e risorse definiti in fase 1; non ricalibrare le soglie per far passare il candidato. Benchmark cold/warm separati da primo download; nessuno speedup promesso.
- Ogni operazione di GPU-PLACEMENT.md ha evidenze distinte `implementation`, `dispatch_observed`, `parity_passed`, `resource_limits_passed`. Un flag GPU, una build o un trace parziale non certificano l'intera fase.
- CPU per controllo, codec/I/O, stringhe, UI, Keychain ed export resta dichiarata. FFT, DP, selezione logits, AHC/VBx e intersezioni numeriche restano nello scope GPU anche quando seriali. Un residuo numerico portabile CPU lascia aperto il requisito, senza eccezioni nascoste.
- Python resta oracle separato durante lo sviluppo, mai fallback automatico del nuovo motore. Ritiro del runtime dal bundle soltanto dopo copertura modelli, lingue, formati, qualità e installer; il pacchetto/API/CLI legacy non viene eliminato.
- Nessun commit, push, tag, branch o worktree automatico: `config.json` deve mantenere `commit_docs: false`, `auto_advance: false` e strategia Git `none`. Pianificare non autorizza l'esecuzione.

## Phase Details

### Phase 1: Baseline, contratti e prove GPU

**Goal:** L'utente può confrontare risultati e costi della pipeline corrente con futuri backend mediante una baseline riproducibile e criteri congelati.
**Depends on:** Nothing (first phase)
**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04
**Success Criteria**:
1. Una run di riferimento produce manifest senza segreti con hardware, memoria, versioni, SHA completo whisper.cpp, modelli/hash, configurazioni e schema/eventi/export correnti; un'altra run può riprodurne le impostazioni. [BASE-01]
2. Il corpus autorizzato e versionabile copre IT/EN, brevi/lunghi, silenzi, overlap, conteggio automatico/esatto e lingue/famiglie rappresentative; annotazioni, licenze e casi di sola parità sono distinguibili. [BASE-02]
3. Per ciascuna fase sono disponibili tempi sincronizzati cold/warm, memoria, copie/readback e attese, con evidenza del lavoro CPU/GPU e primo download separato; il ledger rileva anche un operatore volutamente non coperto. [BASE-03]
4. Prima di qualsiasi promozione sono congelati corpus, policy overlap/collar/permutazione e margini WER/CER, timestamp, DER/JER e speaker count, con criterio di revisione esplicito e nessuna accuratezza dedotta da oracle non annotato. [BASE-04]
**Plans:** 5 plans

Plans:
- [ ] 01-01-PLAN.md — Manifest sanitizzati e contratti brownfield (wave 1).
- [ ] 01-02-PLAN.md — Corpus licenziato, annotazioni e copertura indipendente (wave 1).
- [ ] 01-03-PLAN.md — Provisioning privato e parità worker originale/adapter offline (wave 2).
- [ ] 01-04-PLAN.md — Misure sincronizzate e trace/ledger di placement (wave 3).
- [ ] 01-05-PLAN.md — Metriche, calibrazione reale e freeze immutabile fail-closed (wave 4).

**Scope boundary**: Baseline eseguibile, fixture dei contratti e framework di raccolta prove; non implementare in questa fase l'intero core o i porting. Identificare formati/config/licenze dei checkpoint reali e un proof-of-conversion minimo autorizzato come ricognizione di fattibilità. La soluzione produttiva download/conversione è responsabilità della fase 2; un formato irrisolto è una lacuna dichiarata, non una promessa.
**References**: GPU-PLACEMENT.md; research/SUMMARY.md, gate di promozione; research/PITFALLS.md, baseline/conversione; macOS/Worker/transcribe_worker.py, whisperx/alignment.py e whisperx/diarize.py come oracle da ricertificare.

### Phase 2: Modelli e accesso Hugging Face nativi

**Goal:** L'utente può ottenere i modelli autorizzati dalle Impostazioni Hugging Face e prepararli per il motore nativo senza installare Python.
**Depends on:** Phase 1
**Requirements:** DIAR-01, APP-03
**Success Criteria**:
1. Con token personale e condizioni accettate, il percorso nativo scarica una revisione verificata di community-1 e sotto-modelli/config/PLDA reali; manifest e hash identificano esattamente gli artefatti senza credenziali. [DIAR-01]
2. Un prototipo eseguibile sul target prova lettura/conversione dei formati effettivi necessari alle famiglie neurali previste, senza Python produttivo, pickle generico assunto o hosting proprietario; copertura non provata blocca i porting dipendenti. [DIAR-01]
3. Autorizzazione negata, token invalido, rete interrotta, cancellazione e hash errato producono esiti recuperabili nelle sole Impostazioni Hugging Face; token nel Portachiavi, mai in log, ambiente, argomenti o artefatti. [APP-03]
4. Download verificato riprendibile e successivo uso dei file locali funzionano senza redistribuzione non autorizzata o token condiviso; il helper d'inferenza riceve percorsi locali e non credenziali. [DIAR-01, APP-03]
**Plans:** TBD
**UI hint**: yes
**Scope boundary**: Nessun redesign o nuova superficie nella finestra principale. Stabilire contratto loader/converter e inventario formati per ASR e tutte le famiglie alignment; conversioni e parità specifiche restano ai loro requisiti nelle fasi 5–8. Python di sviluppo può essere oracle, non requisito sul Mac dell'utente.
**References**: research/STACK.md, persistenza/distribuzione/credenziali; research/PITFALLS.md, conversione; PROJECT.md, vincoli Hugging Face.

### Phase 3: Core Metal e helper nativo

**Goal:** Un job nativo isolato può essere avviato, osservato e cancellato con memoria e ownership controllate e senza alterare la CLI legacy.
**Depends on:** Phase 1, Phase 2
**Requirements:** CORE-01, CORE-02, CORE-03, CORE-04, CORE-05
**Success Criteria**:
1. Un client di prova usa ABI C versionata con risultati/errori deterministici; ownership e lifetime sono verificabili e nessuna eccezione attraversa il confine. [CORE-01]
2. Il helper gestisce eventi e cancellazione compatibili con il contratto congelato, isola crash/errori e libera risorse solo dopo completamento dei command buffer, senza esporre credenziali. [CORE-02]
3. Run ripetute e cancellazioni rispettano il budget dichiarato, riusano buffer/pipeline/modelli quando consentito e rendono osservabili copie, fence, layout e sincronizzazioni fra GGML, MPSGraph e Metal. [CORE-03]
4. Il reporting distingue device disponibile e dispatch realmente eseguito; un operatore non coperto genera errore/residuo esplicito e non richiama CPU o worker Python automaticamente. [CORE-05]
5. I comandi/API legacy mantengono argomenti posizionali, flag e comportamenti verificati; eventuali nuove opzioni sono additive e documentate in --help. [CORE-04]
**Plans:** TBD
**References**: research/ARCHITECTURE.md, ABI/helper/memoria; research/PITFALLS.md, interop e scheduler CPU; GPU-PLACEMENT.md.

### Phase 4: Audio e feature GPU a memoria limitata

**Goal:** Registrazioni brevi e lunghe producono input numerici equivalenti ai modelli con DSP su GPU e memoria limitata.
**Depends on:** Phase 2, Phase 3
**Requirements:** FEAT-01, FEAT-02, FEAT-03
**Success Criteria**:
1. Finestre/padding, STFT/FFT, mel/filterbank, log e normalizzazioni Whisper/WeSpeaker hanno parità entro tolleranze congelate e dispatch GPU, compresi silenzio e bordi. [FEAT-01]
2. Downmix/resampling del core sono GPU e conservano sample clock, durata e offset; costo e backend della decodifica codec host sono riportati separatamente. [FEAT-02]
3. Gli stessi audio ripartiti in chunk/batch diversi rispettano parità di feature e memoria massima, mantenendo statistiche globali, contesto/halo e normalizzazioni; nessun troncamento implicito dei file lunghi. [FEAT-03]
**Plans:** TBD
**References**: research/PITFALLS.md, feature streaming; research/ARCHITECTURE.md, flusso/memoria; GPU-PLACEMENT.md, DSP.

### Phase 5: Trascrizione Whisper e selezione token GPU

**Goal:** L'utente ottiene trascrizioni equivalenti con i tre profili e il calcolo numerico ASR, inclusa selezione token, sulla GPU.
**Depends on:** Phase 2, Phase 3, Phase 4
**Requirements:** ASR-01, ASR-02, ASR-03, ASR-04
**Success Criteria**:
1. La build usa whisper.cpp/GGML pinned con SHA completo e licenze, funziona senza checkout personale e registra operatori/backend effettivi invece del solo use_gpu. [ASR-01]
2. base/small/large-v3 conservano lingua automatica/esplicita, hints e policy decoding; Massima usa large-v3 completo non quantizzato e passa i gate congelati contro MLX. [ASR-02]
3. Cache/batch e confini delle finestre preservano testo, segmenti e timestamp su audio breve/lungo, con encoder/decoder GPU osservati e nessun fallback numerico nello scheduler. [ASR-03]
4. Mascheramento logits, normalizzazione, top-k/selezione e parti numeriche del beam applicabile sono GPU; trace/readback provano lettura host di soli risultati compatti senza cambiare la politica di decoding. [ASR-04]
**Plans:** TBD
**References**: research/STACK.md, profili ASR; research/PITFALLS.md, Flash Attention/DTW e beam; whisper.cpp read-only 21411d81. CTC rimane il riferimento per tempi parola, non sostituito dal DTW token.

### Phase 6: Tempi parola GPU in italiano e inglese

**Goal:** Trascrizioni italiane e inglesi ottengono tempi parola e punteggi equivalenti tramite una catena alignment interamente GPU.
**Depends on:** Phase 2, Phase 3, Phase 4, Phase 5
**Requirements:** ALIGN-01, ALIGN-03, ALIGN-04, ALIGN-05, ALIGN-06
**Success Criteria**:
1. Gli stessi pesi/vocabolari dei bundle IT e primo EN producono inferenza batch/log-softmax GPU con padding, maschere, lunghezze e frame-time equivalenti, senza emission roundtrip CPU. [ALIGN-01, ALIGN-03]
2. Trellis CTC FP32 conserva blank/wildcard, bordi, endpoint e tie-breaking su casi sintetici e reali; segmenti piccoli/grandi non richiedono un dispatch host per frame o barriere globali inesistenti. [ALIGN-04]
3. Backtracking e riduzioni caratteri/parole sono GPU e restituiscono soltanto tempi/score/percorso compatti, con gate timestamp/coverage superati e memoria entro budget. [ALIGN-05]
4. Segmenti non allineabili, simboli ignoti, audio fuori range e modello assente producono gli esiti espliciti del contratto corrente senza tempi invalidi o fallback Python. [ALIGN-06]
**Plans:** TBD
**Scope boundary**: IT/EN sono il primo slice, non copertura finale. Lo stesso engine batch sarà verificato su tutte le famiglie nella fase 7.
**References**: whisperx/alignment.py come semantica oracle; research/ARCHITECTURE.md, kernel CTC; research/PITFALLS.md, norm/padding e DP.

### Phase 7: Allineamento multilingue completo

**Goal:** Lingua automatica e ogni lingua/famiglia alignment della baseline continuano a fornire i risultati previsti tramite il percorso nativo GPU.
**Depends on:** Phase 2, Phase 6
**Requirements:** ALIGN-02
**Success Criteria**:
1. Una matrice esaustiva collega ogni mapping corrente a pesi/revisione, vocabolario/tokenizer, blank, normalizzazione, formato/converter, licenza e operazioni GPU, incluse famiglie torchaudio, HF XLSR/XLS-R e modelli 1B. [ALIGN-02]
2. Ogni mapping supportato supera i gate congelati della propria famiglia/lingua e ha prova caricamento/conversione senza Python produttivo; script senza spazi, simboli e rilevamento auto conservano il contratto. [ALIGN-02]
3. Modello realmente assente/non disponibile mantiene un esito esplicito equivalente alla baseline; nessuna lingua precedentemente supportata viene persa o coperta con allineamento CPU nascosto, e lingue/famiglie non certificate impediscono il ritiro del runtime. [ALIGN-02]
**Plans:** TBD
**Scope boundary**: Una sola voce requisito rappresenta numerose famiglie e capacità utente; la separazione evita di dichiarare completa la migrazione globale dopo due modelli. Nessun nuovo modello sostitutivo per nascondere un porting mancante.
**References**: whisperx/alignment.py, mapping correnti; research/ARCHITECTURE.md, copertura lingue; research/PITFALLS.md, eliminazione prematura Python.

### Phase 8: Segmentazione ed embedding delle voci GPU

**Goal:** community-1 individua attività vocale e rappresentazioni delle voci equivalenti eseguendo modelli e feature sulla GPU.
**Depends on:** Phase 2, Phase 3, Phase 4
**Requirements:** DIAR-02, DIAR-03
**Success Criteria**:
1. Segmentazione con pesi/config effettivi conserva output layerwise/finale, batching, convoluzioni, componenti ricorrenti, powerset e postprocessing, con copertura operatori e trace GPU. [DIAR-02]
2. Embedding, pooling e maschere usano feature GPU validate e superano confronti su silenzi, finestre brevi e overlap senza FFT host, conversioni NumPy o risultati numerici intermedi CPU. [DIAR-03]
3. Run lunghe/batch diversi e cancellazioni ripetute rispettano budget, offset e margini congelati; stride, soglie o precisione non sono alterati per ottenere throughput. [DIAR-02, DIAR-03]
**Plans:** TBD
**Scope boundary**: È uno stadio verificabile di un job diagnostico; non dichiarare diarizzazione completa prima della fase 9. Pesi e hyperparameter provengono dai checkpoint verificati nella fase 2, non dai default delle classi.
**References**: research/ARCHITECTURE.md, diarizzazione; research/PITFALLS.md, recurrent kernels/overlap; GPU-PLACEMENT.md.

### Phase 9: Diarizzazione e attribuzione delle voci GPU

**Goal:** L'utente ottiene numero di voci automatico/esatto, sovrapposizioni e attribuzione a segmenti/parole equivalenti con tutta la catena numerica su GPU.
**Depends on:** Phase 6, Phase 8
**Requirements:** DIAR-04, CLUST-01, CLUST-02, CLUST-03, CLUST-04, DIAR-05
**Success Criteria**:
1. Normalizzazione embedding, distanze e trasformazioni PLDA sono GPU e rispettano i margini numerici congelati; precisione rispetto alla baseline FP64 è valutata nei casi vicini alle soglie. [CLUST-01]
2. AHC centroid/euclidean riproduce dendrogramma, inversioni, tie e taglio fcluster con min/update GPU e strategia di memoria verificata; nessuna sostituzione average/ward o arresto alla prima soglia. [CLUST-02]
3. VBx della variante locale, centroidi e ramo K-means del conteggio esatto sono GPU con inizializzazione, stopping e cutoff riproducibili; conteggio automatico non ha un massimo arbitrario e quello esatto conserva semantica corrente. [CLUST-03, DIAR-04]
4. Assegnazione vincolata e aggregazione finestre/count sono GPU e conservano traccia regolare con overlap, vincoli e policy, passando DER/JER e speaker count dopo permutazione delle label. [CLUST-04, DIAR-04]
5. Intersezioni/riduzioni GPU assegnano la voce dominante a segmenti e parole con missing/tie/fill_nearest legacy; word-speaker accuracy e risultato compatto finale passano i gate, senza NumPy roundtrip. [DIAR-05]
**Plans:** TBD
**Scope boundary**: Nessuna eccezione CPU per ricorrenze seriali AHC/VBx/assegnazione. La generalità dell'algoritmo vincolato dipende dai checkpoint, non da un limite inventato alle voci. Ricertificare attribuzione sul timing multilingue prima della release.
**References**: research/PITFALLS.md, AHC centroid/VBx; research/ARCHITECTURE.md, constrained assignment e speaker-word; GPU-PLACEMENT.md.

### Phase 10: App e libreria compatibili con il motore nativo

**Goal:** L'utente usa il nuovo motore dai flussi attuali senza perdere dati, formati o controllo dei file.
**Depends on:** Phase 2, Phase 5, Phase 7, Phase 9
**Requirements:** APP-01, APP-02, APP-04
**Success Criteria**:
1. L'app lancia il helper dal bundle senza Terminale, installazioni manuali o percorsi personali; UI IT/EN, titolo/player nella toolbar e menu/shortcut correnti funzionano, con eventi e cancellazione del nuovo motore. [APP-01]
2. Librerie e trascrizioni preesistenti, revisioni e nomi personalizzati delle voci restano leggibili/modificabili; nuovi risultati ed export JSON/TXT/SRT/VTT rispettano i contratti senza migrazioni distruttive. [APP-02]
3. Importazione/file esterni e spostamento libreria sono verificati: dati gestiti trasferiti e catalogo precedente rimosso solo dopo verifica, nessun backup residuo o cancellazione di file non gestiti. [APP-04]
4. Un job completo multilingue con voci automatiche/esatte, overlap e audio lungo passa i gate congelati nell'app e mantiene le Impostazioni Hugging Face/Modelli/Libreria senza nuove superfici nella finestra principale. [APP-01, APP-02]
**Plans:** TBD
**UI hint**: yes
**References**: AGENTS.md; docs/PROGETTO_APP_MACOS.md; research/ARCHITECTURE.md, contratti app/writer; fixture fase 1. Conservare catalogo attuale, nessun passaggio SQLite implicito.

### Phase 11: Installer autonomo e certificazione finale

**Goal:** L'utente installa e avvia offline una build arm64 autonoma certificata, con copertura GPU e qualità documentate per tutte le capacità correnti.
**Depends on:** Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, Phase 8, Phase 9, Phase 10
**Requirements:** SHIP-01, SHIP-02, SHIP-03, SHIP-04
**Success Criteria**:
1. Installer arm64 riproducibile include helper, librerie native, metallib, tooling modelli e licenze con percorsi relativi; il percorso produttivo non avvia/richiede Python e non dipende da Homebrew o checkout personali. [SHIP-01]
2. La build rilocata si avvia con doppio click su Mac pulito o ambiente equivalente motivato; dopo download verificato completa job offline e distingue firma locale da notarizzazione pubblica non dimostrata. [SHIP-02]
3. Corpus congelato e matrice completa lingue/modelli/formati passano parità/qualità, conteggio/overlap/timestamp, memoria massima e cancellazione; benchmark cold/warm sincronizzati per fase/totale sono associati agli artefatti. [SHIP-03]
4. Il ledger finale collega ogni riga GPU a manifest, trace Metal, test e limiti risorse: nessun residuo numerico portabile CPU o fallback silenzioso viene dichiarato completo, anche con tempi totali migliori. [SHIP-04]
**Plans:** TBD
**Scope boundary**: Rimuovere Python dal bundle soltanto dopo i gate, preservando sorgenti/API legacy e oracle di sviluppo. Pubblicare il rapporto placement fra gli artefatti locali di consegna; non autorizza upload, Git push o pubblicazione del prodotto. Riconciliare macOS minimo prima di adottare API SDK, senza innalzarlo implicitamente.
**References**: GPU-PLACEMENT.md, check completamento; research/SUMMARY.md, release gates; research/STACK.md, packaging; research/PITFALLS.md, lingue/converter.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Baseline, contratti e prove GPU | 0/5 | Planned | - |
| 2. Modelli e accesso Hugging Face nativi | 0/TBD | Pending | - |
| 3. Core Metal e helper nativo | 0/TBD | Pending | - |
| 4. Audio e feature GPU a memoria limitata | 0/TBD | Pending | - |
| 5. Trascrizione Whisper e selezione token GPU | 0/TBD | Pending | - |
| 6. Tempi parola GPU in italiano e inglese | 0/TBD | Pending | - |
| 7. Allineamento multilingue completo | 0/TBD | Pending | - |
| 8. Segmentazione ed embedding delle voci GPU | 0/TBD | Pending | - |
| 9. Diarizzazione e attribuzione delle voci GPU | 0/TBD | Pending | - |
| 10. App e libreria compatibili con il motore nativo | 0/TBD | Pending | - |
| 11. Installer autonomo e certificazione finale | 0/TBD | Pending | - |

**Coverage:** 39/39 requisiti v1 assegnati una sola volta. Le verifiche trasversali non duplicano la responsabilità primaria. Nessun requisito numerico è rinviato implicitamente a v2. Directory del primo piano: `.planning/phases/01-baseline-contratti-e-prove-gpu/`.

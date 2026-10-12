# Requirements: WhisperX macOS — motore nativo GPU

**Defined:** 2026-10-12
**Core Value:** GPU Apple per la pipeline numerica con qualità, word timestamps e diarizzazione mantenuti.

## v1 Requirements

### Baseline e criteri di accettazione

- [ ] **BASE-01**: Registrare versioni, revisione sorgenti, modelli/hash, hardware, memoria e impostazioni della pipeline attuale in manifest riproducibili e senza credenziali.
- [ ] **BASE-02**: Costruire corpus e oracle per italiano, inglese, audio brevi/lunghi, silenzi, speaker automatici/esatti, sovrapposizioni e lingue rappresentative; fixture versionabili prive di registrazioni personali.
- [ ] **BASE-03**: Misurare per fase tempo sincronizzato, memoria, copie/sincronizzazioni e placement osservato GPU/CPU, distinguendo primo download, cold e warm run.
- [ ] **BASE-04**: Calibrare e congelare prima della promozione soglie WER/CER, errore timestamp, DER/JER con policy overlap, speaker-count e speaker-label permutation; i casi senza ground truth misurano parità, non accuratezza.

### Core e interoperabilità

- [ ] **CORE-01**: Costruire core C++/Objective-C++ con ABI C versionata, ownership esplicita, errori e risultati deterministici, senza eccezioni che attraversino l'ABI.
- [ ] **CORE-02**: Eseguire il core in helper nativo con request/event protocol compatibile, progresso e cancellazione; ricevere credenziali soltanto via canale privato.
- [ ] **CORE-03**: Riutilizzare buffer/pipeline Metal e modelli entro budget di memoria e sincronizzare solo quando necessario, con capability reporting basato anche su dispatch reali.
- [ ] **CORE-04**: Conservare API, flag e comportamenti legacy del pacchetto Python; introdurre opzioni soltanto additive dove necessarie.
- [ ] **CORE-05**: Dichiarare e verificare il backend effettivo per fase; operazioni numeriche non GPU non devono essere nascoste come accelerazione completa.

### ASR GPU

- [ ] **ASR-01**: Integrare revisione pinned e riproducibile di whisper.cpp/GGML Metal senza dipendenza da checkout o toolchain personali nel prodotto.
- [ ] **ASR-02**: Conservare profili base/small/large-v3 multilingue, lingua automatica/esplicita e hints; Massima usa large-v3 completo non quantizzato e mantiene semantics documentate di decoding.
- [ ] **ASR-03**: Dimostrare encoder/decoder GPU e corretta gestione cache/batch, timestamp e confini delle finestre con confronto MLX corrente.
- [ ] **ASR-04**: Portare su GPU masking, normalizzazione e selezione dei token/beam numerica, riducendo letture host dei logits senza cambiare politiche di decoding.

### Feature e audio GPU

- [ ] **FEAT-01**: Eseguire sulla GPU finestre, padding, STFT/FFT, mel/filterbank, log e normalizzazioni richieste da Whisper e speaker embedding, con parità delle convenzioni per modello.
- [ ] **FEAT-02**: Portare sulla GPU downmix e resampling numerico quando eseguiti dal nuovo core, lasciando esplicita la decodifica dei codec tramite librerie di sistema o FFmpeg.
- [ ] **FEAT-03**: Elaborare audio lungo con memoria limitata preservando contesto, offset e normalizzazioni globali; evitare che lo streaming cambi implicitamente il risultato.

### Allineamento GPU

- [ ] **ALIGN-01**: Convertire ed eseguire su GPU il modello Wav2Vec2 italiano con stessi pesi/vocabolario, padding, lunghezze e mapping frame-tempo.
- [ ] **ALIGN-02**: Estendere il backend alla matrice delle lingue esposte dall'app e ai modelli supportati dalla baseline; non dichiarare la migrazione finale completa con lingue regredite o fallback nascosti.
- [ ] **ALIGN-03**: Eseguire inferenza alignment e log-softmax GPU in batch con lunghezze/maschere corrette e senza CPU emission roundtrip.
- [ ] **ALIGN-04**: Implementare trellis CTC GPU, blank/wildcard, inizializzazione e tie-breaking equivalenti; supportare segmenti piccoli e grandi senza dispatch host per frame.
- [ ] **ALIGN-05**: Eseguire backtracking e riduzioni caratteri/parole sulla GPU e riportare al lato host soltanto risultati compatti; confrontare timestamp e punteggi.
- [ ] **ALIGN-06**: Gestire segmenti non allineabili, caratteri ignoti, audio fuori range e assenza di modello con esiti espliciti equivalenti al contratto di risultato.

### Diarizzazione e clustering GPU

- [ ] **DIAR-01**: Convertire in modo riproducibile modelli e parametri community-1 con download individuali autorizzati, hash e manifest senza credenziali.
- [ ] **DIAR-02**: Eseguire segmentazione speaker GPU, comprese convoluzioni/componenti ricorrenti effettive, batching e powerset/postprocessing richiesto dai pesi.
- [ ] **DIAR-03**: Eseguire speaker embedding e maschere GPU con feature GPU, senza FFT CPU o NumPy roundtrip nel nuovo percorso.
- [ ] **DIAR-04**: Conservare numero di voci automatico o esatto, intervalli sovrapposti e convenzioni del conteggio; non cambiare stride/soglie per accelerare senza validazione.
- [ ] **CLUST-01**: Eseguire normalizzazione embedding, distanze e trasformazioni PLDA su GPU.
- [ ] **CLUST-02**: Eseguire clustering gerarchico con semantica centroid linkage/soglie corrente su GPU, gestendo dipendenze seriali e memoria; ogni variante richiede confronto e decisione esplicita.
- [ ] **CLUST-03**: Eseguire VBx, centroidi e ramo K-means previsto da conteggio esatto sulla GPU, con stopping criterion e inizializzazione riproducibili.
- [ ] **CLUST-04**: Eseguire assegnazione vincolata e aggregazione delle finestre su GPU mantenendo vincoli speaker/overlap e label permutation equivalence.
- [ ] **DIAR-05**: Assegnare voci a segmenti e parole mediante calcoli di intersezione/riduzione GPU, preservando casi non assegnati e tie-breaking legacy.

### App e persistenza

- [ ] **APP-01**: Avviare il helper nativo dall'app senza Terminale, installazioni manuali o percorsi personali, preservando UI italiana/inglese, player e menu.
- [ ] **APP-02**: Conservare lettura/scrittura di libreria, trascrizioni precedenti, revisioni, nomi voci e formati JSON/TXT/SRT/VTT senza migrazioni distruttive.
- [ ] **APP-03**: Mantenere Hugging Face esclusivamente nelle Impostazioni, token nel Portachiavi e gestione errori/download/cancellazione senza esporre dati sensibili.
- [ ] **APP-04**: Conservare gestione file esterni e spostamento verificato della libreria senza backup residui o cancellazione di file non gestiti.

### Distribuzione e certificazione

- [ ] **SHIP-01**: Produrre installer autonomo arm64 con librerie native/Metal/model tooling necessario, licenze e manifest riproducibile; il percorso di produzione finale non richiede Python.
- [ ] **SHIP-02**: Validare app rilocata e avvio offline con modelli già scaricati su Mac pulito o ambiente esplicitamente equivalente; distinguere firma locale da notarizzazione pubblica.
- [ ] **SHIP-03**: Certificare parità finale sul corpus congelato e benchmark cold/warm per fase con artefatti verificabili, memoria massima e assenza di fallback numerici silenziosi.
- [ ] **SHIP-04**: Pubblicare inventario finale di placement con trace Metal e residui CPU; calcoli numerici portabili ancora CPU restano requisiti aperti, anche se il lavoro totale è più veloce.

## v2 Requirements

Nessuna fase della pipeline numerica sopra è rinviata implicitamente a v2. Eventuali profili turbo/quantizzati aggiuntivi, supporto GPU non Apple e nuovi modelli potranno essere milestone successivi.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cloud, token condivisi, hosting proprietario | Elaborazione locale e credenziali individuali |
| Training o cambio di modello per nascondere limiti di porting | Conservare la qualità del prodotto |
| Riscrittura UI e nuove superfici Hugging Face | Obiettivo è il motore, UI vigente vincolante |
| Eliminazione della CLI/pacchetto Python legacy | Retrocompatibilità richiesta |
| Commit/push/tag automatici | Vietati da AGENTS.md |

## Traceability

Ogni requisito v1 appartiene a una sola fase di consegna. Le verifiche trasversali e finali non duplicano l'assegnazione primaria. Tutti i requisiti sono Pending: pianificazione e ricerca non costituiscono implementazione.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Pending |
| BASE-02 | Phase 1 | Pending |
| BASE-03 | Phase 1 | Pending |
| BASE-04 | Phase 1 | Pending |
| CORE-01 | Phase 3 | Pending |
| CORE-02 | Phase 3 | Pending |
| CORE-03 | Phase 3 | Pending |
| CORE-04 | Phase 3 | Pending |
| CORE-05 | Phase 3 | Pending |
| ASR-01 | Phase 5 | Pending |
| ASR-02 | Phase 5 | Pending |
| ASR-03 | Phase 5 | Pending |
| ASR-04 | Phase 5 | Pending |
| FEAT-01 | Phase 4 | Pending |
| FEAT-02 | Phase 4 | Pending |
| FEAT-03 | Phase 4 | Pending |
| ALIGN-01 | Phase 6 | Pending |
| ALIGN-02 | Phase 7 | Pending |
| ALIGN-03 | Phase 6 | Pending |
| ALIGN-04 | Phase 6 | Pending |
| ALIGN-05 | Phase 6 | Pending |
| ALIGN-06 | Phase 6 | Pending |
| DIAR-01 | Phase 2 | Pending |
| DIAR-02 | Phase 8 | Pending |
| DIAR-03 | Phase 8 | Pending |
| DIAR-04 | Phase 9 | Pending |
| CLUST-01 | Phase 9 | Pending |
| CLUST-02 | Phase 9 | Pending |
| CLUST-03 | Phase 9 | Pending |
| CLUST-04 | Phase 9 | Pending |
| DIAR-05 | Phase 9 | Pending |
| APP-01 | Phase 10 | Pending |
| APP-02 | Phase 10 | Pending |
| APP-03 | Phase 2 | Pending |
| APP-04 | Phase 10 | Pending |
| SHIP-01 | Phase 11 | Pending |
| SHIP-02 | Phase 11 | Pending |
| SHIP-03 | Phase 11 | Pending |
| SHIP-04 | Phase 11 | Pending |

**Coverage:** 39/39 requisiti v1 assegnati, zero orfani e zero assegnazioni duplicate. Vedi ROADMAP.md per dipendenze, confini e gate di promozione.

---
*Last updated: 2026-10-12. Tutti i requisiti sono Pending. Piano approvabile, nessuna implementazione nativa avvenuta.*

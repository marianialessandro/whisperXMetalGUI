# Funzionalità e capacità da conservare

**Dominio:** app macOS locale per trascrizione, word timestamps e diarizzazione. **Data:** 12 ottobre 2026. **Confidenza:** MEDIUM sulle capacità del motore futuro; funzionalità attuali dai sorgenti del checkout.

## Capacità indispensabili

| Funzionalità | Motivo | Complessità | Requisiti |
|---|---|---|---|
| Trascrizione multilingue con lingua auto/esplicita e hints | Contratto corrente | Alta | ASR-01/02/03/04 |
| Profili base/small/large-v3 completo | Conservazione qualità | Alta | ASR-02; Massima non quantizzata |
| Allineamento parola con precisione temporale | Non sostituibile con soli segmenti | Molto alta | ALIGN-01..06 |
| Voci automatiche/esatte e overlap | Conferma esplicita utente | Molto alta | DIAR-01..05, CLUST-01..04 |
| Assegnazione speaker a parole/segmenti | Risultati/export compatibili | Alta | DIAR-05 |
| Risultato senza voci quando disattivate | Trascrizione semplice indipendente HF | Media | APP-01/02/03 |
| Progetti/cartelle, player, nomi voci e revisioni | Prodotto esistente | Media | APP-01/02/04 |
| JSON/TXT/SRT/VTT e lettura vecchi risultati | Nessuna migrazione distruttiva | Media | APP-02 |
| Installazione e doppio click/offline dopo download | Nessun Terminale sul percorso utente | Alta | SHIP-01/02 |
| Portachiavi e Hugging Face solo in Impostazioni | Credenziali e UI vincolanti | Alta | APP-03 |
| Progresso/cancel/errori verificabili | Operazioni lunghe e helper isolato | Alta | CORE-01/02/03/05 |
| CLI Python legacy conservata | Scope app, non rottura upstream | Media | CORE-04 |

La copertura di lingua non coincide con le opzioni visibili IT/EN: auto può selezionare altre lingue. ALIGN-02 comprende mapping già presenti nel pacchetto; i fallimenti allineamento attuali devono restare dichiarati, non trasformarsi in CPU invisibile. Speaker count esatto non implica inventare embedding quando i dati non consentono il conteggio: preservare anche limitazioni e warning del reference.

## Valore distintivo richiesto

| Capacità | Valore | Complessità | Evidenza necessaria |
|---|---|---|---|
| Pipeline numerica su GPU oltre ASR | Obiettivo primario | Molto alta | Inventario e trace per operatori, zero fallback occulto |
| FFT, filterbank, downmix/resampling GPU | Rimuove CPU feature | Alta | Feature/sample parity, bordi, aliasing e durata |
| Trellis/backtrack/merge GPU | Elimina emission roundtrip e cicli host | Molto alta | DP FP32, path tie e word timing |
| AHC/PLDA/VBx/KMeans e assignment GPU | Diarizzazione GPU completa | Molto alta | Dendrogramma, convergenza, count, DER/JER |
| Buffer persistenti e audio lungo bounded | Controllo risorse | Alta | Picchi memoria, test durata/chunk e cancellazione |
| Diagnostica placement affidabile | Consente certificazione | Alta | Backend effettivo, copie/sync e trace correlate ai job |
| Native produzione senza Python | Distribuzione più autonoma | Molto alta | Converter/loader verificati, Mac pulito, tutte le lingue migrate |

## Non costruire

| Anti-feature | Motivo | Sostituzione |
|---|---|---|
| Dashboard GPU o bottoni HF nella finestra principale | Non richiesti, viola prodotto | Diagnostica tecnica in artefatti di verifica; Impostazioni correnti |
| Modelli turbo/quantizzati per far sembrare porting più veloce | Cambia qualità concordata | large-v3 completo e parità |
| Stima speaker con limite arbitrario | Cambia diarizzazione | Budget memoria e algoritmo esatto a tile |
| Fallback numerico CPU automatico | Nasconde incompletezza | Esito supportato/non supportato esplicito |
| Nuovo cloud/hosting modelli/token condiviso | Elaborazione locale e gating individuale | Download personale autorizzato |
| Eliminazione anticipata Python/vecchi output | Perdita oracle/compatibilità | Ritiro solo del runtime produttivo dopo gate |
| Riscrittura player/catalogo/gestione cartelle | Fuori scope GPU | Adapter compatibile |

## Dipendenze

```text
Manifest baseline + corpus + soglie congelate + fattibilità conversione gated senza Python produttivo
  → ABI/helper + MetalSession + tracing
  → Whisper GPU ASR + feature audio GPU
  → Wav2Vec2 italiano/inglese + CTC GPU
  → tutte le famiglie/lingue alignment
  → pyannote segmentazione/embedding con feature GPU
  → AHC/PLDA/VBx/KMeans/constrained assignment GPU
  → aggregazione + speaker-word assignment GPU
  → certificazione end-to-end + runtime produttivo senza Python
```

Feature e ASR hanno dipendenza bidirezionale di parità: si possono prototipare separatamente, ma promuovere solo insieme. Allineamento e diarizzazione condividono il sample clock; kernel clustering richiedono embedding convalidati. Rimozione Python dipende anche dalla conversione e dal download dei modelli, non solo dalle inference.

## Primo slice consigliato

Baseline congelata, core/helper/trace e trascrizione reale `large-v3` tramite whisper.cpp in sviluppo, conservando adapter result e oracolo MLX. È uno slice reviewable, **non** il completamento del requisito pipeline GPU: mel e selezione token host nel reference whisper.cpp vanno ancora migrati. Successivamente alignment IT/EN interamente GPU dimostra la catena neurale + algoritmo custom. Nessuna parte numerica sopra è rinviata implicitamente a v2.

## Fonti

`PROJECT.md`, `REQUIREMENTS.md`, `AGENTS.md`, `macOS/Sources/WhisperXMac/ProjectDetail.swift:119-121`, worker e moduli Python correnti. [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), consultato 2026-10-12: gating, uso offline, conteggio e doppia traccia regolare/esclusiva.

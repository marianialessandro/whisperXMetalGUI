# Sintesi ricerca — WhisperX macOS motore GPU nativo

**Dominio:** trascrizione locale Apple Silicon, allineamento parole, diarizzazione. **Data:** 12 ottobre 2026. **Confidenza complessiva:** MEDIUM. Pianificazione soltanto: niente codice prodotto modificato, benchmark eseguito, modello scaricato o commit.

## Esito

Raccomandazione: core C++17/Objective-C++ con ABI C versionata, helper nativo Swift isolato e Metal come dispositivo numerico esplicito. Whisper.cpp/GGML Metal è la base ASR richiesta, non il motore completo. Usare primitive MPSGraph/MPS verificate per grafi neurali e kernel Metal custom per feature, DP, clustering e aggregazioni. Conservare SwiftUI e i contratti app. Né C++ da solo né il backend MPS disponibile dimostrano collocazione GPU.

La lettura del checkout whisper.cpp `21411d81` ha confermato Metal, Flash Attention e cache/grafi, ma anche mel host, logits readback e presenza backend CPU nello scheduler. Flash Attention esclude DTW: mantenere Wav2Vec2/CTC separato per i tempi parola. Il worker attuale ha MLX per Whisper e MPS per modelli successivi, con emission CTC sulla CPU, WeSpeaker FFT esplicitamente CPU su MPS e postprocessing NumPy/SciPy.

La diarizzazione va migrata integralmente conservando community-1. Il runtime pyannote 4.0.4 usa AHC centroid/euclidean su embedding normalizzati, PLDA, VBx semplificato senza HMM, ramo KMeans per conteggio vincolato e assegnazione vincolata. Il wrapper usa la traccia regolare con overlap; la traccia esclusiva esiste ma non è quella attuale. Tutti questi dettagli sono parte della parità, non libertà di sostituire l'algoritmo con una scorciatoia GPU.

Fattibilità complessiva condizionata: non esiste qui un porting già pronto di tutti i modelli/operatori. CTC, recurrent kernels, centroid AHC, numerica VBx, modello download/conversione senza Python e tutte le lingue richiedono ricerca per fase e gate. Nessun speedup può essere attribuito al progetto prima di misure reali. Lo scopo GPU resta aperto finché calcoli portabili sono ancora CPU, anche se l'app totale termina prima.

## Risultati principali

- **Stack:** whisper.cpp/GGML pinned per ASR, C++/Objective-C++/Metal, MPSGraph come primitiva verificata, SwiftUI/helper e ABI C.
- **Architettura:** DAG GPU con pool a budget e scambi buffer espliciti; CPU per I/O, controllo, tokenizer/strings e serializzazione; copie/readback numerici inventariati e migrati.
- **Rischio maggiore:** conversione incompleta e falsa parità ottenuta cambiando normalizzazione, clustering o modelli. large-v3 deve restare completo non quantizzato.
- **Copertura:** IT/EN iniziali non bastano: lingua automatica e mapping HF/torchaudio globali impongono manifest e prove per famiglie/lingue prima del ritiro runtime Python.
- **Prove:** backend disponibile, dispatch osservato, qualità e tempi sono quattro dimensioni diverse. Gate misurabili, margini qualitativi calibrati e congelati prima di promozione.

## Implicazioni per roadmap

Sequenza raccomandata; il roadmapper può raggruppare slice mantenendo tutte le dipendenze e i 39 requisiti v1.

| Ordine | Fase suggerita | Risultato verificabile | Ricerca ulteriore |
|---|---|---|---|
| 1 | Baseline, corpus e contratti | Versioni/pesi/config, metriche e margini congelati, inventario placement e fattibilità download/conversione dei checkpoint | Corpus annotati, matrici hardware/lingue, soglie, loader nativo e gating |
| 2 | Core/helper/ABI Metal | Job/cancel/eventi, buffer lifetime, tracing e capability reale | Interop GGML-MPSGraph e SDK minimo |
| 3 | ASR GPU | base/small/large-v3, decoding/prompt/lingua, cache e GPU token selection | CPU scheduler e decoder numerico GPU |
| 4 | Audio/features GPU | FFT/mel/Kaldi/downmix/resampler e streaming equivalente | Normalizzazioni globali e chunk boundaries |
| 5 | Wav2Vec2 + CTC IT/EN | Inferenza, DP/backtrack/merge GPU senza emission host | Norm/padding, operatori, DP large/tile |
| 6 | Alignment multilingua | Manifest e equivalenza per tutti i mapping/famiglie | XLSR/XLS-R/1B, alfabeti senza spazi e converter |
| 7 | Diarizzazione neurale | community-1 SincNet/LSTM/powerset ed embedding GPU | Config checkpoint, feature e masked pooling |
| 8 | Clustering e assegnazioni | AHC/PLDA/VBx/KMeans, constrained assignment/aggregation GPU | Centroid inversions, precisione, memoria lunga durata |
| 9 | Integrazione e distribuzione | App/parità totale, converter/loader e package senza Python produttivo | Offline/Mac pulito, relocation, firma e licenze |

La fattibilità di download individuale e lettura/conversione dei checkpoint gated senza Python produttivo e senza hosting proprietario è un **gate iniziale**, prima di investire nei porting dei modelli. La release certifica una soluzione già provata; non è il momento di scoprire che il converter richiede Python o pesi non redistribuibili. In fase 1 identificare formati reali/config/licenze e costruire un proof-of-conversion minimo con artefatti autorizzati; se non possibile, registrare il blocco architetturale senza promettere SHIP-01.

CORE-04 compatibilità legacy è un gate trasversale; non eliminare il pacchetto Python upstream. APP-02/04 conservazione libreria e file esterni va verificata anche senza nuove implementazioni del catalogo. Se roadmap separa speaker-word assignment da clustering, conservarne dipendenza dall'aggregazione diarizzazione e dal timing alignment convalidato.

## Gate di promozione

1. Fixture schema/export/IPC e casi patologici: 100% pass; zero timestamp invalidi e zero segreti nei log/manifest.
2. Placement: zero fallback numerici non dichiarati; dispatch tracciato per operatori/fasi e contatori copie/sync. Residui CPU portabili restano requisiti aperti.
3. Layer/feature/DP: reference pinned; tolleranze per famiglia calibrate e congelate prima di mixed precision/promotion. DP FP32 minimo; pareggi deterministici.
4. Qualità: WER/CER; timestamp start/end p50/p95 e copertura; DER/JER con overlap e collar esplicito; count e word-speaker accuracy dopo permutazione. Margini non-inferiorità da stabilire nella fase 1 con corpus annotato, non inventati in ricerca.
5. Risorse e prestazioni: cold/warm, per-fase e totale, picco memoria, readback e GPU busy time, audio lungo/cancel ripetuto. Nessun OOM/crash entro matrice e budget fissati; nessun downgrade silenzioso di modello o precisione.
6. Release: coverage lingue e download/conversione completi; Mac pulito o equivalente esplicito, avvio offline e rilocato senza Python/Homebrew, firma locale distinta da notarizzazione pubblica.

## Confidenza e lacune

| Area | Livello | Motivo |
|---|---|---|
| Stato corrente | Verificato staticamente | Sorgenti GUI/worker/alignment/diarize, runtime pyannote e whisper.cpp letti |
| Stack | MEDIUM | Componenti esistono; copertura operatori e interop dei buffer da prototipare |
| Funzionalità | MEDIUM | Requisiti approvati e sorgenti; nuova parità non provata |
| Architettura | MEDIUM | Proposta concretamente divisibile, nessun core nuovo costruito |
| Rischi | MEDIUM | Evidenza locale + fonti primarie; gravità e costo quantitativi da misurare |

Il seam `query research-plan` ha selezionato Context7; strumenti MCP e CLI `ctx7` non disponibili. Usato fallback documentazione primaria via web, sorgenti locali e `query classify-confidence --provider websearch --verified` → MEDIUM; digest memorizzati nel research-store. `webfetch` nel seam restituisce LOW: non usato come grado di autorità delle conclusioni. Documentazione Apple dinamica accessibile ma non completa nei dettagli: SDK/header/operator tests sono gate della fase 2. Tool Write non esposto in questa sessione: artefatti creati con apply_patch.

Lacune da risolvere: revision e config effettive dei checkpoint community-1, precisione e soglie corpus, hardware RAM/GPU effettivo, macOS minimo, converter native per tutti i checkpoint e licenze degli artefatti convertiti, limite di memoria per AHC e segmenti CTC. La cache standard HF non è stata trovata: non significa che i modelli non siano installati in un altro percorso.

## Fonti e artefatti

Leggere `STACK.md`, `ARCHITECTURE.md`, `PITFALLS.md`, `FEATURES.md` per fonti precise e rischi per requisito. Primary docs consultate 2026-10-12: [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), [SciPy centroid linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), [Wav2Vec2 bundle](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.Wav2Vec2ASRBundle.html), [Metal synchronization](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work), [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph). Versioni upstream online possono essere più nuove della baseline: non aggiornare automaticamente il reference.

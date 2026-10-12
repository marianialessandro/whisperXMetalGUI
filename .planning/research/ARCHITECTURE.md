# Architettura — pipeline numerica su GPU Apple

**Data:** 12 ottobre 2026. **Confidenza:** MEDIUM sulle implementazioni proposte; sorgenti correnti verificati, kernel non implementati né benchmark eseguiti.

## Confini dei componenti

```text
SwiftUI + catalogo + Portachiavi + ModelManager
  → protocollo job/eventi versionato su canale privato
  → helper nativo Swift persistente
  → ABI C con handle opachi
  → C++ Engine / Objective-C++ MetalSession
      ├─ GGML Metal Whisper
      ├─ Wav2Vec2 GPU
      ├─ CTC Metal
      ├─ feature + segmentation + embedding GPU
      ├─ PLDA + AHC + VBx + constrained assignment GPU
      └─ aggregazione / speaker-word assignment GPU
  → risultati compatti JSON/TXT/SRT/VTT + checkpoint atomici
```

| Componente | Responsabilità | Contratto |
|---|---|---|
| App SwiftUI | UI italiana/inglese, job, player, catalogo, nomi voci, esportazione | Aspetto/flussi preservati; Impostazioni Hugging Face/Modelli/Libreria |
| ModelManager Swift | Accesso HF e conversione/download verificati | Portachiavi e header HTTPS; nessun token all'inferenza |
| Helper Swift | Vita processo, progressi, cancellazione, isolamento crash | IPC versionato; niente elaborazione numerica nascosta in Swift |
| ABI C | Creazione sessione, job, callback eventi, risultati e distruzione | Handle opachi, errori tipizzati, ownership/lifetime espliciti; cancellazione thread-safe |
| Engine C++ | DAG, budget memoria, batch, checkpoint | CPU coordina; trace identifica ogni nodo numerico e backend effettivo |
| MetalSession | Device/queue, pool buffer, dipendenze e capture | Fence/eventi e visibilità CPU espliciti; niente letture premature |
| Backend neurali | Inferenza e preprocessing numerico | Grafi e operatori dichiarati nel manifest; operatore non supportato causa errore, non fallback |
| Writer | Serializzazione e formati | Schema e semantica output attuali; non reinventa allineamento o assegnazione |

Tenere l'helper separato dalla UI; la ABI C collega l'helper alla libreria nello stesso processo, non attraversa IPC. Conservare il protocollo JSON esistente tramite adapter transitorio e versionarlo prima di modificare campi/eventi. Il processo persistente riusa modelli, ma il cache manager può scaricarli quando il budget richiede altre fasi.

## Flusso e placement

1. Decodifica media CPU/FFmpeg in PCM con sample clock globale; conversioni numeriche di downmix/resampling/normalizzazione candidate GPU, I/O dichiarato.
2. FFT, mel, scaling e riduzioni GPU per Whisper. Preservare normalizzazione log-mel globale e padding: elaborare in chunk non significa sostituire un massimo globale con un massimo per chunk.
3. Whisper encoder/decoder GGML Metal. Prima integrazione può mantenere sampling CPU **solo come residuo esplicito temporaneo**, poi GPU logits filtering, softmax, top-k, beam state e cache reorder. Non limitarsi a `use_gpu=true`: scheduler GGML aggiunge backend CPU e ogni nodo deve essere auditato.
4. Segmenti di durata simile → Wav2Vec2 GPU con lunghezze reali, maschere e preprocessing per modello. Padding può alterare group norm: batching corretto non è semplicemente concatenare tensori.
5. Log-probabilità, wildcard max non-blank, trellis e backtrack Metal. Merge caratteri/parole e score tramite scan/riduzioni GPU; tokenizer/Punkt e costruzione delle stringhe restano CPU dichiarata.
6. Finestrelle pyannote con stesso stride/padding/receptive field → SincNet/LSTM/powerset GPU e feature Kaldi compatibili → ResNet/embedding/pooling mascherato GPU. Nessuna scorciatoia sugli overlap.
7. Filtraggio embedding → normalizzazione → AHC centroid euclidean → PLDA → VBx → eventuale KMeans per conteggio imposto → assegnazione vincolata GPU.
8. Ricostruzione delle tracce overlap e aggregazione count GPU → intersezioni parola/segmento con tracce GPU → massimo per voce rispettando pareggi e policy `fill_nearest` attuale. Lettura solo delle label/time compatti finali.
9. Scrittura atomica e output; conservare source language e original/aligned/result secondo contratto corrente.

## Kernel CTC: semantica prima di ottimizzazione

Il trellis locale non è il generico CTC di un tutorial. `alignment.py:get_trellis` inizializza specifici bordi ±inf, cumulata blank e dimensioni extra; `backtrack` sceglie `argmax` sulla colonna finale e usa `changed > stayed` (pareggio = stay). Conservare queste regole, wildcard per simboli/digit/foreign script e probabilità framewise.

Primo kernel: threadgroup per segmento, parallelismo sugli stati testuali, loop temporale e barrier locale solo se gli stati e scratch rientrano nel limite. Per testi lunghi usare tile/wavefront con frontiere tra dispatch: nessuna barrier globale inventata fra threadgroup. Decisioni compatte per backtrack, accumulo FP32, checkpoint/recompute per limitare memoria. Backtrack sequenziale per segmento è comunque computazione GPU; parallelizzare tra segmenti. Argmax e tie-breaking deterministici. Test sintetici devono distinguere successo, fallimento, blank, ripetizioni e timestamp fuori range.

## Diarizzazione: ciò che il porting deve preservare

Il wrapper corrente usa `output.speaker_diarization`, **non** `exclusive_speaker_diarization`. Preservare le sovrapposizioni della traccia regolare e una label dominante per parola come oggi. Non promettere attribuzione contemporanea di una parola a più voci quando lo schema attuale non la rappresenta.

VBx locale `utils/vbx.py` è una variante semplificata con update GMM, senza HMM: non portare arbitrariamente l'algoritmo HMM del paper. Congelare smoothing, `Fa`, `Fb`, ELBO/epsilon, iterazioni, cutoff delle prior e precisione. PLDA: trasformazioni possono essere precalcolate offline dopo equivalenza, evitando eigendecomposition al runtime.

AHC usa centroid linkage euclidea su embedding normalizzati: cluster size, update dei centroidi, min reduction e dendrogramma devono preservare la semantica di `fcluster`. Le distanze di fusione possono non essere monotone; fermarsi alla prima distanza sopra soglia è una scorciatoia errata. Iniziare con implementazione esatta e matrice compatta O(N²), poi ricerca a tile con memoria limitata che mantenga lo stesso minimo globale. Vietato usare nearest-neighbor-chain dei metodi average/ward senza prova per centroid. CPU può inviare dispatch e leggere segnali compatti; distanze/argmin/update restano GPU.

Constrained assignment (`linear_sum_assignment` nel reference) va implementato su GPU per le piccole matrici locali, ad esempio enumerazione limitata soltanto se il massimo effettivo è verificato nel checkpoint, altrimenti algoritmo generale. KMeans forzato deve preservare configurazione e seed reference; identificatori speaker confrontati dopo permutazione ottima, non come stringhe uguali.

## Memoria e sincronizzazione

Memoria unificata non implica interop zero-copy fra GGML, MPSGraph e kernel custom. Specificare device, `MTLBuffer`, offset, stride, tipo, storage mode, hazard tracking, ownership e fence per ogni scambio. Dove GGML non espone un buffer riutilizzabile, ammettere una copia GPU→GPU documentata oppure implementare adapter verificato; non rimuoverla per dichiarazione.

Pool persistenti distinti per pesi, transient e checkpoint. Budget basato sul device e sulla pressione del sistema; batch ridotto senza cambiare pesi/precisione. Niente tutti i modelli sempre residenti per principio. Controllo cancellazione fra dispatch; nessuna libreria rilasciata prima del completamento GPU. Per audio lunghi mantenere frame sample origin assoluta e contesto necessario ai bordi. Normalizzazione globale WeSpeaker richiede riduzione su tutti i frame della finestra prima di centering; rolling center richiede halo esatto.

## Copertura lingue e promozione

GUI offre auto/italiano/inglese; auto può produrre ogni lingua Whisper. Il pacchetto ha cinque bundle torchaudio e ulteriori mapping HF, incluse lingue senza spazi e famiglie 1B. Prototipi IT/EN sono una fase, non il completamento. Manifest per lingua/famiglia: pesi/revision/tokenizer/blank/norm/stride/licenza/placement/parità. Modello non supportato = errore esplicito o risultato solo segmenti dichiarato, mai allineamento CPU nascosto. Mantenere reference finché tutte le capacità correnti sono migrate o una limitazione è formalmente concordata.

## Gate di verifica raccomandati

| Gate | Criterio di uscita |
|---|---|
| Contratti | 100% fixture schema/IPC/export/cancellazione compatibili; segreti assenti da artefatti e log |
| Placement | Zero fallback CPU non dichiarati; trace Metal per ogni stadio numerico incluso CTC, clustering e speaker-word assignment |
| Precisione | FP32 DP; tolleranze layerwise da calibrare su reference e bloccare prima della promozione |
| ASR | WER/CER, silenzi/allucinazioni e prompt per IT/EN e corpus multilingua; margine non-inferiorità da calibrare e bloccare prima di promozione |
| Timing | Errori start/end p50/p95, copertura parole e inversioni; zero timestamp invalidi; margine da bloccare per lingua |
| Voci | DER/JER con overlap incluso e collar esplicito, speaker count error e word-speaker accuracy; margini da calibrare su annotazioni e freeze |
| Risorse | Zero OOM/crash nella matrice dichiarata; budget e durate massime fissati dopo baseline; nessun downgrade nascosto |
| Release | Avvio su Mac pulito senza Python/Homebrew, offline dopo download, firma e package relocation verificati |

Registrare tempi cold/warm per fase, command encoding/dispatch/sync, copie/readback, GPU residency, picco memoria e GPU busy time. Nessun fattore di accelerazione promesso prima dei benchmark. Richiesta GPU non equivale a saturare artificialmente il dispositivo.

## Fonti

Sorgenti locali letti: `whisperx/alignment.py`, `whisperx/diarize.py`, worker/app Swift, `pyannote/audio/pipelines/clustering.py`, `speaker_diarization.py`, `models/segmentation/PyanNet.py`, `models/embedding/wespeaker/__init__.py`, `core/plda.py`, `utils/vbx.py` nel runtime installer; whisper.cpp `src/whisper.cpp` a `21411d81`.

[Modello community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), [centroid linkage SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), [Metal sincronizzazione](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work), consultati 2026-10-12. Le proposte di kernel sono deduzioni progettuali, non capacità già esistenti.

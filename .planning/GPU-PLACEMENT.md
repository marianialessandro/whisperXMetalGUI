# Contratto di placement GPU

**Data:** 2026-10-12. **Stato:** obiettivo pianificato; nessun kernel nuovo implementato o certificato.

Questa matrice è un inventario di calcoli da migrare. Backend disponibile, modello caricato, dispatch osservato e parità verificata devono essere registrati separatamente. Una build C++ o un flag `use_gpu` non provano il placement.

| Calcolo | Baseline da ricertificare | Target | Evidenza richiesta per completamento |
|---------|--------------------------|--------|------------------------------------|
| Decode codec/container | FFmpeg CPU | I/O/decode host esplicito | Distinguere decode da DSP e riportarne il costo |
| Downmix/resampling del core | FFmpeg/CPU | Metal | Confronto sample/offset e dispatch GPU |
| Feature Whisper | MLX; whisper.cpp reference CPU | GPU | Convenzioni FFT/padding/mel equivalenti, trace e confronto valori |
| Encoder/decoder Whisper | MLX GPU | GGML Metal | Grafi osservati, modello/config pinned, confronto testo |
| Filtri logits/top-k/beam numerico | Implementazioni miste, reference whisper.cpp CPU | GPU | Trace selezione e readback ridotto, decoding equivalente |
| Feature speaker | WeSpeaker può forzare FFT CPU su MPS | GPU | Eliminazione FFT host nel nuovo percorso e parità feature |
| Wav2Vec2/log-softmax | MPS per singolo segmento | GPU batched | Padding/maschere/frame mapping, trace e parità logits |
| Trellis/blank/wildcard | PyTorch CPU | Metal | Decisioni/tie-breaking equivalenti e nessun sync host per frame |
| Backtracking CTC | Ciclo Python CPU | Metal | Percorsi equivalenti per segmenti diversi e gestioni fail |
| Riduzione caratteri/parole | CPU/Pandas/Python | GPU per parte numerica | Timestamp/score equivalenti; assemblaggio stringhe host esplicito |
| Segmentazione pyannote | Modello MPS, conversioni NumPy | GPU | Pesi/powerset/mask equivalenti, operator coverage |
| Speaker embedding | MPS, feature/readback CPU | GPU | Feature/model GPU e assenza di intermediate NumPy roundtrip |
| Normalizzazione/distanze/PLDA | NumPy/SciPy | GPU | Confronto matrici e trace GPU |
| AHC centroid linkage | SciPy CPU | GPU | Semantica fusioni/soglia, pareggi e memoria limitata documentati |
| VBx/centroidi/K-means | NumPy/sklearn CPU | GPU | Stopping/init/conteggio equivalenti e trace |
| Assegnazione vincolata/ricostruzione finestre | CPU | GPU | Vincoli, overlap e aggregazione equivalenti |
| Intersezioni speaker-parola | NumPy/Python CPU | GPU | Parità di assegnazione, missing speaker e tie-breaking |
| Stringhe, JSON, export, UI, Keychain | CPU | CPU host | Contratti preservati, nessuna informazione sensibile esposta |

## Regole di certificazione

1. Il manifest di ogni run elenca revisioni, configurazioni, modelli, hardware, operatori e backend effettivi; non contiene token o dati personali.
2. Catturare intervalli CPU e GPU sincronizzati: una misura dell'enqueue asincrono non è una durata di inferenza.
3. Annotare readback, conversioni di layout/dtype, attese e copie reali. Memoria unificata non implica interop senza copie fra framework.
4. Durante lo sviluppo il vecchio worker resta un oracle separato. Non deve essere richiamato automaticamente dal backend nativo in caso di errore.
5. Un fallback diagnostico esplicito non certifica la fase GPU. Un residuo numerico CPU resta requisito aperto fino a porting o revisione di scope concordata.
6. Le ricorrenze temporali di decoder/CTC/LSTM/AHC possono mantenere sequenzialità sulla GPU. Non servono dati host a ogni iterazione; gli schemi threadgroup/tile devono rispettare limiti e barriere di Metal.
7. Saturazione GPU e tempo totale sono misure distinte. Nessuna percentuale fissa o speedup viene promesso prima della baseline.
8. Prima di rimuovere Python dal bundle, verificare copertura delle lingue e dei formati esposti dal prodotto, qualità frozen, error handling, cancellazione e installazione autonoma.

## Numerica e memoria

FP16 può essere usato per pesi/inferenza soltanto quando validato. CTC, log-probabilità, riduzioni VBx/PLDA e decisioni sensibili partono da accumulazione FP32; non trasporre indiscriminatamente calcoli float64 SciPy a float32 senza gate sui risultati. AHC con distanze quadratiche richiede budget, eventuale calcolo tiled/out-of-core e prova che la strategia non cambia fusioni o soglie. La precisione e il tie-breaking devono essere parte del contratto dei kernel.

## Check di completamento

Ogni riga GPU richiede quattro stati: `implementation`, `dispatch_observed`, `parity_passed`, `resource_limits_passed`. Tutti sono inizialmente `pending`. Il ledger finale deve collegare manifest, test e trace reali; non si compila con sola ispezione del codice.

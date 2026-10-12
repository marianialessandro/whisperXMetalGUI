# Rischi del dominio — riscrittura GPU

**Data:** 12 ottobre 2026. **Confidenza:** MEDIUM per fattibilità; criticità corroborate dai sorgenti correnti. Nessuna stima di speedup.

## Rischi critici

### 1. Scambiare backend disponibile per esecuzione GPU completa

**Problema:** il worker seleziona MPS ma torna sulla CPU per emission CTC, feature WeSpeaker e postprocessing; whisper.cpp aggiunge backend CPU al proprio scheduler. `use_gpu` e una build riuscita non dimostrano il placement di ogni operazione.

**Prevenzione:** inventario per nodo, trace Metal, numero/byte di readback, eventi di operatore non supportato. Il percorso nativo deve rifiutare un operatore non coperto oppure dichiarare il residuo e lasciare il requisito aperto; vietato il fallback silenzioso. Non considerare completo un milestone più veloce ma ancora numericamente CPU.

**Rilevazione:** tempo CPU lungo fra dispatch, memcpy di logits/emissions, array NumPy in produzione, fallback nel log o CPU backend nel grafo effettivo.

### 2. Conversione modelli non equivale a compatibilità

**Problema:** GGML non offre già i grafi Wav2Vec2/PyanNet/WeSpeaker del prodotto. Weight layout, group norm, positional convolution con weight norm, LSTM, powerset e pooling mascherato richiedono implementazione. MPSGraph non elimina il lavoro di costruzione dei grafi e verifica della copertura.

**Prevenzione:** manifest pesi/config/vocab; schema converter per famiglia; comparazione input/output layerwise FP32 prima del mixed precision. Modelli gated ottenuti individualmente; nessun token condiviso. Separare runtime inference dal converter e verificare concretamente caricamento PyTorch/NPZ/safetensors senza Python come gate iniziale di fattibilità, prima dei porting estesi; il packaging finale ricertifica la copertura. Nessun hosting proprietario o ridistribuzione non autorizzata può essere assunto per risolvere la conversione.

**Rilevazione:** tensori finali vicini su un file ma divergenti per segmenti brevi, maschere, silenzi o batch; checkpoint letti parzialmente; valori default di classi usati al posto della config reale.

### 3. CTC generico incompatibile col reference locale

**Problema:** inizializzazione ±inf, wildcard max non-blank, endpoint argmax e pareggi cambiano il percorso; FP16 può alterare somme lunghe. Una barrier threadgroup non sincronizza tutti i gruppi del device. Un dispatch host per frame mantiene un collo di bottiglia e viola lo scopo di ALIGN-04.

**Prevenzione:** clone semantico verificato con emission sintetiche; FP32; persistent kernel per segmento dove possibile, wavefront/tile per casi grandi senza ciclo host per frame; decisioni compatte o recompute; backtrack su GPU.

**Rilevazione:** percorsi diversi per emission identiche, NaN/infinito inatteso, parola saltata o tempo negativo, trace con migliaia di wait/dispatch host per singolo segmento.

### 4. AHC centroid trattata come clustering più facile

**Problema:** il reference usa centroid/euclidean su embedding normalizzati. Average/ward/single, ricerca approssimata, stopping alla prima distanza sopra soglia o riduzione arbitraria degli embedding cambiano l'inizializzazione VBx. Centroid può produrre inversioni del dendrogramma; `fcluster` non è semplicemente un confronto con l'ultima distanza.

**Prevenzione:** test dendrogramma/fcluster, cluster size e tie; min globale GPU e aggiornamento esatto; prima implementazione con memoria quantificata, poi ricerca tile esatta. Non imporre massimo arbitrario alle voci per contenere memoria. Distinguere budget embedding/memoria da policy speaker count.

**Rilevazione:** speaker count diverso già prima di VBx; memoria quadratica esplode su meeting lunghi; pareggi rendono la run non riproducibile. La baseline SciPy usa O(N²) memoria; il centroid reference ha anche costo seriale importante.

### 5. Portare il VBx del paper invece del runtime corrente

**Problema:** `utils/vbx.py` locale ha update GMM e annota rimozione supporto HMM. Precisione NumPy, ELBO, prior cutoff, smoothing e convergenza influenzano il numero automatico delle voci. Ramo KMeans vincolato e constrained assignment non sono opzionali.

**Prevenzione:** portare codice/config della versione 4.0.4 bloccata; freeze `Fa/Fb`, maxIters/epsilon, inizializzazione, prior threshold e KMeans n_init/seed. La FP64 reference non è banalmente riproducibile con Metal FP32; analizzare margini e casi vicini alle soglie prima di accettare differenze. Eigendecomposition PLDA può essere convertita offline con verifica.

**Rilevazione:** centroidi simili ma count o fusione diversa, convergenza in iterazioni diverse, coppie di voci sovrapposte assegnate allo stesso cluster senza rispettare policy.

### 6. Feature streaming che cambiano i modelli

**Problema:** Whisper usa normalizzazioni e padding propri; WeSpeaker applica scala `1 << 15`, feature Kaldi e centering globale o rolling; Wav2Vec2 group norm può dipendere dalla lunghezza. Un massimo/media per chunk sostituisce una riduzione globale e altera l'input.

**Prevenzione:** fissare downmix, resampler, sample rate, finestre, FFT, preemphasis, dither, mel, clamp, snip_edges, dtype e receptive field. Due passaggi GPU per statistiche globali o contesto equivalente; halo per running mean. Confrontare feature ai bordi e score modello su audio ripartito in modi diversi.

**Rilevazione:** batch=1 corretto e batch>1 sbagliato; deriva temporale crescente; output dipendente dalla dimensione di chunk; troncamento delle prime/ultime parole.

### 7. Memoria unificata trattata come zero-copy senza contratto

**Problema:** view logiche diverse o lifetime incompatibili fra GGML/MPSGraph/MSL introducono copie o race. Tenere large-v3, alignment e diarizzazione residenti simultaneamente può superare il budget.

**Prevenzione:** ownership buffer, offset/layout e fence specificati; profiling copie GPU-GPU/CPU-GPU; cache a budget; evict esplicito dei modelli e batch adattivo senza cambiare qualità. Cancellazione libera risorse solo dopo completamento dei command buffer.

**Rilevazione:** risultati intermittenti, picchi per compilazione grafo/cache, OOM in warm run, UI bloccata da `waitUntilCompleted`, leak dopo cancellazioni ripetute.

### 8. Eliminare Python con due lingue migrate

**Problema:** UI espone automatico e il mapping alignment comprende lingue molto più numerose di IT/EN, anche lingue senza spazi e modelli 1B. Eliminare il runtime prima della copertura completa causa regressione o fallback occulto.

**Prevenzione:** matrice per tutti i mapping correnti e famiglie, tokenizer e postprocessing; IT/EN primo slice, famiglia XLSR/XLS-R e varianti altre slice. No-model mantiene esito esplicito a tempi di segmento come contratto corrente; nessun modello nuovo scaricato per mascherare una lingua persa.

**Rilevazione:** language auto non trova adapter, manifest non copre config effettiva, test solo in italiano. SHIP resta aperto finché la matrice non passa.

## Rischi moderati

| Rischio | Prevenzione |
|---|---|
| Flash Attention + DTW | whisper.cpp locale disabilita DTW quando Flash Attention attiva; mantenere CTC separato, non sostituire tempi parola con timestamp token |
| Recurrent kernel lento | SincNet/LSTM con dipendenza temporale: batch finestre/direzioni/hidden; verificare gate ordering/bias, non promettere throughput dai soli GEMM |
| Overlap perduto | Il wrapper usa diarizzazione regolare, non esclusiva; mantenere count/powerset e tracce overlap |
| Label speaker diversa ma equivalente | Metriche dopo permutazione; nomi personalizzati restano associati alla revisione corrente, non migrare etichette a caso |
| Beam search cambia testo | Conservare policy corrente e usare gate MLX/whisper.cpp; default upstream diverso non è miglioramento autorizzato |
| Token in output diagnostico | ModelManager separato, redaction prima dell'IPC/log, niente segreti nei process arguments/env/fixture/checkpoint |
| Runtime scaricato non riproducibile | Hash e revision pinned di sorgenti/modelli/toolchain; licenze e gating verificati, niente dipendenza dal checkout personale |
| Deployment minimo incoerente | Riconciliare macOS SwiftPM/installer e funzionalità SDK prima di usare API nuove |

## Warning per fase

| Topic | Ricerca specifica necessaria | Gate |
|---|---|---|
| Baseline | Corpus annotato e soglie non-inferiorità | Calibrazione e freeze prima di promozione |
| Core/Metal | Buffer interop, command dependencies e cancel | No race/leak, trace corrisponde al DAG |
| ASR | Decoder masking/top-k/beam e CPU scheduler | No numerical host fallback, qualità invariata |
| Features | FFT GPU e normalization semantics | Parità feature/chunk/batch/resampler |
| Alignment | Operator coverage famiglia, CTC grande | Parità layer/path/timing e memoria bounded |
| Diarizzazione | Checkpoint/config effettivi, LSTM, powerset | Parità segmentation/embedding/overlap |
| Clustering | Centroid inversions/tie/FP64 baseline | Dendrogramma e DER/count nei margini congelati |
| Packaging | Loader/converter senza Python, firma/helper | Mac pulito, offline e relocation |

## Fonti

Sorgenti locali descritti in `STACK.md`/`ARCHITECTURE.md`; in particolare `clustering.py:550-669`, `utils/vbx.py`, `models/embedding/wespeaker/__init__.py:113-153`, `PyanNet.py`, `alignment.py:431-498`, whisper.cpp `src/whisper.cpp:3708`.

[SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), consultato 2026-10-12, verifica semantica centroid e memoria. [Apple Metal](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work), consultato 2026-10-12, riferimento di sincronizzazione; dettagli API da verificare in SDK.

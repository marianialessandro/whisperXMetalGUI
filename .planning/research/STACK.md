# Stack tecnologico — motore GPU nativo

**Progetto:** WhisperXGUI. **Ricerca:** 12 ottobre 2026. **Confidenza:** MEDIUM per fattibilità del porting; comportamento dei sorgenti locali verificato staticamente, nessuna prova di prestazioni.

## Decisione raccomandata

Usare C++17 per motore e ownership delle risorse, Objective-C++ per integrazione Metal, Metal Shading Language per kernel dedicati e una ABI C versionata verso un helper nativo Swift. Conservare SwiftUI, catalogo e funzionalità dell'app. La priorità è la collocazione sulla GPU della computazione numerica, non il solo tempo complessivo.

| Tecnologia | Versione da fissare | Scopo | Motivazione |
|---|---|---|---|
| whisper.cpp + GGML Metal | Baseline locale `21411d81`; fissare SHA completo e GGML relativo | Encoder/decoder Whisper | Implementazione C/C++ e Metal verificabile; non usare il checkout personale come dipendenza di release |
| Apple Metal/MSL | SDK Xcode fissato in fase iniziale | FFT, feature, CTC, clustering, aggregazioni, selezione token | Controllo diretto del dispositivo e dei buffer |
| MPSGraph/MPS | SDK Apple, operatori da verificare sul minimo macOS supportato | Matrici, convoluzioni, normalizzazioni e grafi Wav2Vec2/pyannote | Riduce kernel riscritti dove le primitive soddisfano semantica e placement; operatore mancante diventa kernel Metal |
| C++17 / Objective-C++ | Apple Clang fissato | Scheduler, manifest, bridge Metal | ABI C limita accoppiamento con Swift; eccezioni non attraversano ABI |
| Swift / SwiftUI | Package corrente Swift tools 6.0 | GUI e helper | Mantiene prodotto, isolamento processi e integrazione Portachiavi |
| CMake e SwiftPM | Versioni da bloccare in build riproducibile | Libreria nativa/helper/app | Non richiede strumenti di sviluppo sul Mac dell'utente |
| FFmpeg già distribuito | Mantenere baseline packaging verificata | Decodifica container/codec non coperti dal sistema | Non restringere formati per eliminare Python; CPU I/O/codec ammessa e dichiarata |
| Python/PyTorch/MLX attuali | Lock locale: pyannote 4.0.4, torch/torchaudio 2.8.0, mlx 0.32.2, mlx-whisper 0.4.3 | Oracolo di conversione e confronto | Solo strumenti di sviluppo fino a equivalenza; non rimuovere prima dei gate |

MPSGraph è una scelta da verificare con un prototipo per ciascuna famiglia neurale, non una promessa di importazione automatica. GGML non è un importatore universale di Wav2Vec2 e pyannote: servono grafi, conversione pesi e copertura degli operatori. Preferire uno stesso dominio Metal di risorse; evitare tre framework concorrenti nella release senza contratto di interoperabilità provato.

## Modelli e precisione

| Famiglia | Contratto |
|---|---|
| ASR Massima | `large-v3` completo, non distillato e non quantizzato; mixed FP16/FP32 verificato, manifest di origine e conversione. Vietato sostituirlo con turbo per far passare un benchmark |
| ASR altri profili | `base` e `small`, stessi significati Rapida/Bilanciata; prompt, lingua automatica/esplicita, segmentazione e opzioni di decoding congelate |
| Allineamento IT/EN | `VOXPOPULI_ASR_BASE_10K_IT` e `WAV2VEC2_ASR_BASE_960H`; medesimi pesi/vocabolari/blank e preprocess corrente |
| Altre lingue | Manifest per tutti i mapping di `whisperx/alignment.py`, inclusi HF XLSR/XLS-R e modelli 1B; non dichiarare copertura da due soli prototipi |
| Voci | `pyannote/speaker-diarization-community-1`, revision/config e sotto-modelli segmentation, embedding, PLDA da acquisire e bloccare con token personale autorizzato |
| DP e clustering | FP32 minimo per CTC; baseline NumPy può essere FP64 in clustering/VBx. Stabilire tolleranze e gestione casi instabili prima di promozione |

Il runtime installato rivela classi e algoritmi, ma la revisione esatta dei pesi pyannote non è stata trovata nella cache standard durante questa ricerca. Non dedurre hyperparameter dei checkpoint dai soli default delle classi.

## Persistenza, distribuzione e credenziali

Conservare catalogo JSON atomico corrente: SQLite non è richiesto dal porting GPU. App e helper firmati, librerie con rpath relativo, metallib e manifest inclusi nel bundle, modelli scaricati separatamente. Il deployment minimo è da riconciliare: SwiftPM dichiara macOS 14, installer attuale usa una baseline diversa; non alzarlo implicitamente per ottenere una API GPU.

Il download nativo deve supportare revisioni, redirect autenticati consentiti, ripresa e hash. Validare formati realmente presenti (checkpoint PyTorch, safetensors, NPZ): un loader C++ capace di leggere qualsiasi pickle non è una scelta implicita. **Gate iniziale di fattibilità:** definire e provare lettura/conversione autorizzata senza Python produttivo, con licenze e senza hosting proprietario assunto, prima dei porting neurali estesi. Prima del ritiro Python certificarne la copertura completa. Token nel Portachiavi, mai nel manifest, negli argomenti, nell'ambiente o nei log; l'helper d'inferenza riceve percorsi locali, non credenziali.

## Alternative escluse dalla prima direzione

| Alternativa | Perché non è la direzione scelta |
|---|---|
| Solo tuning MPS Python | Mantiene FFT CPU, emission readback e clustering NumPy |
| Riscrivere tutto in Swift senza kernel | Linguaggio diverso non prova placement GPU |
| Core ML con `.all` | Può privilegiare ANE/CPU; non soddisfa controllo esplicito GPU richiesto. Rimane alternativa soggetta a nuova decisione e tracing |
| Sostituire pyannote con clustering più semplice | Cambia qualità e numero automatico delle voci |
| Quantizzare/modello turbo per accelerare | Non conserva il profilo Massima concordato |

## Fonti

- Checkout `/Users/marianialessandro/repository/whisper.cpp`, `src/whisper.cpp`, SHA breve `21411d81` (lettura locale 2026-10-12): grafi, backend, logits readback, log-mel e incompatibilità DTW/Flash Attention.
- `macOS/Packaging/requirements.lock`, `macOS/Worker/transcribe_worker.py`, `macOS/Package.swift`, sorgenti runtime pyannote in `macOS/output/installer/WhisperX.app/Contents/Runtime/lib/python3.12/site-packages/pyannote/audio/`.
- [whisper.cpp upstream](https://github.com/ggml-org/whisper.cpp), consultato 2026-10-12: C API e backend Apple; upstream attuale non sostituisce la baseline locale.
- [MPSGraph Apple](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph), [sincronizzazione CPU/GPU Metal](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work), consultati 2026-10-12. Pagine dinamiche: copertura API dettagliata ancora da verificare in SDK.
- [Wav2Vec2ASRBundle](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.Wav2Vec2ASRBundle.html): pesi, modello e label devono migrare insieme.

# WhisperX macOS — motore nativo GPU

## What This Is

Evoluzione dell'app macOS esistente per trascrivere audio e video localmente, allineare le parole e distinguere le voci. Il milestone sostituisce progressivamente il motore numerico Python con componenti C++, Objective-C++ e Metal, conservando UI SwiftUI, libreria, player, risultati, esportazioni e accesso Hugging Face.

## Core Value

Eseguire sulla GPU Apple tutta la computazione numerica portabile della pipeline, mantenendo qualità della trascrizione, precisione dei tempi e diarizzazione completa.

## Requirements

### Validated

Funzionalità esistenti dedotte dal codice; la baseline runtime sarà ricertificata nella fase 1.

- App SwiftUI con progetti, cartelle, importazione, player, nomi delle voci ed esportazioni.
- Worker MLX/Metal ASR e PyTorch/MPS per i modelli di allineamento e diarizzazione.
- `community-1`, conteggio voci automatico o esatto, word timestamps e risultati JSON/TXT/SRT/VTT.
- Credenziali Hugging Face nel Portachiavi, inviate al worker su stdin.
- Installer con runtime autonomo; distribuzione pubblica firmata/notarizzata non dimostrata.

### Active

- [ ] Motore nativo GPU con ABI C, gestione esplicita dei buffer e helper isolato dall'app.
- [ ] ASR whisper.cpp/GGML Metal, modelli completi e qualità comparata con MLX.
- [ ] Wav2Vec2, trellis CTC, backtracking e riduzioni delle parole sulla GPU.
- [ ] Feature, segmentazione, embedding, clustering e assegnazione voci sulla GPU.
- [ ] Streaming a memoria limitata, batching, cancellazione, progresso e diagnostica veritiera.
- [ ] Packaging autonomo senza Python nel percorso finale di produzione, solo dopo parità e copertura.

### Out of Scope

- Rifacimento dell'interfaccia o aggiunta di dashboard/configurazioni GPU nel percorso utente.
- Training di nuovi modelli, elaborazione cloud o hosting proprietario dei modelli.
- Sostituzione implicita di `large-v3` completo con turbo o quantizzato.
- Eliminazione del pacchetto Python upstream o rottura dei suoi flag/API esistenti.
- Commit, push, tag, cambi di branch o modifiche al checkout whisper.cpp di riferimento.

## Context

Decisioni esplicite dell'utente, 12 ottobre 2026: libertà di riscrittura in qualsiasi linguaggio; priorità allo spostamento dei calcoli su GPU; whisper.cpp in `repository` come riferimento; diarizzazione mantenuta. È autorizzata la pianificazione, non l'esecuzione dei piani.

Checkout corrente: `/Users/marianialessandro/Desktop/whisperXGUI`. Prima dell'avvio esistevano `.gitignore` modificato e `.DS_Store`, `AGENTS.md`, `docs/`, `macOS/` non tracciati. Non sono lavoro di questo milestone.

Riferimento read-only: `/Users/marianialessandro/repository/whisper.cpp`, commit osservato `21411d81`. Il checkout presenta file non tracciati preesistenti. L'integrazione futura usa una revisione riproducibile e licenze verificate, senza dipendere dal percorso personale in una build distribuita.

Il worker GUI è `macOS/Worker/transcribe_worker.py`; non coincide con la CLI upstream `whisperx/__main__.py`. Oggi l'allineamento è segment-by-segment e sposta le emissioni sulla CPU; pyannote produce array NumPy e alcune feature WeSpeaker effettuano FFT su CPU in MPS. whisper.cpp offre backend GPU e kernel Metal ma mantiene mel preprocessing e selezione dei token sul lato host: la sola integrazione ASR non completa questo milestone.

L'analisi precedente ha osservato arm64, PyTorch 2.8.0/MPS disponibile, MLX 0.32.2/Metal disponibile, mlx-whisper 0.4.3 e pyannote-audio 4.0.4 nel runtime locale. Questi dati sono un punto di partenza da raccogliere nuovamente nel manifest baseline; non sono una prova di prestazioni o di qualità del nuovo motore.

## Constraints

- **Git:** nessun commit/push per qualsiasi file, comprese `.planning` e documentazione. Questa regola prevale sui workflow GSD. La validazione non richiede commit.
- **Prodotto:** doppio click, installazione autonoma, nessuna installazione manuale di Python/FFmpeg; seguire `AGENTS.md` e `docs/PROGETTO_APP_MACOS.md`, con le direttive recenti prevalenti sulle sezioni storiche.
- **UI:** Hugging Face solo nell'omonimo pannello Impostazioni; Modelli e Libreria mantenuti; niente wizard o pulsanti di configurazione nella finestra principale.
- **Sicurezza:** token nel Portachiavi, mai in argomenti processi, log, manifest, test fixture o file del progetto. Download individuali autorizzati, senza token condivisi.
- **Compatibilità:** libreria e revisioni preesistenti leggibili, word timestamps e speaker assignment esportabili; conteggio automatico senza massimo arbitrario; italiano/inglese e lingue già esposte da valutare nella matrice di copertura.
- **GPU:** Apple Silicon e Metal come target del motore nativo; Neural Engine non sostituisce il requisito GPU. Nessun fallback CPU numerico silenzioso; ogni residuo è inventariato e motivato, e impedisce di dichiarare una copertura GPU completa.
- **Precisione:** stesso modello e stessa semantica prima di ottimizzazioni approssimate; CTC e riduzioni sensibili FP32 inizialmente. Soglie di qualità da calibrare e congelare prima della promozione del backend.
- **Prove:** disponibilità del backend, lavoro realmente GPU, parità di qualità e miglioramento temporale sono quattro proprietà distinte da documentare.
- **Stile:** commenti con a capo soltanto dopo frasi complete; espressioni fra parentesi sulla stessa riga. La knowledge base è stata navigata tramite indici; non sono presenti styleguide Python/C++/Swift nella categoria consultata.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Motore C++/Objective-C++/Metal con ABI C | Controllo su placement GPU, buffer e kernel | Accettata, implementazione pendente |
| Helper nativo con protocollo versionato compatibile con gli eventi attuali | Isolamento, cancellazione e credenziali private; ABI del core riutilizzabile | Proposta operativa verificabile nella fase 1 |
| whisper.cpp/GGML come prima base ASR | Riferimento esplicitamente richiesto e backend Metal esistente | Accettata; performance rispetto a MLX non dimostrate |
| Diarizzazione community-1 completa, inclusi clustering e sovrapposizioni | Conferma esplicita dell'utente | Bloccata come requisito |
| CPU riservata a controllo, I/O e testo finale | GPU placement è obiettivo primario | Residui numerici devono essere visibili e migrati |
| Migrazione per fasi, Python conservato come oracle durante lo sviluppo | Evitare regressioni e perdita del riferimento | Non autorizza fallback silenzioso né rimozioni premature |
| GSD senza commit e senza auto-esecuzione | Direttive del repository e richiesta di pianificazione | Configurato |

## Evolution

Aggiornare questo documento dopo ogni fase verificata. Requisiti completati richiedono implementazione ed evidenze, mai solo piani, disponibilità MPS/Metal o una build riuscita. Eventuali deroghe alla copertura GPU devono restare esplicite e non diventare completamenti impliciti.

---
*Last updated: 2026-10-12 dopo approvazione della direzione e avvio della pianificazione GSD.*

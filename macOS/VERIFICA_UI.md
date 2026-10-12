# Verifica della prima UI

9 ottobre 2026, Apple Silicon, Swift 6.2.3, build Release locale.

## Controlli completati

- Compilazione Release del pacchetto e generazione di `output/WhisperX.app`.
- Avvio del bundle e ispezione visiva della finestra principale e della guida Hugging Face.
- Navigazione nei tre passaggi della guida, chiusura con Più tardi e presenza dei link dedicati.
- Campo token mascherato, pulsante di collegamento inizialmente disabilitato e gestione di un valore di prova non valido senza usare credenziali personali.
- Importazione attraverso il pannello di sistema del WAV sintetico `output/test-media/Anteprima audio.wav`, con spazio nel nome.
- Durata corretta di due secondi, selezione nella sidebar e visualizzazione delle impostazioni del file.
- Riproduzione dell’audio verificata tramite stato del player e avanzamento della timeline.
- Attivazione di Distingui le voci con comparsa dell’azione Configura e dell’opzione per il numero noto di voci.

Durante l’ispezione è stato riprodotto un crash nel framework `_AVKit_SwiftUI` usando `VideoPlayer`. Il player è stato sostituito con `AVPlayerView` attraverso `NSViewRepresentable`; la stessa importazione e la riproduzione sono state poi completate senza crash.

## Verifiche ancora da effettuare

Il percorso riuscito con un token reale e l’accesso al Portachiavi non sono stati esercitati con credenziali personali. Verifica online e gestione del Portachiavi sono implementate, ma non vengono dichiarate validate end-to-end.

Restano da verificare download e inferenza quando saranno integrati, importazione di ulteriori formati e video, drag-and-drop da Finder, VoiceOver completo, tema chiaro e distribuzione su un Mac pulito. L’aspetto osservato in questa sessione è il tema scuro di sistema, senza cambiarne le preferenze.

Nessun commit o push effettuato.

## Collegamento del pulsante Trascrivi

Il pulsante ora avvia il worker reale. Test effettuato con il breve audio sintetico `Prova.aiff`, lingua inglese, qualità Massima e voci disattivate. Il worker ha completato ASR MLX/Metal, allineamento e scrittura dei risultati senza avvisi. Testo ottenuto: “This is a short test. The transcription button now starts the speech recognition engine.” Verificata la presenza di TXT, SRT, VTT e JSON e il contenuto non vuoto della trascrizione.

Build locale collegata al virtualenv già disponibile sul Mac. Il test copre il worker e i suoi output; l’intero avvio dalla UI aggiornata, annullamento, fallback CPU e distinzione delle voci non sono ancora verificati end-to-end. La build precedente già aperta mantiene il vecchio pulsante disabilitato fino alla riapertura.

## Versione 0.3 — progetti e nuova UI

Verifica del 9 ottobre 2026 su macOS 15.6, build Release, con una libreria di prova separata sotto `/private/tmp`. La finestra della vecchia build in uso e i file personali non sono stati chiusi, spostati o eliminati durante questa verifica.

Sette test Swift Testing passati: conservazione delle sorgenti esterne, spostamento del solo file scelto, eliminazione del progetto mantenendo l’audio gestito, persistenza di titoli/cartelle/nomi delle voci, cambio di posizione con aggiornamento dei percorsi e backup, rifiuto di destinazioni non vuote, protezione del catalogo illeggibile e conteggio delle dimensioni senza duplicare i symlink. Alcuni test coprono più condizioni nella stessa esecuzione.

Controlli nell’app:

- Barra superiore con titolo, play/pausa, timeline, salti e volume; rimozione delle diciture Hugging Face dalla finestra principale.
- Impostazioni Voci/Modelli/Libreria; cache trovate, dimensioni e percorsi visualizzati. Pannello Libreria con posizione e opzione di cambiamento.
- Progetto sintetico con speaker: apertura del pannello nomi, salvataggio di Giulia e Marco, verifica della sostituzione nel testo e della persistenza su disco.
- Menu contestuale del progetto e pannello di eliminazione: avviso sulla sorgente esterna e sua conservazione; prova annullata senza eliminare il progetto.
- Importazione di un audio sintetico con domanda sulla posizione e scelta Sposta nella libreria; verificati presenza nella destinazione e rimozione del solo file sintetico dalla posizione precedente.
- Avvio reale da UI, modello large-v3 MLX/Metal già in cache, lingua automatica e voci disattivate: trascrizione e allineamento completati; vista risultante con menu, testo e azioni Nuova trascrizione/Esporta. Nomi delle voci assente come previsto per un risultato senza speaker.
- Rinomina del progetto in Registrazione di prova: titolo della finestra, player e sidebar aggiornati. Creazione della cartella Lezioni e spostamento del progetto dal menu contestuale.
- Riavvio della build di verifica: progetti, parametri e nomi personalizzati mantenuti.

La prova ha rilevato due problemi corretti: lettura sincrona del catalogo all’avvio, ora eseguita in background; dipendenza del worker dal checkout sul Desktop, ora sostituita dal pacchetto WhisperX incluso nelle risorse del bundle. Verificato anche l’annullamento del worker bloccato nella prima prova.

Il test di inferenza dalla UI è stato eseguito con la build di verifica fuori dal Desktop e con file sintetici, senza usare il token personale per la diarizzazione. Restano da validare la diarizzazione reale dalla nuova UI, l’esportazione con nomi personalizzati quando la pagina sarà definita, l’accessibilità completa, ulteriori formati media e la distribuzione autonoma su un Mac pulito. Il cambio di posizione della libreria è coperto dai test sui file; non è stato eseguito sulla libreria personale tramite la UI.

## Versione 0.4 — toolbar e spostamento della libreria

Nove test Swift Testing passati. Aggiornata la verifica della migrazione: i file gestiti e il catalogo sono nella destinazione, la vecchia libreria viene rimossa e le sorgenti esterne restano intatte. Aggiunte regressioni per errore di scrittura del catalogo di destinazione con ripristino dei file, e per conservazione di file non gestiti nella vecchia directory.

Verifica visiva su istanza di prova separata: controlli del player nella toolbar con titolo e Importa, assenza del footer con Impostazioni/Nuova cartella, visualizzazione dei segmenti più compatta. Cmd+Shift+N apre il pannello Nuova cartella; Cmd+, apre le impostazioni con tab Hugging Face, Modelli e Libreria. La libreria personale non è stata spostata durante questi test.

## Versione 0.5 — configurazione modale e toolbar

Nove test passati e build Release ricompilata. Verificato nell’app di prova: importazione di un file sintetico, scelta Lascia nella sorgente, apertura automatica del pannello modale con lingua/qualità/voci, annullamento e riapertura tramite Nuova trascrizione. Avvio reale dal pannello con modello già presente: pannello chiuso, progresso nella finestra principale e risultato completato. Verificata visivamente la toolbar con maggiore altezza, mantenuta nella stessa barra del titolo e delle azioni.

Corretta anche la sostituzione dell’eseguibile durante la build: copia in un file temporaneo e rinomina atomica, per evitare di modificare le pagine dell’eseguibile di un’app già aperta. La build principale aggiornata è stata avviata e il processo è rimasto attivo; nessun nuovo crash dopo la correzione.

## Guida introduttiva e inglese — 11 ottobre 2026

Verificata la build release `output/WhisperX.app` con UI nativa e tema scuro. Al primo avvio la guida compare come sheet in sei passaggi; verificati selezione diretta del passaggio, ultimo pulsante Inizia, chiusura, riapertura da Help → WhisperX guide… e riavvio senza ripresentazione automatica della guida già conclusa.

Il selettore della guida aggiorna subito i testi italiano/inglese mantenendo il passaggio corrente. Verificato anche il cambio dalle Impostazioni → Generali: contenuti, titolo della finestra e schede Generali/Modelli/Libreria si aggiornano senza riavvio. Verificati visivamente i passaggi introduttivo e player, il pannello Modelli in inglese e la configurazione della trascrizione con Language, Quality/Balanced e Identify speakers. La preferenza di lingua è stata riportata a Lingua del sistema dopo i controlli. Non sono stati importati file, avviate trascrizioni o modificati token durante questa verifica.

`swift test --package-path macOS`: 14 test superati. Le nuove verifiche coprono scelta esplicita e fallback della lingua, interpolazioni con contenuto utente, conservazione dei placeholder nel catalogo, precedenza delle risorse incluse nell’app e identificatori persistenti della qualità. Build release e generazione del bundle completate. Il catalogo inglese contiene 243 voci; nessuna chiave usata da `L10n.tr` risulta mancante.

I menu e i pannelli standard di macOS seguono la lingua del sistema. Il controllo visivo non comprende tutte le schermate di errore né una nuova trascrizione completa. Firma Developer ID, notarizzazione e runtime autonomo restano fuori da questa verifica.

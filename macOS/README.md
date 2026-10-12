# WhisperX macOS

App SwiftUI per macOS 14+, senza dipendenze Swift esterne. Il motore Python WhisperX viene incluso nel bundle. La build di sviluppo può usare il runtime locale; il nuovo installer include un runtime autonomo.

## Avvio

Aprire `Package.swift` in Xcode oppure, dalla directory `macOS`, generare il bundle:

```sh
sh build-app.sh
```

Aprire con doppio click `output/WhisperX.app`. Questo comando genera il bundle di sviluppo con firma ad hoc. Per generare il pacchetto completo con Python, FFmpeg e tutte le dipendenze usare `sh build-installer.sh`: vedere [guida dell’installer](Packaging/README.md), anche per firma Developer ID e notarizzazione. La build locale dell’installer richiede Apple Silicon e macOS 15; la distribuzione pubblica resta subordinata ai certificati e alle verifiche indicate nella guida.

## Guida iniziale e lingua

Al primo avvio compare una guida in sei passaggi: orientamento nella finestra, importazione e scelta della posizione, organizzazione, configurazione della trascrizione, player e risultati, Impostazioni. Si può saltare, navigare avanti e indietro o selezionare direttamente un passaggio. La chiusura viene ricordata; per riaprirla usare Aiuto → Guida a WhisperX….

Il selettore nella guida e il pannello Generali delle Impostazioni consentono di scegliere Italiano, English o Lingua del sistema. La lingua cambia subito e viene ricordata tra gli avvii. La modalità sistema segue la lingua preferita di macOS, con inglese come alternativa per lingue non supportate. I menu standard e i pannelli di sistema restano gestiti da macOS. La lingua dell’audio, il contenuto delle trascrizioni e gli identificatori delle qualità salvate restano indipendenti dalla lingua dell’interfaccia.

Le traduzioni inglesi sono in `Sources/WhisperXMac/Resources/English.json`, incluse dal pacchetto Swift e copiate nel bundle da `build-app.sh`. I testi italiani sono le chiavi sorgente; `LocalizedTemplate` mantiene separati i valori interpolati. Nessuna lettura automatica degli appunti, connessione, download o modifica ai progetti viene eseguita dalla guida introduttiva.

## Progetti e libreria

A ogni importazione, tramite pannello macOS, Cmd+O o drag-and-drop, l’app chiede se lasciare il file nella sua sorgente o spostarlo fisicamente nella libreria. Dopo questa scelta apre un pannello modale sopra la finestra per configurare lingua, qualità e voci; le opzioni non occupano l’area del testo. La seconda opzione rimuove il file dalla posizione originale dopo lo spostamento. Ogni progetto ha un UUID e un titolo iniziale derivato dal nome del file, modificabile tramite la matita accanto al titolo o dal menu contestuale.

Nuova cartella è nel menu File con shortcut Cmd+Shift+N; le sottocartelle si creano anche dal menu contestuale. La sidebar permette di rinominare le cartelle e spostare i progetti tramite il menu contestuale. Rimuovere una cartella conserva i progetti e li sposta al livello superiore. Eliminare un progetto apre un pannello che segnala il file sorgente. I file esterni sono conservati; per quelli gestiti si può scegliere se conservarli o spostarli nel Cestino. Le trascrizioni gestite del progetto eliminato vengono spostate nel Cestino.

Titoli, cartelle, impostazioni, revisioni delle trascrizioni e nomi delle voci vengono conservati in `Library.json`, scritto atomicamente. Il catalogo viene letto in background, così l’avvio non blocca la UI mentre macOS verifica l’accesso alla cartella. Un catalogo non leggibile non viene sovrascritto.

La posizione iniziale è `~/Library/Application Support/WhisperX/Library`. Le impostazioni permettono di scegliere una nuova cartella vuota: l’app sposta i dati, aggiorna i percorsi dei file gestiti, verifica il catalogo di destinazione e rimuove quello precedente. La vecchia cartella viene rimossa quando è vuota; eventuali file estranei non gestiti restano intatti. Se la migrazione fallisce, l’app ripristina la posizione dei file e il catalogo originale. I file sorgente esterni non vengono spostati. Durante la copia le modifiche ai progetti sono sospese.

## Riproduzione e risultati

La toolbar unica, con altezza aumentata, contiene titolo del progetto, riproduzione/pausa, salti di 15 secondi, timeline, volume e pulsante Importa. Non esiste una seconda barra del player sotto la toolbar. Nei video, l’anteprima usa la stessa istanza del player, senza duplicare i controlli. Il click sul tempo di un segmento sposta la riproduzione.

Dopo la trascrizione la vista principale cambia: menu per scegliere la trascrizione del progetto e selettore Testo / Con tempi / Per voce, seguiti dal risultato. In alto a destra compaiono Importa, Nuova trascrizione, Nomi delle voci quando il risultato contiene speaker ed Esporta. I segmenti hanno spaziatura e padding ridotti. Nuova trascrizione apre il pannello modale mantenendo il testo precedente visibile dietro di esso. Una nuova esecuzione aggiunge una revisione senza sovrascrivere le precedenti. All’avvio il pannello si chiude e il progresso appare nella finestra principale.

I nomi delle voci si assegnano in un pannello dedicato e vengono salvati per ogni trascrizione. Il risultato originale del motore resta invariato. La pagina di esportazione è ancora da definire; il pulsante apre un pannello provvisorio che permette di raggiungere TXT/SRT/VTT/JSON già generati. Questi file raw conservano gli identificativi originali delle voci; l’app applica i nomi personalizzati nella visualizzazione.

## Impostazioni

Si aprono dal menu dell’app o con Cmd+,; nessun pulsante Impostazioni nella finestra principale.

- **Generali:** scelta della lingua dell’interfaccia; la guida introduttiva è riapribile dal menu Aiuto.
- **Hugging Face:** collegamento Hugging Face e guida in tre passaggi, con token protetto, incollaggio esplicito e Portachiavi. Nessun riferimento Hugging Face nella finestra principale e nessuna configurazione Hugging Face automatica al primo avvio.
- **Modelli:** posizioni e dimensioni dei dati effettivamente presenti nelle cache Hugging Face e PyTorch, totale e apertura nel Finder. I collegamenti simbolici delle snapshot non vengono conteggiati due volte. Un elemento in cache può contenere soltanto metadati: la presenza nell’elenco non significa che il modello sia completo o pronto.
- **Libreria:** posizione, numero di progetti/cartelle, spazio utilizzato, distinzione tra file gestiti ed esterni e cambio della posizione.

Se le voci sono richieste senza collegamento, l’app rimanda al pannello delle impostazioni. Il token viene verificato online e conservato nel Portachiavi, passato al worker tramite stdin ed escluso dai risultati e dagli argomenti del processo.

## Motore locale e limiti

Il worker e il pacchetto WhisperX sono inclusi in `Contents/Resources`. Nell’installer il runtime è esclusivamente quello incluso in `Contents/Runtime`, senza dipendenze da installazioni personali. Nella build di sviluppo il runtime è cercato prima in `Contents/Runtime/bin/python3`, poi nei virtualenv locali `~/repository/whisperX/.venv` e `~/Desktop/whisperXGUI/.venv`. Il codice non dipende più dalla presenza del checkout originale sul Desktop.

ASR tramite MLX/Metal quando disponibile, altrimenti CTranslate2 CPU; allineamento e voci su MPS quando disponibile. Le librerie scaricano i modelli al primo utilizzo. Mancano ancora un gestore dedicato dei download, una coda completa e l’editor del testo. Le fasi opzionali che falliscono conservano il testo con un avviso. I risultati sono salvati in directory distinte sotto `Projects/<project-id>/<run-id>` nella libreria.

Le versioni 0.1 e 0.2 conservavano i progetti solo in memoria: i loro cataloghi non possono essere recuperati dal disco. I risultati precedenti eventualmente salvati in `~/Library/Application Support/WhisperX/Transcriptions` restano intatti e non vengono spostati da questa versione.

## Verifica

```sh
swift test
```

Vedere `VERIFICA_UI.md` per i controlli automatici e visivi. Nessun commit o push: vedere `../AGENTS.md`.

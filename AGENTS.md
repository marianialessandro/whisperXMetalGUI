# Istruzioni per gli agenti

## Direzione del prodotto

Realizzare un’app macOS installabile e avviabile con doppio click, senza richiedere Terminale o installazioni manuali di Python e FFmpeg.

Seguire il progetto in `docs/PROGETTO_APP_MACOS.md`. La configurazione delle voci guida l’utente nella registrazione su Hugging Face, nell’accettazione delle condizioni del modello e nella creazione di un token di lettura. L’interfaccia deve permettere di incollare il token e completare automaticamente verifica e download.

La configurazione Hugging Face deve essere accessibile soltanto dal pannello Hugging Face delle Impostazioni. Non chiamare questo pannello Voci. Non aggiungere indicatori Hugging Face, pulsanti Configura le voci o wizard automatici nella finestra principale. Le Impostazioni devono avere anche pannelli Modelli e Libreria e si aprono dal menu dell’app, senza pulsante nella schermata principale.

Non assumere la disponibilità di un hosting proprietario per i modelli. Non includere token condivisi nell’app. Conservare le credenziali nel Portachiavi macOS ed escluderle da log, argomenti dei processi e file del progetto.

Il player deve essere nella toolbar insieme al titolo del progetto e alle azioni. Nuova cartella deve essere disponibile nel menu File con shortcut Cmd+Shift+N. Il cambio di posizione della libreria deve spostare i dati gestiti e rimuovere il catalogo precedente dopo la verifica, senza conservare backup della vecchia libreria; i file esterni non gestiti devono restare intatti.

# Installer autonomo macOS

Il comando `sh macOS/build-installer.sh` produce un unico `.pkg` che installa `WhisperX.app` in `/Applications`. L’app contiene CPython 3.12.12 standalone, le dipendenze Python bloccate e verificate con SHA-256, FFmpeg 7, FFprobe, le librerie native necessarie e il worker. Nessun download o installazione Python viene eseguito dall’installer sul Mac dell’utente.

## Preparazione sul Mac di build

Occorrono Apple Silicon, Xcode Command Line Tools con Swift 6 e `uv`. FFmpeg 7.1.5 viene scaricato dal sito ufficiale, verificato tramite SHA-256 e compilato automaticamente con Clang in un prefisso privato, senza librerie Homebrew. Questi strumenti servono soltanto a chi genera il pacchetto. Un prefisso FFmpeg 7 già preparato si può indicare facoltativamente con `--ffmpeg-prefix`. La build predefinita disabilita autodetect, rete e componenti esterni; include decodificatori, filtri audio, FFmpeg/FFprobe e le librerie condivise necessarie a TorchCodec, senza abilitare GPL o nonfree. L’archivio sorgente completo, la licenza LGPL e le opzioni di compilazione sono inclusi nelle attribuzioni. Non viene copiata la virtualenv personale di sviluppo.

```sh
sh macOS/build-installer.sh --help
sh macOS/build-installer.sh --output macOS/output/installer
```

Usare una directory di output nuova per ogni release. La build lavora in una directory temporanea separata e non sostituisce il bundle locale eventualmente aperto. Le dipendenze restano nella cache `uv` per le build successive. La directory `macOS/.packaging` contiene il Python standalone di build e i log locali; non va distribuita.

La build include librerie native ricorsivamente, riscrive i collegamenti assoluti con `@loader_path`, controlla l’architettura e calcola il minimo macOS dai binari effettivi. FFmpeg è compilato per macOS 14; il pacchetto installabile corrente richiede **Apple Silicon e macOS 15 o successivo**, in base ai wheel numerici inclusi. L’installer riporta il minimo effettivo nel suo controllo di compatibilità e in `RuntimeManifest.json`. L’interfaccia Swift resta compilabile per macOS 14; distribuire anche a macOS 14 richiede wheel numerici compilati per quel minimo e una nuova verifica.

Il pacchetto usa il normale Installer di macOS, verifica il sistema operativo e l’architettura e installa soltanto l’app. Non usa script privilegiati, non modifica Python di sistema, non crea servizi e non cancella dati dell’utente. Gli aggiornamenti sostituiscono il bundle conservando libreria, modelli e Portachiavi. Chiudere l’app prima di aggiornarla.

## Risorse personali

I modelli vengono scaricati al primo utilizzo, con lo stato del download mostrato dal motore nella finestra della trascrizione. Non sono inclusi nel `.pkg`. Serve Internet durante il primo download; i modelli scaricati restano nella cache dell’utente.

Il modello delle voci richiede la registrazione, l’accettazione delle condizioni e un token personale: questi passaggi non possono essere effettuati dall’installer per conto dell’utente. La configurazione resta esclusivamente in Impostazioni → Hugging Face. Il token viene conservato nel Portachiavi e passato al worker attraverso stdin; la build non legge né copia token o cache personali. Il download delle voci avviene quando si usa la funzione dopo il collegamento.

## Verifiche automatiche

Prima di generare il `.pkg`, la build firma i Mach-O dall’interno verso l’esterno e verifica la firma. Copia l’app in un percorso temporaneo con spazi e avvia il Python incluso in modalità isolata, senza Python o Homebrew nel PATH. Il controllo importa MLX, PyTorch, WhisperX, allineamento, pyannote e TorchCodec; genera un M4A e lo decodifica con FFmpeg/WhisperX e TorchCodec, senza rete o download di modelli.

I test Swift verificano anche l’ambiente del worker: esclusione delle variabili Python esterne, delle librerie iniettate e dei token ereditati, con mantenimento dei percorsi delle cache mostrati nelle Impostazioni.

Questa verifica non sostituisce una prova su un Mac pulito, una trascrizione reale per ogni profilo, l’accettazione delle condizioni delle voci o il primo avvio sotto Gatekeeper dopo un download.

## Firma e notarizzazione pubblica

Senza parametri viene creata una build locale con firma ad hoc e un installer non firmato. Non è una release pubblica notarizzata.

```sh
sh macOS/build-installer.sh --output macOS/output/release --app-identity 'Developer ID Application: Nome (TEAMID)' --installer-identity 'Developer ID Installer: Nome (TEAMID)' --notary-profile 'whisperx-release'
```

I due certificati devono essere già disponibili nel Portachiavi; il profilo `notarytool` deve essere configurato separatamente nel Portachiavi. Nessuna password va inserita nei file del progetto. Il comando applica Hardened Runtime, invia il `.pkg` ad Apple, allega e verifica il ticket e valuta il pacchetto con `spctl`. Le eccezioni del worker permettono JIT e memoria eseguibile per il motore numerico; non viene disabilitata la validazione delle librerie.

Prima di una release pubblica occorre verificare licenze e obblighi di redistribuzione delle dipendenze incluse. Il FFmpeg predefinito include il proprio archivio sorgente completo e la ricetta di compilazione. La build conserva le licenze fornite dai wheel e dal Python standalone e la licenza WhisperX; quando si usa un prefisso Homebrew personalizzato conserva anche le formule con versioni/opzioni/fonti dei componenti nativi. Per un prefisso alternativo, queste formule non sostituiscono l’archivio dei sorgenti corrispondenti eventualmente richiesto dalla sua licenza. Le licenze si trovano in `Contents/Resources/Licenses` e nelle directory delle dipendenze incluse.

Riferimenti: [Python standalone di uv](https://docs.astral.sh/uv/guides/install-python/), [firma Developer ID](https://developer.apple.com/developer-id/), [notarizzazione Apple](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution).

Il risultato della prova locale è documentato in [VERIFICA_INSTALLER.md](VERIFICA_INSTALLER.md).

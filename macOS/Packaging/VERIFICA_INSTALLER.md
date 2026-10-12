# Verifica installer — 11 ottobre 2026

Prodotto locale: `macOS/output/installer/WhisperX-0.6.0-arm64.pkg`, circa 371 MiB. App inclusa: versione 0.6.0, build 7, circa 1,3 GiB. Requisiti verificati nei Mach-O: Apple Silicon e macOS 15.0 o successivo; il minimo è imposto dalle librerie MLX.

SHA-256 del pacchetto:

```text
2a10b03350e5cb0e07d55dca78871e7f2bb5f9b36741afee17408e24faf269d7
```

## Verifiche completate

- `swift test --package-path macOS`: 15 test passati, inclusa l’esclusione delle variabili Python, delle librerie iniettate e dei token ereditati dall’ambiente del worker.
- Script shell verificati con `sh -n`, file Python compilati con `py_compile`, plist verificati con `plutil`.
- Build dell’app release e del runtime autonomo: CPython 3.12.12, PyTorch 2.8.0, TorchCodec 0.7.0, MLX 0.32.2, WhisperX incluso dal checkout, dipendenze bloccate con hash.
- FFmpeg 7.1.5 compilato dai sorgenti ufficiali verificati con SHA-256, senza dipendenze Homebrew, autodetect, rete, GPL o nonfree. Archivio sorgente, licenze e ricetta inclusi nelle risorse dell’app.
- Dipendenze native assolute esterne riscritte; collegamenti TorchAudio/TorchCodec risolti ai file inclusi. Nessuna variabile `DYLD_LIBRARY_PATH` globale: le librerie FFmpeg di PyAV conservano i propri collegamenti, senza essere sostituite da quelle del decoder separato.
- Firma ad hoc applicata ai Mach-O e al bundle; `codesign --verify --deep --strict` passa sia sull’app generata sia sull’app estratta dal PKG.
- App copiata in un percorso temporaneo con spazi, Python avviato con `-I -B`, senza Python/Homebrew nel PATH e con download Hugging Face disabilitati: import di MLX, PyTorch, TorchAudio, TorchVision, CTranslate2, faster-whisper, WhisperX, allineamento, pyannote e TorchCodec completato.
- Generazione di un M4A e decodifica con WhisperX/FFmpeg e TorchCodec: 16.384 campioni audio a 16 kHz ottenuti dal campione di un secondo; Metal disponibile.
- Generazione PKG con `pkgbuild` e `productbuild`, estrazione con `pkgutil --expand-full`, checksum verificato. Distribuzione limitata ad arm64 e macOS 15.0+, destinazione `/Applications`, nessuno script di installazione.
- Trascrizione dal bundle estratto dal PKG, con modello già presente e modalità offline Hugging Face/Transformers: backend MLX/Metal, lingua italiana, profilo Massima, nessuna richiesta di voci. Testo riconosciuto: «Questa è una prova della trascrizione. Il programma funziona sul Mac senza installare Python.» Nessun avviso; generati JSON, TXT, SRT e VTT.

Log, fixture e risultati locali sono in `macOS/.packaging/build.log`, `macOS/.packaging/swift-tests.log` e `macOS/.packaging/qa/`. Questi file e i pacchetti generati sono esclusi da Git. Nessun commit o push eseguito.

## Limiti della verifica

Il PKG è una build locale non firmata come installer; il bundle ha firma ad hoc. Su questo Mac non sono disponibili i certificati Developer ID Application/Installer necessari alla distribuzione pubblica. Il workflow di firma, notarizzazione, stapling e verifica Gatekeeper è predisposto ma non è stato eseguito.

Non è stata effettuata un’installazione privilegiata in `/Applications`, né una prova su un Mac pulito o un primo avvio da download con Gatekeeper. La trascrizione verifica un breve campione italiano con modello già presente; non certifica tutti i profili, lingue, file lunghi, download interrotti o la distinzione delle voci. La configurazione Hugging Face resta nel relativo pannello delle Impostazioni e usa credenziali personali nel Portachiavi. Nessun token personale è stato letto o incluso dalla build.

## Pulizia cache richiesta

Rimossa soltanto la cache temporanea `~/.cache/huggingface/xet`: 3.117.896.108 byte, circa 3,1 GB. Conservati i modelli Whisper large-v3 MLX, Whisper large-v3-turbo e pyannote community-1, i blob condivisi, i token e il catalogo dell’app. Non sono stati trovati download `.incomplete` o revisioni aggiuntive da eliminare nei repository esaminati.

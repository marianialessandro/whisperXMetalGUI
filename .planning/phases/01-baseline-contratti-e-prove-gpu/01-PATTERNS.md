# Fase 1: Baseline, contratti e prove GPU — Pattern Map
**Mappata:** 2026-10-12 · **File classificati:** 15 · **Analoghi:** 11/15.
Scope: strumenti di sviluppo proposti da CONTEXT/RESEARCH; nessuna modifica alla produzione, download o esecuzione test. Percorsi `freeze.py` e bridge sono rispettivamente inferito e condizionale.
## File Classification
| File nuovo | Ruolo | Flusso | Analogo esistente | Qualità |
|---|---|---|---|---|
| `tools/baseline/run.py` | utility | batch | `macOS/Packaging/build_installer.py` | exact |
| `tools/baseline/adapter.py` | service | batch/event-driven | `macOS/Worker/transcribe_worker.py` | exact |
| `tools/baseline/manifest.py` | utility | file-I/O | `macOS/Packaging/build_installer.py` | role-match |
| `tools/baseline/corpus.py` | utility | file-I/O | `macOS/Packaging/smoke_runtime.py` | role-match |
| `tools/baseline/contracts.py` | utility | transform/event-driven | `macOS/Worker/transcribe_worker.py` | role-match |
| `tools/baseline/metrics.py` | utility | transform | nessuno per evaluator completo | nuovo |
| `tools/baseline/profiling.py` | utility | event-driven | nessuno per misure sincronizzate | nuovo |
| `tools/baseline/freeze.py` | utility | file-I/O/transform | nessuno per gate calibrato | nuovo |
| `tools/baseline/credential_bridge.swift` | utility | request-response | nessuno per bridge standalone | nuovo |
| `tests/baseline/test_manifest.py` | test | transform | `macOS/Tests/WhisperXMacTests/WorkerRuntimeTests.swift` | role-match |
| `tests/baseline/test_contracts.py` | test | transform | `tests/test_word_timestamp_interpolation.py` | exact |
| `tests/baseline/test_corpus.py` | test | file-I/O | `macOS/Tests/WhisperXMacTests/LibraryTests.swift` | role-match |
| `tests/baseline/test_metrics.py` | test | transform | `tests/test_word_timestamp_interpolation.py` | role-match |
| `tests/baseline/test_profiling.py` | test | event-driven | `tests/test_word_timestamp_interpolation.py` | role-match |
| `tests/baseline/test_freeze.py` | test | file-I/O | `macOS/Tests/WhisperXMacTests/LibraryTests.swift` | role-match |
## Pattern Assignments
### Adapter e contracts: worker reale, stdout strutturato
Fonte `macOS/Worker/transcribe_worker.py:12-14`:
```python
def emit(kind, **values):
    sys.__stdout__.write(json.dumps({"event": kind, **values}, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()
```
Copiare NDJSON; richiamare `run(request)` (23-107) del worker selezionato, senza riscriverne ASR/align/diarize. Conservare stage/completed/failed, warnings, original/aligned/result e TXT/SRT/VTT. Patch misure scoped con ripristino in finally; verificare parità contro run senza instrumentation.
### Run e manifest: isolamento runtime, CLI separata
Fonte `macOS/Packaging/build_installer.py:106-109`:
```python
def clean_environment(runtime):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "DYLD", "HF_", "HUGGING_FACE"))}
    environment.update(PATH=f"{runtime / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return environment
```
Riutilizzare solo il pattern, non build/sign/install: subprocess argv esplicito (21-26), argparse/main separati (263-277), Python bundled -I -B (120). Manifest baseline è nuovo rispetto al manifest installer: SHA completo + dirty content, versioni loaded, modelli/hash/settings/hardware sanitizzati; niente env grezzo. Provision è comando separato con rete autorizzata, run offline.
### Corpus: audio sintetico temporaneo e FFmpeg bundled
Fonte `macOS/Packaging/smoke_runtime.py:30-34`:
```python
with tempfile.TemporaryDirectory(prefix="WhisperX audio ") as folder:
    audio = Path(folder) / "prova è audio.m4a"
    source = Path(folder) / "source.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
```
Copiare isolamento/nomi Unicode e risoluzione binario dal bundle (36), non sinusoidi come qualità naturale. Aggiungere licenze/hash/annotazioni e ledger atteso indipendente dai casi presenti; missing_license/prerequisite_missing bloccano gate.
### Test contracts/metrics: parole senza tempi sono valide
Fonte `tests/test_word_timestamp_interpolation.py:196-200`:
```python
        sentence_words = [
            {"word": "the", "start": 0.1, "end": 0.3},
            {"word": "99"},  # simulates a word with no CTC timestamps
            {"word": "cats", "start": 0.6, "end": 0.9},
        ]
```
Copiare fixture e asserzioni comportamentali (213-214), non imporre start/end/score obbligatori. I nuovi test usano unittest stdlib, non la struttura pytest legacy; bootstrap root da __file__ per -I. Metrics: edit analitici, denominatore completo, overlap/permutazioni; profiling: clock/drain fallito; freeze: mutazioni/caso obbligatorio rimosso.
### Test manifest: segreti fake e ambiente filtrato
Fonte `macOS/Tests/WhisperXMacTests/WorkerRuntimeTests.swift:14-18`:
```swift
        #expect(environment["DYLD_INSERT_LIBRARIES"] == nil)
        #expect(environment["HF_TOKEN"] == nil)
        #expect(environment["HUGGING_FACE_HUB_TOKEN"] == nil)
        #expect(environment["HF_HOME"] == "/Users/test/model cache")
        #expect(environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1")
```
Trasporre in unittest con sentinel fake anche in errori/log/manifest; cache autorizzata conservata, credenziali mai ereditate o serializzate.
### Test corpus/freeze: dati temporanei e cleanup
Fonte `macOS/Tests/WhisperXMacTests/LibraryTests.swift:20-23`:
```swift
    @Test func externalImportAndDeletePreserveOriginal() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let audio = try source(at: root)
```
Fixture root/UUID e UserDefaults dedicati (7-11); preservare file esterni (31) e rifiutare overwrite (105-119). Trasporre in TemporaryDirectory; nessun catalogo personale.
## Shared Patterns
Worker `save_result` (17-20): file .tmp + replace per JSON atomico; aggiungere validazione e allowlist prima della persistenza. Worker (125-131): failed sanitizzato, token sostituito prima di emit; adapter deve sanitizzare anche stderr/errori prima di salvarli. Token solo pipe privata dal Portachiavi, mai argv/env/file.
Compatibilità: tre profili reali e cleanup ModelHolder/cache (49-60), conteggio min=max (93-94), lingua auto/esplicita, hints e warning invariati. Warm-cache non è warm-model. Swift test esistenti restano analoghi/verification del contratto app, senza nuova UI.
## No Analog Found
Metrics, profiling, freeze e bridge sono nuovi: usare RESEARCH per policy, non inventare API o soglie. DER/JER/mapping riusano dipendenze bundled; synchronize + trace osservata sono distinti da Metal disponibile. Bridge condizionale deve usare il servizio Keychain corrente senza leggere segreti durante pianificazione.
Artefatti corpus/license/annotation manifest, fixture golden ed evaluation-contract.json: nuovi dati versionati, nomi finali al planner; non release o cataloghi personali.
## Metadata
Scope: worker, packaging, tests Python e Swift; 6 analoghi letti, ricerca fermata. Esame statico: nessun test/inferenza/download, commit/push o modifica source. Copertura: 3 exact, 8 role-match, 4 nuovi.

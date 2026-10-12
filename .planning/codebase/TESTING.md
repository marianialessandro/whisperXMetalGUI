# Testing Patterns

**Analysis Date:** 2026-10-12

## Test Framework

**Runner:** pytest (`pyproject.toml`, `.github/workflows/tests.yml`); Swift Testing (`macOS/Tests/WhisperXMacTests`).
**Assertion Library:** assert Python e `#expect` Swift.
**Run Commands:**
```bash
uv run pytest tests/ -v
swift test --package-path macOS
```
Questi sono comandi dedotti dalla configurazione, non eseguiti nella mappatura.

## Test File Organization

**Location:** directory separate `tests/` e `macOS/Tests/WhisperXMacTests/`.
**Naming:** `test_word_timestamp_interpolation.py`; `LibraryTests.swift`, `WorkerRuntimeTests.swift`, `LocalizationTests.swift`.
**Structure:** suite Python per wildcard/interpolazione; strutture Swift con funzioni `@Test`.

## Test Structure

**Suite Organization:**
```swift
import Testing
@testable import WhisperXMac
struct WorkerRuntimeTests {
    @Test func installedWorkerUsesOnlyBundledToolsAndNoInheritedCredentials() {
        // Preparare ambiente ereditato e verificare il filtro di WorkerRuntime.
    }
}
```
Schema ricavato da `macOS/Tests/WhisperXMacTests/WorkerRuntimeTests.swift`.

**Patterns:** `LibraryTests.swift` usa librerie temporanee e operazioni async; verifica file gestiti/esterni, cancellazione, trasferimento e rollback. `LocalizationTests.swift` verifica stringhe/risorse. Non attribuire a questi test precisione dei modelli.

## Mocking

**Framework:** unittest.mock.MagicMock (`tests/test_word_timestamp_interpolation.py`).
**Patterns:** `_make_mock_model` restituisce emissioni deterministiche con batch=1; `_make_emission` favorisce token noti su frame controllati; `torch.manual_seed(0)` nel helper.
**What to Mock:** inferenza per isolare invarianti CTC/wildcard, errori filesystem tramite writer iniettabile di `LibraryStore.swift:relocateLibrary`.
**What NOT to Mock:** nei gate qualità/GPU, modelli, kernel e backend effettivo devono essere reali; questa copertura non è presente nei test attuali.

## Fixtures and Factories

**Test Data:** dizionario sintetico, testo con cifre/caratteri sconosciuti, audio random, logits costruiti (`tests/test_word_timestamp_interpolation.py`).
**Location:** fixture inline nei test Python; directory temporanee in `LibraryTests.swift`; sinusoide WAV/AAC in `macOS/Packaging/smoke_runtime.py`.

## Coverage

**Requirements:** nessuna soglia di copertura rilevata in `pyproject.toml` o `.github/workflows/tests.yml`.
**View Coverage:** comando/configurazione dedicati non rilevati; non dichiarare copertura numerica.

## Test Types

**Unit Tests:** wildcard, interpolazione e ordinamento timestamp; isolamento ambiente, localizzazione (`tests/test_word_timestamp_interpolation.py`, `WorkerRuntimeTests.swift`).
**Integration Tests:** filesystem libreria (`LibraryTests.swift`); runtime installer e audio decode (`smoke_runtime.py`).
**E2E Tests:** nessuna suite automatica completa ASR→allineamento→diarizzazione→export rilevata; `macOS/VERIFICA_UI.md` e `Packaging/VERIFICA_INSTALLER.md` sono documenti di evidenza con limiti propri.

## Common Patterns

**Async Testing:** `@Test func ...() async throws` in `LibraryTests.swift`; attendere `waitUntilLoaded()` prima di usare il catalogo.
**Error Testing:** test del rollback mediante writeManifest che fallisce in `LibraryTests.swift`; preservare esterni e originali.
**Packaging:** `build_installer.py` richiama smoke dopo rilocazione del bundle e codesign verify; `smoke_runtime.py` importa le librerie, decodifica audio e interroga disponibilità Metal. Non esegue modelli ASR/CTC/community-1 e non dimostra copertura GPU.
**GPU migration gap:** servono confronti numerici kernel, WER/CER, errore timestamp, DER/JER/overlap, numero noto/automatico e tracce Metal su corpus documentato. Nessuna soglia o benchmark attuale viene assunto.

---
*Testing analysis: 2026-10-12. Nessun test lanciato durante la mappatura.*

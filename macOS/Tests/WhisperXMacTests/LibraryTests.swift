import Foundation
import Testing
@testable import WhisperXMac

@MainActor
struct LibraryTests {
    private func fixture() throws -> (URL, UserDefaults) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("WhisperXTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let preferences = UserDefaults(suiteName: root.lastPathComponent)!
        return (root, preferences)
    }

    private func source(at root: URL, name: String = "Intervista.wav") throws -> URL {
        let url = root.appendingPathComponent(name)
        try Data("synthetic test audio".utf8).write(to: url)
        return url
    }

    @Test func externalImportAndDeletePreserveOriginal() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let audio = try source(at: root)
        let store = LibraryStore(location: root.appendingPathComponent("Library"), preferences: preferences)
        await store.waitUntilLoaded()
        let id = try store.importFile(audio, move: false)
        #expect(!store.items[0].managed)
        #expect(store.items[0].url == audio)
        try store.deleteProject(id, trashAudio: true)
        #expect(store.items.isEmpty)
        #expect(FileManager.default.fileExists(atPath: audio.path))
    }

    @Test func managedImportMovesOnlySelectedSourceAndDeletionCanKeepIt() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let audio = try source(at: root)
        let other = try source(at: root, name: "Da conservare.wav")
        let store = LibraryStore(location: root.appendingPathComponent("Library"), preferences: preferences)
        await store.waitUntilLoaded()
        let id = try store.importFile(audio, move: true)
        let managedURL = store.items[0].url
        #expect(store.items[0].managed)
        #expect(!FileManager.default.fileExists(atPath: audio.path))
        #expect(FileManager.default.fileExists(atPath: managedURL.path))
        #expect(FileManager.default.fileExists(atPath: other.path))
        try store.deleteProject(id, trashAudio: false)
        #expect(FileManager.default.fileExists(atPath: managedURL.path))
    }

    @Test func titleFoldersAndSpeakerNamesSurviveReload() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let libraryURL = root.appendingPathComponent("Library")
        let store = LibraryStore(location: libraryURL, preferences: preferences)
        await store.waitUntilLoaded()
        let id = try store.importFile(source(at: root), move: false)
        store.renameProject(id, to: "Intervista a Giulia")
        store.createFolder("Interviste")
        let folderID = store.folders[0].id
        store.moveProject(id, to: folderID)
        let record = TranscriptionRecord(outputURL: store.outputDirectory(for: id), language: "it", quality: "Massima", requestedVoices: true, speakers: ["SPEAKER_00"], warnings: [])
        store.addTranscription(to: id, record: record)
        store.saveSpeakerNames(projectID: id, recordID: record.id, names: ["SPEAKER_00": " Giulia "])
        let loaded = LibraryStore(location: libraryURL, preferences: preferences)
        await loaded.waitUntilLoaded()
        #expect(loaded.items[0].title == "Intervista a Giulia")
        #expect(loaded.items[0].folderID == folderID)
        #expect(loaded.items[0].transcriptions[0].speakerNames["SPEAKER_00"] == "Giulia")
        loaded.removeFolder(folderID)
        #expect(loaded.items[0].folderID == nil)
        #expect(loaded.items[0].transcriptions.count == 1)
    }

    @Test func relocationMovesManagedFilesAndRemovesPreviousLibrary() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = root.appendingPathComponent("Old")
        let target = root.appendingPathComponent("New")
        let store = LibraryStore(location: old, preferences: preferences)
        await store.waitUntilLoaded()
        let managed = try store.importFile(source(at: root), move: true)
        let external = try source(at: root, name: "Esterno.wav")
        _ = try store.importFile(external, move: false)
        let output = store.outputDirectory(for: managed)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data("{\"segments\":[]}".utf8).write(to: output.appendingPathComponent("result.json"))
        store.addTranscription(to: managed, record: TranscriptionRecord(outputURL: output, language: "it", quality: "Massima", requestedVoices: false, speakers: [], warnings: []))
        let originalManaged = store.items[0].url
        try await store.relocateLibrary(to: target)
        #expect(store.libraryURL == target)
        #expect(store.items[0].url.path.hasPrefix(target.path + "/"))
        #expect(store.items[0].transcriptions[0].outputURL.path.hasPrefix(target.path + "/"))
        #expect(store.items[1].url == external)
        #expect(!FileManager.default.fileExists(atPath: originalManaged.path))
        #expect(!FileManager.default.fileExists(atPath: old.path))
        #expect(FileManager.default.fileExists(atPath: external.path))
        #expect(FileManager.default.fileExists(atPath: store.items[0].url.path))
        #expect(FileManager.default.fileExists(atPath: store.items[0].transcriptions[0].outputURL.appendingPathComponent("result.json").path))
        let reloaded = LibraryStore(location: target, preferences: preferences)
        await reloaded.waitUntilLoaded()
        #expect(reloaded.items.count == 2)
    }

    @Test func occupiedDestinationIsNotOverwritten() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = root.appendingPathComponent("Old")
        let target = root.appendingPathComponent("New")
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        let protected = try source(at: target)
        let store = LibraryStore(location: old, preferences: preferences)
        await store.waitUntilLoaded()
        do {
            try await store.relocateLibrary(to: target)
            Issue.record("A destination containing user data must be refused")
        } catch {}
        #expect(store.libraryURL == old)
        #expect(try Data(contentsOf: protected) == Data("synthetic test audio".utf8))
    }

    @Test func migrationFailureRestoresFilesAndOriginalCatalog() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = root.appendingPathComponent("Old")
        let target = root.appendingPathComponent("New")
        let store = LibraryStore(location: old, preferences: preferences)
        await store.waitUntilLoaded()
        _ = try store.importFile(source(at: root), move: true)
        let audio = store.items[0].url
        let originalCatalog = try Data(contentsOf: old.appendingPathComponent("Library.json"))
        do {
            try await store.relocateLibrary(to: target, writeManifest: { _, _ in throw CocoaError(.fileWriteOutOfSpace) })
            Issue.record("A failed catalog write must fail migration")
        } catch {}
        #expect(store.libraryURL == old)
        #expect(store.items[0].url == audio)
        #expect(FileManager.default.fileExists(atPath: audio.path))
        #expect(!FileManager.default.fileExists(atPath: target.appendingPathComponent("Media").path))
        #expect(try Data(contentsOf: old.appendingPathComponent("Library.json")) == originalCatalog)
    }

    @Test func migrationDoesNotDeleteUnmanagedFilesInOldDirectory() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = root.appendingPathComponent("Old")
        let target = root.appendingPathComponent("New")
        let store = LibraryStore(location: old, preferences: preferences)
        await store.waitUntilLoaded()
        let external = try source(at: old, name: "Sorgente esterna.wav")
        _ = try store.importFile(external, move: false)
        try await store.relocateLibrary(to: target)
        #expect(FileManager.default.fileExists(atPath: external.path))
        #expect(!FileManager.default.fileExists(atPath: old.appendingPathComponent("Library.json").path))
        #expect(store.items[0].url == external)
    }

    @Test func unreadableManifestIsNeverOverwritten() async throws {
        let (root, preferences) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = root.appendingPathComponent("Library.json")
        let original = Data("broken manifest".utf8)
        try original.write(to: manifest)
        let store = LibraryStore(location: root, preferences: preferences)
        await store.waitUntilLoaded()
        store.renameProject(UUID(), to: "ignored")
        store.createFolder("ignored")
        #expect(store.error != nil)
        #expect(try Data(contentsOf: manifest) == original)
    }

    @Test func modelWeightDoesNotCountSnapshotSymlinksTwice() throws {
        let (root, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let blob = root.appendingPathComponent("weights")
        try Data(repeating: 0, count: 2048).write(to: blob)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("snapshot"), withDestinationURL: blob)
        #expect(StorageInventory.size(of: root) == 2048)
    }
}

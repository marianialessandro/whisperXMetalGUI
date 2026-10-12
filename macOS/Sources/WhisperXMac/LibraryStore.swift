import AppKit
import AVFoundation
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class LibraryStore {
    var items: [MediaItem] = [] { didSet { persist() } }
    var folders: [ProjectFolder] = [] { didSet { persist() } }
    var selection: UUID?
    var search = ""
    var error: String?
    var pendingImports: [URL] = []
    var libraryURL: URL
    var isLoading = true
    var isReady: Bool { ready }
    private let preferences: UserDefaults
    private var loadTask: Task<Void, Never>?
    private var ready = false
    private var relocationInProgress = false
    var isRelocating: Bool { relocationInProgress }
    var filteredItems: [MediaItem] { items.filter { search.isEmpty || $0.title.localizedCaseInsensitiveContains(search) } }

    init(location: URL? = nil, preferences: UserDefaults = .standard) {
        self.preferences = preferences
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let override = ProcessInfo.processInfo.environment["WHISPERX_LIBRARY_PATH"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        libraryURL = location ?? override ?? preferences.url(forKey: "libraryLocation") ?? base.appendingPathComponent("WhisperX/Library", isDirectory: true)
        let root = libraryURL
        loadTask = Task {
            defer { isLoading = false }
            do {
                let content = try await Task.detached(priority: .utility) {
                    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                    let manifest = root.appendingPathComponent("Library.json")
                    guard FileManager.default.fileExists(atPath: manifest.path) else { return LibraryManifest(items: [], folders: []) }
                    let content = try JSONDecoder().decode(LibraryManifest.self, from: Data(contentsOf: manifest))
                    guard content.version == 1 else { throw CocoaError(.fileReadCorruptFile) }
                    return content
                }.value
                items = content.items
                folders = content.folders
                ready = true
                selection = items.first?.id
            } catch {
                self.error = L10n.tr("Non riesco ad aprire la libreria. I dati esistenti sono stati conservati. \(error.localizedDescription)")
            }
        }
    }

    func waitUntilLoaded() async { await loadTask?.value }

    func persist() {
        guard ready, !relocationInProgress else { return }
        do { try JSONEncoder().encode(LibraryManifest(items: items, folders: folders)).write(to: libraryURL.appendingPathComponent("Library.json"), options: .atomic) }
        catch { self.error = L10n.tr("Non riesco a salvare la libreria. \(error.localizedDescription)") }
    }

    func chooseFiles() {
        guard !relocationInProgress else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .movie]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.prompt = L10n.tr("Importa")
        if let window = NSApp.keyWindow {
            panel.beginSheetModal(for: window) { response in if response == .OK { self.requestImport(panel.urls) } }
        } else {
            panel.begin { response in if response == .OK { self.requestImport(panel.urls) } }
        }
    }

    func requestImport(_ urls: [URL]) {
        pendingImports = urls.filter { $0.isFileURL }
    }

    @discardableResult func finishImport(move: Bool) -> UUID? {
        let urls = pendingImports
        pendingImports = []
        var lastImported: UUID?
        for url in urls {
            do { lastImported = try importFile(url, move: move) }
            catch { self.error = L10n.tr("Non riesco a importare \(url.lastPathComponent). \(error.localizedDescription)") }
        }
        return lastImported
    }

    @discardableResult func importFile(_ url: URL, move: Bool) throws -> UUID {
        guard ready else { throw CocoaError(.fileWriteNoPermission) }
        guard url.isFileURL, let type = UTType(filenameExtension: url.pathExtension), type.conforms(to: .audio) || type.conforms(to: .movie) else { throw CocoaError(.fileReadUnsupportedScheme) }
        let source = url.standardizedFileURL
        if let existing = items.first(where: { $0.url.standardizedFileURL == source || $0.originalURL.standardizedFileURL == source }) {
            selection = existing.id
            return existing.id
        }
        guard FileManager.default.fileExists(atPath: source.path) else { throw CocoaError(.fileNoSuchFile) }
        var item = MediaItem(url: source)
        if move {
            let directory = libraryURL.appendingPathComponent("Media/\(item.id.uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent(source.lastPathComponent)
            try FileManager.default.moveItem(at: source, to: destination)
            item.url = destination
            item.managed = true
        }
        items.append(item)
        selection = item.id
        readDuration(id: item.id, url: item.url)
        return item.id
    }

    private func readDuration(id: UUID, url: URL) {
        Task {
            do {
                let asset = AVURLAsset(url: url)
                let duration = try await asset.load(.duration)
                let audioTracks = try await asset.loadTracks(withMediaType: .audio)
                guard let index = items.firstIndex(where: { $0.id == id }) else { return }
                if audioTracks.isEmpty || !duration.seconds.isFinite || duration.seconds <= 0 {
                    items[index].duration = "Audio non disponibile"
                    items[index].problem = L10n.tr("Non riesco a leggere una traccia audio valida in questo file.")
                } else {
                    let seconds = Int(duration.seconds)
                    items[index].durationSeconds = duration.seconds
                    items[index].duration = String(format: "%d:%02d", seconds / 60, seconds % 60)
                }
            } catch {
                guard let index = items.firstIndex(where: { $0.id == id }) else { return }
                items[index].duration = "Anteprima non disponibile"
                items[index].problem = L10n.tr("Anteprima non disponibile. La trascrizione proverà ad aprire l’audio con FFmpeg.")
            }
        }
    }

    func renameProject(_ id: UUID, to title: String) {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].title = title
    }

    func createFolder(_ name: String, parentID: UUID? = nil) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        folders.append(ProjectFolder(name: trimmed, parentID: parentID))
    }


    func removeFolder(_ id: UUID) {
        let parent = folders.first(where: { $0.id == id })?.parentID
        for index in items.indices where items[index].folderID == id { items[index].folderID = parent }
        for index in folders.indices where folders[index].parentID == id { folders[index].parentID = parent }
        folders.removeAll { $0.id == id }
    }

    func moveProject(_ id: UUID, to folderID: UUID?) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].folderID = folderID
    }

    func renameFolder(_ id: UUID, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let index = folders.firstIndex(where: { $0.id == id }) else { return }
        folders[index].name = trimmed
    }

    func outputDirectory(for id: UUID) -> URL {
        libraryURL.appendingPathComponent("Projects/\(id.uuidString)/\(UUID().uuidString)", isDirectory: true)
    }

    func addTranscription(to id: UUID, record: TranscriptionRecord) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].transcriptions.append(record)
    }

    func saveSpeakerNames(projectID: UUID, recordID: UUID, names: [String: String]) {
        guard let project = items.firstIndex(where: { $0.id == projectID }), let record = items[project].transcriptions.firstIndex(where: { $0.id == recordID }) else { return }
        items[project].transcriptions[record].speakerNames = names.mapValues { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.value.isEmpty }
    }

    func deleteProject(_ id: UUID, trashAudio: Bool) throws {
        guard ready, let item = items.first(where: { $0.id == id }) else { return }
        if trashAudio && item.managed && FileManager.default.fileExists(atPath: item.url.path) {
            try FileManager.default.trashItem(at: item.url, resultingItemURL: nil)
        }
        let results = libraryURL.appendingPathComponent("Projects/\(id.uuidString)", isDirectory: true)
        if FileManager.default.fileExists(atPath: results.path) { try FileManager.default.trashItem(at: results, resultingItemURL: nil) }
        items.removeAll { $0.id == id }
        if selection == id { selection = items.first?.id }
    }

    func relocateLibrary(to destination: URL, writeManifest: @escaping @Sendable (Data, URL) throws -> Void = { data, url in try data.write(to: url, options: .atomic) }) async throws {
        guard ready, !relocationInProgress else { throw CocoaError(.fileWriteNoPermission) }
        let destination = destination.standardizedFileURL
        let previous = libraryURL.standardizedFileURL
        let resolvedDestination = destination.resolvingSymlinksInPath()
        let resolvedPrevious = previous.resolvingSymlinksInPath()
        guard resolvedDestination != resolvedPrevious else { return }
        guard !resolvedDestination.path.hasPrefix(resolvedPrevious.path + "/"), !resolvedPrevious.path.hasPrefix(resolvedDestination.path + "/") else { throw CocoaError(.fileWriteInvalidFileName) }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let children = try FileManager.default.contentsOfDirectory(at: destination, includingPropertiesForKeys: nil).filter { $0.lastPathComponent != ".DS_Store" }
        guard children.isEmpty else { throw CocoaError(.fileWriteFileExists) }
        relocationInProgress = true
        defer { relocationInProgress = false }
        let previousItems = items
        let originalManifest = previous.appendingPathComponent("Library.json")
        let originalData = try await Task.detached(priority: .utility) { FileManager.default.fileExists(atPath: originalManifest.path) ? try Data(contentsOf: originalManifest) : nil }.value
        var movedNames: [String] = []
        do {
            for name in ["Media", "Projects"] {
                let moved = try await Task.detached(priority: .utility) {
                    let source = previous.appendingPathComponent(name)
                    guard FileManager.default.fileExists(atPath: source.path) else { return false }
                    try FileManager.default.moveItem(at: source, to: destination.appendingPathComponent(name))
                    return true
                }.value
                if moved { movedNames.append(name) }
            }
            for index in items.indices {
                if items[index].managed { items[index].url = relocated(items[index].url, from: previous, to: destination) }
                for record in items[index].transcriptions.indices { items[index].transcriptions[record].outputURL = relocated(items[index].transcriptions[record].outputURL, from: previous, to: destination) }
            }
            let content = try JSONEncoder().encode(LibraryManifest(items: items, folders: folders))
            try await Task.detached(priority: .utility) {
                let manifest = destination.appendingPathComponent("Library.json")
                try writeManifest(content, manifest)
                _ = try JSONDecoder().decode(LibraryManifest.self, from: Data(contentsOf: manifest))
                if FileManager.default.fileExists(atPath: originalManifest.path) { try FileManager.default.removeItem(at: originalManifest) }
                let remaining = try FileManager.default.contentsOfDirectory(at: previous, includingPropertiesForKeys: nil)
                if remaining.allSatisfy({ $0.lastPathComponent == ".DS_Store" }) { try FileManager.default.removeItem(at: previous) }
            }.value
            libraryURL = destination
            preferences.set(destination, forKey: "libraryLocation")
        } catch {
            items = previousItems
            let names = movedNames
            do {
                try await Task.detached(priority: .utility) {
                    try FileManager.default.createDirectory(at: previous, withIntermediateDirectories: true)
                    for name in names.reversed() { try FileManager.default.moveItem(at: destination.appendingPathComponent(name), to: previous.appendingPathComponent(name)) }
                    if let originalData { try originalData.write(to: originalManifest, options: .atomic) }
                    let targetManifest = destination.appendingPathComponent("Library.json")
                    if FileManager.default.fileExists(atPath: targetManifest.path) { try FileManager.default.removeItem(at: targetManifest) }
                }.value
            } catch {
                self.error = L10n.tr("Lo spostamento è stato interrotto e il ripristino non è completo. I file restano recuperabili nelle cartelle \(previous.path) e \(destination.path).")
                throw error
            }
            throw error
        }
    }

    private func relocated(_ url: URL, from source: URL, to destination: URL) -> URL {
        let prefix = source.path + "/"
        guard url.path.hasPrefix(prefix) else { return url }
        return destination.appendingPathComponent(String(url.path.dropFirst(prefix.count)))
    }

    func relinkProject(_ id: UUID) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .movie]
        panel.prompt = L10n.tr("Ricollega")
        panel.begin { response in
            guard response == .OK, let url = panel.url, let index = self.items.firstIndex(where: { $0.id == id }) else { return }
            self.items[index].url = url
            self.items[index].originalURL = url
            self.items[index].managed = false
            self.items[index].problem = nil
            self.readDuration(id: id, url: url)
        }
    }
}

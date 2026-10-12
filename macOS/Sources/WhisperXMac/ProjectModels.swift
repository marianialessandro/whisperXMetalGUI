import Foundation
import UniformTypeIdentifiers

struct ProjectFolder: Identifiable, Codable, Hashable, Sendable {
    var id = UUID()
    var name: String
    var parentID: UUID?
}

struct TranscriptionRecord: Identifiable, Codable, Sendable {
    var id = UUID()
    var createdAt = Date()
    var outputURL: URL
    var language: String
    var quality: String
    var requestedVoices: Bool
    var speakers: [String]
    var warnings: [String]
    var speakerNames: [String: String] = [:]
    var label: String { "\(createdAt.formatted(.dateTime.day().month(.abbreviated).year().hour().minute().locale(L10n.locale))) · \(L10n.display(quality))\(requestedVoices ? L10n.tr(" · Voci") : "")" }
}

struct MediaItem: Identifiable, Codable, Sendable {
    var id = UUID()
    var url: URL
    var originalURL: URL
    var title: String
    var managed = false
    var folderID: UUID?
    var duration = "Leggo il file…"
    var durationSeconds = 0.0
    var problem: String?
    var language = "auto"
    var quality = "Bilanciata"
    var voices = false
    var knownSpeakers = false
    var speakerCount = 2
    var hints = ""
    var transcriptions: [TranscriptionRecord] = []
    var isVideo: Bool { UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) == true }

    init(url: URL, managed: Bool = false, originalURL: URL? = nil) {
        self.url = url
        self.originalURL = originalURL ?? url
        self.title = url.deletingPathExtension().lastPathComponent
        self.managed = managed
    }
}

struct LibraryManifest: Codable, Sendable {
    var version = 1
    var items: [MediaItem]
    var folders: [ProjectFolder]
}

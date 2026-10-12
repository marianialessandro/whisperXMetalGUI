import AppKit
import Darwin
import Foundation
import Observation

struct TranscriptSegment: Decodable, Identifiable {
    let start: Double
    let end: Double
    let text: String
    let speaker: String?
    var id: String { "\(start)-\(end)-\(text)" }
}

struct Transcript: Decodable {
    let segments: [TranscriptSegment]
}

@MainActor @Observable
final class TranscriptionEngine {
    var activeID: UUID?
    var requestedID: UUID?
    var completedID: UUID?
    var stage = ""
    var backend = ""
    var error: String?
    var warnings: [String] = []
    var segments: [TranscriptSegment] = []
    var outputURL: URL?
    private var process: Process?
    private var task: Task<Void, Never>?
    private var onCompletion: ((URL, [String], [String]) -> Void)?

    var isRunning: Bool { process != nil }

    func cancel() {
        guard let process, process.isRunning else { return }
        if kill(-process.processIdentifier, SIGTERM) != 0 { process.terminate() }
        activeID = nil
        stage = L10n.tr("Trascrizione annullata")
        error = L10n.tr("Trascrizione annullata. Puoi avviarla di nuovo quando vuoi.")
    }

    func start(_ item: MediaItem, outputDirectory: URL, onCompletion: @escaping (URL, [String], [String]) -> Void) {
        guard !isRunning else { return }
        self.onCompletion = onCompletion
        requestedID = item.id
        error = nil
        warnings = []
        stage = L10n.tr("Avvio il motore…")
        backend = ""
        let home = FileManager.default.homeDirectoryForCurrentUser
        let bundledPython = Bundle.main.bundleURL.appendingPathComponent("Contents/Runtime/bin/python3")
        let isInstallerBuild = Bundle.main.url(forResource: "RuntimeManifest", withExtension: "json") != nil
        let candidates = isInstallerBuild ? [bundledPython] : [bundledPython, home.appendingPathComponent("repository/whisperX/.venv/bin/python"), home.appendingPathComponent("Desktop/whisperXGUI/.venv/bin/python")]
        guard let python = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }), let worker = Bundle.main.url(forResource: "transcribe_worker", withExtension: "py") else {
            error = L10n.tr("Il motore Python non è disponibile in questa build. Serve completare il pacchetto dell’app prima di distribuirla su un altro Mac.")
            return
        }
        do {
            let folder = outputDirectory
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var configuration: [String: Any] = ["audio": item.url.path, "output": folder.path, "language": item.language, "quality": item.quality, "voices": item.voices, "hints": item.hints]
            if item.voices {
                guard let token = try TokenVault.load() else { throw ConnectionFailure.invalidToken }
                configuration["token"] = token
                if item.knownSpeakers { configuration["speaker_count"] = item.speakerCount }
            }
            let inputData = try JSONSerialization.data(withJSONObject: configuration)
            let child = Process()
            child.executableURL = python
            child.arguments = ["-I", "-B", "-u", worker.path]
            child.currentDirectoryURL = Bundle.main.resourceURL
            child.environment = WorkerRuntime.environment(bundle: Bundle.main.bundleURL, inherited: ProcessInfo.processInfo.environment, bundled: python == bundledPython)
            let input = Pipe()
            let output = Pipe()
            child.standardInput = input
            child.standardOutput = output
            child.standardError = FileHandle.nullDevice
            try child.run()
            process = child
            activeID = item.id
            try input.fileHandleForWriting.write(contentsOf: inputData)
            try input.fileHandleForWriting.close()
            task = Task {
                do {
                    for try await line in output.fileHandleForReading.bytes.lines {
                        guard !Task.isCancelled else { return }
                        receive(line, for: item.id)
                    }
                    if activeID == item.id {
                        error = L10n.tr("Il motore si è interrotto prima di completare il lavoro. Puoi riprovare.")
                        activeID = nil
                    }
                } catch {
                    if !Task.isCancelled { self.error = L10n.tr("Non riesco a leggere il risultato del motore. Riprova."); activeID = nil }
                }
                process = nil
                task = nil
            }
        } catch {
            process?.terminate()
            process = nil
            activeID = nil
            self.error = error.localizedDescription
        }
    }

    private func receive(_ line: String, for id: UUID) {
        guard let data = line.data(using: .utf8), let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let kind = event["event"] as? String else { return }
        switch kind {
        case "stage":
            stage = event["message"] as? String ?? stage
            backend = event["backend"] as? String ?? backend
        case "failed":
            error = (event["message"] as? String).map(L10n.display) ?? L10n.tr("Trascrizione non riuscita.")
            activeID = nil
        case "completed":
            do {
                guard let path = event["result"] as? String, let output = event["output"] as? String else { throw CocoaError(.fileReadCorruptFile) }
                let content = try Data(contentsOf: URL(fileURLWithPath: path))
                segments = try JSONDecoder().decode(Transcript.self, from: content).segments
                warnings = event["warnings"] as? [String] ?? []
                outputURL = URL(fileURLWithPath: output)
                completedID = id
                onCompletion?(URL(fileURLWithPath: output), Array(Set(segments.compactMap(\.speaker))).sorted(), warnings)
                onCompletion = nil
                stage = L10n.tr("Trascrizione completata")
                activeID = nil
            } catch {
                self.error = L10n.tr("Il risultato non è leggibile. Riprova la trascrizione.")
                activeID = nil
            }
        default: break
        }
    }
}

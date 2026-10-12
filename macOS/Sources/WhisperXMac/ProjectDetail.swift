import AVKit
import SwiftUI

struct ProjectDetail: View {
    @Binding var item: MediaItem
    var library: LibraryStore
    var connection: HuggingFaceConnection
    var engine: TranscriptionEngine
    var playback: PlaybackController
    @Binding var configurationRequest: UUID?
    @State private var selectedRecordID: UUID?
    @State private var presentation = "Con tempi"
    @State private var segments: [TranscriptSegment] = []
    @State private var resultError: String?
    @State private var configureNew = false
    @State private var showNames = false
    @State private var showExport = false
    @State private var names: [String: String] = [:]
    @State private var needsConnection = false

    private var record: TranscriptionRecord? { item.transcriptions.first { $0.id == selectedRecordID } ?? item.transcriptions.last }
    private var showingResult: Bool { record != nil }

    var body: some View {
        VStack(spacing: 0) {
            if item.isVideo {
                NativeMediaPlayer(player: playback.player).frame(height: 200).padding(.horizontal, 28).padding(.top, 18)
            }
            if showingResult, let record {
                resultView(record)
            } else {
                awaitingTranscription
            }
            if engine.isRunning && engine.requestedID == item.id {
                HStack(spacing: 12) {
                    ProgressView().controlSize(.small)
                    Text(L10n.display(engine.stage)).font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button(L10n.tr("Annulla")) { engine.cancel() }
                }.padding(16).background(.bar)
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button { configureNew = true } label: { Label(L10n.tr("Nuova trascrizione"), systemImage: "arrow.clockwise") }.help(L10n.tr("Configura una trascrizione")).disabled(engine.isRunning)
                if let record {
                    if !record.speakers.isEmpty {
                        Button { names = record.speakerNames; showNames = true } label: { Label(L10n.tr("Nomi delle voci"), systemImage: "person.2") }.help(L10n.tr("Assegna nomi alle voci"))
                    }
                    Button { showExport = true } label: { Label(L10n.tr("Esporta"), systemImage: "square.and.arrow.up") }
                }
            }
        }
        .task(id: record?.id) { loadResult() }
        .onChange(of: item.transcriptions.count) { _, _ in
            selectedRecordID = item.transcriptions.last?.id
            configureNew = false
            loadResult()
        }
        .sheet(isPresented: Binding(get: { configureNew || configurationRequest == item.id }, set: { if !$0 { closeConfiguration() } })) {
            configurationView.frame(width: 640, height: 590)
                .alert(L10n.tr("Configura l’accesso per le voci"), isPresented: $needsConnection) {
                    Button("OK", role: .cancel) {}
                } message: { Text(L10n.tr("Dal menu WhisperX → Impostazioni… apri il pannello Hugging Face e completa la configurazione, oppure disattiva Distingui le voci.")) }
        }
        .sheet(isPresented: $showNames) {
            if let record {
                SpeakerNamesView(speakers: record.speakers, names: $names) {
                    library.saveSpeakerNames(projectID: item.id, recordID: record.id, names: names)
                    showNames = false
                } cancel: { showNames = false }
            }
        }
        .sheet(isPresented: $showExport) {
            VStack(alignment: .leading, spacing: 22) {
                Text(L10n.tr("Esporta trascrizione")).font(.title2.weight(.semibold))
                Text(L10n.tr("La pagina di esportazione verrà definita nel prossimo passo. Puoi già aprire i file TXT, SRT, VTT e JSON generati.")).foregroundStyle(.secondary)
                HStack {
                    Button(L10n.tr("Chiudi")) { showExport = false }
                    Spacer()
                    Button(L10n.tr("Apri file generati")) { if let record { NSWorkspace.shared.open(record.outputURL) } }.buttonStyle(.borderedProminent)
                }
            }.padding(28).frame(width: 480)
        }
    }

    private var awaitingTranscription: some View {
        VStack(spacing: 18) {
            Image(systemName: "text.alignleft").font(.system(size: 36)).foregroundStyle(.secondary)
            Text(engine.isRunning && engine.requestedID == item.id ? L10n.tr("La trascrizione è in corso") : L10n.tr("Nessuna trascrizione")).font(.title2.weight(.semibold))
            if !engine.isRunning {
                Button(L10n.tr("Configura trascrizione…")) { configureNew = true }.buttonStyle(.borderedProminent)
            }
            if engine.requestedID == item.id, let error = engine.error { Text(L10n.display(error)).font(.callout).foregroundStyle(.red).padding(.horizontal, 24) }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func closeConfiguration() { configureNew = false; configurationRequest = nil }

    private var configurationView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                HStack {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(item.title).font(.system(size: 28, weight: .semibold, design: .rounded))
                        Label(L10n.display(item.duration), systemImage: "clock").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(L10n.tr("Chiudi"), systemImage: "xmark") { closeConfiguration() }.labelStyle(.iconOnly).buttonStyle(.plain).help(L10n.tr("Chiudi configurazione"))
                }
                if !FileManager.default.fileExists(atPath: item.url.path) {
                    Label(L10n.tr("Il file sorgente non è più nella sua posizione."), systemImage: "exclamationmark.circle").foregroundStyle(.orange)
                    Button(L10n.tr("Ricollega file…")) { library.relinkProject(item.id) }
                }
                VStack(alignment: .leading, spacing: 22) {
                    Text(L10n.tr("Impostazioni della trascrizione")).font(.title3.weight(.semibold))
                    HStack(spacing: 24) {
                        Picker(L10n.tr("Lingua"), selection: $item.language) {
                            Text(L10n.tr("Automatica")).tag("auto")
                            Text(L10n.tr("Italiano")).tag("it")
                            Text(L10n.tr("Inglese")).tag("en")
                            Text(L10n.tr("Francese")).tag("fr")
                            Text(L10n.tr("Tedesco")).tag("de")
                            Text(L10n.tr("Spagnolo")).tag("es")
                        }
                        Picker(L10n.tr("Qualità"), selection: $item.quality) {
                            ForEach(["Rapida", "Bilanciata", "Massima"], id: \.self) { Text(L10n.display($0)).tag($0) }
                        }
                    }
                    Divider()
                    Toggle(isOn: $item.voices) {
                        VStack(alignment: .leading, spacing: 5) {
                            Text(L10n.tr("Distingui le voci")).font(.callout.weight(.medium))
                            Text(L10n.tr("Assegna una voce a ogni persona che parla.")).font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }.toggleStyle(.switch).accessibilityLabel(L10n.tr("Distingui le voci"))
                    if item.voices {
                        Toggle(L10n.tr("Conosco il numero di voci"), isOn: $item.knownSpeakers)
                        if item.knownSpeakers { Stepper(L10n.tr("\(item.speakerCount) voci"), value: $item.speakerCount, in: 1...99) }
                    }
                    TextField(L10n.tr("Parole o nomi utili da riconoscere (facoltativo)"), text: $item.hints).textFieldStyle(.roundedBorder)
                }.padding(24).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 16))
                HStack(spacing: 16) {
                    Button(L10n.tr("Annulla")) { closeConfiguration() }.keyboardShortcut(.cancelAction)
                    Text(engine.isRunning ? L10n.tr("Attendi che la trascrizione in corso termini.") : L10n.tr("I modelli necessari vengono preparati al primo utilizzo.")).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(L10n.tr("Trascrivi"), systemImage: "waveform") { transcribe() }.buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(engine.isRunning || !FileManager.default.fileExists(atPath: item.url.path))
                }
                if engine.requestedID == item.id, let error = engine.error { Label(L10n.display(error), systemImage: "exclamationmark.circle").foregroundStyle(.red).textSelection(.enabled) }
            }.padding(28).frame(maxWidth: .infinity)
        }
    }

    private func resultView(_ record: TranscriptionRecord) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 22) {
                Picker(L10n.tr("Trascrizione"), selection: Binding(get: { self.record?.id ?? record.id }, set: { selectedRecordID = $0 })) {
                    ForEach(item.transcriptions) { Text($0.label).tag($0.id) }
                }.frame(maxWidth: 480)
                Spacer()
                Picker(L10n.tr("Visualizzazione"), selection: $presentation) {
                    Text(L10n.tr("Testo")).tag("Testo")
                    Text(L10n.tr("Con tempi")).tag("Con tempi")
                    if !record.speakers.isEmpty { Text(L10n.tr("Per voce")).tag("Per voce") }
                }.frame(width: 210)
            }.padding(24)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(record.warnings, id: \.self) { Label(L10n.display($0), systemImage: "exclamationmark.circle").font(.callout).foregroundStyle(.secondary) }
                    if let resultError { Label(L10n.display(resultError), systemImage: "exclamationmark.circle").foregroundStyle(.red) }
                    else if segments.isEmpty { Text(L10n.tr("Non ho rilevato parlato in questo file.")).foregroundStyle(.secondary) }
                    else if presentation == "Testo" {
                        Text(segments.map(\.text).joined(separator: "\n")).font(.body).lineSpacing(3).textSelection(.enabled)
                    } else if presentation == "Per voce" {
                        ForEach(record.speakers + (segments.contains { $0.speaker == nil } ? ["__unassigned"] : []), id: \.self) { speaker in
                            VStack(alignment: .leading, spacing: 8) {
                                Text(speaker == "__unassigned" ? L10n.tr("Voce non assegnata") : record.speakerNames[speaker] ?? speaker).font(.title3.weight(.semibold)).foregroundStyle(.indigo)
                                ForEach(Array(segments.filter { speaker == "__unassigned" ? $0.speaker == nil : $0.speaker == speaker }.enumerated()), id: \.offset) { _, segment in
                                    HStack(alignment: .top, spacing: 16) {
                                        Button(timeLabel(segment.start)) { playback.seek(segment.start) }.buttonStyle(.link).font(.callout.monospacedDigit()).frame(width: 65)
                                        Text(segment.text).lineSpacing(2).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                                    }
                                }
                            }.padding(12).frame(maxWidth: .infinity, alignment: .leading).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
                        }
                    } else {
                        ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                            HStack(alignment: .top, spacing: 18) {
                                Button(timeLabel(segment.start)) { playback.seek(segment.start) }.buttonStyle(.link).font(.callout.monospacedDigit()).frame(width: 65, alignment: .leading)
                                VStack(alignment: .leading, spacing: 2) {
                                    if let speaker = segment.speaker { Text(record.speakerNames[speaker] ?? speaker).font(.callout.weight(.semibold)).foregroundStyle(.indigo) }
                                    Text(segment.text).lineSpacing(2).textSelection(.enabled)
                                }.frame(maxWidth: .infinity, alignment: .leading)
                            }.padding(.horizontal, 12).padding(.vertical, 5).background(playback.currentTime >= segment.start && playback.currentTime < segment.end ? Color.indigo.opacity(0.08) : Color.clear, in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
                }.padding(.horizontal, 24).padding(.vertical, 16).frame(maxWidth: 960, alignment: .leading).frame(maxWidth: .infinity)
            }
        }
    }

    private func transcribe() {
        if item.voices && !connection.isConnected { needsConnection = true; return }
        playback.pause()
        let captured = item
        engine.start(item, outputDirectory: library.outputDirectory(for: item.id)) { output, speakers, warnings in
            let record = TranscriptionRecord(outputURL: output, language: captured.language, quality: captured.quality, requestedVoices: captured.voices, speakers: speakers, warnings: warnings)
            library.addTranscription(to: captured.id, record: record)
        }
        if engine.isRunning { closeConfiguration() }
    }

    private func loadResult() {
        segments = []
        resultError = nil
        guard let record else { return }
        if presentation == "Per voce" && record.speakers.isEmpty { presentation = "Con tempi" }
        do { segments = try JSONDecoder().decode(Transcript.self, from: Data(contentsOf: record.outputURL.appendingPathComponent("result.json"))).segments }
        catch { resultError = L10n.tr("Non riesco a leggere questa trascrizione. Puoi crearne una nuova senza perdere le altre.") }
    }
}

struct SpeakerNamesView: View {
    let speakers: [String]
    @Binding var names: [String: String]
    let save: () -> Void
    let cancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(L10n.tr("Dai un nome alle voci")).font(.title2.weight(.semibold))
            Text(L10n.tr("I nomi si applicano a tutti i segmenti di questa trascrizione.")).foregroundStyle(.secondary)
            ScrollView {
                VStack(spacing: 16) {
                    ForEach(speakers, id: \.self) { speaker in
                        HStack {
                            Text(speaker).font(.callout.monospaced()).frame(width: 135, alignment: .leading)
                            TextField(L10n.tr("Nome della persona"), text: Binding(get: { names[speaker] ?? "" }, set: { names[speaker] = $0 })).textFieldStyle(.roundedBorder)
                        }
                    }
                }
            }.frame(maxHeight: 320)
            HStack { Button(L10n.tr("Annulla"), action: cancel).keyboardShortcut(.cancelAction); Spacer(); Button(L10n.tr("Salva nomi"), action: save).buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction) }
        }.padding(28).frame(width: 520)
    }
}

struct NativeMediaPlayer: NSViewRepresentable {
    var player: AVPlayer
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .none
        view.player = player
        return view
    }
    func updateNSView(_ view: AVPlayerView, context: Context) { view.player = player }
    static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) { view.player = nil }
}

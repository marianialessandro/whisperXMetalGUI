import SwiftUI

struct LibraryView: View {
    @Bindable var library: LibraryStore
    var connection: HuggingFaceConnection
    var engine: TranscriptionEngine
    @Bindable var playback: PlaybackController
    @State private var isTargeted = false
    @State private var deleting: MediaItem?
    @State private var renaming: MediaItem?
    @State private var titleDraft = ""
    @State private var folderSheet = false
    @State private var folderDraft = ""
    @State private var folderParent: UUID?
    @State private var renamingFolder: UUID?
    @State private var expanded: Set<UUID> = []
    @State private var pendingConfiguration: UUID?
    @State private var configurationRequest: UUID?

    var body: some View {
        VStack(spacing: 0) {
            NavigationSplitView {
                List(selection: $library.selection) {
                    Section(L10n.tr("Progetti")) {
                        ForEach(library.filteredItems.filter { $0.folderID == nil }) { projectRow($0) }
                    }
                    Section {
                        ForEach(library.folders.filter { $0.parentID == nil }) { folderNode($0) }
                    } header: { Text(L10n.tr("Cartelle")) }
                }
                .listStyle(.sidebar)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 360)
                .contextMenu { Button(L10n.tr("Nuova cartella…")) { newFolder(parent: nil) } }
            } detail: {
                if let index = library.items.firstIndex(where: { $0.id == library.selection }) {
                    ProjectDetail(item: $library.items[index], library: library, connection: connection, engine: engine, playback: playback, configurationRequest: $configurationRequest)
                        .id(library.items[index].id)
                } else { emptyLibrary }
            }
        }
        .navigationTitle(selected?.title ?? "WhisperX")
        .disabled(library.isRelocating || !library.isReady)
        .overlay {
            if library.isLoading { ProgressView(L10n.tr("Apro la libreria…")).padding(24).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16)) }
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                PlaybackBar(playback: playback)
            }
            ToolbarItem(placement: .navigation) {
                if let selected {
                    Button { renaming = selected; titleDraft = selected.title } label: { Image(systemName: "pencil") }.help(L10n.tr("Rinomina progetto"))
                }
            }
            ToolbarItem(placement: .automatic) {
                Button { library.chooseFiles() } label: { Label(L10n.tr("Importa"), systemImage: "plus") }
            }
        }
        .searchable(text: $library.search, placement: .sidebar, prompt: L10n.tr("Cerca un progetto"))
        .dropDestination(for: URL.self) { urls, _ in library.requestImport(urls); return !urls.isEmpty } isTargeted: { isTargeted = $0 }
        .overlay {
            if isTargeted { RoundedRectangle(cornerRadius: 16).strokeBorder(.indigo, style: StrokeStyle(lineWidth: 3, dash: [9])).padding(12).allowsHitTesting(false) }
        }
        .sheet(isPresented: Binding(get: { !library.pendingImports.isEmpty }, set: { if !$0 { library.pendingImports = [] } }), onDismiss: {
            configurationRequest = pendingConfiguration
            pendingConfiguration = nil
        }) {
            ImportChoiceView(urls: library.pendingImports, location: library.libraryURL) { move in pendingConfiguration = library.finishImport(move: move) } cancel: { library.pendingImports = []; pendingConfiguration = nil }
        }
        .sheet(item: $deleting) { item in
            DeleteProjectView(item: item) { trash in
                do {
                    try library.deleteProject(item.id, trashAudio: trash)
                    deleting = nil
                    playback.select(selected)
                } catch { library.error = L10n.tr("Non riesco a eliminare il progetto. \(error.localizedDescription)"); deleting = nil }
            } cancel: { deleting = nil }
        }
        .sheet(item: $renaming) { item in
            NameEntryView(title: L10n.tr("Rinomina progetto"), value: $titleDraft) {
                library.renameProject(item.id, to: titleDraft)
                playback.select(selected)
                renaming = nil
            } cancel: { renaming = nil }
        }
        .sheet(isPresented: $folderSheet) {
            NameEntryView(title: renamingFolder == nil ? L10n.tr("Nuova cartella") : L10n.tr("Rinomina cartella"), value: $folderDraft) {
                if let id = renamingFolder { library.renameFolder(id, to: folderDraft) }
                else { library.createFolder(folderDraft, parentID: folderParent) }
                folderSheet = false
            } cancel: { folderSheet = false }
        }
        .onAppear { playback.select(selected) }
        .focusedSceneValue(\.newFolderAction, { newFolder(parent: nil) })
        .onChange(of: library.selection) { _, _ in playback.select(selected) }
        .onChange(of: selected?.title) { _, _ in playback.select(selected) }
        .onChange(of: selected?.durationSeconds) { _, _ in playback.select(selected) }
        .onChange(of: selected?.url) { _, _ in playback.select(nil); playback.select(selected) }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in engine.cancel() }
        .alert(L10n.tr("Libreria"), isPresented: Binding(get: { library.error != nil }, set: { if !$0 { library.error = nil } })) {
            Button("OK") { library.error = nil }
        } message: { Text(L10n.display(library.error ?? "")) }
    }

    private var selected: MediaItem? { library.items.first { $0.id == library.selection } }

    private func projectRow(_ item: MediaItem) -> some View {
        HStack(spacing: 10) {
            Image(systemName: item.transcriptions.isEmpty ? "waveform" : "text.alignleft").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                Text(item.title).lineLimit(1)
                Text(item.transcriptions.isEmpty ? L10n.display(item.duration) : L10n.tr("\(item.transcriptions.count) \(item.transcriptions.count == 1 ? L10n.tr("trascrizione") : L10n.tr("trascrizioni")) · \(item.duration)")).font(.caption).foregroundStyle(.secondary)
            }
        }.padding(.vertical, 5).tag(item.id)
            .contextMenu {
                Button(L10n.tr("Rinomina…")) { renaming = item; titleDraft = item.title }
                Button(L10n.tr("Mostra file sorgente nel Finder")) { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                Menu(L10n.tr("Sposta nella cartella")) {
                    Button(L10n.tr("Senza cartella")) { library.moveProject(item.id, to: nil) }
                    ForEach(library.folders) { folder in Button(folder.name) { library.moveProject(item.id, to: folder.id); expanded.insert(folder.id) } }
                }
                Divider()
                Button(L10n.tr("Elimina progetto…"), role: .destructive) { deleting = item }.disabled(engine.isRunning && engine.requestedID == item.id)
            }
    }

    private func folderNode(_ folder: ProjectFolder) -> AnyView {
        AnyView(DisclosureGroup(isExpanded: Binding(get: { expanded.contains(folder.id) || !library.search.isEmpty }, set: { if $0 { expanded.insert(folder.id) } else { expanded.remove(folder.id) } })) {
            ForEach(library.folders.filter { $0.parentID == folder.id }) { folderNode($0) }
            ForEach(library.filteredItems.filter { $0.folderID == folder.id }) { projectRow($0) }
        } label: {
            Label(folder.name, systemImage: "folder")
                .contextMenu {
                    Button(L10n.tr("Nuova sottocartella…")) { newFolder(parent: folder.id); expanded.insert(folder.id) }
                    Button(L10n.tr("Rinomina…")) { renamingFolder = folder.id; folderDraft = folder.name; folderSheet = true }
                    Button(L10n.tr("Rimuovi cartella"), role: .destructive) { library.removeFolder(folder.id) }
                }
        })
    }

    private func newFolder(parent: UUID?) {
        folderParent = parent
        folderDraft = ""
        renamingFolder = nil
        folderSheet = true
    }

    private var emptyLibrary: some View {
        VStack(spacing: 22) {
            Image(systemName: "waveform").font(.system(size: 48)).foregroundStyle(.indigo)
            Text(L10n.tr("Un file. Un nuovo progetto.")).font(.system(size: 30, weight: .semibold, design: .rounded))
            Text(L10n.tr("Importa un audio o un video per iniziare.")).font(.title3).foregroundStyle(.secondary)
            Button(L10n.tr("Importa file…")) { library.chooseFiles() }.buttonStyle(.borderedProminent).controlSize(.large)
            Text(L10n.tr("Puoi anche trascinarlo qui")).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct NameEntryView: View {
    let title: String
    @Binding var value: String
    let save: () -> Void
    let cancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(title).font(.title2.weight(.semibold))
            TextField(L10n.tr("Nome"), text: $value).textFieldStyle(.roundedBorder).onSubmit { if !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { save() } }
            HStack { Button(L10n.tr("Annulla"), action: cancel).keyboardShortcut(.cancelAction); Spacer(); Button(L10n.tr("Salva"), action: save).buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction).disabled(value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) }
        }.padding(28).frame(width: 420)
    }
}

struct ImportChoiceView: View {
    let urls: [URL]
    let location: URL
    let importFiles: (Bool) -> Void
    let cancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text(L10n.tr("Dove conserviamo il file?")).font(.title2.weight(.semibold))
            Text(urls.count == 1 ? urls[0].lastPathComponent : L10n.tr("\(urls.count) file selezionati")).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 10) {
                Text(L10n.tr("Lascia nella posizione originale")).font(.headline)
                Text(L10n.tr("Il progetto farà riferimento al file attuale. Se lo sposti in seguito, dovrai ricollegarlo.")).foregroundStyle(.secondary)
                Button(L10n.tr("Lascia nella sorgente")) { importFiles(false) }
            }.padding(18).frame(maxWidth: .infinity, alignment: .leading).background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
            VStack(alignment: .leading, spacing: 10) {
                Text(L10n.tr("Sposta nella libreria")).font(.headline)
                Text(L10n.tr("Il file verrà spostato nella libreria dell’app e non resterà nella posizione originale.")).foregroundStyle(.secondary)
                Text(location.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                Button(L10n.tr("Sposta nella libreria")) { importFiles(true) }.buttonStyle(.borderedProminent)
            }.padding(18).frame(maxWidth: .infinity, alignment: .leading).background(.indigo.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
            Button(L10n.tr("Annulla"), action: cancel).keyboardShortcut(.cancelAction)
        }.padding(28).frame(width: 530)
    }
}

struct DeleteProjectView: View {
    let item: MediaItem
    let delete: (Bool) -> Void
    let cancel: () -> Void
    @State private var trashAudio = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(L10n.tr("Eliminare “\(item.title)”?")).font(.title2.weight(.semibold))
            Text(L10n.tr("Il progetto verrà rimosso dalla libreria. Le sue trascrizioni gestite saranno spostate nel Cestino.")).foregroundStyle(.secondary)
            Label(item.managed ? L10n.tr("Questo progetto include un file audio/video nella libreria.") : L10n.tr("Il progetto fa riferimento a un file nella sorgente originale."), systemImage: "waveform").font(.headline)
            Text(item.url.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            if item.managed {
                Toggle(L10n.tr("Sposta anche il file audio/video nel Cestino"), isOn: $trashAudio)
                Text(L10n.tr("Se non selezioni questa opzione, il file resterà nella libreria.")).font(.caption).foregroundStyle(.secondary)
            } else {
                Text(L10n.tr("Il file sorgente esterno sarà conservato nella sua posizione.")).font(.callout).foregroundStyle(.secondary)
            }
            HStack { Button(L10n.tr("Annulla"), action: cancel).keyboardShortcut(.cancelAction); Spacer(); Button(L10n.tr("Elimina progetto"), role: .destructive) { delete(trashAudio) }.keyboardShortcut(.defaultAction) }
        }.padding(28).frame(width: 540)
    }
}

import SwiftUI

struct AppSettingsView: View {
    @AppStorage(L10n.preferenceKey) private var interfaceLanguage = "system"
    @State private var selectedTab = "general"
    var library: LibraryStore
    var connection: HuggingFaceConnection
    var engine: TranscriptionEngine

    var body: some View {
        TabView(selection: $selectedTab) {
            GeneralSettings().tabItem { Label(L10n.tr("Generali"), systemImage: "gearshape") }.tag("general")
            ConnectionSettings(connection: connection).tabItem { Label(L10n.tr("Hugging Face"), systemImage: "link") }.tag("huggingface")
            ModelSettings().tabItem { Label(L10n.tr("Modelli"), systemImage: "cpu") }.tag("models")
            LibrarySettings(library: library, engine: engine).tabItem { Label(L10n.tr("Libreria"), systemImage: "folder") }.tag("library")
        }.id(interfaceLanguage).padding(12)
    }
}

struct ModelSettings: View {
    @State private var inventory = ModelInventory()

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text(L10n.tr("Modelli sul Mac")).font(.title2.weight(.semibold))
                    Text(L10n.tr("\(inventory.entries.count) modelli · \(ByteCountFormatter.string(fromByteCount: inventory.total, countStyle: .file)) complessivi")).foregroundStyle(.secondary)
                }
                Spacer()
                if inventory.loading { ProgressView().controlSize(.small) }
                Button(L10n.tr("Aggiorna")) { Task { await inventory.refresh() } }.disabled(inventory.loading)
            }
            Text(L10n.tr("Sono incluse le cache usate dal motore locale, anche quando condivise con altre app. I collegamenti simbolici ai pesi non vengono contati due volte.")).font(.caption).foregroundStyle(.secondary)
            List(inventory.entries) { model in
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Text(model.name).font(.callout.weight(.medium))
                        Spacer()
                        Text(ByteCountFormatter.string(fromByteCount: model.bytes, countStyle: .file)).monospacedDigit().foregroundStyle(.secondary)
                        Button { NSWorkspace.shared.activateFileViewerSelecting([model.url]) } label: { Image(systemName: "folder") }.buttonStyle(.plain).help(L10n.tr("Mostra nel Finder"))
                    }
                    Text(model.url.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }.padding(.vertical, 7)
            }
            if inventory.entries.isEmpty && !inventory.loading { Text(L10n.tr("Non sono stati trovati modelli nelle cache del motore.")).foregroundStyle(.secondary) }
            VStack(alignment: .leading, spacing: 5) {
                Text(L10n.tr("Posizioni controllate")).font(.caption.weight(.semibold))
                ForEach(StorageInventory.roots(), id: \.path) { Text($0.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
            }
        }.padding(20).task { await inventory.refresh() }
    }
}

struct LibrarySettings: View {
    @Bindable var library: LibraryStore
    var engine: TranscriptionEngine
    @State private var bytes: Int64 = 0
    @State private var moving = false
    @State private var message: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text(L10n.tr("La tua libreria")).font(.title2.weight(.semibold))
            Text(L10n.tr("\(library.items.count) \(library.items.count == 1 ? L10n.tr("progetto") : L10n.tr("progetti")) · \(library.folders.count) \(library.folders.count == 1 ? L10n.tr("cartella") : L10n.tr("cartelle")) · \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))")).foregroundStyle(.secondary)
            GroupBox(L10n.tr("Posizione della libreria")) {
                VStack(alignment: .leading, spacing: 15) {
                    Text(library.libraryURL.path).font(.callout).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    HStack {
                        Button(L10n.tr("Mostra nel Finder")) { NSWorkspace.shared.open(library.libraryURL) }
                        Button(L10n.tr("Cambia posizione…")) { chooseLocation() }.disabled(engine.isRunning || moving)
                        if moving { ProgressView().controlSize(.small) }
                    }
                }.padding(12)
            }
            Text(L10n.tr("Cambiare posizione sposta i file gestiti, le trascrizioni e l’organizzazione nella nuova cartella, senza conservare una copia della libreria precedente. I file sorgente esterni restano dove sono.")).font(.callout).foregroundStyle(.secondary)
            GroupBox(L10n.tr("File gestiti e sorgenti esterne")) {
                VStack(alignment: .leading, spacing: 10) {
                    Text(L10n.tr("\(library.items.filter(\.managed).count) file nella libreria"))
                    Text(L10n.tr("\(library.items.filter { !$0.managed }.count) file nella posizione originale"))
                    Text(L10n.tr("La scelta viene richiesta a ogni importazione. Puoi individuare la sorgente dal menu contestuale di un progetto.")).font(.caption).foregroundStyle(.secondary)
                }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            }
            if let message { Text(message).font(.callout).foregroundStyle(.secondary) }
            Spacer()
        }.padding(24).task(id: library.libraryURL) {
            let url = library.libraryURL
            bytes = await Task.detached(priority: .utility) { StorageInventory.size(of: url) }.value
        }
    }

    private func chooseLocation() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.message = L10n.tr("Scegli una cartella vuota per la nuova libreria.")
        panel.prompt = L10n.tr("Scegli libreria")
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            Task {
                moving = true
                defer { moving = false }
                do {
                    try await library.relocateLibrary(to: destination)
                    message = L10n.tr("Libreria spostata. Non è stata conservata una copia nella posizione precedente.")
                } catch { message = L10n.tr("Non riesco a cambiare posizione. Scegli una cartella vuota e verifica lo spazio disponibile. \(error.localizedDescription)") }
            }
        }
    }
}

struct GeneralSettings: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text(L10n.tr("Lingua e interfaccia")).font(.title2.weight(.semibold))
            InterfaceLanguagePicker().frame(maxWidth: 420)
            Text(L10n.tr("La lingua dell’interfaccia non cambia la lingua dell’audio o il contenuto delle trascrizioni.")).foregroundStyle(.secondary)
            Text(L10n.tr("Per rivedere l’introduzione, scegli Aiuto → Guida a WhisperX….")).font(.callout).foregroundStyle(.secondary)
            Spacer()
        }.padding(32)
    }
}

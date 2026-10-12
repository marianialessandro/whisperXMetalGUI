import SwiftUI

@main
struct WhisperXApp: App {
    @AppStorage(L10n.preferenceKey) private var interfaceLanguage = "system"
    @State private var library = LibraryStore()
    @State private var connection = HuggingFaceConnection()
    @State private var engine = TranscriptionEngine()
    @State private var playback = PlaybackController()

    var body: some Scene {
        WindowGroup("WhisperX") {
            GuidedLibraryView(library: library, connection: connection, engine: engine, playback: playback)
                .frame(minWidth: 920, minHeight: 640)
                .tint(.indigo)
                .environment(\.locale, Locale(identifier: L10n.language(preference: interfaceLanguage)))
        }
        .defaultSize(width: 1180, height: 780)
        .windowToolbarStyle(.unified)
        .commands { ProjectCommands(library: library) }
        Settings {
            AppSettingsView(library: library, connection: connection, engine: engine)
                .frame(width: 760, height: 560)
                .tint(.indigo)
                .environment(\.locale, Locale(identifier: L10n.language(preference: interfaceLanguage)))
        }
    }
}

struct NewFolderActionKey: FocusedValueKey {
    typealias Value = () -> Void
}

extension FocusedValues {
    var newFolderAction: NewFolderActionKey.Value? {
        get { self[NewFolderActionKey.self] }
        set { self[NewFolderActionKey.self] = newValue }
    }
}

struct ProjectCommands: Commands {
    @AppStorage(L10n.preferenceKey) private var interfaceLanguage = "system"
    var library: LibraryStore
    @FocusedValue(\.newFolderAction) private var newFolder
    @FocusedValue(\.introGuideAction) private var openGuide

    var body: some Commands {
        CommandGroup(after: .help) {
            Button(L10n.tr("Guida a WhisperX…", language: L10n.language(preference: interfaceLanguage))) { openGuide?() }.disabled(openGuide == nil)
        }
        CommandGroup(replacing: .newItem) {
            Button(L10n.tr("Aggiungi file…", language: L10n.language(preference: interfaceLanguage))) { library.chooseFiles() }.keyboardShortcut("o").disabled(!library.isReady || library.isRelocating)
            Button(L10n.tr("Nuova cartella…", language: L10n.language(preference: interfaceLanguage))) { newFolder?() }.keyboardShortcut("n", modifiers: [.command, .shift]).disabled(newFolder == nil || !library.isReady || library.isRelocating)
        }
    }
}

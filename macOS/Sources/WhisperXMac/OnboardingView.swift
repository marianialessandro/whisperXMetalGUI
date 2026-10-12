import SwiftUI

struct GuideStep: Identifiable {
    let id: Int
    let symbol: String
    let title: String
    let detail: String
    let tips: [(String, String)]

    static var all: [GuideStep] {
        [
            GuideStep(id: 0, symbol: "waveform", title: L10n.tr("Benvenuto in WhisperX"), detail: L10n.tr("Dall’audio al testo, sul tuo Mac. Questa guida ti mostra dove trovare gli strumenti per lavorare sui tuoi file."), tips: [("sidebar.left", L10n.tr("A sinistra trovi progetti e cartelle. Al centro lavori sulla trascrizione selezionata.")), ("menubar.rectangle", L10n.tr("La toolbar contiene il titolo, il player e le azioni del progetto."))]),
            GuideStep(id: 1, symbol: "square.and.arrow.down", title: L10n.tr("Importa il tuo primo file"), detail: L10n.tr("Usa Importa nella toolbar, File → Aggiungi file… (⌘O), oppure trascina un audio o un video nella finestra."), tips: [("link", L10n.tr("Lascia nella sorgente mantiene il file dov’è. Se lo sposti in seguito, usa Ricollega file….")), ("folder", L10n.tr("Sposta nella libreria trasferisce il file nella libreria dell’app e lo rimuove dalla posizione originale."))]),
            GuideStep(id: 2, symbol: "sidebar.left", title: L10n.tr("Organizza i progetti"), detail: L10n.tr("Ogni file diventa un progetto. Selezionalo nella barra laterale e usa la ricerca per ritrovarlo."), tips: [("folder.badge.plus", L10n.tr("Crea una cartella da File → Nuova cartella… (⇧⌘N). Puoi creare sottocartelle dal menu contestuale.")), ("cursorarrow", L10n.tr("Con un clic destro puoi rinominare, spostare un progetto in una cartella o mostrare la sorgente nel Finder."))]),
            GuideStep(id: 3, symbol: "text.badge.plus", title: L10n.tr("Configura la trascrizione"), detail: L10n.tr("Dopo l’importazione scegli lingua dell’audio e qualità, aggiungi eventuali parole utili e premi Trascrivi."), tips: [("globe", L10n.tr("Automatica rileva la lingua dell’audio. Questa scelta è indipendente dalla lingua dell’interfaccia.")), ("arrow.clockwise", L10n.tr("Nuova trascrizione nella toolbar crea un’altra versione nello stesso progetto. Puoi annullare il lavoro in corso.")), ("arrow.down.circle", L10n.tr("I modelli necessari vengono scaricati al primo utilizzo: serve una connessione Internet."))]),
            GuideStep(id: 4, symbol: "play.rectangle", title: L10n.tr("Ascolta e consulta il testo"), detail: L10n.tr("Il player nella toolbar permette di riprodurre, mettere in pausa e spostarsi nell’audio. Nei video l’immagine appare sopra il testo."), tips: [("clock", L10n.tr("Scegli Testo, Con tempi o Per voce. Clicca un tempo per ascoltare quel punto; Per voce è disponibile se sono state riconosciute le voci.")), ("person.2", L10n.tr("Nomi delle voci assegna un nome alle persone riconosciute. Il selettore Trascrizione permette di passare tra le versioni.")), ("square.and.arrow.up", L10n.tr("Esporta → Apri file generati apre la cartella con TXT, SRT, VTT e JSON. La pagina di esportazione dedicata è ancora in preparazione."))]),
            GuideStep(id: 5, symbol: "gearshape", title: L10n.tr("Personalizza dalle Impostazioni"), detail: L10n.tr("Apri WhisperX → Impostazioni… (⌘,) dal menu dell’app."), tips: [("globe", L10n.tr("Generali cambia la lingua dell’interfaccia. Modelli mostra le risorse sul Mac e lo spazio occupato.")), ("link", L10n.tr("Hugging Face guida la configurazione del token per Distingui le voci. Puoi trascrivere anche senza attivare questa opzione.")), ("folder", L10n.tr("Libreria mostra la posizione dei dati e permette di spostarli. Puoi riaprire questa guida dal menu Aiuto in qualsiasi momento."))])
        ]
    }
}

struct OnboardingView: View {
    @State private var page = 0
    let finish: () -> Void
    private var steps: [GuideStep] { GuideStep.all }
    private var step: GuideStep { steps[page] }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(L10n.tr("Guida a WhisperX")).font(.headline)
                Spacer()
                InterfaceLanguagePicker().frame(width: 280)
            }.padding(24)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    HStack(spacing: 18) {
                        Image(systemName: step.symbol).font(.system(size: 34)).foregroundStyle(.indigo).frame(width: 76, height: 76).background(.indigo.opacity(0.1), in: RoundedRectangle(cornerRadius: 20)).accessibilityHidden(true)
                        VStack(alignment: .leading, spacing: 8) {
                            Text(L10n.tr("Passaggio \(page + 1) di \(steps.count)")).font(.caption).foregroundStyle(.secondary)
                            Text(step.title).font(.system(size: 27, weight: .semibold, design: .rounded)).accessibilityAddTraits(.isHeader)
                        }
                    }
                    Text(step.detail).font(.title3).fixedSize(horizontal: false, vertical: true)
                    VStack(alignment: .leading, spacing: 20) {
                        ForEach(step.tips.indices, id: \.self) { index in
                            Label { Text(step.tips[index].1).fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: step.tips[index].0).foregroundStyle(.indigo).frame(width: 26) }
                        }
                    }.padding(22).frame(maxWidth: .infinity, alignment: .leading).background(.quaternary, in: RoundedRectangle(cornerRadius: 16))
                }.padding(30)
            }
            HStack(spacing: 8) {
                ForEach(steps) { candidate in
                    Button { page = candidate.id } label: {
                        Capsule().fill(candidate.id == page ? Color.indigo : Color.secondary.opacity(0.25)).frame(width: 28, height: 6).padding(.vertical, 8)
                    }.buttonStyle(.plain).accessibilityLabel(candidate.title).accessibilityAddTraits(candidate.id == page ? .isSelected : [])
                }
            }.padding(.bottom, 12)
            Divider()
            HStack {
                Button(L10n.tr("Salta la guida"), action: finish).keyboardShortcut(.cancelAction)
                Spacer()
                if page > 0 { Button(L10n.tr("Indietro")) { page -= 1 } }
                Button(page == steps.count - 1 ? L10n.tr("Inizia") : L10n.tr("Continua")) {
                    if page == steps.count - 1 { finish() } else { page += 1 }
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            }.padding(24)
        }.frame(width: 700, height: 580)
    }
}

struct GuidedLibraryView: View {
    var library: LibraryStore
    var connection: HuggingFaceConnection
    var engine: TranscriptionEngine
    var playback: PlaybackController
    @AppStorage("introGuideCompleted") private var completed = false
    @State private var showGuide = false

    var body: some View {
        LibraryView(library: library, connection: connection, engine: engine, playback: playback)
            .task { if !completed { showGuide = true } }
            .focusedSceneValue(\.introGuideAction, { showGuide = true })
            .sheet(isPresented: $showGuide, onDismiss: { completed = true }) {
                OnboardingView { completed = true; showGuide = false }
            }
    }
}

struct IntroGuideActionKey: FocusedValueKey { typealias Value = () -> Void }

extension FocusedValues {
    var introGuideAction: IntroGuideActionKey.Value? {
        get { self[IntroGuideActionKey.self] }
        set { self[IntroGuideActionKey.self] = newValue }
    }
}

import AppKit
import SwiftUI

struct VoiceSetupView: View {
    var connection: HuggingFaceConnection
    var replacingToken = false
    @Environment(\.dismiss) private var dismiss
    @State private var step = 0
    @State private var token = ""
    @State private var showToken = false
    @State private var success = false
    @State private var pasteError: String?

    private let titles = [L10n.tr("Un piccolo passo, tante voci."), L10n.tr("Prepariamo il modello."), L10n.tr("Incolla. E ci siamo.")]
    private let subtitles = [L10n.tr("Ti accompagniamo a collegare Hugging Face.\nPoi potrai distinguere chi parla nei tuoi file."), L10n.tr("Il modello che distingue le voci è gratuito.\nDevi solo accettare le sue condizioni."), L10n.tr("Il token collega l’app al tuo account.\nLo crei una volta, al resto pensiamo noi.")]

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label(L10n.tr("CONFIGURA LE VOCI"), systemImage: "person.2").font(.caption.weight(.semibold)).tracking(1)
                Spacer()
                Button { dismiss() } label: { Image(systemName: "xmark").foregroundStyle(.secondary) }
                    .buttonStyle(.plain).accessibilityLabel(L10n.tr("Chiudi configurazione"))
            }.padding(24).foregroundStyle(.secondary)
            if success {
                successContent
            } else {
                ScrollView {
                    VStack(spacing: 22) {
                        Image(systemName: ["person.crop.circle.badge.plus", "person.2.wave.2", "link"][step])
                            .font(.system(size: 36, weight: .medium)).foregroundStyle(.indigo)
                            .frame(width: 86, height: 86).background(.indigo.opacity(0.09), in: RoundedRectangle(cornerRadius: 26))
                        VStack(spacing: 10) {
                            Text(titles[step]).font(.system(size: 27, weight: .semibold, design: .rounded))
                            Text(subtitles[step]).foregroundStyle(.secondary).multilineTextAlignment(.center).lineSpacing(4)
                        }
                        HStack(spacing: 8) {
                            ForEach(0..<3) { index in
                                Capsule().fill(index <= step ? Color.indigo : Color.secondary.opacity(0.16)).frame(width: 32, height: 4)
                            }
                            Text(L10n.tr("\(step + 1) di 3")).font(.caption).foregroundStyle(.secondary).padding(.leading, 6)
                        }
                        stepContent.padding(22).frame(maxWidth: .infinity).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 18))
                    }.padding(.horizontal, 36).padding(.top, 8).padding(.bottom, 24)
                }
                if let error = connection.error ?? pasteError {
                    Label(L10n.display(error), systemImage: "exclamationmark.circle").font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 36).padding(.bottom, 16)
                }
                Divider()
                HStack {
                    if step > 0 { Button(L10n.tr("Indietro")) { step -= 1; connection.error = nil; pasteError = nil }.disabled(connection.isChecking) }
                    Button(L10n.tr("Più tardi")) { dismiss() }.buttonStyle(.plain).foregroundStyle(.secondary)
                    Spacer()
                    if step < 2 {
                        Button(L10n.tr("Continua")) { step += 1 }.buttonStyle(.borderedProminent).controlSize(.large)
                    } else {
                        Button {
                            pasteError = nil
                            connection.connect(token) { token = ""; showToken = false; success = true }
                        } label: {
                            HStack(spacing: 8) {
                                if connection.isChecking { ProgressView().controlSize(.small) }
                                Text(connection.isChecking ? L10n.tr("Verifico…") : L10n.tr("Collega e continua"))
                            }
                        }.buttonStyle(.borderedProminent).controlSize(.large)
                            .disabled(token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || connection.isChecking)
                    }
                }.padding(24)
            }
        }
        .frame(width: 600, height: 650)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            if replacingToken { step = 2 }
            else if connection.isConnected { success = true }
        }
        .onDisappear { connection.cancelVerification(); token = ""; showToken = false }
    }

    @ViewBuilder private var stepContent: some View {
        switch step {
        case 0:
            VStack(alignment: .leading, spacing: 18) {
                Text(L10n.tr("1. Crea il tuo account")).font(.headline)
                Text(L10n.tr("Apri Hugging Face e registrati. Se richiesto, conferma la tua email, poi torna qui.")).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button(L10n.tr("Crea un account gratuito ↗")) { open("https://huggingface.co/join") }.controlSize(.large)
                Button(L10n.tr("Ho già un account ↗")) { open("https://huggingface.co/login") }.buttonStyle(.link)
            }.frame(maxWidth: .infinity, alignment: .leading)
        case 1:
            VStack(alignment: .leading, spacing: 18) {
                Text(L10n.tr("2. Abilita il modello delle voci")).font(.headline)
                Text(L10n.tr("Accedi con il tuo account e accetta le condizioni nella pagina del modello. Il sito può chiederti informazioni di contatto.")).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button(L10n.tr("Apri la pagina del modello ↗")) { open("https://huggingface.co/pyannote/speaker-diarization-community-1") }.controlSize(.large)
                Label(L10n.tr("Questa operazione si fa una sola volta."), systemImage: "checkmark.circle").font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
        default:
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text(L10n.tr("3. Collega Hugging Face")).font(.headline)
                    Spacer()
                    Button(L10n.tr("Crea il token ↗")) { open("https://huggingface.co/settings/tokens") }.buttonStyle(.link)
                }
                Text(L10n.tr("Scegli “Create new token”, tipo “Read” e un nome come “WhisperX Mac”. Poi copia il token.")).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Text(L10n.tr("Token Hugging Face")).font(.callout.weight(.medium))
                HStack {
                    Group {
                        if showToken { TextField(L10n.tr("Incolla qui il token"), text: $token) }
                        else { SecureField(L10n.tr("Incolla qui il token"), text: $token) }
                    }.textFieldStyle(.plain).disabled(connection.isChecking)
                    Button { showToken.toggle() } label: { Image(systemName: showToken ? "eye.slash" : "eye") }
                        .buttonStyle(.plain).accessibilityLabel(showToken ? L10n.tr("Nascondi token") : L10n.tr("Mostra token"))
                    Button(L10n.tr("Incolla")) {
                        if let value = NSPasteboard.general.string(forType: .string) { token = value; pasteError = nil }
                        else { pasteError = L10n.tr("Negli appunti non c’è testo. Copia il token da Hugging Face e riprova.") }
                    }.disabled(connection.isChecking)
                }.padding(12).background(Color(nsColor: .windowBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
                Label(L10n.tr("Salvato nel Portachiavi del Mac. I tuoi audio restano sul computer."), systemImage: "lock.shield").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if connection.error != nil {
                    HStack {
                        Button(L10n.tr("Apri il modello ↗")) { open("https://huggingface.co/pyannote/speaker-diarization-community-1") }
                        Button(L10n.tr("Apri i token ↗")) { open("https://huggingface.co/settings/tokens") }
                    }.buttonStyle(.link).font(.caption)
                }
            }
        }
    }

    private var successContent: some View {
        VStack(spacing: 24) {
            Spacer()
            Image(systemName: "checkmark.circle.fill").font(.system(size: 64)).foregroundStyle(.indigo)
            Text(L10n.tr("Collegato. Un passo in meno.")).font(.system(size: 27, weight: .semibold, design: .rounded))
            Text(L10n.tr("Il token è nel Portachiavi del tuo Mac.\nNon dovrai incollarlo a ogni apertura.")).multilineTextAlignment(.center).foregroundStyle(.secondary).lineSpacing(4)
            Text(L10n.tr("I modelli necessari verranno scaricati automaticamente alla prima trascrizione con le voci attive.")).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center).padding(20)
                .background(.indigo.opacity(0.06), in: RoundedRectangle(cornerRadius: 16))
            Spacer()
            Button(L10n.tr("Fine")) { dismiss() }.buttonStyle(.borderedProminent).controlSize(.large)
        }.padding(36)
    }

    private func open(_ address: String) { if let url = URL(string: address) { NSWorkspace.shared.open(url) } }
}

struct ConnectionSettings: View {
    var connection: HuggingFaceConnection
    @State private var showSetup = false

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text(L10n.tr("Hugging Face")).font(.title2.weight(.semibold))
            Label(connection.isConnected ? L10n.tr("Hugging Face collegato") : L10n.tr("Hugging Face da collegare"), systemImage: connection.isConnected ? "checkmark.circle.fill" : "link").foregroundStyle(connection.isConnected ? .indigo : .secondary)
            Text(L10n.tr("Il collegamento serve a scaricare il modello delle voci. I file audio e video vengono elaborati sul tuo Mac.")).foregroundStyle(.secondary)
            HStack {
                Button(connection.isConnected ? L10n.tr("Sostituisci token…") : L10n.tr("Configura…")) { showSetup = true }
                if connection.isConnected { Button(L10n.tr("Scollega"), role: .destructive) { connection.disconnect() } }
            }
            if let error = connection.error { Text(L10n.display(error)).foregroundStyle(.red).font(.callout) }
            Spacer()
        }.padding(32)
            .sheet(isPresented: $showSetup) { VoiceSetupView(connection: connection, replacingToken: connection.isConnected).onAppear { connection.error = nil } }
    }
}

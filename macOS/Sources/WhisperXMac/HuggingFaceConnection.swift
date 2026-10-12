import Foundation
import Observation
import Security

enum ConnectionFailure: LocalizedError {
    case invalidToken, modelAccess, server, keychain
    var errorDescription: String? {
        switch self {
        case .invalidToken: L10n.tr("Questo token non funziona. Controlla di averlo copiato per intero oppure creane uno nuovo.")
        case .modelAccess: L10n.tr("Non riesco ad accedere al modello. Controlla di aver accettato le condizioni e che il token abbia i permessi di lettura.")
        case .server: L10n.tr("Hugging Face non è disponibile in questo momento. Riprova tra poco.")
        case .keychain: L10n.tr("Non riesco ad accedere al Portachiavi. Riprova per completare il collegamento.")
        }
    }
}

enum TokenVault {
    static var query: [String: Any] { [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "it.whisperx.mac.huggingface", kSecAttrAccount as String: "user-token"] }

    static func containsToken() throws -> Bool {
        var lookup = query
        lookup[kSecReturnData as String] = false
        let status = SecItemCopyMatching(lookup as CFDictionary, nil)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw ConnectionFailure.keychain }
        return status == errSecSuccess
    }

    static func save(_ token: String) throws {
        let data = Data(token.utf8)
        let status = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var entry = query
            entry[kSecValueData as String] = data
            entry[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            guard SecItemAdd(entry as CFDictionary, nil) == errSecSuccess else { throw ConnectionFailure.keychain }
        } else if status != errSecSuccess {
            throw ConnectionFailure.keychain
        }
    }

    static func load() throws -> String? {
        var lookup = query
        lookup[kSecReturnData as String] = true
        var result: CFTypeRef?
        let status = SecItemCopyMatching(lookup as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data, let token = String(data: data, encoding: .utf8) else { throw ConnectionFailure.keychain }
        return token
    }

    static func remove() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw ConnectionFailure.keychain }
    }
}

@MainActor @Observable
final class HuggingFaceConnection {
    var isConnected = false
    var isChecking = false
    var error: String?
    private var verification: Task<Void, Never>?

    init() {
        do { isConnected = try TokenVault.containsToken() } catch { self.error = error.localizedDescription }
    }

    func connect(_ rawToken: String, onSuccess: @escaping @MainActor () -> Void) {
        guard !isChecking else { return }
        let token = rawToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard token.hasPrefix("hf_"), !token.contains(where: { $0.isWhitespace }) else {
            error = ConnectionFailure.invalidToken.localizedDescription
            return
        }
        error = nil
        isChecking = true
        verification = Task {
            defer { isChecking = false; verification = nil }
            do {
                let config = URLSessionConfiguration.ephemeral
                config.timeoutIntervalForRequest = 25
                config.timeoutIntervalForResource = 40
                let session = URLSession(configuration: config)
                defer { session.invalidateAndCancel() }
                var identity = URLRequest(url: URL(string: "https://huggingface.co/api/whoami-v2")!)
                identity.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                let (_, response) = try await session.data(for: identity)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                if status == 401 || status == 403 { throw ConnectionFailure.invalidToken }
                guard status == 200 else { throw ConnectionFailure.server }
                var model = URLRequest(url: URL(string: "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml")!)
                model.httpMethod = "HEAD"
                model.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                let (_, access) = try await session.data(for: model)
                let modelStatus = (access as? HTTPURLResponse)?.statusCode ?? 0
                if [401, 403, 404].contains(modelStatus) { throw ConnectionFailure.modelAccess }
                guard modelStatus == 200 else { throw ConnectionFailure.server }
                try Task.checkCancellation()
                try TokenVault.save(token)
                isConnected = true
                onSuccess()
            } catch is CancellationError {
                return
            } catch let network as URLError {
                if network.code != .cancelled { self.error = L10n.tr("Non riesco a collegarmi a Hugging Face. Controlla la connessione e riprova.") }
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    func cancelVerification() { verification?.cancel() }

    func disconnect() {
        do {
            try TokenVault.remove()
            isConnected = false
            error = nil
        } catch { self.error = error.localizedDescription }
    }
}

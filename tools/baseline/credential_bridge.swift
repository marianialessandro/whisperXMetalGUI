import Darwin
import Foundation
import LocalAuthentication
import Security

// Credentials may leave this process only through a private pipe, never a terminal or regular file.
var destination = stat()
guard fstat(STDOUT_FILENO, &destination) == 0, destination.st_mode & S_IFMT == S_IFIFO else {
    FileHandle.standardError.write(Data("{\"code\":\"private_pipe_required\"}\n".utf8))
    exit(64)
}
let authentication = LAContext()
authentication.interactionNotAllowed = true
let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: "it.whisperx.mac.huggingface", kSecAttrAccount as String: "user-token", kSecReturnData as String: true, kSecUseAuthenticationContext as String: authentication]
var item: CFTypeRef?
let status = SecItemCopyMatching(query as CFDictionary, &item)
guard status == errSecSuccess, let data = item as? Data, !data.isEmpty, data.count <= 4096 else {
    FileHandle.standardError.write(Data("{\"code\":\"keychain_unavailable\"}\n".utf8))
    exit(2)
}
do {
    try FileHandle.standardOutput.write(contentsOf: data)
} catch {
    FileHandle.standardError.write(Data("{\"code\":\"credential_pipe_failed\"}\n".utf8))
    exit(2)
}

import Foundation
import Observation

struct StorageEntry: Identifiable, Sendable {
    var id: String { url.path }
    var name: String
    var url: URL
    var bytes: Int64
    var kind: String
}

enum StorageInventory {
    static func size(of root: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
        if let values = try? root.resourceValues(forKeys: keys), values.isRegularFile == true { return Int64(values.fileSize ?? 0) }
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: Array(keys)) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            guard let values = try? url.resourceValues(forKeys: keys), values.isSymbolicLink != true, values.isRegularFile == true else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }

    static func roots(environment: [String: String] = ProcessInfo.processInfo.environment, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> [URL] {
        var candidates: [URL] = []
        if let cache = environment["HF_HUB_CACHE"] ?? environment["HUGGINGFACE_HUB_CACHE"] { candidates.append(URL(fileURLWithPath: cache)) }
        if let directory = environment["HF_HOME"] { candidates.append(URL(fileURLWithPath: directory).appendingPathComponent("hub")) }
        let cacheRoot = environment["XDG_CACHE_HOME"].map { URL(fileURLWithPath: $0) } ?? home.appendingPathComponent(".cache")
        candidates.append(cacheRoot.appendingPathComponent("huggingface/hub"))
        let torch = environment["TORCH_HOME"].map { URL(fileURLWithPath: $0) } ?? cacheRoot.appendingPathComponent("torch")
        candidates.append(torch.appendingPathComponent("hub/checkpoints"))
        var seen: Set<String> = []
        return candidates.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    static func models() -> [StorageEntry] {
        var result: [StorageEntry] = []
        for root in roots() {
            guard let entries = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
            for entry in entries {
                if entry.lastPathComponent.hasPrefix("models--") {
                    result.append(StorageEntry(name: String(entry.lastPathComponent.dropFirst(8)).replacingOccurrences(of: "--", with: "/"), url: entry, bytes: size(of: entry), kind: L10n.tr("Hugging Face")))
                } else if root.lastPathComponent == "checkpoints" {
                    result.append(StorageEntry(name: entry.lastPathComponent, url: entry, bytes: size(of: entry), kind: L10n.tr("Allineamento / PyTorch")))
                }
            }
        }
        return result.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}

@MainActor @Observable
final class ModelInventory {
    var entries: [StorageEntry] = []
    var loading = false
    var total: Int64 { entries.reduce(0) { $0 + $1.bytes } }
    func refresh() async {
        guard !loading else { return }
        loading = true
        entries = await Task.detached(priority: .utility) { StorageInventory.models() }.value
        loading = false
    }
}

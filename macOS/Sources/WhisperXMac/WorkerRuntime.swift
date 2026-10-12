import Foundation

enum WorkerRuntime {
    static func environment(bundle: URL, inherited: [String: String], bundled: Bool) -> [String: String] {
        var environment = inherited.filter { key, _ in !key.hasPrefix("PYTHON") && !key.hasPrefix("DYLD") && !key.hasPrefix("HF_") && !key.hasPrefix("HUGGING_FACE") }
        let runtime = bundle.appendingPathComponent("Contents/Runtime")
        environment["PATH"] = bundled ? "\(runtime.path)/bin:/usr/bin:/bin:/usr/sbin:/sbin" : "\(runtime.path)/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        // Cache paths are shared with the Models settings panel; credentials are supplied only through stdin.
        for key in ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"] {
            if let value = inherited[key] { environment[key] = value }
        }
        return environment
    }
}

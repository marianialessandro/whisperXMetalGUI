import Foundation
import Testing
@testable import WhisperXMac

struct WorkerRuntimeTests {
    @Test func installedWorkerUsesOnlyBundledToolsAndNoInheritedCredentials() {
        let bundle = URL(fileURLWithPath: "/Applications/Whisper X.app")
        let inherited = ["HOME": "/Users/test", "PYTHONPATH": "/external/packages", "PYTHONHOME": "/external/python", "DYLD_INSERT_LIBRARIES": "/external/inject.dylib", "HF_TOKEN": "private", "HUGGING_FACE_HUB_TOKEN": "private", "HF_HOME": "/Users/test/model cache"]
        let environment = WorkerRuntime.environment(bundle: bundle, inherited: inherited, bundled: true)
        #expect(environment["PATH"] == "/Applications/Whisper X.app/Contents/Runtime/bin:/usr/bin:/bin:/usr/sbin:/sbin")
        #expect(environment["DYLD_LIBRARY_PATH"] == nil)
        #expect(environment["PYTHONPATH"] == nil)
        #expect(environment["PYTHONHOME"] == nil)
        #expect(environment["DYLD_INSERT_LIBRARIES"] == nil)
        #expect(environment["HF_TOKEN"] == nil)
        #expect(environment["HUGGING_FACE_HUB_TOKEN"] == nil)
        #expect(environment["HF_HOME"] == "/Users/test/model cache")
        #expect(environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1")
        #expect(environment["HOME"] == "/Users/test")
    }
}

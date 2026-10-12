import Foundation
import Testing
@testable import WhisperXMac

struct LocalizationTests {
    @Test func interfaceLanguageHonorsExplicitChoiceAndSystemFallback() {
        #expect(L10n.language(preference: "en", preferredLanguages: ["it-IT"]) == "en")
        #expect(L10n.language(preference: "it", preferredLanguages: ["en-GB"]) == "it")
        #expect(L10n.language(preference: "system", preferredLanguages: ["it-IT", "en"]) == "it")
        #expect(L10n.language(preference: "system", preferredLanguages: ["fr-FR"]) == "en")
        #expect(L10n.language(preference: "unknown", preferredLanguages: []) == "en")
    }

    @Test func interpolationPreservesUserContentAndTranslatesCounts() {
        #expect(L10n.tr("Passaggio \(2) di \(6)", language: "en") == "Step 2 of 6")
        #expect(L10n.tr("Passaggio \(2) di \(6)", language: "it") == "Passaggio 2 di 6")
        let filename = "Interview {1} 日本語.wav"
        #expect(L10n.tr("Non riesco a importare \(filename). \("Disk full")", language: "en") == "Could not import Interview {1} 日本語.wav. Disk full")
        #expect(L10n.tr("Una chiave sconosciuta", language: "en") == "Una chiave sconosciuta")
    }

    @Test func packagedCatalogIsPreferredToDevelopmentResources() throws {
        let resources = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: resources) }
        let catalog = resources.appendingPathComponent("WhisperXMac_WhisperXMac.bundle/English.json")
        try FileManager.default.createDirectory(at: catalog.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: catalog)
        #expect(L10n.catalogURL(resourceDirectory: resources) == catalog)
    }

    @Test func catalogPreservesAllInterpolationPlaceholders() {
        #expect(L10n.english.count > 200)
        let regex = try! NSRegularExpression(pattern: #"\{\d+\}"#)
        func placeholders(_ value: String) -> [String] {
            regex.matches(in: value, range: NSRange(value.startIndex..., in: value)).map { String(value[Range($0.range, in: value)!]) }.sorted()
        }
        for (key, translation) in L10n.english {
            #expect(!translation.isEmpty)
            #expect(placeholders(key) == placeholders(translation), "Invalid placeholders: \(key)")
        }
    }

    @Test func guideHasOrderedStepsAndQualityIdentifiersRemainStable() {
        #expect(GuideStep.all.map(\.id) == Array(0..<6))
        #expect(GuideStep.all.allSatisfy { !$0.tips.isEmpty })
        let item = MediaItem(url: URL(fileURLWithPath: "/tmp/example.wav"))
        #expect(item.quality == "Bilanciata")
        #expect(item.language == "auto")
        #expect(L10n.tr("Bilanciata", language: "en") == "Balanced")
    }
}

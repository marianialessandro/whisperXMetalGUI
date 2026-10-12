import Foundation
import SwiftUI

enum InterfaceLanguage: String, CaseIterable, Identifiable {
    case system, italian = "it", english = "en"
    var id: String { rawValue }
    var name: String {
        switch self {
        case .system: L10n.tr("Lingua del sistema")
        case .italian: "Italiano"
        case .english: "English"
        }
    }
}

struct LocalizedTemplate: ExpressibleByStringLiteral, ExpressibleByStringInterpolation {
    var key: String
    var arguments: [String] = []
    init(stringLiteral value: String) { key = value }
    init(stringInterpolation: StringInterpolation) { key = stringInterpolation.key; arguments = stringInterpolation.arguments }

    struct StringInterpolation: StringInterpolationProtocol {
        var key = ""
        var arguments: [String] = []
        init(literalCapacity: Int, interpolationCount: Int) { key.reserveCapacity(literalCapacity) }
        mutating func appendLiteral(_ literal: String) { key += literal }
        mutating func appendInterpolation<T>(_ value: T) {
            key += "{\(arguments.count)}"
            arguments.append(String(describing: value))
        }
    }
}

enum L10n {
    static let preferenceKey = "interfaceLanguage"
    static let english: [String: String] = {
        guard let url = catalogURL(resourceDirectory: Bundle.main.resourceURL), let data = try? Data(contentsOf: url), let catalog = try? JSONDecoder().decode([String: String].self, from: data) else { return [:] }
        return catalog
    }()

    static func catalogURL(resourceDirectory: URL?) -> URL? {
        if let packaged = resourceDirectory?.appendingPathComponent("WhisperXMac_WhisperXMac.bundle/English.json"), FileManager.default.fileExists(atPath: packaged.path) { return packaged }
        return Bundle.module.url(forResource: "English", withExtension: "json")
    }

    static func language(preference: String, preferredLanguages: [String] = Locale.preferredLanguages) -> String {
        if preference == "it" || preference == "en" { return preference }
        return preferredLanguages.first?.hasPrefix("it") == true ? "it" : "en"
    }

    static var currentLanguage: String { language(preference: UserDefaults.standard.string(forKey: preferenceKey) ?? "system") }
    static var locale: Locale { Locale(identifier: currentLanguage == "it" ? "it_IT" : "en_US") }

    static func tr(_ template: LocalizedTemplate, language: String? = nil) -> String {
        let format = (language ?? currentLanguage) == "en" ? english[template.key] ?? template.key : template.key
        // Replace placeholders in one pass so that user text containing a placeholder is preserved.
        return format.split(separator: "{", omittingEmptySubsequences: false).enumerated().map { offset, part in
            guard offset > 0, let close = part.firstIndex(of: "}"), let index = Int(part[..<close]), template.arguments.indices.contains(index) else { return (offset > 0 ? "{" : "") + part }
            return template.arguments[index] + part[part.index(after: close)...]
        }.joined()
    }

    static func display(_ value: String) -> String { tr(LocalizedTemplate(stringLiteral: value)) }
}

struct InterfaceLanguagePicker: View {
    @AppStorage(L10n.preferenceKey) private var language = "system"
    var body: some View {
        Picker(L10n.tr("Lingua dell’interfaccia"), selection: $language) {
            ForEach(InterfaceLanguage.allCases) { Text($0.name).tag($0.rawValue) }
        }
    }
}

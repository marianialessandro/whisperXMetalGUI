// swift-tools-version: 6.0
import PackageDescription

let package = Package(name: "WhisperXMac", platforms: [.macOS(.v14)], products: [.executable(name: "WhisperXMac", targets: ["WhisperXMac"])], targets: [.executableTarget(name: "WhisperXMac", resources: [.process("Resources")]), .testTarget(name: "WhisperXMacTests", dependencies: ["WhisperXMac"])])

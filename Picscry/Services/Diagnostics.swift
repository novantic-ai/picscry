import Foundation
import MetricKit
import OSLog

final class Diagnostics: NSObject {
    static let shared = Diagnostics()

    private let logger = Logger(subsystem: "com.novanticai.picscry", category: "diagnostics")
    private let queue = DispatchQueue(label: "com.novanticai.picscry.diagnostics")
    private let fileManager = FileManager.default
    private lazy var logURL: URL = {
        let baseURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directoryURL = baseURL.appendingPathComponent("Picscry", isDirectory: true)
        try? fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        return directoryURL.appendingPathComponent("diagnostics.log")
    }()

    private override init() {
        super.init()
    }

    func start() {
        MXMetricManager.shared.add(self)
        log("Diagnostics started. Build \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown")).")
    }

    func log(_ message: String) {
        logger.info("\(message, privacy: .public)")
        append("[\(Self.timestamp())] \(message)\n")
    }

    func log(error: Error, context: String) {
        log("\(context): \(error.localizedDescription)")
    }

    func recentText(maxBytes: Int = 80_000) -> String {
        queue.sync {
            guard let data = try? Data(contentsOf: logURL), !data.isEmpty else {
                return "No diagnostics recorded yet."
            }
            let suffix = data.count > maxBytes ? data.suffix(maxBytes) : data[...]
            return String(decoding: suffix, as: UTF8.self)
        }
    }

    func clear() {
        queue.sync {
            try? fileManager.removeItem(at: logURL)
        }
        log("Diagnostics cleared.")
    }

    func exportURL() -> URL {
        logURL
    }

    private func append(_ text: String) {
        queue.async { [logURL] in
            guard let data = text.data(using: .utf8) else { return }
            if FileManager.default.fileExists(atPath: logURL.path) {
                if let handle = try? FileHandle(forWritingTo: logURL) {
                    try? handle.seekToEnd()
                    try? handle.write(contentsOf: data)
                    try? handle.close()
                }
            } else {
                try? data.write(to: logURL, options: .atomic)
            }
        }
    }

    private static func timestamp() -> String {
        ISO8601DateFormatter().string(from: Date())
    }
}

extension Diagnostics: MXMetricManagerSubscriber {
    nonisolated func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for payload in payloads {
            if let data = try? payload.jsonRepresentation(),
               let text = String(data: data, encoding: .utf8) {
                Diagnostics.shared.log("MetricKit diagnostic payload received:\n\(text)")
            } else {
                Diagnostics.shared.log("MetricKit diagnostic payload received but could not be serialized.")
            }
        }
    }

    nonisolated func didReceive(_ payloads: [MXMetricPayload]) {
        Diagnostics.shared.log("MetricKit metric payload received: \(payloads.count) payload(s).")
    }
}

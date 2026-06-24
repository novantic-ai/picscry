import Foundation

enum FaceEmbeddingDebugExportError: LocalizedError {
    case debugDirectoryMissing
    case noDebugFiles
    case archiveCreationFailed

    var errorDescription: String? {
        switch self {
        case .debugDirectoryMissing:
            return "No face embedding debug directory exists yet."
        case .noDebugFiles:
            return "No face embedding debug files are available yet."
        case .archiveCreationFailed:
            return "The face embedding debug bundle could not be created."
        }
    }
}

enum FaceEmbeddingDebugExportService {
    static func makeDebugBundle() throws -> URL {
        guard let debugDirectory = debugDirectoryURL(),
              FileManager.default.fileExists(atPath: debugDirectory.path) else {
            throw FaceEmbeddingDebugExportError.debugDirectoryMissing
        }

        let debugFiles = try FileManager.default.contentsOfDirectory(
            at: debugDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        .filter { url in
            (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !debugFiles.isEmpty else {
            throw FaceEmbeddingDebugExportError.noDebugFiles
        }

        let exportDirectory = try exportDirectoryURL()
        try FileManager.default.createDirectory(at: exportDirectory, withIntermediateDirectories: true)
        let archiveURL = exportDirectory.appendingPathComponent("FaceEmbeddingDebug-\(timestamp()).zip")
        try? FileManager.default.removeItem(at: archiveURL)

        var archive = StoredZipArchive()
        for file in debugFiles {
            let data = try Data(contentsOf: file)
            archive.addFile(
                path: "FaceEmbeddingDebug/\(file.lastPathComponent)",
                data: data
            )
        }

        let manifest = makeManifest(for: debugFiles)
        let manifestData = try JSONEncoder.pretty.encode(manifest)
        archive.addFile(path: "FaceEmbeddingDebug/manifest.json", data: manifestData)

        let diagnosticsData = (try? Data(contentsOf: Diagnostics.shared.exportURL())) ??
            Data(Diagnostics.shared.recentText().utf8)
        archive.addFile(path: "FaceEmbeddingDebug/diagnostics.log", data: diagnosticsData)

        guard archive.write(to: archiveURL) else {
            throw FaceEmbeddingDebugExportError.archiveCreationFailed
        }
        Diagnostics.shared.log("Face embedding debug bundle exported: \(archiveURL.path)")
        return archiveURL
    }

    private static func makeManifest(for files: [URL]) -> FaceEmbeddingDebugManifest {
        let items = files
            .filter { $0.pathExtension.lowercased() == "png" }
            .enumerated()
            .map { index, imageURL in
                let jsonName = imageURL.deletingPathExtension().appendingPathExtension("json").lastPathComponent
                let embeddingURL = imageURL.deletingPathExtension().appendingPathExtension("json")
                let debugIdentifier = debugIdentifier(from: embeddingURL)
                let parsed = parsedDebugIdentifier(debugIdentifier)
                return FaceEmbeddingDebugManifest.Item(
                    index: index + 1,
                    assetID: parsed.assetID,
                    faceIndex: parsed.faceIndex,
                    debugIdentifier: debugIdentifier,
                    imageFile: imageURL.lastPathComponent,
                    embeddingFile: files.contains(where: { $0.lastPathComponent == jsonName }) ? jsonName : nil
                )
            }

        return FaceEmbeddingDebugManifest(
            appVersion: "\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"))",
            schemaVersion: 7,
            modelInputDescription: "data shape [3,112,112]",
            modelOutputDescription: "fc1 shape [128]",
            inputLayout: "CHW [3,112,112] RGB 0...255",
            items: items
        )
    }

    private static func debugDirectoryURL() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("Picscry", isDirectory: true)
            .appendingPathComponent("FaceEmbeddingDebug", isDirectory: true)
    }

    private static func exportDirectoryURL() throws -> URL {
        guard let baseURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            throw FaceEmbeddingDebugExportError.archiveCreationFailed
        }
        return baseURL.appendingPathComponent("PicscryExports", isDirectory: true)
    }

    private static func debugIdentifier(from url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
              let rawObject = try? JSONSerialization.jsonObject(with: data),
              let object = rawObject as? [String: Any] else {
            return nil
        }
        return object["debugIdentifier"] as? String
    }

    private static func parsedDebugIdentifier(_ value: String?) -> (assetID: String?, faceIndex: Int?) {
        guard let value else { return (nil, nil) }
        guard let range = value.range(of: "_face", options: .backwards) else {
            return (value, nil)
        }
        let assetID = String(value[..<range.lowerBound])
        let faceText = String(value[range.upperBound...])
        return (assetID, Int(faceText))
    }

    private static func timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
    }
}

private struct FaceEmbeddingDebugManifest: Encodable {
    struct Item: Encodable {
        let index: Int
        let assetID: String?
        let faceIndex: Int?
        let debugIdentifier: String?
        let imageFile: String
        let embeddingFile: String?
    }

    let appVersion: String
    let schemaVersion: Int
    let modelInputDescription: String
    let modelOutputDescription: String
    let inputLayout: String
    let items: [Item]
}

private struct StoredZipArchive {
    private struct CentralDirectoryEntry {
        let pathData: Data
        let crc32: UInt32
        let compressedSize: UInt32
        let uncompressedSize: UInt32
        let localHeaderOffset: UInt32
    }

    private var data = Data()
    private var entries: [CentralDirectoryEntry] = []

    mutating func addFile(path: String, data fileData: Data) {
        guard let pathData = path.data(using: .utf8),
              let fileSize = UInt32(exactly: fileData.count),
              let headerOffset = UInt32(exactly: data.count) else {
            return
        }

        let crc = CRC32.checksum(fileData)
        data.appendLittleEndian(UInt32(0x04034b50))
        data.appendLittleEndian(UInt16(20))
        data.appendLittleEndian(UInt16(0x0800))
        data.appendLittleEndian(UInt16(0))
        data.appendLittleEndian(UInt16(0))
        data.appendLittleEndian(UInt16(0))
        data.appendLittleEndian(crc)
        data.appendLittleEndian(fileSize)
        data.appendLittleEndian(fileSize)
        data.appendLittleEndian(UInt16(pathData.count))
        data.appendLittleEndian(UInt16(0))
        data.append(pathData)
        data.append(fileData)

        entries.append(CentralDirectoryEntry(
            pathData: pathData,
            crc32: crc,
            compressedSize: fileSize,
            uncompressedSize: fileSize,
            localHeaderOffset: headerOffset
        ))
    }

    mutating func write(to url: URL) -> Bool {
        guard let centralDirectoryOffset = UInt32(exactly: data.count) else { return false }

        for entry in entries {
            data.appendLittleEndian(UInt32(0x02014b50))
            data.appendLittleEndian(UInt16(20))
            data.appendLittleEndian(UInt16(20))
            data.appendLittleEndian(UInt16(0x0800))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(entry.crc32)
            data.appendLittleEndian(entry.compressedSize)
            data.appendLittleEndian(entry.uncompressedSize)
            data.appendLittleEndian(UInt16(entry.pathData.count))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(UInt16(0))
            data.appendLittleEndian(UInt32(0))
            data.appendLittleEndian(entry.localHeaderOffset)
            data.append(entry.pathData)
        }

        guard let centralDirectorySize = UInt32(exactly: data.count - Int(centralDirectoryOffset)),
              let entryCount = UInt16(exactly: entries.count) else {
            return false
        }

        data.appendLittleEndian(UInt32(0x06054b50))
        data.appendLittleEndian(UInt16(0))
        data.appendLittleEndian(UInt16(0))
        data.appendLittleEndian(entryCount)
        data.appendLittleEndian(entryCount)
        data.appendLittleEndian(centralDirectorySize)
        data.appendLittleEndian(centralDirectoryOffset)
        data.appendLittleEndian(UInt16(0))

        do {
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            Diagnostics.shared.log("Failed to write face embedding debug ZIP: \(error.localizedDescription)")
            return false
        }
    }
}

private enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { index in
        var crc = UInt32(index)
        for _ in 0..<8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xedb88320
            } else {
                crc >>= 1
            }
        }
        return crc
    }

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xffffffff
        for byte in data {
            let index = Int((crc ^ UInt32(byte)) & 0xff)
            crc = (crc >> 8) ^ table[index]
        }
        return crc ^ 0xffffffff
    }
}

private extension Data {
    mutating func appendLittleEndian(_ value: UInt16) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }

    mutating func appendLittleEndian(_ value: UInt32) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

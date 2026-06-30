import CoreGraphics
import CoreImage
import CoreML
import Foundation
import UIKit

actor FaceEmbeddingService {
    private static let inputSize = 112
    private let model: MLModel?
    private let modelInputShape: [Int]
    private let modelInputDataType: MLMultiArrayDataType
    private var didLogInputLayout = false
    private var didLogModelDescription = false
    private var didLogModelOutput = false
    private var didLogEmbeddingStats = false
    private var loggedInputStatsCount = 0
    private var debugExportCount = 0
    private var rawFeatureSamples: [[Float]] = []

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .all
        if let modelURL = Bundle.main.url(forResource: "FaceEmbeddingModel", withExtension: "mlmodelc") {
            model = try? MLModel(contentsOf: modelURL, configuration: modelConfiguration)
        } else {
            model = nil
        }
        modelInputShape = Self.inputShape(from: model) ?? [3, Self.inputSize, Self.inputSize]
        modelInputDataType = Self.inputDataType(from: model) ?? .double
    }

    var isModelAvailable: Bool {
        model != nil
    }

    func embedding(
        for faceImage: CGImage,
        debugIdentifier: String? = nil,
        debugMetadata: FaceEmbeddingDebugMetadata? = nil
    ) async throws -> [Float] {
        guard let model else {
            throw FaceEmbeddingServiceError.modelUnavailable
        }

        if !didLogModelDescription {
            didLogModelDescription = true
            Diagnostics.shared.log("Face model input descriptions: \(Self.featureDescriptionText(model.modelDescription.inputDescriptionsByName)).")
            Diagnostics.shared.log("Face model output descriptions: \(Self.featureDescriptionText(model.modelDescription.outputDescriptionsByName)).")
        }

        let input = try Self.multiArrayInput(from: faceImage, shape: modelInputShape, dataType: modelInputDataType)
        if !didLogInputLayout {
            didLogInputLayout = true
            Diagnostics.shared.log("Face embedding input layout: \(input.layoutDescription), RGB, value range 0...255, OpenCV-equivalent swapRB=true path.")
        }
        if loggedInputStatsCount < 10 {
            loggedInputStatsCount += 1
            Diagnostics.shared.log("Face embedding input stats: rMean \(input.stats.red.mean), gMean \(input.stats.green.mean), bMean \(input.stats.blue.mean), rMin \(input.stats.red.min), rMax \(input.stats.red.max), gMin \(input.stats.green.min), gMax \(input.stats.green.max), bMin \(input.stats.blue.min), bMax \(input.stats.blue.max), blackPixelRatio \(input.stats.blackPixelRatio).")
        }

        let output = try await model.prediction(from: FaceEmbeddingFeatureProvider(input: input.array))
        if !didLogModelOutput {
            didLogModelOutput = true
            Diagnostics.shared.log("Face embedding model output feature names: \(output.featureNames.sorted().joined(separator: ", ")).")
        }
        guard let embedding = output.featureValue(for: "fc1")?.multiArrayValue else {
            throw FaceEmbeddingServiceError.outputUnavailable
        }
        let raw = Self.floatArray(from: embedding)
        if !didLogEmbeddingStats {
            didLogEmbeddingStats = true
            let norm = sqrt(raw.reduce(Float(0)) { $0 + ($1 * $1) })
            let minValue = raw.min() ?? 0
            let maxValue = raw.max() ?? 0
            let mean = raw.isEmpty ? 0 : raw.reduce(Float(0), +) / Float(raw.count)
            Diagnostics.shared.log("Face embedding output shape \(embedding.shape), strides \(embedding.strides), dataType \(embedding.dataType).")
            Diagnostics.shared.log("Face embedding raw stats: count \(raw.count), min \(minValue), max \(maxValue), mean \(mean), norm \(norm).")
        }
        recordRawFeatureSample(raw)

        let normalized = raw.l2Normalized()
        exportDebugArtifactsIfNeeded(
            faceImage: faceImage,
            rawEmbedding: raw,
            normalizedEmbedding: normalized,
            debugIdentifier: debugIdentifier,
            debugMetadata: debugMetadata
        )
        return normalized
    }

    private static func multiArrayInput(
        from image: CGImage,
        shape: [Int] = [3, inputSize, inputSize],
        dataType: MLMultiArrayDataType = .double
    ) throws -> FaceEmbeddingInput {
        let size = inputSize
        let bytesPerPixel = 4
        let bytesPerRow = size * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: size * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &pixels,
                width: size,
                height: size,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: rgbaBitmapInfo
              ) else {
            throw FaceEmbeddingServiceError.inputCreationFailed
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        let normalizedShape = normalizedInputShape(shape)
        let arrayDataType: MLMultiArrayDataType = dataType == .float32 ? .float32 : .double
        let array = try MLMultiArray(shape: normalizedShape.map(NSNumber.init(value:)), dataType: arrayDataType)
        var stats = FaceEmbeddingInputStats()

        for y in 0..<size {
            for x in 0..<size {
                let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                let red = Double(pixels[offset])
                let green = Double(pixels[offset + 1])
                let blue = Double(pixels[offset + 2])
                stats.add(red: red, green: green, blue: blue)

                // OpenCV SFace uses blobFromImage(..., swapRB: true) on BGR OpenCV images,
                // so the model receives RGB channel values in the 0...255 range.
                writeModelInput(red: red, green: green, blue: blue, x: x, y: y, into: array)
            }
        }

        return FaceEmbeddingInput(
            array: array,
            stats: stats.finalized(),
            layoutDescription: normalizedShape.count == 4 ? "NCHW batch \(normalizedShape)" : "CHW \(normalizedShape)"
        )
    }

    nonisolated static func debugRGBChannelsForFirstPixel(
        from image: CGImage,
        shape: [Int] = [3, inputSize, inputSize]
    ) throws -> (red: Double, green: Double, blue: Double) {
        let input = try multiArrayInput(from: image, shape: shape)
        let array = input.array
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let channelDimension = shape.count == 4 ? 1 : 0
        let channelStride = strides[channelDimension]
        let baseOffset = shape.count == 4 ? 0 * strides[0] : 0
        switch array.dataType {
        case .float32:
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            return (
                red: Double(pointer[baseOffset]),
                green: Double(pointer[baseOffset + channelStride]),
                blue: Double(pointer[baseOffset + (channelStride * 2)])
            )
        default:
            let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
            return (
                red: pointer[baseOffset],
                green: pointer[baseOffset + channelStride],
                blue: pointer[baseOffset + (channelStride * 2)]
            )
        }
    }

    nonisolated static func debugInputShape(from image: CGImage, shape: [Int]) throws -> [Int] {
        try multiArrayInput(from: image, shape: shape).array.shape.map(\.intValue)
    }

    private static var rgbaBitmapInfo: UInt32 {
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    }

    private static func floatArray(from multiArray: MLMultiArray) -> [Float] {
        let shape = multiArray.shape.map(\.intValue)
        let strides = multiArray.strides.map(\.intValue)
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else { return [] }

        let offsets = flattenedOffsets(shape: shape, strides: strides)
        switch multiArray.dataType {
        case .double:
            let pointer = multiArray.dataPointer.bindMemory(to: Double.self, capacity: multiArray.count)
            return offsets.map { Float(pointer[$0]) }
        case .float32:
            let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
            return offsets.map { pointer[$0] }
        default:
            return []
        }
    }

    private static func flattenedOffsets(shape: [Int], strides: [Int]) -> [Int] {
        guard shape.count == strides.count else { return Array(0..<shape.reduce(1, *)) }

        var offsets: [Int] = []
        offsets.reserveCapacity(shape.reduce(1, *))

        func appendOffsets(dimension: Int, baseOffset: Int) {
            if dimension == shape.count {
                offsets.append(baseOffset)
                return
            }

            for index in 0..<shape[dimension] {
                appendOffsets(
                    dimension: dimension + 1,
                    baseOffset: baseOffset + (index * strides[dimension])
                )
            }
        }

        appendOffsets(dimension: 0, baseOffset: 0)
        return offsets
    }

    private static func inputShape(from model: MLModel?) -> [Int]? {
        guard let constraint = model?.modelDescription.inputDescriptionsByName["data"]?.multiArrayConstraint else {
            return nil
        }
        return normalizedInputShape(constraint.shape.map(\.intValue))
    }

    private static func inputDataType(from model: MLModel?) -> MLMultiArrayDataType? {
        model?.modelDescription.inputDescriptionsByName["data"]?.multiArrayConstraint?.dataType
    }

    private static func normalizedInputShape(_ shape: [Int]) -> [Int] {
        if shape == [1, 3, inputSize, inputSize] || shape == [3, inputSize, inputSize] {
            return shape
        }
        if shape.count == 4, shape[1] == 3, shape[2] == inputSize, shape[3] == inputSize {
            return [max(shape[0], 1), 3, inputSize, inputSize]
        }
        return [3, inputSize, inputSize]
    }

    private static func writeModelInput(
        red: Double,
        green: Double,
        blue: Double,
        x: Int,
        y: Int,
        into array: MLMultiArray
    ) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        func offsets() -> (red: Int, green: Int, blue: Int) {
            if shape.count == 4 {
                let base = 0 * strides[0]
                return (
                    red: base + (0 * strides[1]) + (y * strides[2]) + (x * strides[3]),
                    green: base + (1 * strides[1]) + (y * strides[2]) + (x * strides[3]),
                    blue: base + (2 * strides[1]) + (y * strides[2]) + (x * strides[3])
                )
            }
            return (
                red: (0 * strides[0]) + (y * strides[1]) + (x * strides[2]),
                green: (1 * strides[0]) + (y * strides[1]) + (x * strides[2]),
                blue: (2 * strides[0]) + (y * strides[1]) + (x * strides[2])
            )
        }
        let channelOffsets = offsets()
        if array.dataType == .float32 {
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            pointer[channelOffsets.red] = Float(red)
            pointer[channelOffsets.green] = Float(green)
            pointer[channelOffsets.blue] = Float(blue)
            return
        }
        let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        if shape.count == 4 {
            pointer[channelOffsets.red] = red
            pointer[channelOffsets.green] = green
            pointer[channelOffsets.blue] = blue
            return
        }
        pointer[channelOffsets.red] = red
        pointer[channelOffsets.green] = green
        pointer[channelOffsets.blue] = blue
    }

    private static func featureDescriptionText(_ descriptions: [String: MLFeatureDescription]) -> String {
        descriptions
            .keys
            .sorted()
            .map { name in
                guard let description = descriptions[name] else { return name }
                let shape = description.multiArrayConstraint?.shape.map(\.intValue) ?? []
                let dataType = description.multiArrayConstraint?.dataType
                return "\(name): type \(description.type), shape \(shape), dataType \(String(describing: dataType))"
            }
            .joined(separator: "; ")
    }

    private func recordRawFeatureSample(_ raw: [Float]) {
        guard rawFeatureSamples.count < 32, !raw.isEmpty else { return }
        rawFeatureSamples.append(raw)
        guard rawFeatureSamples.count == 32 else { return }

        var stds: [Float] = []
        stds.reserveCapacity(raw.count)
        for dimension in raw.indices {
            let values = rawFeatureSamples.map { $0[dimension] }
            let mean = values.reduce(Float(0), +) / Float(values.count)
            let variance = values.reduce(Float(0)) { $0 + (($1 - mean) * ($1 - mean)) } / Float(values.count)
            stds.append(sqrt(variance))
        }
        stds.sort()

        var distances: [Float] = []
        for leftIndex in rawFeatureSamples.indices {
            for rightIndex in rawFeatureSamples.indices where rightIndex > leftIndex {
                let squared = zip(rawFeatureSamples[leftIndex], rawFeatureSamples[rightIndex]).reduce(Float(0)) {
                    let delta = $1.0 - $1.1
                    return $0 + (delta * delta)
                }
                distances.append(sqrt(squared))
            }
        }
        distances.sort()

        Diagnostics.shared.log("Face embedding vector diversity: dimensionStd min \(stds.first ?? 0), median \(stds[stds.count / 2]), max \(stds.last ?? 0).")
        Diagnostics.shared.log("Face raw feature pairwise L2: min \(distances.first ?? 0), median \(distances[distances.count / 2]), max \(distances.last ?? 0).")
    }

    private func exportDebugArtifactsIfNeeded(
        faceImage: CGImage,
        rawEmbedding: [Float],
        normalizedEmbedding: [Float],
        debugIdentifier: String?,
        debugMetadata: FaceEmbeddingDebugMetadata?
    ) {
        guard debugExportCount < 20 else { return }
        debugExportCount += 1
        let safeIdentifier = (debugIdentifier ?? "face").map { character in
            character.isLetter || character.isNumber ? character : "_"
        }.reduce(into: "") { $0.append($1) }
        let prefix = String(format: "aligned_%03d_%@", debugExportCount, safeIdentifier)
        guard let directory = Self.debugDirectoryURL() else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let imageURL = directory.appendingPathComponent("\(prefix).png")
        if let data = UIImage(cgImage: faceImage).pngData() {
            try? data.write(to: imageURL, options: .atomic)
        }

        let vectorURL = directory.appendingPathComponent("\(prefix).json")
        let payload = FaceEmbeddingDebugPayload(
            debugIdentifier: debugIdentifier,
            debugMetadata: debugMetadata,
            rawEmbedding: rawEmbedding,
            normalizedEmbedding: normalizedEmbedding
        )
        if let data = try? JSONEncoder().encode(payload) {
            try? data.write(to: vectorURL, options: .atomic)
        }
        Diagnostics.shared.log("Face debug aligned crop saved: \(imageURL.path)")
        Diagnostics.shared.log("Face debug embedding saved: \(vectorURL.path)")
    }

    private static func debugDirectoryURL() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("Picscry", isDirectory: true)
            .appendingPathComponent("FaceEmbeddingDebug", isDirectory: true)
    }
}

private struct FaceEmbeddingInput {
    let array: MLMultiArray
    let stats: FinalFaceEmbeddingInputStats
    let layoutDescription: String
}

private struct FaceEmbeddingInputStats {
    private(set) var red = RunningPixelStats()
    private(set) var green = RunningPixelStats()
    private(set) var blue = RunningPixelStats()
    private var blackPixelCount = 0
    private var pixelCount = 0

    mutating func add(red: Double, green: Double, blue: Double) {
        self.red.add(red)
        self.green.add(green)
        self.blue.add(blue)
        if red < 2, green < 2, blue < 2 {
            blackPixelCount += 1
        }
        pixelCount += 1
    }

    func finalized() -> FinalFaceEmbeddingInputStats {
        FinalFaceEmbeddingInputStats(
            red: red.finalized(),
            green: green.finalized(),
            blue: blue.finalized(),
            blackPixelRatio: pixelCount == 0 ? 0 : Double(blackPixelCount) / Double(pixelCount)
        )
    }
}

private struct RunningPixelStats {
    private var count = 0
    private var sum = 0.0
    private var minimum = Double.greatestFiniteMagnitude
    private var maximum = -Double.greatestFiniteMagnitude

    mutating func add(_ value: Double) {
        count += 1
        sum += value
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    }

    func finalized() -> FinalChannelStats {
        FinalChannelStats(
            min: count == 0 ? 0 : minimum,
            max: count == 0 ? 0 : maximum,
            mean: count == 0 ? 0 : sum / Double(count)
        )
    }
}

private struct FinalFaceEmbeddingInputStats {
    let red: FinalChannelStats
    let green: FinalChannelStats
    let blue: FinalChannelStats
    let blackPixelRatio: Double
}

private struct FinalChannelStats {
    let min: Double
    let max: Double
    let mean: Double
}

private struct FaceEmbeddingDebugPayload: Encodable {
    let debugIdentifier: String?
    let debugMetadata: FaceEmbeddingDebugMetadata?
    let rawEmbedding: [Float]
    let normalizedEmbedding: [Float]
}

private enum FaceEmbeddingServiceError: LocalizedError {
    case modelUnavailable
    case inputCreationFailed
    case outputUnavailable

    var errorDescription: String? {
        switch self {
        case .modelUnavailable:
            return "Face recognition model could not be loaded."
        case .inputCreationFailed:
            return "Face recognition input could not be created."
        case .outputUnavailable:
            return "Face recognition model output was unavailable."
        }
    }
}

private final class FaceEmbeddingFeatureProvider: MLFeatureProvider {
    let input: MLMultiArray

    init(input: MLMultiArray) {
        self.input = input
    }

    var featureNames: Set<String> {
        ["data"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "data" else { return nil }
        return MLFeatureValue(multiArray: input)
    }
}

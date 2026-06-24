import CoreGraphics
import CoreImage
import CoreML
import Foundation
import UIKit

actor FaceEmbeddingService {
    private static let inputSize = 112
    private let model: MLModel?
    private var didLogInputLayout = false
    private var didLogModelOutput = false
    private var didLogEmbeddingStats = false

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .all
        if let modelURL = Bundle.main.url(forResource: "FaceEmbeddingModel", withExtension: "mlmodelc") {
            model = try? MLModel(contentsOf: modelURL, configuration: modelConfiguration)
        } else {
            model = nil
        }
    }

    var isModelAvailable: Bool {
        model != nil
    }

    func embedding(for faceImage: CGImage) async throws -> [Float] {
        guard let model else {
            throw FaceEmbeddingServiceError.modelUnavailable
        }

        let input = try Self.multiArrayInput(from: faceImage)
        if !didLogInputLayout {
            didLogInputLayout = true
            Diagnostics.shared.log("Face embedding input layout: NCHW RGB, value range 0...255, OpenCV-equivalent swapRB=true path.")
        }
        let output = try await model.prediction(from: FaceEmbeddingFeatureProvider(input: input))
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
        return raw.l2Normalized()
    }

    private static func multiArrayInput(from image: CGImage) throws -> MLMultiArray {
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

        let array = try MLMultiArray(shape: [3, NSNumber(value: size), NSNumber(value: size)], dataType: .double)
        let channelStride = array.strides[0].intValue
        let rowStride = array.strides[1].intValue
        let columnStride = array.strides[2].intValue
        let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)

        for y in 0..<size {
            for x in 0..<size {
                let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                let red = Double(pixels[offset])
                let green = Double(pixels[offset + 1])
                let blue = Double(pixels[offset + 2])

                // OpenCV SFace uses blobFromImage(..., swapRB: true) on BGR OpenCV images,
                // so the model receives NCHW RGB values in the 0...255 range. The ONNX/Core ML
                // model contains its own early normalization layers.
                pointer[(0 * channelStride) + (y * rowStride) + (x * columnStride)] = red
                pointer[(1 * channelStride) + (y * rowStride) + (x * columnStride)] = green
                pointer[(2 * channelStride) + (y * rowStride) + (x * columnStride)] = blue
            }
        }

        return array
    }

    static func debugRGBChannelsForFirstPixel(from image: CGImage) throws -> (red: Double, green: Double, blue: Double) {
        let array = try multiArrayInput(from: image)
        let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        let channelStride = array.strides[0].intValue
        return (
            red: pointer[0],
            green: pointer[channelStride],
            blue: pointer[channelStride * 2]
        )
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

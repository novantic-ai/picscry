import CoreGraphics
import CoreImage
import CoreML
import Foundation
import UIKit

actor FaceEmbeddingService {
    private static let inputSize = 112
    private let model: MLModel?

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
        let output = try await model.prediction(from: FaceEmbeddingFeatureProvider(input: input))
        guard let embedding = output.featureValue(for: "fc1")?.multiArrayValue else {
            throw FaceEmbeddingServiceError.outputUnavailable
        }
        return Self.floatArray(from: embedding).l2Normalized()
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
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
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

                // OpenCV SFace expects NCHW BGR input in the 0...255 range. The model's
                // first layers apply (value - 127.5) * 0.0078125.
                pointer[(0 * channelStride) + (y * rowStride) + (x * columnStride)] = blue
                pointer[(1 * channelStride) + (y * rowStride) + (x * columnStride)] = green
                pointer[(2 * channelStride) + (y * rowStride) + (x * columnStride)] = red
            }
        }

        return array
    }

    private static func floatArray(from multiArray: MLMultiArray) -> [Float] {
        switch multiArray.dataType {
        case .double:
            let pointer = multiArray.dataPointer.bindMemory(to: Double.self, capacity: multiArray.count)
            return (0..<multiArray.count).map { Float(pointer[$0]) }
        case .float32:
            let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
            return (0..<multiArray.count).map { pointer[$0] }
        default:
            return []
        }
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

import CoreGraphics
import CoreML
import Foundation
import ImageIO
import UIKit
import Vision

struct DetectedFace {
    let normalizedBoundingBox: CGRect
    let confidence: Float
    let quality: Float?
    let landmarks: FaceLandmarkFivePoint?
    let backend: FaceDetectionBackend
    let detectorRow: [Float]?
}

final class FaceDetectionService {
    private let yunet: YuNetFaceDetectionService?
    private let vision = VisionFaceDetectionService()
    private(set) var lastDetectionUsedVisionFallback = false
    private(set) var lastDetectionYuNetFailureCount = 0

    init() {
        do {
            yunet = try YuNetFaceDetectionService()
            Diagnostics.shared.log("YuNet face detector loaded as primary detector.")
        } catch {
            yunet = nil
            Diagnostics.shared.log("YuNet face detector unavailable at startup; Vision fallback will be used. Error: \(error.localizedDescription)")
        }
    }

    func detectFaces(in cgImage: CGImage, orientation: CGImagePropertyOrientation) async throws -> [DetectedFace] {
        lastDetectionUsedVisionFallback = false
        lastDetectionYuNetFailureCount = 0
        guard let yunet else {
            lastDetectionUsedVisionFallback = true
            lastDetectionYuNetFailureCount = 1
            return try await vision.detectFaces(in: cgImage, orientation: orientation)
        }

        do {
            return try await yunet.detectFaces(in: cgImage)
        } catch {
            lastDetectionUsedVisionFallback = true
            lastDetectionYuNetFailureCount = 1
            Diagnostics.shared.log("YuNet face detection failed; using Vision fallback for this image. Error: \(error.localizedDescription)")
            return try await vision.detectFaces(in: cgImage, orientation: orientation)
        }
    }
}

final class YuNetFaceDetectionService {
    static let inputSize = CGSize(width: 640, height: 640)
    static let scoreThreshold: Float = 0.9
    static let nmsThreshold: Float = 0.3
    static let topK = 5_000

    private let model: MLModel
    private var didLogModelDescription = false

    init() throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        guard let modelURL = Bundle.main.url(forResource: "FaceDetectionModel", withExtension: "mlmodelc") else {
            throw YuNetFaceDetectionError.modelUnavailable
        }
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    func detectFaces(in cgImage: CGImage) async throws -> [DetectedFace] {
        let model = self.model
        let shouldLogDescription = !didLogModelDescription
        didLogModelDescription = true

        return try await Task.detached(priority: .utility) {
            if shouldLogDescription {
                Diagnostics.shared.log("YuNet model input descriptions: \(Self.featureDescriptionText(model.modelDescription.inputDescriptionsByName)).")
                Diagnostics.shared.log("YuNet model output descriptions: \(Self.featureDescriptionText(model.modelDescription.outputDescriptionsByName)).")
            }

            let prepared = try Self.modelInput(from: cgImage)
            let output = try model.prediction(from: YuNetFeatureProvider(input: prepared.array))
            let rows = try Self.decodeRows(
                output: output,
                originalImageSize: CGSize(width: cgImage.width, height: cgImage.height),
                inputScale: prepared.scale,
                scoreThreshold: Self.scoreThreshold,
                nmsThreshold: Self.nmsThreshold,
                topK: Self.topK
            )
            Diagnostics.shared.log("YuNet detected \(rows.count) faces after NMS for \(cgImage.width)x\(cgImage.height) image.")
            return rows.map { row in
                DetectedFace(
                    normalizedBoundingBox: row.normalizedBoundingBox,
                    confidence: row.score,
                    quality: nil,
                    landmarks: row.landmarks,
                    backend: .yunet,
                    detectorRow: row.detectorRow
                )
            }
            .sorted { $0.normalizedBoundingBox.minX < $1.normalizedBoundingBox.minX }
        }.value
    }

    nonisolated static func decodedDetectionForTesting(
        rawRow: [Float],
        originalImageSize: CGSize,
        inputScale: CGFloat
    ) -> YuNetDecodedDetection? {
        guard rawRow.count == 15 else { return nil }
        return decodedDetection(from: rawRow, originalImageSize: originalImageSize, inputScale: inputScale)
    }

    private static func modelInput(from image: CGImage) throws -> YuNetPreparedInput {
        let size = Int(inputSize.width)
        let bytesPerPixel = 4
        let bytesPerRow = size * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: size * bytesPerRow)
        let imageSize = CGSize(width: image.width, height: image.height)
        let scale = min(inputSize.width / max(imageSize.width, 1), inputSize.height / max(imageSize.height, 1))
        let drawSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &pixels,
                width: size,
                height: size,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
              ) else {
            throw YuNetFaceDetectionError.inputCreationFailed
        }

        context.interpolationQuality = .high
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        context.draw(image, in: CGRect(origin: .zero, size: drawSize))

        let array = try MLMultiArray(
            shape: [NSNumber(value: 3), NSNumber(value: size), NSNumber(value: size)],
            dataType: .double
        )
        let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        let strides = array.strides.map(\.intValue)

        for y in 0..<size {
            for x in 0..<size {
                let pixelOffset = (y * bytesPerRow) + (x * bytesPerPixel)
                let red = Double(pixels[pixelOffset])
                let green = Double(pixels[pixelOffset + 1])
                let blue = Double(pixels[pixelOffset + 2])
                pointer[(0 * strides[0]) + (y * strides[1]) + (x * strides[2])] = blue
                pointer[(1 * strides[0]) + (y * strides[1]) + (x * strides[2])] = green
                pointer[(2 * strides[0]) + (y * strides[1]) + (x * strides[2])] = red
            }
        }

        return YuNetPreparedInput(array: array, scale: scale)
    }

    private static func decodeRows(
        output: MLFeatureProvider,
        originalImageSize: CGSize,
        inputScale: CGFloat,
        scoreThreshold: Float,
        nmsThreshold: Float,
        topK: Int
    ) throws -> [YuNetDecodedDetection] {
        let rawOutputs = try YuNetRawOutputs(output: output)
        var candidates: [YuNetDecodedDetection] = []

        for spec in YuNetHeadSpec.all {
            for row in 0..<spec.rows {
                for column in 0..<spec.columns {
                    let cls = sigmoid(rawOutputs.score(for: spec, row: row, column: column))
                    let obj = sigmoid(rawOutputs.objectness(for: spec, row: row, column: column))
                    let score = sqrt(max(0, cls) * max(0, obj))
                    guard score >= scoreThreshold else { continue }

                    let bbox = rawOutputs.bbox(for: spec, row: row, column: column)
                    let stride = Float(spec.stride)
                    let centerX = (Float(column) + bbox[0]) * stride
                    let centerY = (Float(row) + bbox[1]) * stride
                    let width = exp(bbox[2]) * stride
                    let height = exp(bbox[3]) * stride
                    let x = centerX - (width / 2)
                    let y = centerY - (height / 2)

                    let keypoints = rawOutputs.keypoints(for: spec, row: row, column: column)
                    let detectorRow: [Float] = [
                        x, y, width, height,
                        (Float(column) + keypoints[0]) * stride,
                        (Float(row) + keypoints[1]) * stride,
                        (Float(column) + keypoints[2]) * stride,
                        (Float(row) + keypoints[3]) * stride,
                        (Float(column) + keypoints[4]) * stride,
                        (Float(row) + keypoints[5]) * stride,
                        (Float(column) + keypoints[6]) * stride,
                        (Float(row) + keypoints[7]) * stride,
                        (Float(column) + keypoints[8]) * stride,
                        (Float(row) + keypoints[9]) * stride,
                        score
                    ]
                    if let detection = decodedDetection(
                        from: detectorRow,
                        originalImageSize: originalImageSize,
                        inputScale: inputScale
                    ) {
                        candidates.append(detection)
                    }
                }
            }
        }

        return nonMaximumSuppression(
            candidates.sorted { $0.score > $1.score }.prefix(topK).map { $0 },
            threshold: nmsThreshold
        )
    }

    private static func decodedDetection(
        from detectorRow: [Float],
        originalImageSize: CGSize,
        inputScale: CGFloat
    ) -> YuNetDecodedDetection? {
        guard detectorRow.count == 15, inputScale > 0 else { return nil }
        let inverseScale = 1 / inputScale
        let imageRect = CGRect(origin: .zero, size: originalImageSize)
        let pixelRect = CGRect(
            x: CGFloat(detectorRow[0]) * inverseScale,
            y: CGFloat(detectorRow[1]) * inverseScale,
            width: CGFloat(detectorRow[2]) * inverseScale,
            height: CGFloat(detectorRow[3]) * inverseScale
        )
        .intersection(imageRect)

        guard pixelRect.width > 0, pixelRect.height > 0 else { return nil }

        let rightEye = CGPoint(x: CGFloat(detectorRow[4]) * inverseScale, y: CGFloat(detectorRow[5]) * inverseScale)
        let leftEye = CGPoint(x: CGFloat(detectorRow[6]) * inverseScale, y: CGFloat(detectorRow[7]) * inverseScale)
        let nose = CGPoint(x: CGFloat(detectorRow[8]) * inverseScale, y: CGFloat(detectorRow[9]) * inverseScale)
        let rightMouth = CGPoint(x: CGFloat(detectorRow[10]) * inverseScale, y: CGFloat(detectorRow[11]) * inverseScale)
        let leftMouth = CGPoint(x: CGFloat(detectorRow[12]) * inverseScale, y: CGFloat(detectorRow[13]) * inverseScale)
        let normalized = CGRect(
            x: pixelRect.minX / max(originalImageSize.width, 1),
            y: 1 - (pixelRect.maxY / max(originalImageSize.height, 1)),
            width: pixelRect.width / max(originalImageSize.width, 1),
            height: pixelRect.height / max(originalImageSize.height, 1)
        )

        return YuNetDecodedDetection(
            normalizedBoundingBox: normalized,
            pixelBoundingBox: pixelRect,
            landmarks: FaceLandmarkFivePoint(
                rightEye: rightEye,
                leftEye: leftEye,
                noseTip: nose,
                rightMouth: rightMouth,
                leftMouth: leftMouth
            ),
            score: detectorRow[14],
            detectorRow: detectorRow
        )
    }

    private static func nonMaximumSuppression(
        _ detections: [YuNetDecodedDetection],
        threshold: Float
    ) -> [YuNetDecodedDetection] {
        var selected: [YuNetDecodedDetection] = []
        for detection in detections {
            if selected.allSatisfy({ intersectionOverUnion(detection.pixelBoundingBox, $0.pixelBoundingBox) <= CGFloat(threshold) }) {
                selected.append(detection)
            }
        }
        return selected
    }

    private static func intersectionOverUnion(_ left: CGRect, _ right: CGRect) -> CGFloat {
        let intersection = left.intersection(right)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = (left.width * left.height) + (right.width * right.height) - intersectionArea
        return unionArea <= 0 ? 0 : intersectionArea / unionArea
    }

    private static func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
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
}

private final class VisionFaceDetectionService {
    func detectFaces(in cgImage: CGImage, orientation: CGImagePropertyOrientation) async throws -> [DetectedFace] {
        try await withCheckedThrowingContinuation { continuation in
            Task.detached(priority: .utility) {
                let rectangleRequest = VNDetectFaceRectanglesRequest()
                if #available(iOS 15.0, *) {
                    rectangleRequest.revision = VNDetectFaceRectanglesRequestRevision3
                }

                let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
                do {
                    try handler.perform([rectangleRequest])
                    let rectangles = (rectangleRequest.results ?? []).filter { $0.confidence > 0.3 }

                    let landmarkRequest = VNDetectFaceLandmarksRequest()
                    landmarkRequest.inputFaceObservations = rectangles
                    try handler.perform([landmarkRequest])

                    let landmarkResults = landmarkRequest.results ?? rectangles
                    let faces = landmarkResults
                        .map {
                            DetectedFace(
                                normalizedBoundingBox: $0.boundingBox,
                                confidence: $0.confidence,
                                quality: $0.faceCaptureQuality,
                                landmarks: Self.fivePointLandmarks(
                                    from: $0.landmarks,
                                    faceBox: $0.boundingBox,
                                    imageSize: CGSize(width: cgImage.width, height: cgImage.height)
                                ),
                                backend: .visionFallback,
                                detectorRow: nil
                            )
                        }
                        .sorted { $0.normalizedBoundingBox.minX < $1.normalizedBoundingBox.minX }

                    Diagnostics.shared.log("Vision fallback detected \(faces.count) faces for \(cgImage.width)x\(cgImage.height) image.")
                    continuation.resume(returning: faces)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func fivePointLandmarks(
        from landmarks: VNFaceLandmarks2D?,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> FaceLandmarkFivePoint? {
        guard let landmarks,
              let visionLeftEye = landmarks.leftEye,
              let visionRightEye = landmarks.rightEye,
              let rightEye = landmarkCenter(visionRightEye, faceBox: faceBox, imageSize: imageSize),
              let leftEye = landmarkCenter(visionLeftEye, faceBox: faceBox, imageSize: imageSize),
              let noseTip = noseTip(from: landmarks, faceBox: faceBox, imageSize: imageSize),
              let mouthCorners = mouthCorners(from: landmarks, faceBox: faceBox, imageSize: imageSize) else {
            return nil
        }

        return FaceLandmarkFivePoint(
            rightEye: rightEye,
            leftEye: leftEye,
            noseTip: noseTip,
            rightMouth: mouthCorners.right,
            leftMouth: mouthCorners.left
        )
    }

    private static func noseTip(
        from landmarks: VNFaceLandmarks2D,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> CGPoint? {
        if let noseCrest = landmarks.noseCrest,
           let bottom = landmarkPoints(noseCrest, faceBox: faceBox, imageSize: imageSize).max(by: { $0.y < $1.y }) {
            return bottom
        }
        if let nose = landmarks.nose,
           let bottom = landmarkPoints(nose, faceBox: faceBox, imageSize: imageSize).max(by: { $0.y < $1.y }) {
            return bottom
        }
        return nil
    }

    private static func mouthCorners(
        from landmarks: VNFaceLandmarks2D,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> (left: CGPoint, right: CGPoint)? {
        let region = landmarks.outerLips ?? landmarks.innerLips
        guard let region else { return nil }
        let points = landmarkPoints(region, faceBox: faceBox, imageSize: imageSize)
        guard let left = points.min(by: { $0.x < $1.x }),
              let right = points.max(by: { $0.x < $1.x }) else {
            return nil
        }
        return (left, right)
    }

    private static func landmarkCenter(
        _ region: VNFaceLandmarkRegion2D,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> CGPoint? {
        guard region.pointCount > 0 else { return nil }
        let points = region.normalizedPoints
        let sum = points.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + CGFloat(point.x), y: partial.y + CGFloat(point.y))
        }
        let average = CGPoint(
            x: sum.x / CGFloat(region.pointCount),
            y: sum.y / CGFloat(region.pointCount)
        )
        return CGPoint(
            x: (faceBox.minX + (average.x * faceBox.width)) * imageSize.width,
            y: (1 - (faceBox.minY + (average.y * faceBox.height))) * imageSize.height
        )
    }

    private static func landmarkPoints(
        _ region: VNFaceLandmarkRegion2D,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> [CGPoint] {
        region.normalizedPoints.map { point in
            CGPoint(
                x: (faceBox.minX + (CGFloat(point.x) * faceBox.width)) * imageSize.width,
                y: (1 - (faceBox.minY + (CGFloat(point.y) * faceBox.height))) * imageSize.height
            )
        }
    }
}

struct YuNetDecodedDetection {
    let normalizedBoundingBox: CGRect
    let pixelBoundingBox: CGRect
    let landmarks: FaceLandmarkFivePoint
    let score: Float
    let detectorRow: [Float]
}

private struct YuNetPreparedInput {
    let array: MLMultiArray
    let scale: CGFloat
}

private enum YuNetFaceDetectionError: LocalizedError {
    case modelUnavailable
    case inputCreationFailed
    case missingOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelUnavailable:
            return "YuNet face detection model could not be loaded."
        case .inputCreationFailed:
            return "YuNet face detection input could not be created."
        case let .missingOutput(name):
            return "YuNet face detection output \(name) was unavailable."
        }
    }
}

private struct YuNetHeadSpec {
    let stride: Int
    let rows: Int
    let columns: Int

    var suffix: String { "\(stride)" }

    static let all = [
        YuNetHeadSpec(stride: 8, rows: 80, columns: 80),
        YuNetHeadSpec(stride: 16, rows: 40, columns: 40),
        YuNetHeadSpec(stride: 32, rows: 20, columns: 20)
    ]
}

private struct YuNetRawOutputs {
    private let arrays: [String: MLMultiArray]

    init(output: MLFeatureProvider) throws {
        var arrays: [String: MLMultiArray] = [:]
        for name in [
            "cls_8_raw", "cls_16_raw", "cls_32_raw",
            "obj_8_raw", "obj_16_raw", "obj_32_raw",
            "bbox_8_raw", "bbox_16_raw", "bbox_32_raw",
            "kps_8_raw", "kps_16_raw", "kps_32_raw"
        ] {
            guard let value = output.featureValue(for: name)?.multiArrayValue else {
                throw YuNetFaceDetectionError.missingOutput(name)
            }
            arrays[name] = value
        }
        self.arrays = arrays
    }

    func score(for spec: YuNetHeadSpec, row: Int, column: Int) -> Float {
        value("cls_\(spec.suffix)_raw", channel: 0, row: row, column: column)
    }

    func objectness(for spec: YuNetHeadSpec, row: Int, column: Int) -> Float {
        value("obj_\(spec.suffix)_raw", channel: 0, row: row, column: column)
    }

    func bbox(for spec: YuNetHeadSpec, row: Int, column: Int) -> [Float] {
        (0..<4).map { value("bbox_\(spec.suffix)_raw", channel: $0, row: row, column: column) }
    }

    func keypoints(for spec: YuNetHeadSpec, row: Int, column: Int) -> [Float] {
        (0..<10).map { value("kps_\(spec.suffix)_raw", channel: $0, row: row, column: column) }
    }

    private func value(_ name: String, channel: Int, row: Int, column: Int) -> Float {
        guard let array = arrays[name] else { return 0 }
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let offset: Int
        if shape.count == 4 {
            offset = (0 * strides[0]) + (channel * strides[1]) + (row * strides[2]) + (column * strides[3])
        } else if shape.count == 3 {
            offset = (channel * strides[0]) + (row * strides[1]) + (column * strides[2])
        } else {
            offset = min(array.count - 1, max(0, (channel * row * column)))
        }

        switch array.dataType {
        case .float32:
            return array.dataPointer.bindMemory(to: Float.self, capacity: array.count)[offset]
        case .double:
            return Float(array.dataPointer.bindMemory(to: Double.self, capacity: array.count)[offset])
        default:
            return 0
        }
    }
}

private final class YuNetFeatureProvider: MLFeatureProvider {
    let input: MLMultiArray

    init(input: MLMultiArray) {
        self.input = input
    }

    var featureNames: Set<String> {
        ["input"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "input" else { return nil }
        return MLFeatureValue(multiArray: input)
    }
}

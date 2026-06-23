import CoreGraphics
import Foundation
import ImageIO
import UIKit

struct FaceRecognitionConfiguration {
    var faceProcessingMaxDimension: CGFloat = 1_600
    var minimumFacePixelSize: CGFloat = 48
    var autoMatchThreshold: Float = 0.84
    var possibleMatchThreshold: Float = 0.76
    var mergeThreshold: Float = 0.82
    var singleSampleAutoMatchThreshold: Float = 0.88
    var disallowMultipleFacesFromSameAssetForSamePerson: Bool = true
    var minimumBestSecondBestMargin: Float = 0.025
    var representativeThumbnailSize: CGFloat = 256
    var avatarPaddingRatio: CGFloat = 0.85
    var modelInputPaddingRatio: CGFloat = 0.25
    var indexingBatchSize: Int = 20
    var embeddingDimension = 128
    var faceImageRequestTimeoutSeconds: TimeInterval = 45
    var peopleRefreshBatchSize: Int = 50
    var databaseSaveBatchSize: Int = 100
    var embeddingCalibrationSampleCount: Int = 32
    var collapsedEmbeddingMedianSimilarityThreshold: Float = 0.90
    var collapsedEmbeddingMinimumSimilarityThreshold: Float = 0.75
    var disableAutoClusteringWhenEmbeddingHealthSuspicious = true
    var clusterRebuildBatchSize: Int = 100
    var maximumAllPairsClusteringFaceCount: Int = 5_000
    var graphEdgeSimilarityThreshold: Float = 0.92
    var graphEdgeSimilarityThresholdForSingleSample: Float = 0.94
}

struct PersonSummary: Identifiable, Hashable {
    let id: UUID
    let displayName: String
    let isUnknown: Bool
    let photoCount: Int
    let faceCount: Int
    let representativeFaceImageData: Data?
    let isProvisional: Bool

    init(
        id: UUID,
        displayName: String,
        isUnknown: Bool,
        photoCount: Int,
        faceCount: Int,
        representativeFaceImageData: Data?,
        isProvisional: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.isUnknown = isUnknown
        self.photoCount = photoCount
        self.faceCount = faceCount
        self.representativeFaceImageData = representativeFaceImageData
        self.isProvisional = isProvisional
    }
}

struct PhotoFaceSummary: Identifiable, Hashable {
    let id: UUID
    let personID: UUID
    let assetLocalIdentifier: String
    let displayName: String
    let isUnknown: Bool
    let normalizedBoundingBox: CGRect
    let leftToRightIndex: Int
    let confidence: Float
    let representativeFaceImageData: Data?
    let isManuallyCorrected: Bool
}

struct FaceObservationInput {
    let assetLocalIdentifier: String
    let assetModificationDate: Date?
    let assetPixelWidth: Int
    let assetPixelHeight: Int
    let normalizedBoundingBox: CGRect
    let leftToRightIndex: Int
    let detectionConfidence: Float
    let faceQuality: Float?
    let embedding: [Float]
    let faceCropImageData: Data?
}

enum FaceIndexingState: Equatable {
    case idle
    case indexing(processed: Int, total: Int)
    case paused
    case failed(String)

    var isIndexing: Bool {
        if case .indexing = self { return true }
        return false
    }
}

enum RenamePersonResult: Equatable {
    case renamed
    case needsMergeConfirmation(existingPersonID: UUID, existingName: String)
}

struct FaceProcessingImage {
    let image: UIImage
    let cgImage: CGImage
    let orientation: CGImagePropertyOrientation
    let pixelWidth: Int
    let pixelHeight: Int
}

extension Array where Element == Float {
    func l2Normalized() -> [Float] {
        let magnitude = sqrt(reduce(Float(0)) { $0 + ($1 * $1) })
        guard magnitude > 0 else { return self }
        return map { $0 / magnitude }
    }

    func cosineSimilarity(to other: [Float]) -> Float {
        guard count == other.count else { return 0 }
        return zip(self, other).reduce(Float(0)) { $0 + ($1.0 * $1.1) }
    }
}

extension Data {
    init(float32Array values: [Float]) {
        self = values.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return Data() }
            return Data(
                bytes: baseAddress,
                count: buffer.count * MemoryLayout<Float>.stride
            )
        }
    }

    func float32Array() -> [Float] {
        guard count >= MemoryLayout<Float>.stride else { return [] }
        let valueCount = count / MemoryLayout<Float>.stride
        return withUnsafeBytes { rawBuffer -> [Float] in
            (0..<valueCount).map { index in
                rawBuffer.load(
                    fromByteOffset: index * MemoryLayout<Float>.stride,
                    as: Float.self
                )
            }
        }
    }
}

enum PeopleOrdering {
    static func sorted(_ people: [PersonSummary]) -> [PersonSummary] {
        let named = people
            .filter { !$0.isUnknown }
            .sorted {
                $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending
            }

        let unknown = people
            .filter(\.isUnknown)
            .sorted {
                if $0.photoCount != $1.photoCount { return $0.photoCount > $1.photoCount }
                return $0.faceCount > $1.faceCount
            }

        return named + unknown
    }
}

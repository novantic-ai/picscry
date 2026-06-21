import CoreLocation
import Foundation
import Photos

struct PhotoAssetSummary: Identifiable {
    let id: String
    let creationDate: Date?
    let modificationDate: Date?
    let pixelWidth: Int
    let pixelHeight: Int
    let duration: TimeInterval
    let mediaType: PHAssetMediaType
    let mediaSubtypes: PHAssetMediaSubtype
    let sourceType: PHAssetSourceType
    let isFavorite: Bool
    let isHidden: Bool
    let burstIdentifier: String?
    let location: CLLocation?

    init(asset: PHAsset) {
        id = asset.localIdentifier
        creationDate = asset.creationDate
        modificationDate = asset.modificationDate
        pixelWidth = asset.pixelWidth
        pixelHeight = asset.pixelHeight
        duration = asset.duration
        mediaType = asset.mediaType
        mediaSubtypes = asset.mediaSubtypes
        sourceType = asset.sourceType
        isFavorite = asset.isFavorite
        isHidden = asset.isHidden
        burstIdentifier = asset.burstIdentifier
        location = asset.location
    }

    var displayTitle: String {
        creationDate?.formatted(date: .abbreviated, time: .shortened) ?? "Undated Photo"
    }

    var dimensionsText: String {
        "\(pixelWidth) x \(pixelHeight)"
    }

    var accessibilitySummary: String {
        let date = creationDate?.formatted(date: .complete, time: .shortened) ?? "unknown date"
        let favorite = isFavorite ? ", favorite" : ""
        let locationText = location.map { ", location \($0.coordinate.latitude.formatted(.number.precision(.fractionLength(4)))), \($0.coordinate.longitude.formatted(.number.precision(.fractionLength(4))))" } ?? ""
        return "\(mediaType.displayName), \(dimensionsText), created \(date)\(favorite)\(locationText)"
    }
}

struct PhotoResourceSummary: Identifiable {
    let id = UUID()
    let originalFilename: String
    let uniformTypeIdentifier: String
    let resourceType: PHAssetResourceType

    init(resource: PHAssetResource) {
        originalFilename = resource.originalFilename
        uniformTypeIdentifier = resource.uniformTypeIdentifier
        resourceType = resource.type
    }
}

extension PHAssetMediaType {
    var displayName: String {
        switch self {
        case .image: "Image"
        case .video: "Video"
        case .audio: "Audio"
        case .unknown: "Unknown"
        @unknown default: "Unknown"
        }
    }
}

extension PHAssetResourceType {
    var displayName: String {
        switch self {
        case .photo: "Photo"
        case .video: "Video"
        case .audio: "Audio"
        case .alternatePhoto: "Alternate Photo"
        case .fullSizePhoto: "Full Size Photo"
        case .fullSizeVideo: "Full Size Video"
        case .adjustmentBaseVideo: "Adjustment Base Video"
        case .adjustmentData: "Adjustment Data"
        case .adjustmentBasePhoto: "Adjustment Base Photo"
        case .pairedVideo: "Live Photo Video"
        case .fullSizePairedVideo: "Full Size Live Photo Video"
        case .adjustmentBasePairedVideo: "Adjustment Base Live Photo Video"
        case .photoProxy: "Photo Proxy"
        @unknown default: "Unknown"
        }
    }
}

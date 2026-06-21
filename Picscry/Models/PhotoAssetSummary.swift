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
        creationDate?.formatted(date: .abbreviated, time: .shortened) ?? "Undated \(mediaType.accessibilityName)"
    }

    var isVideo: Bool {
        mediaType == .video
    }

    var durationText: String {
        guard duration.isFinite, duration > 0 else { return "0:00" }

        let totalSeconds = Int(duration.rounded())
        let hours = totalSeconds / 3_600
        let minutes = (totalSeconds % 3_600) / 60
        let seconds = totalSeconds % 60

        if hours > 0 {
            return "\(hours):\(String(format: "%02d", minutes)):\(String(format: "%02d", seconds))"
        }

        return "\(minutes):\(String(format: "%02d", seconds))"
    }

    var dimensionsText: String {
        "\(pixelWidth) x \(pixelHeight)"
    }

    var accessibilitySummary: String {
        let date = creationDate?.formatted(date: .complete, time: .shortened) ?? "unknown date and time"
        return "\(mediaType.accessibilityName), \(date)"
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
        case .image: "Photo"
        case .video: "Video"
        case .audio: "Audio"
        case .unknown: "Unknown"
        @unknown default: "Unknown"
        }
    }

    var accessibilityName: String {
        displayName.lowercased()
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

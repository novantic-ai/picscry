import CoreGraphics
import CoreLocation
import Foundation
import ImageIO
import Photos
import UIKit

enum PhotoQualityMode {
    case fastPreview
    case fullResolutionRendered
    case originalAssetData
}

enum LoadedImageQuality {
    case preview
    case fullResolutionRendered
}

struct PhotoDisplayImageUpdate {
    let image: UIImage
    let quality: LoadedImageQuality
}

struct PhotoDisplayQualityState {
    static func nextQuality(current: LoadedImageQuality?, incoming: LoadedImageQuality) -> LoadedImageQuality {
        if current == .fullResolutionRendered && incoming == .preview {
            return .fullResolutionRendered
        }
        return incoming
    }
}

struct OriginalPhotoData {
    let data: Data
    let uniformTypeIdentifier: String
    let originalFilename: String
    let orientation: CGImagePropertyOrientation?
    let pixelWidth: Int
    let pixelHeight: Int
    let resourceType: PHAssetResourceType
}

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
        creationDate?.formatted(date: .abbreviated, time: .shortened) ?? "Undated \(mediaKind.accessibilityName)"
    }

    var isVideo: Bool {
        mediaType == .video
    }

    var isScreenshot: Bool {
        mediaType == .image && mediaSubtypes.contains(.photoScreenshot)
    }

    var mediaKind: LibraryMediaKind {
        if isVideo { return .video }
        if isScreenshot { return .screenshot }
        return .photo
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

    var aspectRatio: CGFloat {
        guard pixelWidth > 0, pixelHeight > 0 else { return 1 }
        return CGFloat(pixelWidth) / CGFloat(pixelHeight)
    }

    var orientationText: String {
        if pixelWidth > pixelHeight { return "landscape" }
        if pixelHeight > pixelWidth { return "portrait" }
        return "square"
    }

    var thumbnailTargetSize: CGSize {
        let width: CGFloat = 720
        return CGSize(width: width, height: max(1, width / aspectRatio))
    }

    var originalTargetSize: CGSize {
        PHImageManagerMaximumSize
    }

    var dimensionsText: String {
        "\(pixelWidth) x \(pixelHeight)"
    }

    var accessibilitySummary: String {
        var parts = [mediaKind.accessibilityName, orientationText]
        if isFavorite { parts.append("favorite") }
        if isVideo { parts.append("duration \(durationText)") }
        parts.append(creationDate?.formatted(date: .complete, time: .shortened) ?? "unknown date and time")
        return parts.joined(separator: ", ")
    }
}

enum LibraryMediaKind: String, CaseIterable, Identifiable {
    case photo
    case screenshot
    case video

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .photo: "Photo"
        case .screenshot: "Screenshot"
        case .video: "Video"
        }
    }

    var accessibilityName: String { displayName.lowercased() }
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

    init(originalFilename: String, uniformTypeIdentifier: String, resourceType: PHAssetResourceType) {
        self.originalFilename = originalFilename
        self.uniformTypeIdentifier = uniformTypeIdentifier
        self.resourceType = resourceType
    }

    static func preferredOriginalPhotoResource(in resources: [PhotoResourceSummary]) -> PhotoResourceSummary? {
        resources
            .filter { $0.resourceType.isOriginalPhotoDataCandidate }
            .min { lhs, rhs in
                let lhsPriority = lhs.resourceType.originalPhotoDataPriority
                let rhsPriority = rhs.resourceType.originalPhotoDataPriority
                if lhsPriority != rhsPriority { return lhsPriority < rhsPriority }
                return lhs.originalFilename.localizedStandardCompare(rhs.originalFilename) == .orderedAscending
            }
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
    var isOriginalPhotoDataCandidate: Bool {
        switch self {
        case .fullSizePhoto, .photo, .alternatePhoto:
            return true
        default:
            return false
        }
    }

    var originalPhotoDataPriority: Int {
        switch self {
        case .fullSizePhoto: 0
        case .photo: 1
        case .alternatePhoto: 2
        default: Int.max
        }
    }

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

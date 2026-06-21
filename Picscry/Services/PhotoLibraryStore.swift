import ImageIO
import Observation
import Photos
import SwiftUI

@MainActor
@Observable
final class PhotoLibraryStore: NSObject {
    enum AuthorizationState: Equatable {
        case unknown
        case requesting
        case authorized
        case limited
        case denied
        case restricted

        var canReadLibrary: Bool {
            self == .authorized || self == .limited
        }
    }

    var authorizationState: AuthorizationState = .unknown
    var assets: [PhotoAssetSummary] = []
    var isLoading = false
    var errorMessage: String?

    private let imageManager = PHCachingImageManager()
    private var hasRegisteredChangeObserver = false

    override init() {
        super.init()
        authorizationState = Self.authorizationState(from: PHPhotoLibrary.authorizationStatus(for: .readWrite))
        registerForChangesIfNeeded()
    }

    deinit {
        PHPhotoLibrary.shared().unregisterChangeObserver(self)
    }

    func prepareLibrary() async {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        authorizationState = Self.authorizationState(from: status)

        if authorizationState == .unknown {
            authorizationState = .requesting
            let requestedStatus = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            authorizationState = Self.authorizationState(from: requestedStatus)
        }

        guard authorizationState.canReadLibrary else {
            assets = []
            return
        }

        registerForChangesIfNeeded()
        await reloadAssets()
    }

    func reloadAssets() async {
        guard authorizationState.canReadLibrary else { return }
        isLoading = true
        errorMessage = nil

        let fetchedAssets = fetchImageAssets()
        assets = fetchedAssets
        imageManager.stopCachingImagesForAllAssets()
        imageManager.startCachingImages(
            for: fetchedAssets.compactMap { Self.asset(with: $0.id) },
            targetSize: CGSize(width: 320, height: 320),
            contentMode: .aspectFill,
            options: nil
        )
        isLoading = false
    }

    func thumbnail(for summary: PhotoAssetSummary, targetSize: CGSize) async -> UIImage? {
        guard let asset = Self.asset(with: summary.id) else { return nil }

        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true

        return await withCheckedContinuation { continuation in
            var didResume = false
            imageManager.requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFill,
                options: options
            ) { image, info in
                let isDegraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
                guard !didResume, !isDegraded else { return }
                didResume = true
                continuation.resume(returning: image)
            }
        }
    }

    func metadata(for summary: PhotoAssetSummary) async -> PhotoMetadata {
        guard let asset = Self.asset(with: summary.id) else { return .empty }

        var sections = [
            PhotoMetadataSection(
                id: "photo-library",
                title: "Photo Library",
                items: libraryItems(for: summary)
            ),
            PhotoMetadataSection(
                id: "resources",
                title: "Resources",
                items: resourceItems(for: summary.resourceSummaries)
            )
        ]

        if let imageProperties = await imageProperties(for: asset), !imageProperties.isEmpty {
            sections.append(
                PhotoMetadataSection(
                    id: "image-properties",
                    title: "Image Metadata",
                    items: flatten(metadata: imageProperties)
                )
            )
        }

        return PhotoMetadata(sections: sections.filter { !$0.items.isEmpty })
    }

    private func fetchImageAssets() -> [PhotoAssetSummary] {
        let options = PHFetchOptions()
        options.includeHiddenAssets = true
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        let result = PHAsset.fetchAssets(with: .image, options: options)

        var summaries: [PhotoAssetSummary] = []
        summaries.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in
            summaries.append(PhotoAssetSummary(asset: asset))
        }
        return summaries
    }

    private func libraryItems(for summary: PhotoAssetSummary) -> [PhotoMetadataItem] {
        var items: [PhotoMetadataItem] = [
            item("Identifier", summary.id),
            item("Media Type", summary.mediaType.displayName),
            item("Dimensions", summary.dimensionsText),
            item("Favorite", summary.isFavorite ? "Yes" : "No"),
            item("Hidden", summary.isHidden ? "Yes" : "No"),
            item("Source", sourceTypeText(summary.sourceType))
        ]

        if let creationDate = summary.creationDate {
            items.append(item("Created", creationDate.formatted(date: .complete, time: .standard)))
        }
        if let modificationDate = summary.modificationDate {
            items.append(item("Modified", modificationDate.formatted(date: .complete, time: .standard)))
        }
        if let location = summary.location {
            items.append(item("Latitude", location.coordinate.latitude.formatted(.number.precision(.fractionLength(6)))))
            items.append(item("Longitude", location.coordinate.longitude.formatted(.number.precision(.fractionLength(6)))))
            if location.altitude.isFinite {
                items.append(item("Altitude", "\(location.altitude.formatted(.number.precision(.fractionLength(1)))) m"))
            }
        }
        if let burstIdentifier = summary.burstIdentifier {
            items.append(item("Burst Identifier", burstIdentifier))
        }
        if !summary.mediaSubtypes.isEmpty {
            items.append(item("Subtypes", mediaSubtypeText(summary.mediaSubtypes)))
        }
        return items
    }

    private func resourceItems(for resources: [PhotoResourceSummary]) -> [PhotoMetadataItem] {
        resources.flatMap { resource in
            [
                item("Filename", resource.originalFilename),
                item("Resource Type", resource.resourceType.displayName),
                item("Uniform Type", resource.uniformTypeIdentifier)
            ]
        }
    }

    private func imageProperties(for asset: PHAsset) async -> [String: Any]? {
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.version = .current

        return await withCheckedContinuation { continuation in
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, _, _, info in
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.resume(returning: nil)
                    return
                }
                guard let data,
                      let source = CGImageSourceCreateWithData(data as CFData, nil),
                      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: properties)
            }
        }
    }

    private func flatten(metadata: [String: Any]) -> [PhotoMetadataItem] {
        metadata
            .flatMap { key, value -> [PhotoMetadataItem] in
                if let nested = value as? [String: Any] {
                    return nested.map { nestedKey, nestedValue in
                        item("\(cleanKey(key)) \(cleanKey(nestedKey))", stringValue(nestedValue))
                    }
                }
                return [item(cleanKey(key), stringValue(value))]
            }
            .sorted { $0.label.localizedStandardCompare($1.label) == .orderedAscending }
    }

    private func item(_ label: String, _ value: String) -> PhotoMetadataItem {
        PhotoMetadataItem(label: label, value: value)
    }

    private func stringValue(_ value: Any) -> String {
        switch value {
        case let number as NSNumber:
            return number.stringValue
        case let string as String:
            return string
        case let array as [Any]:
            return array.map(stringValue).joined(separator: ", ")
        default:
            return "\(value)"
        }
    }

    private func cleanKey(_ key: String) -> String {
        key
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: "Exif", with: "EXIF")
            .replacingOccurrences(of: "TIFF", with: "TIFF ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func sourceTypeText(_ sourceType: PHAssetSourceType) -> String {
        var values: [String] = []
        if sourceType.contains(.typeUserLibrary) { values.append("User Library") }
        if sourceType.contains(.typeCloudShared) { values.append("iCloud Shared") }
        if sourceType.contains(.typeiTunesSynced) { values.append("Synced") }
        return values.isEmpty ? "Unknown" : values.joined(separator: ", ")
    }

    private func mediaSubtypeText(_ subtypes: PHAssetMediaSubtype) -> String {
        var values: [String] = []
        if subtypes.contains(.photoPanorama) { values.append("Panorama") }
        if subtypes.contains(.photoHDR) { values.append("HDR") }
        if subtypes.contains(.photoScreenshot) { values.append("Screenshot") }
        if subtypes.contains(.photoLive) { values.append("Live Photo") }
        if subtypes.contains(.photoDepthEffect) { values.append("Depth Effect") }
        return values.isEmpty ? "None" : values.joined(separator: ", ")
    }

    private func registerForChangesIfNeeded() {
        guard !hasRegisteredChangeObserver else { return }
        PHPhotoLibrary.shared().register(self)
        hasRegisteredChangeObserver = true
    }

    private static func authorizationState(from status: PHAuthorizationStatus) -> AuthorizationState {
        switch status {
        case .notDetermined: .unknown
        case .restricted: .restricted
        case .denied: .denied
        case .authorized: .authorized
        case .limited: .limited
        @unknown default: .denied
        }
    }

    private static func asset(with localIdentifier: String) -> PHAsset? {
        PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil).firstObject
    }
}

extension PhotoLibraryStore: PHPhotoLibraryChangeObserver {
    nonisolated func photoLibraryDidChange(_ changeInstance: PHChange) {
        Task { @MainActor in
            await reloadAssets()
        }
    }
}

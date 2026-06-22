import ImageIO
import Observation
import AVFoundation
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
    var totalAssetCount = 0

    private let imageManager = PHCachingImageManager()
    private var hasRegisteredChangeObserver = false
    private var reloadTask: Task<Void, Never>?
    private var detailCacheIdentifiers: Set<String> = []
    private var detailCacheTargetSize: CGSize?

    override init() {
        super.init()
        authorizationState = Self.authorizationState(from: PHPhotoLibrary.authorizationStatus(for: .readWrite))
    }

    deinit {
        PHPhotoLibrary.shared().unregisterChangeObserver(self)
    }

    func prepareLibrary() async {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        authorizationState = Self.authorizationState(from: status)

        if authorizationState == .unknown {
            authorizationState = .requesting
            Diagnostics.shared.log("Requesting PhotoKit read/write authorization.")
            let requestedStatus = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            authorizationState = Self.authorizationState(from: requestedStatus)
            Diagnostics.shared.log("PhotoKit authorization completed: \(authorizationState.diagnosticName).")
        }

        guard authorizationState.canReadLibrary else {
            assets = []
            totalAssetCount = 0
            Diagnostics.shared.log("Photo library unavailable: \(authorizationState.diagnosticName).")
            return
        }

        registerForChangesIfNeeded()
        await reloadAssets()
    }

    func reloadAssets() async {
        guard authorizationState.canReadLibrary else { return }
        reloadTask?.cancel()
        isLoading = true
        errorMessage = nil
        let shouldPublishIncrementally = assets.isEmpty

        let result = fetchLibraryAssets()
        totalAssetCount = result.count
        Diagnostics.shared.log("Starting PhotoKit fetch for \(result.count) photo and video assets.")
        imageManager.stopCachingImagesForAllAssets()
        detailCacheIdentifiers.removeAll()
        detailCacheTargetSize = nil

        let task = Task { @MainActor in
            let batchSize = 150
            var batch: [PhotoAssetSummary] = []
            var refreshedAssets: [PhotoAssetSummary] = []
            batch.reserveCapacity(batchSize)
            refreshedAssets.reserveCapacity(result.count)

            guard result.count > 0 else {
                assets = []
                isLoading = false
                Diagnostics.shared.log("Finished PhotoKit fetch with 0 media summaries.")
                return
            }

            for index in 0..<result.count {
                guard !Task.isCancelled else {
                    let loadedCount = shouldPublishIncrementally ? assets.count : refreshedAssets.count
                    Diagnostics.shared.log("PhotoKit fetch cancelled after loading \(loadedCount) summaries.")
                    return
                }

                let asset = result.object(at: index)
                batch.append(PhotoAssetSummary(asset: asset))

                if batch.count == batchSize || index == result.count - 1 {
                    if shouldPublishIncrementally {
                        assets.append(contentsOf: batch)
                    } else {
                        refreshedAssets.append(contentsOf: batch)
                    }
                    batch.removeAll(keepingCapacity: true)
                    let loadedCount = shouldPublishIncrementally ? assets.count : refreshedAssets.count
                    Diagnostics.shared.log("Loaded \(loadedCount) of \(result.count) media summaries.")
                    await Task.yield()
                }
            }

            if !shouldPublishIncrementally {
                assets = refreshedAssets
            }
            isLoading = false
            Diagnostics.shared.log("Finished PhotoKit fetch with \(assets.count) media summaries.")
        }

        reloadTask = task
        await task.value
    }

    func thumbnail(
        for summary: PhotoAssetSummary,
        targetSize: CGSize,
        deliveryMode: PHImageRequestOptionsDeliveryMode = .highQualityFormat,
        contentMode: PHImageContentMode = .aspectFit
    ) async -> UIImage? {
        guard let asset = Self.asset(with: summary.id) else { return nil }
        guard !Task.isCancelled else { return nil }

        let options = PHImageRequestOptions()
        options.deliveryMode = deliveryMode
        options.resizeMode = targetSize == PHImageManagerMaximumSize ? .none : .fast
        options.isNetworkAccessAllowed = true
        options.version = .current

        return await withCheckedContinuation { continuation in
            var didResume = false
            imageManager.requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: contentMode,
                options: options
            ) { image, info in
                guard !didResume else { return }
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    didResume = true
                    continuation.resume(returning: nil)
                    return
                }
                if let error = info?[PHImageErrorKey] as? Error {
                    Diagnostics.shared.log("PhotoKit image request failed: \(error.localizedDescription)")
                    didResume = true
                    continuation.resume(returning: nil)
                    return
                }
                let isDegraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
                if isDegraded {
                    guard deliveryMode == .opportunistic, image != nil else { return }
                }
                didResume = true
                continuation.resume(returning: image)
            }
        }
    }

    // PHImageManager.requestImage returns a rendered UIImage suitable for display.
    // It is not guaranteed to be the original file bytes. Use originalPhotoData(for:)
    // for the original asset resource without recompression.
    func fullQualityImageUpdates(for summary: PhotoAssetSummary) -> AsyncStream<UIImage> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                for await update in fullResolutionRenderedImageUpdates(for: summary) {
                    guard !Task.isCancelled else {
                        continuation.finish()
                        return
                    }
                    continuation.yield(update.image)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // Returns display-rendered UIImages: a fast preview followed by the best full-resolution
    // rendered image PhotoKit can provide. This stream does not expose original file bytes.
    func displayThenFullQualityImageUpdates(for summary: PhotoAssetSummary, displayTargetSize: CGSize) -> AsyncStream<PhotoDisplayImageUpdate> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                // Preview rendering is capped for speed while paging through detail photos.
                if let displayImage = await thumbnail(
                    for: summary,
                    targetSize: Self.normalizedTargetSize(displayTargetSize),
                    deliveryMode: .highQualityFormat,
                    contentMode: .aspectFit
                ) {
                    Diagnostics.shared.log("Loaded preview image for \(summary.id): asset \(summary.pixelWidth)x\(summary.pixelHeight), UIImage \(Self.pixelSizeText(for: displayImage)).")
                    continuation.yield(PhotoDisplayImageUpdate(image: displayImage, quality: .preview))
                }

                guard !Task.isCancelled else {
                    continuation.finish()
                    return
                }

                // Final rendered image requests must stay uncapped so normal HEIC/JPEG photos
                // can match the PHAsset pixel dimensions when PhotoKit provides full size.
                if let renderedImage = await originalRenderedImage(for: summary) {
                    guard !Task.isCancelled else { return }
                    Diagnostics.shared.log("Loaded full-resolution rendered image for \(summary.id): asset \(summary.pixelWidth)x\(summary.pixelHeight), UIImage \(Self.pixelSizeText(for: renderedImage)).")
                    continuation.yield(PhotoDisplayImageUpdate(image: renderedImage, quality: .fullResolutionRendered))
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // Full-resolution rendered UIImage for display. This is not the original file bytes.
    // Use originalPhotoData(for:) when exporting, sharing, or inspecting the exact asset
    // resource bytes without recompression.
    func fullResolutionRenderedImageUpdates(for summary: PhotoAssetSummary) -> AsyncStream<PhotoDisplayImageUpdate> {
        fullResolutionRenderedImageUpdates(for: summary, includeDegradedResults: true)
    }

    func originalPhotoData(for summary: PhotoAssetSummary) async throws -> OriginalPhotoData {
        guard let asset = Self.asset(with: summary.id) else {
            throw PhotoLibraryStoreError.assetNotFound
        }

        let resources = PHAssetResource.assetResources(for: asset)
        guard let selectedResource = Self.preferredOriginalPhotoResource(in: resources) else {
            throw PhotoLibraryStoreError.originalPhotoResourceNotFound
        }

        Diagnostics.shared.log("Selected original asset resource for \(summary.id): type \(selectedResource.type.displayName), filename \(selectedResource.originalFilename), UTI \(selectedResource.uniformTypeIdentifier), asset \(asset.pixelWidth)x\(asset.pixelHeight).")

        let options = PHAssetResourceRequestOptions()
        options.isNetworkAccessAllowed = true

        let data: Data = try await withCheckedThrowingContinuation { continuation in
            var accumulatedData = Data()
            let resourceManager = PHAssetResourceManager.default()
            resourceManager.requestData(
                for: selectedResource,
                options: options,
                dataReceivedHandler: { chunk in
                    accumulatedData.append(chunk)
                },
                completionHandler: { error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume(returning: accumulatedData)
                    }
                }
            )
        }

        let orientation = Self.imageOrientation(in: data)
        Diagnostics.shared.log("Fetched original asset data for \(summary.id): \(data.count) bytes, type \(selectedResource.type.displayName), filename \(selectedResource.originalFilename), UTI \(selectedResource.uniformTypeIdentifier), orientation \(orientation?.rawValue.description ?? "unknown").")
        Self.logImageIOMetadata(
            in: data,
            context: "original asset data",
            summary: summary,
            uniformTypeIdentifier: selectedResource.uniformTypeIdentifier
        )

        return OriginalPhotoData(
            data: data,
            uniformTypeIdentifier: selectedResource.uniformTypeIdentifier,
            originalFilename: selectedResource.originalFilename,
            orientation: orientation,
            pixelWidth: asset.pixelWidth,
            pixelHeight: asset.pixelHeight,
            resourceType: selectedResource.type
        )
    }

    func originalRenderedImage(for summary: PhotoAssetSummary) async -> UIImage? {
        guard let asset = Self.asset(with: summary.id) else { return nil }

        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .none
        options.version = .current

        let image: UIImage? = await withCheckedContinuation { continuation in
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, _, orientation, info in
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.resume(returning: nil)
                    return
                }
                if let error = info?[PHImageErrorKey] as? Error {
                    Diagnostics.shared.log("PhotoKit original rendered image data request failed: \(error.localizedDescription)")
                    continuation.resume(returning: nil)
                    return
                }
                guard let data else {
                    continuation.resume(returning: nil)
                    return
                }
                Self.logImageIOMetadata(
                    in: data,
                    context: "PhotoKit rendered image data",
                    summary: summary,
                    uniformTypeIdentifier: nil
                )
                guard let image = Self.decodeDisplayImage(data: data, orientation: orientation, summary: summary) else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: image)
            }
        }

        if let image {
            Diagnostics.shared.log("Decoded original rendered image data for \(summary.id): asset \(asset.pixelWidth)x\(asset.pixelHeight), UIImage \(Self.pixelSizeText(for: image)).")
            return image
        }

        Diagnostics.shared.log("Falling back to PHImageManagerMaximumSize rendered image for \(summary.id).")
        for await update in fullResolutionRenderedImageUpdates(for: summary, includeDegradedResults: false) {
            return update.image
        }
        return nil
    }

    func updateDetailImageCache(for summaries: [PhotoAssetSummary], targetSize: CGSize) {
        let imageSummaries = summaries.filter { !$0.isVideo }
        let nextIdentifiers = Set(imageSummaries.map(\.id))
        let cacheTargetSize = Self.normalizedTargetSize(targetSize)
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true
        options.version = .current

        if let previousTargetSize = detailCacheTargetSize,
           previousTargetSize != cacheTargetSize,
           !detailCacheIdentifiers.isEmpty {
            imageManager.stopCachingImages(
                for: Self.assets(with: Array(detailCacheIdentifiers)),
                targetSize: previousTargetSize,
                contentMode: .aspectFit,
                options: options
            )
            detailCacheIdentifiers.removeAll()
        }

        let identifiersToStop = detailCacheIdentifiers.subtracting(nextIdentifiers)
        let identifiersToStart = nextIdentifiers.subtracting(detailCacheIdentifiers)

        if !identifiersToStop.isEmpty {
            let assetsToStop = Self.assets(with: Array(identifiersToStop))
            imageManager.stopCachingImages(
                for: assetsToStop,
                targetSize: cacheTargetSize,
                contentMode: .aspectFit,
                options: options
            )
        }

        if !identifiersToStart.isEmpty {
            let assetsToStart = Self.assets(with: Array(identifiersToStart))
            imageManager.startCachingImages(
                for: assetsToStart,
                targetSize: cacheTargetSize,
                contentMode: .aspectFit,
                options: options
            )
        }

        detailCacheIdentifiers = nextIdentifiers
        detailCacheTargetSize = cacheTargetSize
    }

    private func fullResolutionRenderedImageUpdates(for summary: PhotoAssetSummary, includeDegradedResults: Bool) -> AsyncStream<PhotoDisplayImageUpdate> {
        guard let asset = Self.asset(with: summary.id) else {
            return AsyncStream { continuation in
                continuation.finish()
            }
        }

        let options = PHImageRequestOptions()
        options.deliveryMode = includeDegradedResults ? .opportunistic : .highQualityFormat
        options.resizeMode = .none
        options.isNetworkAccessAllowed = true
        options.version = .current

        return AsyncStream { continuation in
            let imageManager = PHImageManager.default()
            let requestToken = ImageRequestToken()
            requestToken.requestID = imageManager.requestImage(
                for: asset,
                targetSize: PHImageManagerMaximumSize,
                contentMode: .aspectFit,
                options: options
            ) { image, info in
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.finish()
                    return
                }
                if let error = info?[PHImageErrorKey] as? Error {
                    Diagnostics.shared.log("PhotoKit full-quality image request failed: \(error.localizedDescription)")
                    continuation.finish()
                    return
                }
                guard let image else {
                    let isDegraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
                    if !isDegraded {
                        continuation.finish()
                    }
                    return
                }

                let isDegraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
                if !isDegraded || includeDegradedResults {
                    continuation.yield(PhotoDisplayImageUpdate(
                        image: image,
                        quality: isDegraded ? .preview : .fullResolutionRendered
                    ))
                }

                if !isDegraded {
                    continuation.finish()
                }
            }

            continuation.onTermination = { _ in
                guard requestToken.requestID != PHInvalidImageRequestID else { return }
                imageManager.cancelImageRequest(requestToken.requestID)
            }
        }
    }

    func playerItem(for summary: PhotoAssetSummary) async -> AVPlayerItem? {
        guard summary.isVideo, let asset = Self.asset(with: summary.id) else { return nil }

        let options = PHVideoRequestOptions()
        options.deliveryMode = .automatic
        options.isNetworkAccessAllowed = true
        options.version = .current

        return await withCheckedContinuation { continuation in
            PHImageManager.default().requestPlayerItem(forVideo: asset, options: options) { playerItem, info in
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.resume(returning: nil)
                    return
                }
                if let error = info?[PHImageErrorKey] as? Error {
                    Diagnostics.shared.log("PhotoKit video request failed: \(error.localizedDescription)")
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: playerItem)
            }
        }
    }

    func metadata(for summary: PhotoAssetSummary, renderMode: PhotoRenderMode = .preferHDR) async -> PhotoMetadata {
        guard let asset = Self.asset(with: summary.id) else { return .empty }
        let resourceSummaries = PHAssetResource.assetResources(for: asset).map(PhotoResourceSummary.init(resource:))

        var sections = [
            PhotoMetadataSection(
                id: "photo-library",
                title: "Photo Library",
                items: libraryItems(for: summary)
            ),
            PhotoMetadataSection(
                id: "resources",
                title: "Resources",
                items: resourceItems(for: resourceSummaries)
            )
        ]

        if let selectedOriginalResource = PhotoResourceSummary.preferredOriginalPhotoResource(in: resourceSummaries) {
            Diagnostics.shared.log("Metadata selected original resource candidate for \(summary.id): type \(selectedOriginalResource.resourceType.displayName), filename \(selectedOriginalResource.originalFilename), UTI \(selectedOriginalResource.uniformTypeIdentifier), asset \(asset.pixelWidth)x\(asset.pixelHeight).")
            sections.append(
                PhotoMetadataSection(
                    id: "selected-original-resource",
                    title: "Original Data Candidate",
                    items: [
                        item("Resource Type", selectedOriginalResource.resourceType.displayName),
                        item("Filename", selectedOriginalResource.originalFilename),
                        item("Uniform Type", selectedOriginalResource.uniformTypeIdentifier)
                    ]
                )
            )
        }

        if asset.mediaType == .image, let imageProperties = await imageProperties(for: asset), !imageProperties.isEmpty {
            sections.append(
                PhotoMetadataSection(
                    id: "image-properties",
                    title: "Image Metadata",
                    items: flatten(metadata: imageProperties)
                )
            )
        }

        if asset.mediaType == .image {
            let renderingItems = await renderingDiagnosticItems(
                for: summary,
                asset: asset,
                resourceSummaries: resourceSummaries,
                renderMode: renderMode
            )
            sections.append(
                PhotoMetadataSection(
                    id: "rendering-diagnostics",
                    title: "Rendering Diagnostics",
                    items: renderingItems
                )
            )
        } else {
            sections.append(
                PhotoMetadataSection(
                    id: "video-rendering-diagnostics",
                    title: "Rendering Diagnostics",
                    items: [
                        item("Video HDR/EDR Follow-up", "Investigate AVPlayerLayer HDR/EDR behavior, original/current video asset quality, and AVAsset track color primaries, transfer function, and HDR format.")
                    ]
                )
            )
        }

        return PhotoMetadata(sections: sections.filter { !$0.items.isEmpty })
    }

    private func fetchLibraryAssets() -> PHFetchResult<PHAsset> {
        let options = PHFetchOptions()
        options.predicate = NSPredicate(
            format: "mediaType == %d || mediaType == %d",
            PHAssetMediaType.image.rawValue,
            PHAssetMediaType.video.rawValue
        )
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        return PHAsset.fetchAssets(with: options)
    }

    private func libraryItems(for summary: PhotoAssetSummary) -> [PhotoMetadataItem] {
        var items: [PhotoMetadataItem] = [
            item("Identifier", summary.id),
            item("Media Type", summary.mediaKind.displayName),
            item("Dimensions", summary.dimensionsText),
            item("Orientation", summary.orientationText),
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

    private func renderingDiagnosticItems(
        for summary: PhotoAssetSummary,
        asset: PHAsset,
        resourceSummaries: [PhotoResourceSummary],
        renderMode: PhotoRenderMode
    ) async -> [PhotoMetadataItem] {
        var items: [PhotoMetadataItem] = [
            item("Asset Pixel Dimensions", "\(asset.pixelWidth) x \(asset.pixelHeight)"),
            item("Render Mode", renderMode.diagnosticName),
            item("Display Gamut", UIScreen.main.traitCollection.displayGamut.diagnosticName),
            item("iOS Version", UIDevice.current.systemVersion),
            item("HDR Dynamic Range Requested", renderMode == .preferHDR ? "Yes on iOS 17+" : "No; standard dynamic range requested on iOS 17+")
        ]

        let previewTargetSize = Self.normalizedTargetSize(CGSize(width: asset.pixelWidth, height: asset.pixelHeight))
        if let previewImage = await thumbnail(for: summary, targetSize: previewTargetSize, deliveryMode: .highQualityFormat, contentMode: .aspectFit) {
            items.append(item("Preview UIImage Pixel Dimensions", Self.pixelSizeText(for: previewImage)))
        } else {
            items.append(item("Preview UIImage Pixel Dimensions", "Unavailable"))
        }

        if let fullImage = await originalRenderedImage(for: summary) {
            items.append(item("Full Rendered UIImage Pixel Dimensions", Self.pixelSizeText(for: fullImage)))
            items.append(contentsOf: Self.cgImageDiagnosticItems(for: fullImage).map { item($0.label, $0.value) })
        } else {
            items.append(item("Full Rendered UIImage Pixel Dimensions", "Unavailable"))
        }

        if let selectedOriginalResource = PhotoResourceSummary.preferredOriginalPhotoResource(in: resourceSummaries) {
            items.append(item("Resource Type Selected", selectedOriginalResource.resourceType.displayName))
            items.append(item("UTI", selectedOriginalResource.uniformTypeIdentifier))
        } else {
            items.append(item("Resource Type Selected", "Unavailable"))
            items.append(item("UTI", "Unavailable"))
        }

        return items
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
        if subtypes.contains(.videoHighFrameRate) { values.append("High Frame Rate") }
        if subtypes.contains(.videoTimelapse) { values.append("Timelapse") }
        if subtypes.contains(.videoCinematic) { values.append("Cinematic") }
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

    private static func assets(with localIdentifiers: [String]) -> [PHAsset] {
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: localIdentifiers, options: nil)
        var assets: [PHAsset] = []
        assets.reserveCapacity(fetchResult.count)
        fetchResult.enumerateObjects { asset, _, _ in
            assets.append(asset)
        }
        return assets
    }

    private static func normalizedTargetSize(_ targetSize: CGSize) -> CGSize {
        guard targetSize.width.isFinite, targetSize.height.isFinite, targetSize.width > 0, targetSize.height > 0 else {
            return CGSize(width: 1_600, height: 1_600)
        }

        return CGSize(
            width: max(1, min(targetSize.width, 3_000)),
            height: max(1, min(targetSize.height, 3_000))
        )
    }

    private nonisolated static func preferredOriginalPhotoResource(in resources: [PHAssetResource]) -> PHAssetResource? {
        resources
            .filter { $0.type.isOriginalPhotoDataCandidate }
            .min { lhs, rhs in
                let lhsPriority = lhs.type.originalPhotoDataPriority
                let rhsPriority = rhs.type.originalPhotoDataPriority
                if lhsPriority != rhsPriority { return lhsPriority < rhsPriority }
                return lhs.originalFilename.localizedStandardCompare(rhs.originalFilename) == .orderedAscending
            }
    }

    private nonisolated static func imageOrientation(in data: Data) -> CGImagePropertyOrientation? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let rawValue = properties[kCGImagePropertyOrientation] as? UInt32 else {
            return nil
        }
        return CGImagePropertyOrientation(rawValue: rawValue)
    }

    private nonisolated static func decodeDisplayImage(data: Data, orientation: CGImagePropertyOrientation, summary: PhotoAssetSummary) -> UIImage? {
        guard let image = UIImage(data: data) else {
            Diagnostics.shared.log("UIImage(data:) failed for \(summary.id).")
            return nil
        }

        logCGImageDiagnostics(for: image, context: "UIImage(data:) decoded image", summary: summary)

        if #available(iOS 17.0, *) {
            // TODO: If the project adopts a public SDK API such as UIImageReader with explicit
            // preferred dynamic range configuration, prefer that decoder here. Do not convert to
            // JPEG/PNG, redraw with UIGraphicsImageRenderer, or flatten to sRGB for the HDR path.
            Diagnostics.shared.log("Decoded \(summary.id) with UIImage(data:). HDR display depends on UIImageView.preferredImageDynamicRange in this build.")
        }

        return Self.image(image, applying: orientation)
    }

    private nonisolated static func logImageIOMetadata(
        in data: Data,
        context: String,
        summary: PhotoAssetSummary,
        uniformTypeIdentifier: String?
    ) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            Diagnostics.shared.log("ImageIO \(context) metadata unavailable for \(summary.id): could not create image source.")
            return
        }

        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] ?? [:]
        let fileUTI = CGImageSourceGetType(source).map { $0 as String } ?? uniformTypeIdentifier ?? "unknown"
        let profileName = properties[kCGImagePropertyProfileName as String].map(diagnosticStringValue) ?? "unknown"
        let colorModel = properties[kCGImagePropertyColorModel as String].map(diagnosticStringValue) ?? "unknown"
        let depth = properties[kCGImagePropertyDepth as String].map(diagnosticStringValue) ?? "unknown"
        let pixelWidth = properties[kCGImagePropertyPixelWidth as String].map(diagnosticStringValue) ?? "unknown"
        let pixelHeight = properties[kCGImagePropertyPixelHeight as String].map(diagnosticStringValue) ?? "unknown"
        let hdrKeys = hdrRelatedKeys(in: properties)
        let exif = properties[kCGImagePropertyExifDictionary as String] as? [String: Any] ?? [:]
        let brightness = exif[kCGImagePropertyExifBrightnessValue as String].map(diagnosticStringValue) ?? "unknown"
        let exposureTime = exif[kCGImagePropertyExifExposureTime as String].map(diagnosticStringValue) ?? "unknown"
        let fNumber = exif[kCGImagePropertyExifFNumber as String].map(diagnosticStringValue) ?? "unknown"
        let iso = exif[kCGImagePropertyExifISOSpeedRatings as String].map(diagnosticStringValue) ?? "unknown"

        Diagnostics.shared.log("ImageIO \(context) for \(summary.id): fileUTI \(fileUTI), inputUTI \(uniformTypeIdentifier ?? "unknown"), profile \(profileName), colorModel \(colorModel), depth \(depth), pixels \(pixelWidth)x\(pixelHeight), HDR/gain-map keys \(hdrKeys.isEmpty ? "none" : hdrKeys.joined(separator: ", ")), EXIF brightness \(brightness), exposure \(exposureTime), fNumber \(fNumber), ISO \(iso).")
    }

    private nonisolated static func logCGImageDiagnostics(for image: UIImage, context: String, summary: PhotoAssetSummary) {
        let cgImage = image.cgImage
        let colorSpaceName = cgImage?.colorSpace?.name.map { "\($0)" } ?? "unknown"
        let imageAssetStatus = image.imageAsset == nil ? "none" : "present"
        Diagnostics.shared.log("\(context) for \(summary.id): UIImage \(pixelSizeText(for: image)), scale \(image.scale), imageAsset \(imageAssetStatus), CGImage bitsPerComponent \(cgImage?.bitsPerComponent.description ?? "unknown"), bitsPerPixel \(cgImage?.bitsPerPixel.description ?? "unknown"), colorSpace \(colorSpaceName).")
    }

    private nonisolated static func cgImageDiagnosticItems(for image: UIImage) -> [(label: String, value: String)] {
        let cgImage = image.cgImage
        let colorSpaceName = cgImage?.colorSpace?.name.map { "\($0)" } ?? "unknown"
        return [
            ("CGImage Color Space", colorSpaceName),
            ("Bits Per Component", cgImage?.bitsPerComponent.description ?? "unknown"),
            ("Bits Per Pixel", cgImage?.bitsPerPixel.description ?? "unknown")
        ]
    }

    private nonisolated static func hdrRelatedKeys(in metadata: [String: Any]) -> [String] {
        var matches: [String] = []

        func visit(keyPath: String, value: Any) {
            let lowered = keyPath.lowercased()
            if lowered.contains("hdr") ||
                lowered.contains("gain") ||
                lowered.contains("headroom") ||
                lowered.contains("dynamicrange") {
                matches.append(keyPath)
            }

            if let dictionary = value as? [String: Any] {
                for (key, nestedValue) in dictionary {
                    visit(keyPath: keyPath.isEmpty ? key : "\(keyPath).\(key)", value: nestedValue)
                }
            } else if let dictionary = value as? [CFString: Any] {
                for (key, nestedValue) in dictionary {
                    let stringKey = key as String
                    visit(keyPath: keyPath.isEmpty ? stringKey : "\(keyPath).\(stringKey)", value: nestedValue)
                }
            }
        }

        for (key, value) in metadata {
            visit(keyPath: key, value: value)
        }

        return Array(Set(matches)).sorted()
    }

    private nonisolated static func image(_ image: UIImage, applying orientation: CGImagePropertyOrientation) -> UIImage {
        guard image.imageOrientation == .up, orientation != .up, let cgImage = image.cgImage else {
            return image
        }
        return UIImage(cgImage: cgImage, scale: image.scale, orientation: UIImage.Orientation(orientation))
    }

    private nonisolated static func pixelSizeText(for image: UIImage) -> String {
        let width = Int((image.size.width * image.scale).rounded())
        let height = Int((image.size.height * image.scale).rounded())
        return "\(width)x\(height)"
    }

    private nonisolated static func diagnosticStringValue(_ value: Any) -> String {
        switch value {
        case let number as NSNumber:
            return number.stringValue
        case let string as String:
            return string
        case let array as [Any]:
            return array.map(diagnosticStringValue).joined(separator: ", ")
        default:
            return "\(value)"
        }
    }
}

enum PhotoLibraryStoreError: LocalizedError {
    case assetNotFound
    case originalPhotoResourceNotFound

    var errorDescription: String? {
        switch self {
        case .assetNotFound:
            return "The selected PhotoKit asset could not be found."
        case .originalPhotoResourceNotFound:
            return "The selected PhotoKit asset does not have an original photo resource."
        }
    }
}

private extension UIImage.Orientation {
    init(_ cgImageOrientation: CGImagePropertyOrientation) {
        switch cgImageOrientation {
        case .up:
            self = .up
        case .upMirrored:
            self = .upMirrored
        case .down:
            self = .down
        case .downMirrored:
            self = .downMirrored
        case .left:
            self = .left
        case .leftMirrored:
            self = .leftMirrored
        case .right:
            self = .right
        case .rightMirrored:
            self = .rightMirrored
        }
    }
}

private final class ImageRequestToken: @unchecked Sendable {
    var requestID = PHInvalidImageRequestID
}

private extension PhotoLibraryStore.AuthorizationState {
    var diagnosticName: String {
        switch self {
        case .unknown: "unknown"
        case .requesting: "requesting"
        case .authorized: "authorized"
        case .limited: "limited"
        case .denied: "denied"
        case .restricted: "restricted"
        }
    }
}

private extension UIDisplayGamut {
    var diagnosticName: String {
        switch self {
        case .SRGB: "sRGB"
        case .P3: "P3"
        case .unspecified: "unspecified"
        @unknown default: "unknown"
        }
    }
}

extension PhotoLibraryStore: PHPhotoLibraryChangeObserver {
    nonisolated func photoLibraryDidChange(_ changeInstance: PHChange) {
        Task { @MainActor in
            await reloadAssets()
        }
    }
}

import XCTest
@testable import Picscry

final class PicscryTests: XCTestCase {
    func testEmptyMetadataHasNoSections() {
        XCTAssertTrue(PhotoMetadata.empty.sections.isEmpty)
    }

    func testMetadataItemsHaveUniqueIdentityWhenValuesMatch() {
        let first = PhotoMetadataItem(label: "Filename", value: "IMG_0001.HEIC")
        let second = PhotoMetadataItem(label: "Filename", value: "IMG_0001.HEIC")

        XCTAssertNotEqual(first.id, second.id)
        XCTAssertEqual(first.label, second.label)
        XCTAssertEqual(first.value, second.value)
    }

    func testOriginalPhotoResourceSelectionPrefersFullSizePhoto() {
        let resources = [
            PhotoResourceSummary(
                originalFilename: "IMG_0001.HEIC",
                uniformTypeIdentifier: "public.heic",
                resourceType: .photo
            ),
            PhotoResourceSummary(
                originalFilename: "IMG_0001.DNG",
                uniformTypeIdentifier: "com.adobe.raw-image",
                resourceType: .alternatePhoto
            ),
            PhotoResourceSummary(
                originalFilename: "IMG_0001_FULL.HEIC",
                uniformTypeIdentifier: "public.heic",
                resourceType: .fullSizePhoto
            )
        ]

        let selected = PhotoResourceSummary.preferredOriginalPhotoResource(in: resources)

        XCTAssertEqual(selected?.resourceType, .fullSizePhoto)
        XCTAssertEqual(selected?.originalFilename, "IMG_0001_FULL.HEIC")
    }

    func testOriginalPhotoResourceSelectionKeepsAlternatePhotoFallback() {
        let resources = [
            PhotoResourceSummary(
                originalFilename: "IMG_0002.DNG",
                uniformTypeIdentifier: "com.adobe.raw-image",
                resourceType: .alternatePhoto
            )
        ]

        let selected = PhotoResourceSummary.preferredOriginalPhotoResource(in: resources)

        XCTAssertEqual(selected?.resourceType, .alternatePhoto)
        XCTAssertEqual(selected?.uniformTypeIdentifier, "com.adobe.raw-image")
    }

    func testOriginalPhotoResourceSelectionIgnoresNonPhotoResources() {
        let resources = [
            PhotoResourceSummary(
                originalFilename: "IMG_0003.MOV",
                uniformTypeIdentifier: "com.apple.quicktime-movie",
                resourceType: .pairedVideo
            ),
            PhotoResourceSummary(
                originalFilename: "IMG_0003.AAE",
                uniformTypeIdentifier: "com.apple.photos.adjustment-data",
                resourceType: .adjustmentData
            )
        ]

        XCTAssertNil(PhotoResourceSummary.preferredOriginalPhotoResource(in: resources))
    }

    func testLoadedImageQualityTransitionsFromPreviewToFullResolutionRendered() {
        var quality: LoadedImageQuality?

        quality = PhotoDisplayQualityState.nextQuality(current: quality, incoming: .preview)
        XCTAssertEqual(quality, .preview)

        quality = PhotoDisplayQualityState.nextQuality(current: quality, incoming: .fullResolutionRendered)
        XCTAssertEqual(quality, .fullResolutionRendered)
    }

    func testLoadedImageQualityDoesNotDowngradeAfterFullResolutionRendered() {
        let quality = PhotoDisplayQualityState.nextQuality(
            current: .fullResolutionRendered,
            incoming: .preview
        )

        XCTAssertEqual(quality, .fullResolutionRendered)
    }
}

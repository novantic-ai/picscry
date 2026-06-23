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

    func testPhotoDetailPreloadRangeCentersAroundSelectedAsset() {
        XCTAssertEqual(PhotoDetailPaging.preloadRange(centeredOn: 50, assetCount: 100, radius: 12), 38..<63)
    }

    func testPhotoDetailPreloadRangeClampsAtStart() {
        XCTAssertEqual(PhotoDetailPaging.preloadRange(centeredOn: 2, assetCount: 100, radius: 12), 0..<15)
    }

    func testPhotoDetailPreloadRangeClampsAtEnd() {
        XCTAssertEqual(PhotoDetailPaging.preloadRange(centeredOn: 98, assetCount: 100, radius: 12), 86..<100)
    }

    func testPhotoDetailPreloadRangeHandlesEmptyLibrary() {
        XCTAssertEqual(PhotoDetailPaging.preloadRange(centeredOn: nil, assetCount: 0, radius: 12), 0..<0)
    }

    func testFaceEmbeddingDataRoundTrip() {
        let values: [Float] = [0.1, 0.2, 0.3, 0.4]
        XCTAssertEqual(Data(float32Array: values).float32Array(), values)
    }

    func testCosineSimilarityForNormalizedVectors() {
        XCTAssertEqual([Float(1), 0].cosineSimilarity(to: [1, 0]), 1, accuracy: 0.0001)
        XCTAssertEqual([Float(1), 0].cosineSimilarity(to: [0, 1]), 0, accuracy: 0.0001)
    }

    func testCentroidUpdateProducesNormalizedVector() {
        let engine = FaceClusteringEngine()
        let centroid = engine.updatedCentroid(existing: [1, 0], existingCount: 1, adding: [0, 1])
        let magnitude = sqrt(centroid.reduce(Float(0)) { $0 + ($1 * $1) })

        XCTAssertEqual(magnitude, 1, accuracy: 0.0001)
        XCTAssertEqual(centroid[0], centroid[1], accuracy: 0.0001)
    }

    func testClusterAssignmentAboveAutoThresholdUsesExistingPerson() {
        let personID = UUID()
        var configuration = FaceRecognitionConfiguration()
        configuration.autoMatchThreshold = 0.72
        let engine = FaceClusteringEngine(configuration: configuration)
        let cluster = FaceCluster(personID: personID, centroid: [1, 0], faceCount: 1, name: nil)
        let assignment = engine.assignment(for: [0.8, 0.6].l2Normalized(), clusters: [cluster])

        if case let .existingPerson(assignedID, _) = assignment.kind {
            XCTAssertEqual(assignedID, personID)
        } else {
            XCTFail("Expected existing person assignment.")
        }
    }

    func testClusterAssignmentBelowThresholdCreatesNewPerson() {
        let engine = FaceClusteringEngine()
        let cluster = FaceCluster(personID: UUID(), centroid: [1, 0], faceCount: 1, name: nil)
        let assignment = engine.assignment(for: [0, 1], clusters: [cluster])

        XCTAssertEqual(assignment.kind, .newPerson)
    }

    func testMergedCentroidUsesFaceCounts() {
        let engine = FaceClusteringEngine()
        let merged = engine.mergedCentroid(clusters: [
            (centroid: [1, 0], count: 3),
            (centroid: [0, 1], count: 1)
        ])

        XCTAssertGreaterThan(merged[0], merged[1])
    }

    func testPeopleSortingNamedFirstThenUnknownByPhotoCount() {
        let alpha = PersonSummary(id: UUID(), displayName: "Ana", isUnknown: false, photoCount: 1, faceCount: 1, representativeFaceImageData: nil)
        let zed = PersonSummary(id: UUID(), displayName: "Zed", isUnknown: false, photoCount: 1, faceCount: 1, representativeFaceImageData: nil)
        let unknownSmall = PersonSummary(id: UUID(), displayName: "Unknown", isUnknown: true, photoCount: 2, faceCount: 4, representativeFaceImageData: nil)
        let unknownLarge = PersonSummary(id: UUID(), displayName: "Unknown", isUnknown: true, photoCount: 8, faceCount: 8, representativeFaceImageData: nil)

        XCTAssertEqual(PeopleOrdering.sorted([unknownSmall, zed, unknownLarge, alpha]).map(\.id), [
            alpha.id,
            zed.id,
            unknownLarge.id,
            unknownSmall.id
        ])
    }
}

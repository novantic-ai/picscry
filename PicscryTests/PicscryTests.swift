import XCTest
import UIKit
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

    func testAssignmentCreatesNewPersonWhenNoClusters() {
        let engine = FaceClusteringEngine()
        let assignment = engine.assignment(for: [1, 0, 0], clusters: [])

        XCTAssertEqual(assignment.kind, .newPerson)
    }

    func testAssignmentExcludesPersonIDs() {
        let engine = FaceClusteringEngine()
        let excludedID = UUID()
        let cluster = FaceCluster(personID: excludedID, centroid: [1, 0, 0], faceCount: 4, name: nil)
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [cluster],
            excluding: [excludedID]
        )

        XCTAssertEqual(assignment.kind, .newPerson)
    }

    func testAssignmentUsesNextCandidateWhenBestClusterExcluded() {
        let engine = FaceClusteringEngine()
        let excludedID = UUID()
        let candidateID = UUID()
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [
                FaceCluster(personID: excludedID, centroid: [1, 0, 0], faceCount: 4, name: nil),
                FaceCluster(personID: candidateID, centroid: [0.99, 0.1, 0].l2Normalized(), faceCount: 4, name: nil)
            ],
            excluding: [excludedID]
        )

        if case let .existingPerson(assignedID, _) = assignment.kind {
            XCTAssertEqual(assignedID, candidateID)
        } else {
            XCTFail("Expected next candidate assignment.")
        }
    }

    func testSingleSampleClusterRequiresHigherThreshold() {
        let personID = UUID()
        var configuration = FaceRecognitionConfiguration()
        configuration.autoMatchThreshold = 0.84
        configuration.singleSampleAutoMatchThreshold = 0.88
        configuration.possibleMatchThreshold = 0.76
        let engine = FaceClusteringEngine(configuration: configuration)
        let cluster = FaceCluster(personID: personID, centroid: [1, 0, 0], faceCount: 1, name: nil)
        let assignment = engine.assignment(for: [0.86, 0.510294, 0], clusters: [cluster])

        if case let .ambiguous(bestPersonID, _) = assignment.kind {
            XCTAssertEqual(bestPersonID, personID)
        } else {
            XCTFail("Expected ambiguous assignment below the single-sample auto threshold.")
        }
    }

    func testMultiSampleClusterUsesAutoMatchThreshold() {
        let personID = UUID()
        var configuration = FaceRecognitionConfiguration()
        configuration.autoMatchThreshold = 0.84
        configuration.singleSampleAutoMatchThreshold = 0.88
        let engine = FaceClusteringEngine(configuration: configuration)
        let cluster = FaceCluster(personID: personID, centroid: [1, 0, 0], faceCount: 2, name: nil)
        let assignment = engine.assignment(for: [0.86, 0.510294, 0], clusters: [cluster])

        if case let .existingPerson(assignedID, _) = assignment.kind {
            XCTAssertEqual(assignedID, personID)
        } else {
            XCTFail("Expected existing person assignment.")
        }
    }

    func testClusterAssignmentBelowThresholdCreatesNewPerson() {
        let engine = FaceClusteringEngine()
        let cluster = FaceCluster(personID: UUID(), centroid: [1, 0, 0], faceCount: 2, name: nil)
        let assignment = engine.assignment(for: [0, 1, 0], clusters: [cluster])

        XCTAssertEqual(assignment.kind, .newPerson)
    }

    func testMarginPreventsCollapsedEmbeddingAssignment() {
        var configuration = FaceRecognitionConfiguration()
        configuration.autoMatchThreshold = 0.90
        configuration.minimumBestSecondBestMargin = 0.025
        let engine = FaceClusteringEngine(configuration: configuration)
        let bestID = UUID()
        let secondID = UUID()
        let bestCentroid = [Float(0.997), sqrt(1 - Float(0.997 * 0.997)), 0]
        let secondCentroid = [Float(0.996), 0, sqrt(1 - Float(0.996 * 0.996))]
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [
                FaceCluster(personID: bestID, centroid: bestCentroid, faceCount: 3, name: nil),
                FaceCluster(personID: secondID, centroid: secondCentroid, faceCount: 3, name: nil)
            ]
        )

        if case let .deferredProvisional(bestPersonID, similarity) = assignment.kind {
            XCTAssertEqual(bestPersonID, bestID)
            XCTAssertEqual(similarity ?? 0, 0.997, accuracy: 0.0001)
        } else {
            XCTFail("Expected deferred provisional assignment for collapsed best/second-best margin.")
        }
    }

    func testHealthyMarginCanAssignExistingPerson() {
        var configuration = FaceRecognitionConfiguration()
        configuration.autoMatchThreshold = 0.90
        configuration.minimumBestSecondBestMargin = 0.025
        let engine = FaceClusteringEngine(configuration: configuration)
        let bestID = UUID()
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [
                FaceCluster(personID: bestID, centroid: [Float(0.94), sqrt(1 - Float(0.94 * 0.94)), 0], faceCount: 3, name: nil),
                FaceCluster(personID: UUID(), centroid: [Float(0.75), 0, sqrt(1 - Float(0.75 * 0.75))], faceCount: 3, name: nil)
            ]
        )

        if case let .existingPerson(assignedID, _) = assignment.kind {
            XCTAssertEqual(assignedID, bestID)
        } else {
            XCTFail("Expected existing person assignment when best match has a healthy margin.")
        }
    }

    func testUpdatedCentroidNormalizesResult() {
        let engine = FaceClusteringEngine()
        let centroid = engine.updatedCentroid(existing: [1, 0], existingCount: 1, adding: [0, 1])
        let magnitude = sqrt(centroid.reduce(Float(0)) { $0 + ($1 * $1) })

        XCTAssertEqual(magnitude, 1, accuracy: 0.0001)
        XCTAssertEqual(centroid[0], centroid[1], accuracy: 0.0001)
    }

    func testMergedCentroidUsesFaceCounts() {
        let engine = FaceClusteringEngine()
        let merged = engine.mergedCentroid(clusters: [
            (centroid: [1, 0], count: 3),
            (centroid: [0, 1], count: 1)
        ])

        XCTAssertGreaterThan(merged[0], merged[1])
    }

    func testConstrainedGraphClusteringKeepsSamePhotoFacesSeparate() {
        let engine = FaceClusteringEngine()
        let firstSamePhoto = UUID()
        let secondSamePhoto = UUID()
        let laterMatch = UUID()
        let nodes = [
            FaceClusteringObservationNode(id: firstSamePhoto, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
            FaceClusteringObservationNode(id: secondSamePhoto, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
            FaceClusteringObservationNode(id: laterMatch, assetLocalIdentifier: "asset-b", embedding: [1, 0, 0])
        ]

        let components = engine.constrainedComponents(for: nodes, similarityThreshold: 0.92)

        XCTAssertEqual(components.count, 2)
        XCTAssertFalse(components.contains { component in
            component.nodeIDs.contains(firstSamePhoto) && component.nodeIDs.contains(secondSamePhoto)
        })
        XCTAssertTrue(components.contains { $0.nodeIDs.count == 2 && $0.nodeIDs.contains(laterMatch) })
    }

    func testIncrementalConstrainedClusteringKeepsSamePhotoFacesSeparate() {
        let engine = FaceClusteringEngine()
        let firstSamePhoto = UUID()
        let secondSamePhoto = UUID()
        let laterMatch = UUID()
        let components = engine.constrainedIncrementalComponents(
            for: [
                FaceClusteringObservationNode(id: firstSamePhoto, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
                FaceClusteringObservationNode(id: secondSamePhoto, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
                FaceClusteringObservationNode(id: laterMatch, assetLocalIdentifier: "asset-b", embedding: [1, 0, 0])
            ],
            similarityThreshold: 0.92,
            singleSampleThreshold: 0.94
        )

        XCTAssertEqual(components.count, 2)
        XCTAssertFalse(components.contains { component in
            component.nodeIDs.contains(firstSamePhoto) && component.nodeIDs.contains(secondSamePhoto)
        })
    }

    func testEmbeddingHealthDetectsCollapse() {
        let monitor = FaceEmbeddingHealthMonitor()
        let sample = Array(repeating: Float(1) / sqrt(Float(128)), count: 128)
        for _ in 0..<32 {
            monitor.add(sample)
        }

        XCTAssertEqual(monitor.report().status, .suspiciousCollapsed)
    }

    func testEmbeddingHealthDoesNotFlagDiverseSamplesAsCollapsed() {
        let monitor = FaceEmbeddingHealthMonitor()
        for index in 0..<32 {
            var sample = Array(repeating: Float(0), count: 128)
            sample[index] = 1
            monitor.add(sample)
        }

        XCTAssertNotEqual(monitor.report().status, .suspiciousCollapsed)
    }

    func testPixelChannelOrderWritesRedToBGRRedChannel() throws {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 112, height: 112))
        let image = renderer.image { context in
            UIColor.red.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 112, height: 112))
        }
        let cgImage = try XCTUnwrap(image.cgImage)

        let channels = try FaceEmbeddingService.debugBGRChannelsForFirstPixel(from: cgImage)

        XCTAssertEqual(channels.blue, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.green, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.red, 255, accuracy: 0.0001)
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

    func testPeopleOrderingStillNamedFirstUnknownsAfter() {
        let named = PersonSummary(id: UUID(), displayName: "Mina", isUnknown: false, photoCount: 1, faceCount: 1, representativeFaceImageData: nil)
        let unknown = PersonSummary(id: UUID(), displayName: "Unknown", isUnknown: true, photoCount: 9, faceCount: 12, representativeFaceImageData: nil)

        XCTAssertEqual(PeopleOrdering.sorted([unknown, named]).map(\.id), [
            named.id,
            unknown.id
        ])
    }

    func testSquareAvatarRectReturnsSquareRectInsideImageBounds() {
        let imageSize = CGSize(width: 400, height: 300)
        let rect = FaceCropService.squareAvatarRect(
            around: CGRect(x: 100, y: 80, width: 50, height: 80),
            paddingRatio: 0.85,
            imageSize: imageSize
        )

        XCTAssertEqual(rect.width, rect.height, accuracy: 0.0001)
        XCTAssertGreaterThan(rect.width, 80)
        XCTAssertGreaterThanOrEqual(rect.minX, 0)
        XCTAssertGreaterThanOrEqual(rect.minY, 0)
        XCTAssertLessThanOrEqual(rect.maxX, imageSize.width)
        XCTAssertLessThanOrEqual(rect.maxY, imageSize.height)
    }

    func testSquareAvatarRectNearEdgeDoesNotProduceZeroSize() {
        let rect = FaceCropService.squareAvatarRect(
            around: CGRect(x: 0, y: 0, width: 48, height: 48),
            paddingRatio: 0.85,
            imageSize: CGSize(width: 120, height: 100)
        )

        XCTAssertEqual(rect.width, rect.height, accuracy: 0.0001)
        XCTAssertGreaterThan(rect.width, 0)
        XCTAssertGreaterThanOrEqual(rect.minX, 0)
        XCTAssertGreaterThanOrEqual(rect.minY, 0)
    }
}

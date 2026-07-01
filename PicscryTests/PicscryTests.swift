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

    func testSingleSampleClusterUsesTunedThreshold() {
        let personID = UUID()
        let engine = FaceClusteringEngine()
        let cluster = FaceCluster(personID: personID, centroid: [1, 0, 0], faceCount: 1, name: nil)
        let assignment = engine.assignment(for: vector(similarityToXAxis: 0.72), clusters: [cluster])

        if case let .existingPerson(assignedID, _) = assignment.kind {
            XCTAssertEqual(assignedID, personID)
        } else {
            XCTFail("Expected same person assignment at the tuned single-sample threshold.")
        }
    }

    func testPermissiveVerificationSimilarityDoesNotAutoAssign() {
        let personID = UUID()
        let engine = FaceClusteringEngine()
        let cluster = FaceCluster(personID: personID, centroid: [1, 0, 0], faceCount: 2, name: nil)
        let assignment = engine.assignment(for: vector(similarityToXAxis: 0.52), clusters: [cluster])

        if case let .ambiguous(bestPersonID, similarity) = assignment.kind {
            XCTAssertEqual(bestPersonID, personID)
            XCTAssertEqual(similarity, 0.52, accuracy: 0.0001)
        } else {
            XCTFail("Expected 0.52 similarity to stay ambiguous, not auto-assign.")
        }
    }

    func testStableClusterAutoAssignsAtPrecisionThresholdWithSafeMargin() {
        let bestID = UUID()
        let engine = FaceClusteringEngine()
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [
                FaceCluster(personID: bestID, centroid: vector(similarityToXAxis: 0.66), faceCount: 3, name: nil),
                FaceCluster(personID: UUID(), centroid: vector(similarityToXAxis: 0.45), faceCount: 3, name: nil)
            ]
        )

        if case let .existingPerson(assignedID, similarity) = assignment.kind {
            XCTAssertEqual(assignedID, bestID)
            XCTAssertEqual(similarity, 0.66, accuracy: 0.0001)
        } else {
            XCTFail("Expected stable multi-face cluster to auto-assign at 0.66 with a safe margin.")
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

    func testModerateSingleSampleSimilarityClustersSamePerson() {
        let personID = UUID()
        let engine = FaceClusteringEngine()
        let assignment = engine.assignment(
            for: vector(similarityToXAxis: 0.72),
            clusters: [FaceCluster(personID: personID, centroid: [1, 0, 0], faceCount: 1, name: nil)]
        )

        if case let .existingPerson(assignedID, similarity) = assignment.kind {
            XCTAssertEqual(assignedID, personID)
            XCTAssertEqual(similarity, 0.72, accuracy: 0.0001)
        } else {
            XCTFail("Expected existing person for moderate same-person similarity.")
        }
    }

    func testWeakSecondBestDoesNotBlockAssignment() {
        let bestID = UUID()
        let engine = FaceClusteringEngine()
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [
                FaceCluster(personID: bestID, centroid: vector(similarityToXAxis: 0.82), faceCount: 2, name: nil),
                FaceCluster(personID: UUID(), centroid: vector(similarityToXAxis: 0.35), faceCount: 2, name: nil)
            ]
        )

        if case let .existingPerson(assignedID, _) = assignment.kind {
            XCTAssertEqual(assignedID, bestID)
        } else {
            XCTFail("Expected weak second-best candidate to be ignored.")
        }
    }

    func testRealAmbiguityDefersProvisionalAssignment() {
        let bestID = UUID()
        let engine = FaceClusteringEngine()
        let assignment = engine.assignment(
            for: [1, 0, 0],
            clusters: [
                FaceCluster(personID: bestID, centroid: vector(similarityToXAxis: 0.82), faceCount: 2, name: nil),
                FaceCluster(personID: UUID(), centroid: vector(similarityToXAxis: 0.815), faceCount: 2, name: nil)
            ]
        )

        if case let .deferredProvisional(bestPersonID, similarity) = assignment.kind {
            XCTAssertEqual(bestPersonID, bestID)
            XCTAssertEqual(similarity ?? 0, 0.82, accuracy: 0.0001)
        } else {
            XCTFail("Expected close best and second-best candidates to defer.")
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

    func testWeightedCentroidFavorsHigherQualityEmbedding() {
        let engine = FaceClusteringEngine()
        let merged = engine.weightedCentroid(embeddings: [
            (embedding: [1, 0], weight: 3),
            (embedding: [0, 1], weight: 1)
        ])

        XCTAssertGreaterThan(merged[0], merged[1])
        XCTAssertEqual(sqrt(merged.reduce(Float(0)) { $0 + ($1 * $1) }), 1, accuracy: 0.0001)
    }

    func testConstrainedClusteringKeepsSamePhotoFacesSeparate() {
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

        XCTAssertEqual(components.count, 3)
        XCTAssertFalse(components.contains { component in
            component.nodeIDs.contains(firstSamePhoto) && component.nodeIDs.contains(secondSamePhoto)
        })
        XCTAssertTrue(components.contains { $0.nodeIDs.contains(laterMatch) })
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

        XCTAssertFalse(components.contains { component in
            component.nodeIDs.contains(firstSamePhoto) && component.nodeIDs.contains(secondSamePhoto)
        })
        XCTAssertTrue(components.contains { $0.nodeIDs.contains(firstSamePhoto) })
        XCTAssertTrue(components.contains { $0.nodeIDs.contains(secondSamePhoto) })
        XCTAssertTrue(components.contains { $0.nodeIDs.contains(laterMatch) })
    }

    func testConstrainedClusteringDoesNotMergeThroughWeakTransitiveChain() {
        let engine = FaceClusteringEngine()
        let first = UUID()
        let second = UUID()
        let third = UUID()
        let angle = acos(Float(0.72))
        let nodes = [
            FaceClusteringObservationNode(id: first, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
            FaceClusteringObservationNode(id: second, assetLocalIdentifier: "asset-b", embedding: [cos(angle), sin(angle), 0]),
            FaceClusteringObservationNode(id: third, assetLocalIdentifier: "asset-c", embedding: [cos(angle * 2), sin(angle * 2), 0])
        ]

        let components = engine.constrainedComponents(for: nodes, similarityThreshold: 0.66)

        XCTAssertEqual(components.count, 2)
        XCTAssertFalse(components.contains { component in
            component.nodeIDs.contains(first) && component.nodeIDs.contains(third)
        })

        let mergeResult = engine.mergedConstrainedComponentResult(
            from: components,
            nodes: nodes,
            mergeThreshold: 0.68
        )

        XCTAssertEqual(mergeResult.components.count, 2)
    }

    func testMergePassKeepsSamePhotoFacesSeparate() {
        let engine = FaceClusteringEngine()
        let firstSamePhoto = UUID()
        let secondSamePhoto = UUID()
        let laterMatch = UUID()
        let nodes = [
            FaceClusteringObservationNode(id: firstSamePhoto, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
            FaceClusteringObservationNode(id: secondSamePhoto, assetLocalIdentifier: "asset-a", embedding: [1, 0, 0]),
            FaceClusteringObservationNode(id: laterMatch, assetLocalIdentifier: "asset-b", embedding: [1, 0, 0])
        ]

        let initial = engine.constrainedComponents(for: nodes, similarityThreshold: 0.80)
        let merged = engine.mergedConstrainedComponents(from: initial, nodes: nodes, mergeThreshold: 0.79)

        XCTAssertEqual(merged.count, 2)
        XCTAssertFalse(merged.contains { component in
            component.nodeIDs.contains(firstSamePhoto) && component.nodeIDs.contains(secondSamePhoto)
        })
    }

    func testUnknownClustersMergeAtSFaceCalibratedThreshold() {
        let engine = FaceClusteringEngine()
        let first = FaceCluster(
            personID: UUID(),
            centroid: [1, 0, 0],
            faceCount: 1,
            name: nil,
            assetLocalIdentifiers: ["asset-a"]
        )
        let second = FaceCluster(
            personID: UUID(),
            centroid: vector(similarityToXAxis: 0.68),
            faceCount: 1,
            name: nil,
            assetLocalIdentifiers: ["asset-b"]
        )
        let score = FaceClusterPairScore(
            centroidSimilarity: 0.68,
            bestPairSimilarity: 0.72,
            topKAverageSimilarity: 0.68,
            bestSecondBestMargin: nil
        )

        XCTAssertTrue(engine.mergeDecision(left: first, right: second, score: score).accepted)
    }

    func testPermissiveSFaceVerificationRangeDoesNotAutoMerge() {
        let engine = FaceClusteringEngine()
        let first = FaceCluster(
            personID: UUID(),
            centroid: [1, 0, 0],
            faceCount: 1,
            name: nil,
            assetLocalIdentifiers: ["asset-a"]
        )
        let second = FaceCluster(
            personID: UUID(),
            centroid: vector(similarityToXAxis: 0.64),
            faceCount: 1,
            name: nil,
            assetLocalIdentifiers: ["asset-b"]
        )
        let score = FaceClusterPairScore(
            centroidSimilarity: 0.64,
            bestPairSimilarity: 0.64,
            topKAverageSimilarity: 0.64,
            bestSecondBestMargin: nil
        )

        let decision = engine.mergeDecision(left: first, right: second, score: score)

        XCTAssertFalse(decision.accepted)
        XCTAssertEqual(decision.rejectReason, .belowThreshold)
    }

    func testUnknownMergeRequiresCentroidAndPairEvidence() {
        let engine = FaceClusteringEngine()
        let first = FaceCluster(
            personID: UUID(),
            centroid: [1, 0, 0],
            faceCount: 2,
            name: nil,
            assetLocalIdentifiers: ["asset-a", "asset-b"]
        )
        let second = FaceCluster(
            personID: UUID(),
            centroid: vector(similarityToXAxis: 0.70),
            faceCount: 2,
            name: nil,
            assetLocalIdentifiers: ["asset-c", "asset-d"]
        )
        let score = FaceClusterPairScore(
            centroidSimilarity: 0.70,
            bestPairSimilarity: 0.71,
            topKAverageSimilarity: 0.69,
            bestSecondBestMargin: nil
        )

        let decision = engine.mergeDecision(left: first, right: second, score: score)

        XCTAssertFalse(decision.accepted)
        XCTAssertEqual(decision.rejectReason, .belowThreshold)
    }

    func testSingleSampleFragmentationRepairMakesNextAssignmentStable() {
        var configuration = FaceRecognitionConfiguration()
        configuration.singleSampleAutoMatchThreshold = 0.74
        let engine = FaceClusteringEngine(configuration: configuration)
        let firstID = UUID()
        let secondID = UUID()
        let firstEmbedding: [Float] = [1, 0, 0]
        let secondEmbedding = vector(similarityToXAxis: 0.72)
        let firstCluster = FaceCluster(personID: firstID, centroid: firstEmbedding, faceCount: 1, name: nil)

        let firstAssignment = engine.assignment(for: secondEmbedding, clusters: [firstCluster])
        if case .existingPerson = firstAssignment.kind {
            XCTFail("Expected online assignment to avoid auto-joining below the single-sample threshold.")
        }

        let nodes = [
            FaceClusteringObservationNode(id: firstID, assetLocalIdentifier: "asset-a", embedding: firstEmbedding),
            FaceClusteringObservationNode(id: secondID, assetLocalIdentifier: "asset-b", embedding: secondEmbedding)
        ]
        let mergeResult = engine.mergedConstrainedComponentResult(
            from: nodes.map { FaceClusteringComponent(nodeIDs: [$0.id]) },
            nodes: nodes,
            mergeThreshold: 0.68
        )

        XCTAssertEqual(mergeResult.components.count, 1)
        XCTAssertEqual(mergeResult.stats.acceptedMerges, 1)

        let repairedCentroid = engine.mergedCentroid(clusters: [
            (centroid: firstEmbedding, count: 1),
            (centroid: secondEmbedding, count: 1)
        ])
        let repairedCluster = FaceCluster(personID: firstID, centroid: repairedCentroid, faceCount: 2, name: nil)
        let nextAssignment = engine.assignment(for: secondEmbedding, clusters: [repairedCluster])

        if case let .existingPerson(assignedID, _) = nextAssignment.kind {
            XCTAssertEqual(assignedID, firstID)
        } else {
            XCTFail("Expected the repaired multi-face cluster to receive the next same-person face.")
        }
    }

    func testNamedPersonCanAbsorbUnknownAboveThreshold() {
        let engine = FaceClusteringEngine()
        let named = FaceCluster(
            personID: UUID(),
            centroid: [1, 0, 0],
            faceCount: 4,
            name: "Sujana",
            assetLocalIdentifiers: ["asset-a"]
        )
        let unknown = FaceCluster(
            personID: UUID(),
            centroid: vector(similarityToXAxis: 0.74),
            faceCount: 2,
            name: nil,
            assetLocalIdentifiers: ["asset-b", "asset-c"]
        )
        let score = FaceClusterPairScore(
            centroidSimilarity: 0.74,
            bestPairSimilarity: 0.74,
            topKAverageSimilarity: 0.74,
            bestSecondBestMargin: nil
        )

        XCTAssertTrue(engine.mergeDecision(left: named, right: unknown, score: score).accepted)
    }

    func testUnsafeMergeMarginIsRejected() {
        let engine = FaceClusteringEngine()
        let first = FaceCluster(personID: UUID(), centroid: [1, 0, 0], faceCount: 2, name: nil)
        let second = FaceCluster(personID: UUID(), centroid: vector(similarityToXAxis: 0.80), faceCount: 2, name: nil)
        let score = FaceClusterPairScore(
            centroidSimilarity: 0.80,
            bestPairSimilarity: 0.80,
            topKAverageSimilarity: 0.80,
            bestSecondBestMargin: 0.01
        )

        let decision = engine.mergeDecision(left: first, right: second, score: score)

        XCTAssertFalse(decision.accepted)
        XCTAssertEqual(decision.rejectReason, .unsafeMargin)
    }

    func testManualCorrectionPreventsAutomaticMerge() {
        let engine = FaceClusteringEngine()
        let corrected = FaceCluster(
            personID: UUID(),
            centroid: [1, 0, 0],
            faceCount: 2,
            name: nil,
            manuallyCorrectedFaceCount: 1
        )
        let unknown = FaceCluster(personID: UUID(), centroid: [1, 0, 0], faceCount: 2, name: nil)
        let score = FaceClusterPairScore(
            centroidSimilarity: 1,
            bestPairSimilarity: 1,
            topKAverageSimilarity: 1,
            bestSecondBestMargin: nil
        )

        let decision = engine.mergeDecision(left: corrected, right: unknown, score: score)

        XCTAssertFalse(decision.accepted)
        XCTAssertEqual(decision.rejectReason, .manualCorrection)
    }

    func testDifferentNamedPeopleAreNotAutomaticMergeCandidates() {
        let engine = FaceClusteringEngine()
        let left = FaceCluster(personID: UUID(), centroid: [1, 0, 0], faceCount: 4, name: "Ana")
        let right = FaceCluster(personID: UUID(), centroid: [1, 0, 0], faceCount: 4, name: "Bo")

        XCTAssertFalse(engine.canAutomaticallyMerge(left: left, right: right, similarity: 0.99))
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

    func testFaceEmbeddingInputUsesRGBChannelOrderForRedImage() throws {
        let cgImage = try XCTUnwrap(makeSolidColorImage(red: 1, green: 0, blue: 0))

        let channels = try FaceEmbeddingService.debugRGBChannelsForFirstPixel(from: cgImage)

        XCTAssertEqual(channels.red, 255, accuracy: 0.0001)
        XCTAssertEqual(channels.green, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.blue, 0, accuracy: 0.0001)
    }

    func testFaceEmbeddingInputUsesRGBChannelOrderForGreenImage() throws {
        let cgImage = try XCTUnwrap(makeSolidColorImage(red: 0, green: 1, blue: 0))

        let channels = try FaceEmbeddingService.debugRGBChannelsForFirstPixel(from: cgImage)

        XCTAssertEqual(channels.red, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.green, 255, accuracy: 0.0001)
        XCTAssertEqual(channels.blue, 0, accuracy: 0.0001)
    }

    func testFaceEmbeddingInputUsesRGBChannelOrderForBlueImage() throws {
        let cgImage = try XCTUnwrap(makeSolidColorImage(red: 0, green: 0, blue: 1))

        let channels = try FaceEmbeddingService.debugRGBChannelsForFirstPixel(from: cgImage)

        XCTAssertEqual(channels.red, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.green, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.blue, 255, accuracy: 0.0001)
    }

    func testFaceEmbeddingInputSupportsFourDimensionalModelShape() throws {
        let cgImage = try XCTUnwrap(makeSolidColorImage(red: 1, green: 0, blue: 0))

        let shape = try FaceEmbeddingService.debugInputShape(from: cgImage, shape: [1, 3, 112, 112])
        let channels = try FaceEmbeddingService.debugRGBChannelsForFirstPixel(
            from: cgImage,
            shape: [1, 3, 112, 112]
        )

        XCTAssertEqual(shape, [1, 3, 112, 112])
        XCTAssertEqual(channels.red, 255, accuracy: 0.0001)
        XCTAssertEqual(channels.green, 0, accuracy: 0.0001)
        XCTAssertEqual(channels.blue, 0, accuracy: 0.0001)
    }

    func testYuNetRowParsingConvertsOpenCVRowToVisionStyleBoundingBox() throws {
        let row: [Float] = [
            64, 96, 160, 200,
            104, 160,
            184, 160,
            144, 210,
            112, 260,
            176, 260,
            0.93
        ]

        let detection = try XCTUnwrap(YuNetFaceDetectionService.decodedDetectionForTesting(
            rawRow: row,
            originalImageSize: CGSize(width: 640, height: 480),
            inputScale: 1
        ))

        XCTAssertEqual(detection.pixelBoundingBox.origin.x, 64, accuracy: 0.0001)
        XCTAssertEqual(detection.pixelBoundingBox.origin.y, 96, accuracy: 0.0001)
        XCTAssertEqual(detection.pixelBoundingBox.width, 160, accuracy: 0.0001)
        XCTAssertEqual(detection.pixelBoundingBox.height, 200, accuracy: 0.0001)
        XCTAssertEqual(detection.normalizedBoundingBox.minX, 0.1, accuracy: 0.0001)
        XCTAssertEqual(detection.normalizedBoundingBox.minY, 1 - (296.0 / 480.0), accuracy: 0.0001)
        XCTAssertEqual(detection.landmarks.rightEye, CGPoint(x: 104, y: 160))
        XCTAssertEqual(detection.landmarks.leftEye, CGPoint(x: 184, y: 160))
        XCTAssertEqual(detection.landmarks.rightMouth, CGPoint(x: 112, y: 260))
        XCTAssertEqual(detection.landmarks.leftMouth, CGPoint(x: 176, y: 260))
        XCTAssertEqual(detection.score, 0.93, accuracy: 0.0001)
    }

    func testYuNetRowParsingScalesFromFixedModelInputToOriginalImage() throws {
        let row: [Float] = [
            64, 96, 160, 200,
            104, 160,
            184, 160,
            144, 210,
            112, 260,
            176, 260,
            0.91
        ]

        let detection = try XCTUnwrap(YuNetFaceDetectionService.decodedDetectionForTesting(
            rawRow: row,
            originalImageSize: CGSize(width: 1280, height: 960),
            inputScale: 0.5
        ))

        XCTAssertEqual(detection.pixelBoundingBox.origin.x, 128, accuracy: 0.0001)
        XCTAssertEqual(detection.pixelBoundingBox.origin.y, 192, accuracy: 0.0001)
        XCTAssertEqual(detection.pixelBoundingBox.width, 320, accuracy: 0.0001)
        XCTAssertEqual(detection.pixelBoundingBox.height, 400, accuracy: 0.0001)
        XCTAssertEqual(detection.landmarks.noseTip, CGPoint(x: 288, y: 420))
    }

    func testFaceLandmarkFivePointUsesOpenCVSFaceRowOrder() {
        let landmarks = FaceLandmarkFivePoint(
            rightEye: CGPoint(x: 1, y: 2),
            leftEye: CGPoint(x: 3, y: 4),
            noseTip: CGPoint(x: 5, y: 6),
            rightMouth: CGPoint(x: 7, y: 8),
            leftMouth: CGPoint(x: 9, y: 10)
        )

        XCTAssertEqual(landmarks.sfaceSourcePoints, [
            CGPoint(x: 1, y: 2),
            CGPoint(x: 3, y: 4),
            CGPoint(x: 5, y: 6),
            CGPoint(x: 7, y: 8),
            CGPoint(x: 9, y: 10)
        ])
    }

    func testAdaptiveWorkerPolicyUsesOneWorkerForBackgroundLowPowerOrHotDevice() {
        XCTAssertEqual(FaceIndexingWorkerPolicy.workerLimit(
            context: .backgroundTask,
            thermalState: .nominal,
            isLowPowerModeEnabled: false,
            processorCount: 8
        ), 1)
        XCTAssertEqual(FaceIndexingWorkerPolicy.workerLimit(
            context: .foregroundAutomatic,
            thermalState: .nominal,
            isLowPowerModeEnabled: true,
            processorCount: 8
        ), 1)
        XCTAssertEqual(FaceIndexingWorkerPolicy.workerLimit(
            context: .foregroundAutomatic,
            thermalState: .serious,
            isLowPowerModeEnabled: false,
            processorCount: 8
        ), 1)
    }

    func testAdaptiveWorkerPolicyKeepsAutomaticForegroundSerial() {
        XCTAssertEqual(FaceIndexingWorkerPolicy.workerLimit(
            context: .foregroundAutomatic,
            thermalState: .nominal,
            isLowPowerModeEnabled: false,
            processorCount: 8
        ), 1)
    }

    func testAdaptiveWorkerPolicyUsesTwoWorkersOnlyForCoolManualRefreshCapableDevices() {
        XCTAssertEqual(FaceIndexingWorkerPolicy.workerLimit(
            context: .manualRefresh,
            thermalState: .fair,
            isLowPowerModeEnabled: false,
            processorCount: 8
        ), 2)
        XCTAssertEqual(FaceIndexingWorkerPolicy.workerLimit(
            context: .manualRefresh,
            thermalState: .nominal,
            isLowPowerModeEnabled: false,
            processorCount: 4
        ), 2)
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

    private func makeSolidColorImage(red: CGFloat, green: CGFloat, blue: CGFloat, size: Int = 112) -> CGImage? {
        let bytesPerPixel = 4
        let bytesPerRow = size * bytesPerPixel
        var pixels = [UInt8](repeating: 255, count: size * bytesPerRow)
        for y in 0..<size {
            for x in 0..<size {
                let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                pixels[offset] = UInt8(red * 255)
                pixels[offset + 1] = UInt8(green * 255)
                pixels[offset + 2] = UInt8(blue * 255)
                pixels[offset + 3] = 255
            }
        }

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
            return nil
        }

        return context.makeImage()
    }

    private func vector(similarityToXAxis similarity: Float) -> [Float] {
        [similarity, sqrt(max(Float(0), 1 - (similarity * similarity))), 0]
    }
}

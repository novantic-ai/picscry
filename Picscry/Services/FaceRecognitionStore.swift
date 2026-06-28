import BackgroundTasks
import Foundation
import Observation
import UIKit

@MainActor
@Observable
final class FaceRecognitionStore {
    nonisolated static let backgroundTaskIdentifier = "com.novanticai.picscry.face-indexing"

    var people: [PersonSummary] = []
    var indexingState: FaceIndexingState = .idle
    var errorMessage: String?
    var currentIndexingMessage: String?
    var lastIndexingSummary: String?
    var faceRecognitionHealthMessage: String?
    var hiddenProvisionalUnknownCount = 0

    private let configuration: FaceRecognitionConfiguration
    private let detectionService: FaceDetectionService
    private let embeddingService: FaceEmbeddingService
    private let cropService: FaceCropService
    private let clusteringEngine: FaceClusteringEngine
    private let embeddingHealthMonitor: FaceEmbeddingHealthMonitor

    private var persons: [UUID: StoredPerson] = [:]
    private var facesByID: [UUID: StoredFaceObservation] = [:]
    private var faceIDsByAssetID: [String: [UUID]] = [:]
    private var indexRecords: [String: AssetIndexRecord] = [:]
    private var indexingTask: Task<Void, Never>?
    private var activeIndexingAssetFingerprint: String?
    private var activeIndexingRunID: UUID?
    private var lastLoggedEmbeddingHealthSampleCount = 0
    private var didLogSuspiciousEmbeddingHealthThisRun = false
    private var assignmentDecisionCounts = FaceAssignmentDecisionCounts()

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        self.configuration = configuration
        detectionService = FaceDetectionService()
        embeddingService = FaceEmbeddingService(configuration: configuration)
        cropService = FaceCropService()
        clusteringEngine = FaceClusteringEngine(configuration: configuration)
        embeddingHealthMonitor = FaceEmbeddingHealthMonitor(configuration: configuration)
        loadPersistedState()
    }

    func prepare(photoLibraryStore: PhotoLibraryStore) async {
        await startIndexing(photoLibraryStore: photoLibraryStore, reason: "foreground", waitForCompletion: false)
    }

    func runBackgroundIndexing(photoLibraryStore: PhotoLibraryStore) async {
        await startIndexing(photoLibraryStore: photoLibraryStore, reason: "background task", waitForCompletion: true)
    }

    func retry(photoLibraryStore: PhotoLibraryStore) async {
        await startIndexing(photoLibraryStore: photoLibraryStore, reason: "manual refresh", waitForCompletion: false, forceRestart: true)
    }

    func scheduleBackgroundIndexing(reason: String) {
        let request = BGProcessingTaskRequest(identifier: Self.backgroundTaskIdentifier)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)

        do {
            BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: Self.backgroundTaskIdentifier)
            try BGTaskScheduler.shared.submit(request)
            Diagnostics.shared.log("Scheduled face indexing background task: \(reason).")
        } catch {
            Diagnostics.shared.log("Failed to schedule face indexing background task (\(reason)): \(error.localizedDescription)")
        }
    }

    private func startIndexing(
        photoLibraryStore: PhotoLibraryStore,
        reason: String,
        waitForCompletion: Bool,
        forceRestart: Bool = false
    ) async {
        guard await embeddingService.isModelAvailable else {
            indexingState = .failed("Face recognition model could not be loaded.")
            currentIndexingMessage = nil
            Diagnostics.shared.log("Face indexing not started (\(reason)): model unavailable.")
            return
        }

        let assets = photoLibraryStore.assets.filter { !$0.isVideo }
        let currentFingerprint = indexingFingerprint(for: assets)
        if let indexingTask,
           !indexingTask.isCancelled,
           indexingState.isIndexing,
           activeIndexingAssetFingerprint == currentFingerprint,
           !forceRestart {
            Diagnostics.shared.log("Face indexing prepare ignored (\(reason)): existing run active for same fingerprint.")
            return
        }

        if indexingState.isIndexing,
           activeIndexingAssetFingerprint != currentFingerprint {
            Diagnostics.shared.log("Face indexing cancelling active run (\(reason)): library fingerprint changed.")
            indexingTask?.cancel()
        } else if forceRestart {
            Diagnostics.shared.log("Face indexing cancelling active run (\(reason)): forced restart.")
            indexingTask?.cancel()
        }

        let totalEligibleImageCount = assets.count
        let liveIDs = Set(assets.map(\.id))
        removeDeletedAssets(liveIDs: liveIDs)

        let pending = assets.filter { assetNeedsIndex($0) }
        let alreadyIndexedCount = max(0, totalEligibleImageCount - pending.count)
        guard !pending.isEmpty else {
            if configuration.backgroundMergeEnabled {
                rebuildAutomaticClusters(reason: "no pending photos after \(reason)")
            }
            refreshPeople(persist: true, allowReorder: true)
            indexingState = .idle
            currentIndexingMessage = nil
            lastIndexingSummary = "All \(totalEligibleImageCount) photos are indexed. \(facesByID.count) faces across \(people.count) people."
            Diagnostics.shared.log("Face indexing skipped (\(reason)): no pending photos out of \(assets.count) image assets. Existing observations: \(facesByID.count).")
            scheduleBackgroundIndexing(reason: "no pending photos after \(reason)")
            return
        }

        let runID = UUID()
        activeIndexingRunID = runID
        activeIndexingAssetFingerprint = currentFingerprint
        didLogSuspiciousEmbeddingHealthThisRun = false
        assignmentDecisionCounts = FaceAssignmentDecisionCounts()
        Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing run started (\(reason)): \(pending.count) pending photos out of \(assets.count) image assets. Existing index records: \(indexRecords.count), people: \(persons.count), faces: \(facesByID.count).")
        scheduleBackgroundIndexing(reason: "indexing started from \(reason)")

        let task = Task { @MainActor in
            await index(
                pending: pending,
                totalEligibleImageCount: totalEligibleImageCount,
                alreadyIndexedCount: alreadyIndexedCount,
                photoLibraryStore: photoLibraryStore,
                reason: reason,
                runID: runID
            )
        }
        indexingTask = task
        if waitForCompletion {
            await task.value
        }
    }

    func faces(for asset: PhotoAssetSummary) async -> [PhotoFaceSummary] {
        summaries(forAssetID: asset.id)
    }

    func person(with id: UUID) async -> PersonSummary? {
        summary(for: id)
    }

    func assets(forPersonID personID: UUID, from allAssets: [PhotoAssetSummary]) async -> [PhotoAssetSummary] {
        let assetIDs = Set(facesByID.values.filter { $0.personID == personID }.map(\.assetLocalIdentifier))
        return allAssets.filter { assetIDs.contains($0.id) }
    }

    func renamePerson(_ personID: UUID, to proposedName: String) async -> RenamePersonResult {
        let normalizedName = Self.normalizedName(proposedName)
        guard var person = persons[personID] else { return .renamed }

        if let normalizedName,
           let existing = persons.values.first(where: { other in
               other.id != personID && Self.normalizedName(other.name ?? "") == normalizedName
           }) {
            return .needsMergeConfirmation(existingPersonID: existing.id, existingName: existing.name ?? "Unknown")
        }

        person.name = normalizedName
        person.isAutomaticCluster = false
        person.isProvisional = false
        person.updatedAt = .now
        persons[personID] = person
        refreshPeople()
        return .renamed
    }

    func confirmRenameMerge(sourcePersonID: UUID, targetPersonID: UUID, finalName: String) async {
        await mergePeople(sourcePersonID: sourcePersonID, targetPersonID: targetPersonID)
        if var target = persons[targetPersonID] {
            target.name = Self.normalizedName(finalName)
            target.isAutomaticCluster = false
            target.isProvisional = false
            target.updatedAt = .now
            persons[targetPersonID] = target
        }
        recompute(personID: targetPersonID)
        refreshPeople()
    }

    func mergePeople(sourcePersonID: UUID, targetPersonID: UUID) async {
        guard sourcePersonID != targetPersonID, persons[sourcePersonID] != nil, persons[targetPersonID] != nil else { return }
        for faceID in facesByID.values.filter({ $0.personID == sourcePersonID }).map(\.id) {
            facesByID[faceID]?.personID = targetPersonID
        }
        persons[sourcePersonID] = nil
        recompute(personID: targetPersonID)
        refreshPeople()
    }

    func moveFaceObservation(_ faceID: UUID, toExistingPersonID personID: UUID) async {
        guard let original = facesByID[faceID], persons[personID] != nil else { return }
        let oldPersonID = original.personID
        facesByID[faceID]?.personID = personID
        facesByID[faceID]?.isManuallyCorrected = true
        recompute(personID: oldPersonID)
        recompute(personID: personID)
        deletePersonIfEmpty(oldPersonID)
        refreshPeople()
    }

    func moveFaceObservation(_ faceID: UUID, toNewPersonNamed name: String?) async {
        guard let original = facesByID[faceID] else { return }
        let oldPersonID = original.personID
        let newPersonID = UUID()
        persons[newPersonID] = StoredPerson(
            id: newPersonID,
            name: Self.normalizedName(name ?? ""),
            isAutomaticCluster: false,
            isProvisional: false
        )
        facesByID[faceID]?.personID = newPersonID
        facesByID[faceID]?.isManuallyCorrected = true
        recompute(personID: oldPersonID)
        recompute(personID: newPersonID)
        deletePersonIfEmpty(oldPersonID)
        refreshPeople()
    }

    func markFaceIncorrect(_ faceID: UUID) async {
        facesByID[faceID]?.isManuallyCorrected = true
        refreshPeople()
    }

    private func index(
        pending: [PhotoAssetSummary],
        totalEligibleImageCount: Int,
        alreadyIndexedCount: Int,
        photoLibraryStore: PhotoLibraryStore,
        reason: String,
        runID: UUID
    ) async {
        let indexingStartedAt = Date()
        var processedThisRun = 0
        var processedOverall = alreadyIndexedCount
        indexingState = .indexing(processed: processedOverall, total: totalEligibleImageCount)
        currentIndexingMessage = "Starting face indexing..."

        for (offset, asset) in pending.enumerated() {
            guard activeIndexingRunID == runID else {
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Ignoring stale face indexing run before processing \(asset.id).")
                return
            }
            guard !Task.isCancelled else {
                indexingState = .paused
                currentIndexingMessage = "Face indexing paused at \(processedOverall) of \(totalEligibleImageCount)."
                savePersistedState()
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing run cancelled (\(reason)) at \(processedOverall) of \(totalEligibleImageCount) photos in \(Self.durationText(since: indexingStartedAt)).")
                return
            }

            let assetStartedAt = Date()
            currentIndexingMessage = "Processing photo \(min(processedOverall + 1, totalEligibleImageCount)) of \(totalEligibleImageCount)"
            Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing asset \(offset + 1)/\(pending.count) started: \(asset.id), pixels \(asset.pixelWidth)x\(asset.pixelHeight), modified \(asset.modificationDate?.description ?? "unknown").")

            do {
                let observations = try await process(asset: asset, photoLibraryStore: photoLibraryStore)
                guard activeIndexingRunID == runID else {
                    Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Ignoring stale face indexing run after processing \(asset.id).")
                    return
                }
                try Task.checkCancellation()
                saveAndCluster(observations, asset: asset)
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing asset \(offset + 1)/\(pending.count) finished in \(Self.durationText(since: assetStartedAt)): \(asset.id), saved \(observations.count) face observations.")
            } catch is CancellationError {
                indexingState = .paused
                currentIndexingMessage = "Face indexing paused at \(processedOverall) of \(totalEligibleImageCount)."
                savePersistedState()
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing cancelled while processing \(asset.id); not recording index record.")
                return
            } catch {
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing failed for \(asset.id): \(error.localizedDescription)")
            }

            processedThisRun += 1
            processedOverall = min(totalEligibleImageCount, alreadyIndexedCount + processedThisRun)
            indexingState = .indexing(processed: processedOverall, total: totalEligibleImageCount)
            if processedThisRun == 1 || processedThisRun.isMultiple(of: 10) {
                currentIndexingMessage = "Indexed \(processedOverall) of \(totalEligibleImageCount) photos"
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing progress (\(reason)): \(processedOverall)/\(totalEligibleImageCount) photos in \(Self.durationText(since: indexingStartedAt)).")
            }

            if processedThisRun.isMultiple(of: configuration.peopleRefreshBatchSize) {
                refreshPeople(persist: false, allowReorder: false)
            }

            if processedThisRun.isMultiple(of: configuration.indexingBatchSize) {
                await Task.yield()
            }

            if processedThisRun.isMultiple(of: configuration.databaseSaveBatchSize) {
                savePersistedState()
            }
        }

        if configuration.fullMergeAfterIndexing {
            rebuildAutomaticClusters(reason: "indexing finished")
        }
        refreshPeople(persist: true, allowReorder: true)
        indexingState = .idle
        activeIndexingRunID = nil
        activeIndexingAssetFingerprint = nil
        currentIndexingMessage = nil
        lastIndexingSummary = "Indexed \(processedThisRun) photos in \(Self.durationText(since: indexingStartedAt)). \(facesByID.count) faces across \(people.count) people."
        Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing finished (\(reason)) at \(processedOverall)/\(totalEligibleImageCount) photos in \(Self.durationText(since: indexingStartedAt)) with \(facesByID.count) face observations across \(people.count) people.")
    }

    private func process(asset: PhotoAssetSummary, photoLibraryStore: PhotoLibraryStore) async throws -> [FaceObservationInput] {
        let imageStartedAt = Date()
        guard let processingImage = await photoLibraryStore.imageForFaceProcessing(
            for: asset,
            maxDimension: configuration.faceProcessingMaxDimension,
            timeoutSeconds: configuration.faceImageRequestTimeoutSeconds
        ) else {
            Diagnostics.shared.log("Face indexing asset \(asset.id): no processing image after \(Self.durationText(since: imageStartedAt)); recording zero faces.")
            return []
        }
        Diagnostics.shared.log("Face indexing asset \(asset.id): processing image ready in \(Self.durationText(since: imageStartedAt)), decoded \(processingImage.pixelWidth)x\(processingImage.pixelHeight).")

        let detectionStartedAt = Date()
        let detectedFaces = try await detectionService.detectFaces(
            in: processingImage.cgImage,
            orientation: processingImage.orientation
        )
        Diagnostics.shared.log("Face indexing asset \(asset.id): Vision detected \(detectedFaces.count) faces in \(Self.durationText(since: detectionStartedAt)).")

        var observations: [FaceObservationInput] = []
        observations.reserveCapacity(detectedFaces.count)
        for (index, detectedFace) in detectedFaces.enumerated() {
            let faceStartedAt = Date()
            guard let crop = cropService.cropFace(
                from: processingImage.cgImage,
                detectedFace: detectedFace,
                configuration: configuration
            ) else {
                Diagnostics.shared.log("Face indexing asset \(asset.id): face \(index + 1)/\(detectedFaces.count) crop skipped.")
                continue
            }

            let embeddingStartedAt = Date()
            let embedding = try await embeddingService.embedding(
                for: crop.modelInputImage,
                debugIdentifier: "\(asset.id)_face\(index + 1)"
            )
            Diagnostics.shared.log("Face crop diagnostics asset \(asset.id), face \(index + 1): modelCrop \(crop.modelInputImage.width)x\(crop.modelInputImage.height), alignment \(crop.alignmentMethod.rawValue), quality \(crop.alignmentQuality), confidence \(detectedFace.confidence).")
            Diagnostics.shared.log("Face indexing asset \(asset.id): face \(index + 1)/\(detectedFaces.count) embedded in \(Self.durationText(since: embeddingStartedAt)); total face time \(Self.durationText(since: faceStartedAt)), confidence \(detectedFace.confidence), alignment \(crop.alignmentMethod.rawValue), alignment quality \(crop.alignmentQuality).")
            observations.append(FaceObservationInput(
                assetLocalIdentifier: asset.id,
                assetModificationDate: asset.modificationDate,
                assetPixelWidth: asset.pixelWidth,
                assetPixelHeight: asset.pixelHeight,
                normalizedBoundingBox: detectedFace.normalizedBoundingBox,
                leftToRightIndex: index,
                detectionConfidence: detectedFace.confidence,
                faceQuality: detectedFace.quality ?? crop.qualityScore,
                embedding: embedding,
                faceCropImageData: crop.avatarImageData
            ))
        }

        return observations
    }

    private func saveAndCluster(_ observations: [FaceObservationInput], asset: PhotoAssetSummary) {
        let previousPersonIDs = Set((faceIDsByAssetID[asset.id] ?? []).compactMap { facesByID[$0]?.personID })
        removeFaces(forAssetID: asset.id)
        for personID in previousPersonIDs {
            deletePersonIfEmpty(personID)
        }

        var personIDsAssignedInCurrentAsset = Set<UUID>()
        var savedObservationCount = 0
        for observation in observations {
            guard observation.embedding.count == configuration.embeddingDimension,
                  observation.embedding.allSatisfy(\.isFinite) else {
                Diagnostics.shared.log("Skipping invalid face embedding for asset \(asset.id), face \(observation.leftToRightIndex + 1).")
                continue
            }

            let norm = sqrt(observation.embedding.reduce(Float(0)) { $0 + ($1 * $1) })
            guard norm > 0.95, norm < 1.05 else {
                Diagnostics.shared.log("Skipping non-normalized face embedding for asset \(asset.id), face \(observation.leftToRightIndex + 1), norm \(norm).")
                continue
            }
            embeddingHealthMonitor.add(observation.embedding)
            let healthReport = embeddingHealthMonitor.report()
            logEmbeddingHealthIfReady(healthReport)

            let clusters = persons.values.map { person in
                FaceCluster(
                    personID: person.id,
                    centroid: person.centroid,
                    faceCount: faceCount(for: person.id),
                    name: person.name,
                    isProvisional: person.isProvisional,
                    assetLocalIdentifiers: assetIDs(for: person.id),
                    manuallyCorrectedFaceCount: manuallyCorrectedFaceCount(for: person.id),
                    representativeQuality: representativeQuality(for: person.id)
                )
            }
            let excludedPersonIDs: Set<UUID> = configuration.disallowMultipleFacesFromSameAssetForSamePerson
                ? personIDsAssignedInCurrentAsset
                : []
            let bestCandidate = clusteringEngine.bestCandidate(
                for: observation.embedding,
                clusters: clusters,
                excluding: excludedPersonIDs
            )
            let shouldDisableAutoClustering = configuration.disableAutoClusteringWhenEmbeddingHealthSuspicious &&
                (healthReport.status == .suspiciousCollapsed || healthReport.status == .suspiciousNoisy)
            let assignment = shouldDisableAutoClustering
                ? FaceClusterAssignment(kind: .deferredProvisional(bestPersonID: bestCandidate?.personID, similarity: bestCandidate?.similarity))
                : clusteringEngine.assignment(
                    for: observation.embedding,
                    clusters: clusters,
                    excluding: excludedPersonIDs
                )
            let personID: UUID
            let isProvisional: Bool

            switch assignment.kind {
            case let .existingPerson(existingID, _):
                assignmentDecisionCounts.existingPerson += 1
                personID = existingID
                isProvisional = false
            case .newPerson, .ambiguous:
                if case .newPerson = assignment.kind {
                    assignmentDecisionCounts.newPerson += 1
                } else {
                    assignmentDecisionCounts.ambiguous += 1
                }
                personID = UUID()
                isProvisional = observations.count > 1
                persons[personID] = StoredPerson(id: personID, name: nil, isProvisional: isProvisional)
            case .deferredProvisional:
                assignmentDecisionCounts.deferredProvisional += 1
                personID = UUID()
                isProvisional = true
                persons[personID] = StoredPerson(id: personID, name: nil, isProvisional: true)
                if shouldDisableAutoClustering {
                    faceRecognitionHealthMessage = "Face recognition embeddings look too similar. Picscry paused auto-grouping to avoid incorrect people."
                    if !didLogSuspiciousEmbeddingHealthThisRun {
                        didLogSuspiciousEmbeddingHealthThisRun = true
                        Diagnostics.shared.log("Face embedding health suspicious: \(healthReport.status), sampleCount \(healthReport.sampleCount), median \(healthReport.medianSimilarity?.description ?? "unknown"), min \(healthReport.minSimilarity?.description ?? "unknown"), max \(healthReport.maxSimilarity?.description ?? "unknown"); disabling auto-clustering.")
                    }
                } else {
                    Diagnostics.shared.log("Face clustering deferred provisional for asset \(asset.id), face \(observation.leftToRightIndex + 1): insufficient best-vs-second-best margin.")
                }
            }

            let face = StoredFaceObservation(input: observation, personID: personID)
            facesByID[face.id] = face
            faceIDsByAssetID[asset.id, default: []].append(face.id)
            recompute(personID: personID)
            personIDsAssignedInCurrentAsset.insert(personID)
            savedObservationCount += 1
            Diagnostics.shared.log("Face clustering asset \(asset.id), face \(observation.leftToRightIndex + 1): best similarity \(bestCandidate?.similarity.description ?? "none"), excluded \(excludedPersonIDs.count), assigned person \(personID), provisional \(isProvisional), assignment \(assignment.kind).")
        }

        indexRecords[asset.id] = AssetIndexRecord(asset: asset, faceCount: savedObservationCount)
        if savedObservationCount > 0,
           (facesByID.count.isMultiple(of: configuration.clusterRebuildBatchSize) ||
            facesByID.count.isMultiple(of: configuration.batchMergeIntervalFaceCount)) {
            rebuildAutomaticClusters(reason: "batch \(facesByID.count)")
        }
    }

    private func assetNeedsIndex(_ asset: PhotoAssetSummary) -> Bool {
        guard let record = indexRecords[asset.id] else { return true }
        return record.assetModificationDate != asset.modificationDate ||
            record.assetPixelWidth != asset.pixelWidth ||
            record.assetPixelHeight != asset.pixelHeight
    }

    private func summaries(forAssetID assetID: String) -> [PhotoFaceSummary] {
        (faceIDsByAssetID[assetID] ?? [])
            .compactMap { faceID -> PhotoFaceSummary? in
                guard let face = facesByID[faceID], let person = persons[face.personID] else { return nil }
                return PhotoFaceSummary(
                    id: face.id,
                    personID: person.id,
                    assetLocalIdentifier: face.assetLocalIdentifier,
                    displayName: person.displayName,
                    isUnknown: person.isUnknown,
                    normalizedBoundingBox: face.normalizedBoundingBox,
                    leftToRightIndex: face.leftToRightIndex,
                    confidence: face.detectionConfidence,
                    representativeFaceImageData: person.representativeImageData ?? face.faceCropImageData,
                    isManuallyCorrected: face.isManuallyCorrected
                )
            }
            .sorted { $0.leftToRightIndex < $1.leftToRightIndex }
    }

    private func summary(for personID: UUID) -> PersonSummary? {
        guard let person = persons[personID] else { return nil }
        let faceIDs = facesByID.values.filter { $0.personID == personID }
        return PersonSummary(
            id: person.id,
            displayName: person.displayName,
            isUnknown: person.isUnknown,
            photoCount: Set(faceIDs.map(\.assetLocalIdentifier)).count,
            faceCount: faceIDs.count,
            representativeFaceImageData: person.representativeImageData,
            isProvisional: person.isProvisional
        )
    }

    private func refreshPeople(persist: Bool = true, allowReorder: Bool = true) {
        let summaries = makePeopleSummaries()
        let refreshedPeople = allowReorder
            ? PeopleOrdering.sorted(summaries)
            : stablePeopleUpdate(existing: people, refreshed: summaries)

        if Self.peopleHaveMeaningfulDifference(people, refreshedPeople) {
            people = refreshedPeople
        }

        if persist {
            savePersistedState()
        }
    }

    private func makePeopleSummaries() -> [PersonSummary] {
        let summaries = persons.keys.compactMap(summary(for:))
        hiddenProvisionalUnknownCount = summaries.filter { $0.isUnknown && $0.isProvisional && $0.photoCount < 2 }.count
        return summaries.filter { !($0.isUnknown && $0.isProvisional && $0.photoCount < 2) }
    }

    private func stablePeopleUpdate(existing: [PersonSummary], refreshed: [PersonSummary]) -> [PersonSummary] {
        let refreshedByID = Dictionary(uniqueKeysWithValues: refreshed.map { ($0.id, $0) })
        var result = existing.compactMap { refreshedByID[$0.id] }

        let existingIDs = Set(existing.map(\.id))
        let newPeople = refreshed.filter { !existingIDs.contains($0.id) }
        result.append(contentsOf: PeopleOrdering.sorted(newPeople))
        return result
    }

    private static func peopleHaveMeaningfulDifference(_ lhs: [PersonSummary], _ rhs: [PersonSummary]) -> Bool {
        guard lhs.count == rhs.count else { return true }
        for (left, right) in zip(lhs, rhs) {
            if left.id != right.id ||
                left.displayName != right.displayName ||
                left.isUnknown != right.isUnknown ||
                left.isProvisional != right.isProvisional ||
                left.photoCount != right.photoCount ||
                left.faceCount != right.faceCount ||
                left.representativeFaceImageData?.count != right.representativeFaceImageData?.count {
                return true
            }
        }
        return false
    }

    private func recompute(personID: UUID) {
        guard var person = persons[personID] else { return }
        let faces = facesByID.values.filter { $0.personID == personID }
        guard !faces.isEmpty else {
            persons[personID] = person
            return
        }

        let clusters = faces.map { (centroid: $0.embedding, count: 1) }
        person.centroid = clusteringEngine.mergedCentroid(clusters: clusters)
        person.representativeImageData = faces.max(by: { score($0) < score($1) })?.faceCropImageData
        if person.isProvisional && Set(faces.map(\.assetLocalIdentifier)).count >= 2 {
            person.isProvisional = false
        }
        person.updatedAt = .now
        persons[personID] = person
    }

    private func score(_ face: StoredFaceObservation) -> Float {
        let boxArea = Float(face.normalizedBoundingBox.width * face.normalizedBoundingBox.height)
        let manualBonus: Float = face.isManuallyCorrected ? 0.15 : 0
        return (face.detectionConfidence * 0.35) + (min(boxArea * 12, 1) * 0.25) + ((face.faceQuality ?? 0.5) * 0.25) + manualBonus
    }

    private func faceCount(for personID: UUID) -> Int {
        facesByID.values.filter { $0.personID == personID }.count
    }

    private func assetIDs(for personID: UUID) -> Set<String> {
        Set(facesByID.values.filter { $0.personID == personID }.map(\.assetLocalIdentifier))
    }

    private func manuallyCorrectedFaceCount(for personID: UUID) -> Int {
        facesByID.values.filter { $0.personID == personID && $0.isManuallyCorrected }.count
    }

    private func representativeQuality(for personID: UUID) -> Float {
        facesByID.values
            .filter { $0.personID == personID }
            .map(score)
            .max() ?? 0
    }

    private func rebuildAutomaticClusters(reason: String) {
        let candidates = facesByID.values
            .filter { face in
                guard let person = persons[face.personID] else { return false }
                return person.isAutomaticCluster &&
                    person.isUnknown &&
                    !face.isManuallyCorrected &&
                    face.embedding.count == configuration.embeddingDimension &&
                    face.embedding.allSatisfy(\.isFinite)
            }
            .sorted {
                if $0.assetLocalIdentifier != $1.assetLocalIdentifier {
                    return $0.assetLocalIdentifier < $1.assetLocalIdentifier
                }
                return $0.leftToRightIndex < $1.leftToRightIndex
            }

        guard !candidates.isEmpty else { return }

        let healthReport = embeddingHealthMonitor.report()
        let shouldDisableAutoClustering = configuration.disableAutoClusteringWhenEmbeddingHealthSuspicious &&
            (healthReport.status == .suspiciousCollapsed || healthReport.status == .suspiciousNoisy)
        let oldAutomaticPersonIDs = Set(candidates.map(\.personID))
        let components: [FaceClusteringComponent]
        let initialComponents: [FaceClusteringComponent]
        var mergeStats = FaceClusterMergeStats()
        let nodes = candidates.map {
            FaceClusteringObservationNode(
                id: $0.id,
                assetLocalIdentifier: $0.assetLocalIdentifier,
                embedding: $0.embedding
            )
        }

        if shouldDisableAutoClustering {
            initialComponents = candidates.map { FaceClusteringComponent(nodeIDs: [$0.id]) }
            components = initialComponents
            faceRecognitionHealthMessage = "Face recognition embeddings look too similar. Picscry paused auto-grouping to avoid incorrect people."
            Diagnostics.shared.log("Face clustering rebuild skipped grouping (\(reason)): embedding health \(healthReport.status), observations \(candidates.count).")
        } else {
            if nodes.count <= configuration.maximumAllPairsClusteringFaceCount {
                initialComponents = clusteringEngine.constrainedComponents(
                    for: nodes,
                    similarityThreshold: configuration.graphEdgeSimilarityThreshold
                )
                Diagnostics.shared.log("Face clustering rebuild using all-pairs graph (\(reason)): observations \(nodes.count), threshold \(configuration.graphEdgeSimilarityThreshold).")
            } else {
                initialComponents = clusteringEngine.constrainedIncrementalComponents(
                    for: nodes,
                    similarityThreshold: configuration.graphEdgeSimilarityThreshold,
                    singleSampleThreshold: configuration.graphEdgeSimilarityThresholdForSingleSample
                )
                Diagnostics.shared.log("Face clustering rebuild using incremental constrained fallback (\(reason)): observations \(nodes.count), threshold \(configuration.graphEdgeSimilarityThreshold).")
            }
            let mergeResult = clusteringEngine.mergedConstrainedComponentResult(
                from: initialComponents,
                nodes: nodes,
                mergeThreshold: configuration.mergeThreshold
            )
            components = mergeResult.components
            mergeStats = mergeResult.stats
            if components.count != initialComponents.count {
                Diagnostics.shared.log("Face clustering merge pass (\(reason)): reduced clusters from \(initialComponents.count) to \(components.count), accepted \(mergeStats.acceptedMerges), threshold \(configuration.mergeThreshold).")
            }
        }

        for component in components {
            let componentFaces = component.nodeIDs.compactMap { facesByID[$0] }
            guard !componentFaces.isEmpty else { continue }

            let personID = UUID()
            let assetIDs = Set(componentFaces.map(\.assetLocalIdentifier))
            persons[personID] = StoredPerson(
                id: personID,
                name: nil,
                isAutomaticCluster: true,
                isProvisional: assetIDs.count < 2
            )

            for faceID in component.nodeIDs {
                facesByID[faceID]?.personID = personID
                facesByID[faceID]?.updatedAt = .now
            }
            recompute(personID: personID)
        }

        for personID in oldAutomaticPersonIDs {
            recompute(personID: personID)
            deletePersonIfEmpty(personID)
        }

        let provisionalCount = components.filter { component in
            let assetIDs = Set(component.nodeIDs.compactMap { facesByID[$0]?.assetLocalIdentifier })
            return assetIDs.count < 2
        }.count
        let diagnostics = makeClusteringDiagnostics(
            reason: reason,
            nodes: nodes,
            initialComponents: initialComponents,
            components: components,
            provisionalCount: provisionalCount,
            mergeStats: mergeStats
        )
        saveClusteringDiagnostics(diagnostics)
        Diagnostics.shared.log("Face clustering rebuild completed (\(reason)): observations \(candidates.count), clusters \(components.count), provisional \(provisionalCount), largest \(diagnostics.largestClusterSize), median \(diagnostics.medianClusterSize), singleton \(diagnostics.singletonClusterCount), top20 \(diagnostics.topClusterSizes.prefix(20)).")
    }

    private func makeClusteringDiagnostics(
        reason: String,
        nodes: [FaceClusteringObservationNode],
        initialComponents: [FaceClusteringComponent],
        components: [FaceClusteringComponent],
        provisionalCount: Int,
        mergeStats: FaceClusterMergeStats
    ) -> FaceClusteringDiagnosticsSnapshot {
        let initialClusterSizes = initialComponents.map(\.nodeIDs.count).sorted(by: >)
        let clusterSizes = components.map(\.nodeIDs.count).sorted(by: >)
        let medianClusterSize = clusterSizes.isEmpty ? 0 : clusterSizes[clusterSizes.count / 2]
        let pairStats = pairSimilarityStats(for: nodes)
        return FaceClusteringDiagnosticsSnapshot(
            generatedAt: Date(),
            reason: reason,
            thresholdProfile: FaceThresholdDiagnostics(
                possibleMatchThreshold: configuration.possibleMatchThreshold,
                autoMatchThreshold: configuration.autoMatchThreshold,
                singleSampleAutoMatchThreshold: configuration.singleSampleAutoMatchThreshold,
                mergeThreshold: configuration.mergeThreshold,
                pairMergeThreshold: configuration.pairMergeThreshold,
                namedAbsorbUnknownThreshold: configuration.namedAbsorbUnknownThreshold,
                largeNamedAbsorbUnknownThreshold: configuration.largeNamedAbsorbUnknownThreshold,
                minimumBestSecondBestMargin: configuration.minimumBestSecondBestMargin,
                graphEdgeSimilarityThreshold: configuration.graphEdgeSimilarityThreshold,
                graphEdgeSimilarityThresholdForSingleSample: configuration.graphEdgeSimilarityThresholdForSingleSample,
                batchMergeIntervalFaceCount: configuration.batchMergeIntervalFaceCount
            ),
            totalFaces: nodes.count,
            totalPeopleClusters: components.count,
            largestClusterSize: clusterSizes.first ?? 0,
            medianClusterSize: medianClusterSize,
            singletonClusterCount: clusterSizes.filter { $0 == 1 }.count,
            provisionalClusterCount: provisionalCount,
            topClusterSizes: Array(clusterSizes.prefix(20)),
            assignmentDecisionCounts: assignmentDecisionCounts,
            acceptedMergeCount: mergeStats.acceptedMerges,
            rejectedMergeCountsByReason: Dictionary(uniqueKeysWithValues: FaceClusterMergeRejectReason.allCases.map {
                ($0.rawValue, mergeStats.rejectedByReason[$0] ?? 0)
            }),
            beforeClusterSizeDistribution: initialClusterSizes,
            afterClusterSizeDistribution: clusterSizes,
            similarityHistogram: pairStats.similarityHistogram,
            bestSecondBestMarginHistogram: pairStats.marginHistogram
        )
    }

    private func pairSimilarityStats(
        for nodes: [FaceClusteringObservationNode]
    ) -> (similarityHistogram: [String: Int], marginHistogram: [String: Int]) {
        var similarityHistogram = FaceClusteringDiagnosticsSnapshot.emptySimilarityHistogram()
        var marginHistogram = FaceClusteringDiagnosticsSnapshot.emptyMarginHistogram()
        guard nodes.count <= configuration.maximumAllPairsClusteringFaceCount else {
            return (similarityHistogram, marginHistogram)
        }

        var topSimilaritiesByNode = Array(
            repeating: (best: Optional<Float>.none, second: Optional<Float>.none),
            count: nodes.count
        )
        for leftIndex in nodes.indices {
            for rightIndex in nodes.indices where rightIndex > leftIndex {
                guard nodes[leftIndex].assetLocalIdentifier != nodes[rightIndex].assetLocalIdentifier else {
                    continue
                }
                let similarity = nodes[leftIndex].embedding.cosineSimilarity(to: nodes[rightIndex].embedding)
                similarityHistogram[Self.similarityBucket(for: similarity), default: 0] += 1
                Self.recordTopSimilarity(similarity, for: leftIndex, in: &topSimilaritiesByNode)
                Self.recordTopSimilarity(similarity, for: rightIndex, in: &topSimilaritiesByNode)
            }
        }

        for pair in topSimilaritiesByNode {
            guard let best = pair.best, let second = pair.second else { continue }
            marginHistogram[Self.marginBucket(for: best - second), default: 0] += 1
        }

        return (similarityHistogram, marginHistogram)
    }

    private static func recordTopSimilarity(
        _ similarity: Float,
        for index: Int,
        in topSimilaritiesByNode: inout [(best: Float?, second: Float?)]
    ) {
        let current = topSimilaritiesByNode[index]
        if current.best.map({ similarity > $0 }) ?? true {
            topSimilaritiesByNode[index] = (similarity, current.best)
        } else if current.second.map({ similarity > $0 }) ?? true {
            topSimilaritiesByNode[index].second = similarity
        }
    }

    private static func similarityBucket(for similarity: Float) -> String {
        switch similarity {
        case ..<0.60: return "0.50-0.60"
        case ..<0.70: return "0.60-0.70"
        case ..<0.75: return "0.70-0.75"
        case ..<0.80: return "0.75-0.80"
        case ..<0.85: return "0.80-0.85"
        case ..<0.90: return "0.85-0.90"
        case ..<0.95: return "0.90-0.95"
        default: return "0.95-1.00"
        }
    }

    private static func marginBucket(for margin: Float) -> String {
        switch margin {
        case ..<0.005: return "0.000-0.005"
        case ..<0.010: return "0.005-0.010"
        case ..<0.015: return "0.010-0.015"
        case ..<0.025: return "0.015-0.025"
        case ..<0.050: return "0.025-0.050"
        default: return "0.050+"
        }
    }

    private func saveClusteringDiagnostics(_ diagnostics: FaceClusteringDiagnosticsSnapshot) {
        guard let url = Self.clusteringDiagnosticsURL() else { return }
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try JSONEncoder.pretty.encode(diagnostics)
            try data.write(to: url, options: .atomic)
        } catch {
            Diagnostics.shared.log("Failed to save face clustering diagnostics: \(error.localizedDescription)")
        }
    }

    private func logEmbeddingHealthIfReady(_ report: FaceEmbeddingHealthReport) {
        guard report.sampleCount > lastLoggedEmbeddingHealthSampleCount,
              (report.sampleCount == configuration.embeddingCalibrationSampleCount ||
               report.sampleCount.isMultiple(of: 8)) else {
            return
        }
        lastLoggedEmbeddingHealthSampleCount = report.sampleCount
        Diagnostics.shared.log("Face embedding health: \(report.status), sampleCount \(report.sampleCount), pairCount \(report.pairCount), min \(report.minSimilarity?.description ?? "unknown"), median \(report.medianSimilarity?.description ?? "unknown"), max \(report.maxSimilarity?.description ?? "unknown").")
    }

    private func indexingFingerprint(for assets: [PhotoAssetSummary]) -> String {
        let firstIDs = assets.prefix(20).map(\.id).joined(separator: "|")
        let lastIDs = assets.suffix(20).map(\.id).joined(separator: "|")
        let maxModified = assets.compactMap(\.modificationDate).max()?.timeIntervalSince1970 ?? 0
        return "\(assets.count)#\(Int(maxModified))#\(firstIDs)#\(lastIDs)"
    }

    private func removeDeletedAssets(liveIDs: Set<String>) {
        for assetID in Set(indexRecords.keys).subtracting(liveIDs) {
            removeFaces(forAssetID: assetID)
            indexRecords[assetID] = nil
        }
        for personID in Array(persons.keys) {
            deletePersonIfEmpty(personID)
        }
        refreshPeople()
    }

    private func removeFaces(forAssetID assetID: String) {
        for faceID in faceIDsByAssetID[assetID] ?? [] {
            if let personID = facesByID[faceID]?.personID {
                facesByID[faceID] = nil
                recompute(personID: personID)
            }
        }
        faceIDsByAssetID[assetID] = []
    }

    private func deletePersonIfEmpty(_ personID: UUID) {
        if !facesByID.values.contains(where: { $0.personID == personID }) {
            persons[personID] = nil
        }
    }

    private func loadPersistedState() {
        guard let url = Self.databaseURL(),
              FileManager.default.fileExists(atPath: url.path) else {
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let snapshot = try JSONDecoder().decode(FaceDatabaseSnapshot.self, from: data)
            if [7, 8, 9].contains(snapshot.schemaVersion), FaceDatabaseSchema.currentVersion == 10 {
                persons = Dictionary(uniqueKeysWithValues: snapshot.persons.map { ($0.id, $0) })
                facesByID = Dictionary(uniqueKeysWithValues: snapshot.faces.map { ($0.id, $0) })
                faceIDsByAssetID = snapshot.faceIDsByAssetID
                indexRecords = snapshot.indexRecords
                rebuildAutomaticClusters(reason: "schema 10 precision threshold migration")
                refreshPeople(persist: true, allowReorder: true)
                lastIndexingSummary = "Face clustering thresholds were tightened. Picscry rebuilt automatic unknown people while preserving named people and manual corrections."
                Diagnostics.shared.log("Migrated face database schema \(snapshot.schemaVersion) to 10 with precision-first thresholds. People \(persons.count), faces \(facesByID.count).")
                return
            }

            guard snapshot.schemaVersion == FaceDatabaseSchema.currentVersion else {
                Diagnostics.shared.log("Discarding old face database schema \(snapshot.schemaVersion); current schema is \(FaceDatabaseSchema.currentVersion). Reindex required.")
                resetPersistedState(at: url)
                lastIndexingSummary = "Face recognition clustering was upgraded. Picscry will reindex faces on this device."
                return
            }

            persons = Dictionary(uniqueKeysWithValues: snapshot.persons.map { ($0.id, $0) })
            facesByID = Dictionary(uniqueKeysWithValues: snapshot.faces.map { ($0.id, $0) })
            faceIDsByAssetID = snapshot.faceIDsByAssetID
            indexRecords = snapshot.indexRecords
            refreshPeople(persist: false, allowReorder: true)
            Diagnostics.shared.log("Loaded persisted face database with \(persons.count) people and \(facesByID.count) faces.")
        } catch {
            Diagnostics.shared.log("Face database is incompatible or unreadable: \(error.localizedDescription). Resetting face index.")
            resetPersistedState(at: url)
            lastIndexingSummary = "Face recognition was upgraded. Picscry will reindex faces on this device."
        }
    }

    private func resetPersistedState(at url: URL) {
        persons = [:]
        facesByID = [:]
        faceIDsByAssetID = [:]
        indexRecords = [:]
        try? FileManager.default.removeItem(at: url)
        refreshPeople(persist: false, allowReorder: true)
    }

    private func savePersistedState() {
        guard let url = Self.databaseURL() else { return }

        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let snapshot = FaceDatabaseSnapshot(
                schemaVersion: FaceDatabaseSchema.currentVersion,
                persons: Array(persons.values),
                faces: Array(facesByID.values),
                faceIDsByAssetID: faceIDsByAssetID,
                indexRecords: indexRecords
            )
            let data = try JSONEncoder().encode(snapshot)
            try data.write(to: url, options: .atomic)
        } catch {
            Diagnostics.shared.log("Failed to save face database: \(error.localizedDescription)")
        }
    }

    private static func databaseURL() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("FaceRecognition", isDirectory: true)
            .appendingPathComponent("faces.json", isDirectory: false)
    }

    private static func clusteringDiagnosticsURL() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("FaceRecognition", isDirectory: true)
            .appendingPathComponent("clustering-diagnostics.json", isDirectory: false)
    }

    private static func normalizedName(_ name: String) -> String? {
        let collapsed = name
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return collapsed.isEmpty ? nil : collapsed
    }

    private static func durationText(since startDate: Date) -> String {
        String(format: "%.2fs", Date().timeIntervalSince(startDate))
    }

    private static func shortRunID(_ runID: UUID) -> String {
        String(runID.uuidString.prefix(8))
    }
}

private enum FaceDatabaseSchema {
    static let currentVersion = 10
}

private struct FaceThresholdDiagnostics: Codable {
    let possibleMatchThreshold: Float
    let autoMatchThreshold: Float
    let singleSampleAutoMatchThreshold: Float
    let mergeThreshold: Float
    let pairMergeThreshold: Float
    let namedAbsorbUnknownThreshold: Float
    let largeNamedAbsorbUnknownThreshold: Float
    let minimumBestSecondBestMargin: Float
    let graphEdgeSimilarityThreshold: Float
    let graphEdgeSimilarityThresholdForSingleSample: Float
    let batchMergeIntervalFaceCount: Int
}

private struct FaceAssignmentDecisionCounts: Codable {
    var existingPerson = 0
    var newPerson = 0
    var ambiguous = 0
    var deferredProvisional = 0
}

private struct FaceClusteringDiagnosticsSnapshot: Codable {
    let generatedAt: Date
    let reason: String
    let thresholdProfile: FaceThresholdDiagnostics
    let totalFaces: Int
    let totalPeopleClusters: Int
    let largestClusterSize: Int
    let medianClusterSize: Int
    let singletonClusterCount: Int
    let provisionalClusterCount: Int
    let topClusterSizes: [Int]
    let assignmentDecisionCounts: FaceAssignmentDecisionCounts
    let acceptedMergeCount: Int
    let rejectedMergeCountsByReason: [String: Int]
    let beforeClusterSizeDistribution: [Int]
    let afterClusterSizeDistribution: [Int]
    let similarityHistogram: [String: Int]
    let bestSecondBestMarginHistogram: [String: Int]

    static func emptySimilarityHistogram() -> [String: Int] {
        [
            "0.50-0.60": 0,
            "0.60-0.70": 0,
            "0.70-0.75": 0,
            "0.75-0.80": 0,
            "0.80-0.85": 0,
            "0.85-0.90": 0,
            "0.90-0.95": 0,
            "0.95-1.00": 0
        ]
    }

    static func emptyMarginHistogram() -> [String: Int] {
        [
            "0.000-0.005": 0,
            "0.005-0.010": 0,
            "0.010-0.015": 0,
            "0.015-0.025": 0,
            "0.025-0.050": 0,
            "0.050+": 0
        ]
    }
}

private struct FaceDatabaseSnapshot: Codable {
    let schemaVersion: Int
    let persons: [StoredPerson]
    let faces: [StoredFaceObservation]
    let faceIDsByAssetID: [String: [UUID]]
    let indexRecords: [String: AssetIndexRecord]
}

private struct StoredPerson: Codable {
    let id: UUID
    var name: String?
    var createdAt = Date()
    var updatedAt = Date()
    var centroid: [Float] = []
    var representativeImageData: Data?
    var isAutomaticCluster = true
    var isProvisional = false

    var displayName: String { name ?? "Unknown" }
    var isUnknown: Bool { name == nil }

    init(
        id: UUID,
        name: String?,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        centroid: [Float] = [],
        representativeImageData: Data? = nil,
        isAutomaticCluster: Bool = true,
        isProvisional: Bool = false
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.centroid = centroid
        self.representativeImageData = representativeImageData
        self.isAutomaticCluster = isAutomaticCluster
        self.isProvisional = isProvisional
    }
}

private struct StoredFaceObservation: Codable {
    let id: UUID
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
    var personID: UUID
    let createdAt = Date()
    var updatedAt = Date()
    var isManuallyCorrected = false

    init(input: FaceObservationInput, personID: UUID) {
        id = UUID()
        assetLocalIdentifier = input.assetLocalIdentifier
        assetModificationDate = input.assetModificationDate
        assetPixelWidth = input.assetPixelWidth
        assetPixelHeight = input.assetPixelHeight
        normalizedBoundingBox = input.normalizedBoundingBox
        leftToRightIndex = input.leftToRightIndex
        detectionConfidence = input.detectionConfidence
        faceQuality = input.faceQuality
        embedding = input.embedding
        faceCropImageData = input.faceCropImageData
        self.personID = personID
    }
}

private struct AssetIndexRecord: Codable {
    let assetModificationDate: Date?
    let assetPixelWidth: Int
    let assetPixelHeight: Int
    let indexedAt = Date()
    let faceCount: Int

    init(asset: PhotoAssetSummary, faceCount: Int) {
        assetModificationDate = asset.modificationDate
        assetPixelWidth = asset.pixelWidth
        assetPixelHeight = asset.pixelHeight
        self.faceCount = faceCount
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

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
    private var faceIDsByPersonID: [UUID: Set<UUID>] = [:]
    private var assetIDsByPersonID: [UUID: Set<String>] = [:]
    private var manuallyCorrectedFaceCountsByPersonID: [UUID: Int] = [:]
    private var representativeQualityByPersonID: [UUID: Float] = [:]
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

        if shouldPauseAfterUncleanIndexingExit(
            reason: reason,
            fingerprint: currentFingerprint,
            forceRestart: forceRestart
        ) {
            indexingState = .paused
            currentIndexingMessage = nil
            lastIndexingSummary = "Face indexing was paused because the previous indexing run did not exit cleanly. Use manual refresh to resume."
            Diagnostics.shared.log("Face indexing paused on launch (\(reason)): previous run did not exit cleanly for the same library fingerprint.")
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
            let unclusteredCount = unclusteredObservationCount()
            if configuration.backgroundMergeEnabled || unclusteredCount > 0 {
                rebuildAutomaticClusters(reason: "no pending photos after \(reason)")
            }
            refreshPeople(persist: true, allowReorder: true)
            indexingState = .idle
            currentIndexingMessage = nil
            lastIndexingSummary = "All \(totalEligibleImageCount) photos are indexed. \(facesByID.count) faces across \(people.count) people."
            Diagnostics.shared.log("Face indexing skipped (\(reason)): no pending photos out of \(assets.count) image assets. Existing observations: \(facesByID.count).")
            clearActiveIndexingRunMarker()
            scheduleBackgroundIndexing(reason: "no pending photos after \(reason)")
            return
        }

        let runID = UUID()
        activeIndexingRunID = runID
        activeIndexingAssetFingerprint = currentFingerprint
        didLogSuspiciousEmbeddingHealthThisRun = false
        assignmentDecisionCounts = FaceAssignmentDecisionCounts()
        markIndexingRunActive(fingerprint: currentFingerprint, reason: reason, processedImageCount: alreadyIndexedCount)
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
        for faceID in Array(faceIDsByPersonID[sourcePersonID] ?? []) {
            moveFaceIndex(faceID, to: targetPersonID)
        }
        persons[sourcePersonID] = nil
        recompute(personID: targetPersonID)
        refreshPeople()
    }

    func moveFaceObservation(_ faceID: UUID, toExistingPersonID personID: UUID) async {
        guard let original = facesByID[faceID], persons[personID] != nil else { return }
        let oldPersonID = original.personID
        moveFaceIndex(faceID, to: personID, isManuallyCorrected: true)
        if let oldPersonID {
            recompute(personID: oldPersonID)
            deletePersonIfEmpty(oldPersonID)
        }
        recompute(personID: personID)
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
        moveFaceIndex(faceID, to: newPersonID, isManuallyCorrected: true)
        if let oldPersonID {
            recompute(personID: oldPersonID)
            deletePersonIfEmpty(oldPersonID)
        }
        recompute(personID: newPersonID)
        refreshPeople()
    }

    func markFaceIncorrect(_ faceID: UUID) async {
        guard var face = facesByID[faceID], let personID = face.personID else { return }
        face.isManuallyCorrected = true
        facesByID[faceID] = face
        rebuildAggregateIndex(for: personID)
        recompute(personID: personID)
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
        var performanceMetrics = FaceIndexingPerformanceMetrics(
            totalEligibleImageCount: totalEligibleImageCount,
            alreadyIndexedCount: alreadyIndexedCount,
            pendingImageCount: pending.count
        )
        var processedThisRun = 0
        var processedOverall = alreadyIndexedCount
        indexingState = .indexing(processed: processedOverall, total: totalEligibleImageCount)
        currentIndexingMessage = "Starting face indexing..."
        let extractionStartedAt = Date()
        let runContext = indexingRunContext(for: reason)
        let workers = (0..<3).map { FaceIndexingWorker(id: $0 + 1, configuration: configuration) }
        var chunkStart = pending.startIndex
        while chunkStart < pending.endIndex {
            guard activeIndexingRunID == runID else {
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Ignoring stale face indexing run before chunk \(chunkStart).")
                return
            }
            guard !Task.isCancelled else {
                indexingState = .paused
                currentIndexingMessage = "Face indexing paused at \(processedOverall) of \(totalEligibleImageCount)."
                savePersistedState()
                clearActiveIndexingRunMarker()
                saveIndexingPerformanceDiagnostics(performanceMetrics.snapshot(
                    reason: reason,
                    completed: false,
                    duration: Date().timeIntervalSince(indexingStartedAt),
                    extractionDuration: Date().timeIntervalSince(extractionStartedAt),
                    clusteringDuration: 0,
                    totalFaces: facesByID.count,
                    visiblePeople: people.count,
                    unclusteredObservationCount: unclusteredObservationCount()
                ))
                Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing run cancelled (\(reason)) at \(processedOverall) of \(totalEligibleImageCount) photos in \(Self.durationText(since: indexingStartedAt)).")
                return
            }

            let workerLimit = FaceIndexingWorkerPolicy.workerLimit(context: runContext)
            performanceMetrics.recordWorkerLimit(workerLimit, processedImageCount: processedThisRun)
            let chunkEnd = min(pending.endIndex, chunkStart + 25)
            let chunk = Array(pending[chunkStart..<chunkEnd].enumerated().map { localOffset, asset in
                PendingFaceIndexingAsset(offset: chunkStart + localOffset, asset: asset)
            })
            currentIndexingMessage = "Extracting faces \(processedOverall + 1)-\(min(totalEligibleImageCount, alreadyIndexedCount + processedThisRun + chunk.count)) of \(totalEligibleImageCount)"
            Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face extraction chunk \(chunkStart + 1)-\(chunkEnd) using \(workerLimit) workers.")

            let completions = await processExtractionChunk(
                chunk,
                workerLimit: workerLimit,
                workers: workers,
                photoLibraryStore: photoLibraryStore
            )

            for completion in completions {
                guard activeIndexingRunID == runID else {
                    Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Ignoring stale face indexing run after processing \(completion.asset.id).")
                    return
                }

                switch completion.result {
                case let .success(result):
                    performanceMetrics.record(result)
                    saveExtractedObservations(result.observations, asset: completion.asset)
                    Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face extraction asset \(completion.offset + 1)/\(pending.count) finished: \(completion.asset.id), saved \(result.observations.count) unclustered observations.")
                case let .failure(error as CancellationError):
                    indexingState = .paused
                    currentIndexingMessage = "Face indexing paused at \(processedOverall) of \(totalEligibleImageCount)."
                    savePersistedState()
                    clearActiveIndexingRunMarker()
                    saveIndexingPerformanceDiagnostics(performanceMetrics.snapshot(
                        reason: reason,
                        completed: false,
                        duration: Date().timeIntervalSince(indexingStartedAt),
                        extractionDuration: Date().timeIntervalSince(extractionStartedAt),
                        clusteringDuration: 0,
                        totalFaces: facesByID.count,
                        visiblePeople: people.count,
                        unclusteredObservationCount: unclusteredObservationCount()
                    ))
                    Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing cancelled while processing \(completion.asset.id); not recording index record. \(error.localizedDescription)")
                    return
                case let .failure(error):
                    Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing failed for \(completion.asset.id): \(error.localizedDescription)")
                }

                processedThisRun += 1
                processedOverall = min(totalEligibleImageCount, alreadyIndexedCount + processedThisRun)
                indexingState = .indexing(processed: processedOverall, total: totalEligibleImageCount)
                if processedThisRun == 1 || processedThisRun.isMultiple(of: 10) {
                    currentIndexingMessage = "Extracted faces from \(processedOverall) of \(totalEligibleImageCount) photos"
                    Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face extraction progress (\(reason)): \(processedOverall)/\(totalEligibleImageCount) photos in \(Self.durationText(since: indexingStartedAt)).")
                }

                if processedThisRun.isMultiple(of: configuration.peopleRefreshBatchSize) {
                    refreshPeople(persist: false, allowReorder: false)
                }

                if processedThisRun.isMultiple(of: configuration.databaseSaveBatchSize) {
                    savePersistedState()
                    markIndexingRunActive(fingerprint: activeIndexingAssetFingerprint ?? "", reason: reason, processedImageCount: processedOverall)
                }
            }

            chunkStart = chunkEnd
            savePersistedState()
            markIndexingRunActive(fingerprint: activeIndexingAssetFingerprint ?? "", reason: reason, processedImageCount: processedOverall)
            await Task.yield()
        }

        let extractionDuration = Date().timeIntervalSince(extractionStartedAt)
        let clusteringStartedAt = Date()
        if configuration.fullMergeAfterIndexing {
            rebuildAutomaticClusters(reason: "indexing finished")
        }
        let clusteringDuration = Date().timeIntervalSince(clusteringStartedAt)
        refreshPeople(persist: true, allowReorder: true)
        saveIndexingPerformanceDiagnostics(performanceMetrics.snapshot(
            reason: reason,
            completed: true,
            duration: Date().timeIntervalSince(indexingStartedAt),
            extractionDuration: extractionDuration,
            clusteringDuration: clusteringDuration,
            totalFaces: facesByID.count,
            visiblePeople: people.count,
            unclusteredObservationCount: unclusteredObservationCount()
        ))
        indexingState = .idle
        activeIndexingRunID = nil
        activeIndexingAssetFingerprint = nil
        currentIndexingMessage = nil
        lastIndexingSummary = "Indexed \(processedThisRun) photos in \(Self.durationText(since: indexingStartedAt)). \(facesByID.count) faces across \(people.count) people."
        clearActiveIndexingRunMarker()
        Diagnostics.shared.log("[FaceRun \(Self.shortRunID(runID))] Face indexing finished (\(reason)) at \(processedOverall)/\(totalEligibleImageCount) photos in \(Self.durationText(since: indexingStartedAt)) with \(facesByID.count) face observations across \(people.count) people.")
    }

    private func processExtractionChunk(
        _ chunk: [PendingFaceIndexingAsset],
        workerLimit: Int,
        workers: [FaceIndexingWorker],
        photoLibraryStore: PhotoLibraryStore
    ) async -> [FaceIndexingWorkerCompletion] {
        guard !chunk.isEmpty else { return [] }
        var completions: [FaceIndexingWorkerCompletion] = []
        completions.reserveCapacity(chunk.count)
        var nextIndex = 0
        let activeWorkerCount = min(workerLimit, workers.count, chunk.count)

        await withTaskGroup(of: FaceIndexingWorkerCompletion.self) { group in
            func enqueue(workerIndex: Int) {
                guard nextIndex < chunk.count else { return }
                let pendingAsset = chunk[nextIndex]
                let worker = workers[workerIndex]
                nextIndex += 1
                group.addTask {
                    let result: Result<FaceAssetProcessingResult, Error>
                    do {
                        result = .success(try await worker.process(
                            asset: pendingAsset.asset,
                            photoLibraryStore: photoLibraryStore
                        ))
                    } catch {
                        result = .failure(error)
                    }
                    return FaceIndexingWorkerCompletion(
                        workerIndex: workerIndex,
                        offset: pendingAsset.offset,
                        asset: pendingAsset.asset,
                        result: result
                    )
                }
            }

            for workerIndex in 0..<activeWorkerCount {
                enqueue(workerIndex: workerIndex)
            }

            while let completion = await group.next() {
                completions.append(completion)
                enqueue(workerIndex: completion.workerIndex)
            }
        }

        return completions.sorted { $0.offset < $1.offset }
    }

    private func indexingRunContext(for reason: String) -> FaceIndexingRunContext {
        if reason.localizedCaseInsensitiveContains("background") {
            return .backgroundTask
        }
        if reason.localizedCaseInsensitiveContains("manual") {
            return .manualRefresh
        }
        return .foregroundAutomatic
    }

    private func process(asset: PhotoAssetSummary, photoLibraryStore: PhotoLibraryStore) async throws -> FaceAssetProcessingResult {
        let imageStartedAt = Date()
        guard let processingImage = await photoLibraryStore.imageForFaceProcessing(
            for: asset,
            maxDimension: configuration.faceProcessingMaxDimension,
            timeoutSeconds: configuration.faceImageRequestTimeoutSeconds
        ) else {
            Diagnostics.shared.log("Face indexing asset \(asset.id): no processing image after \(Self.durationText(since: imageStartedAt)); recording zero faces.")
            return FaceAssetProcessingResult(
                observations: [],
                imageRequestDuration: Date().timeIntervalSince(imageStartedAt),
                detectionDuration: 0,
                embeddingDuration: 0,
                detectedFaceCount: 0,
                skippedCropCount: 0,
                imageUnavailable: true,
                detectorBackendCounts: [:],
                yunetFailureCount: 0,
                visionFallbackCount: 0
            )
        }
        let imageRequestDuration = Date().timeIntervalSince(imageStartedAt)
        Diagnostics.shared.log("Face indexing asset \(asset.id): processing image ready in \(Self.durationText(since: imageStartedAt)), decoded \(processingImage.pixelWidth)x\(processingImage.pixelHeight).")

        let detectionStartedAt = Date()
        let detectedFaces = try await detectionService.detectFaces(
            in: processingImage.cgImage,
            orientation: processingImage.orientation
        )
        let detectionDuration = Date().timeIntervalSince(detectionStartedAt)
        Diagnostics.shared.log("Face indexing asset \(asset.id): detected \(detectedFaces.count) faces in \(Self.durationText(since: detectionStartedAt)); backends \(Self.backendCountsText(for: detectedFaces)).")

        var observations: [FaceObservationInput] = []
        observations.reserveCapacity(detectedFaces.count)
        var embeddingDuration: TimeInterval = 0
        var skippedCropCount = 0
        for (index, detectedFace) in detectedFaces.enumerated() {
            let faceStartedAt = Date()
            guard let crop = cropService.cropFace(
                from: processingImage.cgImage,
                detectedFace: detectedFace,
                configuration: configuration
            ) else {
                skippedCropCount += 1
                Diagnostics.shared.log("Face indexing asset \(asset.id): face \(index + 1)/\(detectedFaces.count) crop skipped.")
                continue
            }

            let embeddingStartedAt = Date()
            let embedding = try await embeddingService.embedding(
                for: crop.modelInputImage,
                debugIdentifier: "\(asset.id)_face\(index + 1)",
                debugMetadata: FaceEmbeddingDebugMetadata(
                    detectorBackend: detectedFace.backend,
                    detectorRow: detectedFace.detectorRow,
                    alignmentMethod: crop.alignmentMethod,
                    alignmentQuality: crop.alignmentQuality
                )
            )
            embeddingDuration += Date().timeIntervalSince(embeddingStartedAt)
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
                detectorBackend: detectedFace.backend,
                detectorRow: detectedFace.detectorRow,
                embedding: embedding,
                faceCropImageData: crop.avatarImageData
            ))
        }

        return FaceAssetProcessingResult(
            observations: observations,
            imageRequestDuration: imageRequestDuration,
            detectionDuration: detectionDuration,
            embeddingDuration: embeddingDuration,
            detectedFaceCount: detectedFaces.count,
            skippedCropCount: skippedCropCount,
            imageUnavailable: false,
            detectorBackendCounts: Self.backendCounts(for: detectedFaces),
            yunetFailureCount: detectionService.lastDetectionYuNetFailureCount,
            visionFallbackCount: detectionService.lastDetectionUsedVisionFallback ? 1 : 0
        )
    }

    private func saveExtractedObservations(_ observations: [FaceObservationInput], asset: PhotoAssetSummary) {
        let previousPersonIDs = Set((faceIDsByAssetID[asset.id] ?? []).compactMap { facesByID[$0]?.personID })
        removeFaces(forAssetID: asset.id)
        for personID in previousPersonIDs {
            deletePersonIfEmpty(personID)
        }

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

            if configuration.disableAutoClusteringWhenEmbeddingHealthSuspicious &&
                (healthReport.status == .suspiciousCollapsed || healthReport.status == .suspiciousNoisy) {
                faceRecognitionHealthMessage = "Face recognition embeddings look too similar. Picscry paused auto-grouping to avoid incorrect people."
                if !didLogSuspiciousEmbeddingHealthThisRun {
                    didLogSuspiciousEmbeddingHealthThisRun = true
                    Diagnostics.shared.log("Face embedding health suspicious: \(healthReport.status), sampleCount \(healthReport.sampleCount), median \(healthReport.medianSimilarity?.description ?? "unknown"), min \(healthReport.minSimilarity?.description ?? "unknown"), max \(healthReport.maxSimilarity?.description ?? "unknown"); clustering will keep extracted observations provisional.")
                }
            }

            let face = StoredFaceObservation(input: observation, personID: nil)
            facesByID[face.id] = face
            faceIDsByAssetID[asset.id, default: []].append(face.id)
            savedObservationCount += 1
            Diagnostics.shared.log("Face extraction asset \(asset.id), face \(observation.leftToRightIndex + 1): saved unclustered observation, detector \(observation.detectorBackend.rawValue), confidence \(observation.detectionConfidence).")
        }

        indexRecords[asset.id] = AssetIndexRecord(asset: asset, faceCount: savedObservationCount)
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
                guard let face = facesByID[faceID],
                      let personID = face.personID,
                      let person = persons[personID] else { return nil }
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
        let faceCount = faceIDsByPersonID[personID]?.count ?? 0
        return PersonSummary(
            id: person.id,
            displayName: person.displayName,
            isUnknown: person.isUnknown,
            photoCount: assetIDsByPersonID[personID]?.count ?? 0,
            faceCount: faceCount,
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
        let faces = sortedFaces(for: personID)
        guard !faces.isEmpty else {
            persons[personID] = person
            representativeQualityByPersonID[personID] = 0
            return
        }

        person.centroid = clusteringEngine.weightedCentroid(
            embeddings: faces.map { (embedding: $0.embedding, weight: max(score($0), 0.01)) }
        )
        let representative = faces.max(by: { score($0) < score($1) })
        representativeQualityByPersonID[personID] = representative.map(score) ?? 0
        person.representativeImageData = representative?.faceCropImageData
        if person.isProvisional && (assetIDsByPersonID[personID]?.count ?? 0) >= 2 {
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
        faceIDsByPersonID[personID]?.count ?? 0
    }

    private func assetIDs(for personID: UUID) -> Set<String> {
        assetIDsByPersonID[personID] ?? []
    }

    private func manuallyCorrectedFaceCount(for personID: UUID) -> Int {
        manuallyCorrectedFaceCountsByPersonID[personID] ?? 0
    }

    private func representativeQuality(for personID: UUID) -> Float {
        representativeQualityByPersonID[personID] ?? 0
    }

    private func clusterSnapshots() -> [FaceCluster] {
        persons.values.map { person in
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
    }

    private func sortedFaces(for personID: UUID) -> [StoredFaceObservation] {
        (faceIDsByPersonID[personID] ?? [])
            .compactMap { facesByID[$0] }
            .sorted {
                if $0.assetLocalIdentifier != $1.assetLocalIdentifier {
                    return $0.assetLocalIdentifier < $1.assetLocalIdentifier
                }
                return $0.leftToRightIndex < $1.leftToRightIndex
            }
    }

    private func addFaceToPersonIndex(_ face: StoredFaceObservation) {
        guard let personID = face.personID else { return }
        faceIDsByPersonID[personID, default: []].insert(face.id)
        assetIDsByPersonID[personID, default: []].insert(face.assetLocalIdentifier)
        if face.isManuallyCorrected {
            manuallyCorrectedFaceCountsByPersonID[personID, default: 0] += 1
        }
        representativeQualityByPersonID[personID] = max(
            representativeQualityByPersonID[personID] ?? 0,
            score(face)
        )
    }

    private func removeFaceFromPersonIndex(_ face: StoredFaceObservation) {
        guard let personID = face.personID else { return }
        faceIDsByPersonID[personID]?.remove(face.id)
        manuallyCorrectedFaceCountsByPersonID[personID] = max(
            0,
            (manuallyCorrectedFaceCountsByPersonID[personID] ?? 0) - (face.isManuallyCorrected ? 1 : 0)
        )
        rebuildAggregateIndex(for: personID)
    }

    private func moveFaceIndex(_ faceID: UUID, to personID: UUID, isManuallyCorrected: Bool? = nil) {
        guard var face = facesByID[faceID] else { return }
        let oldPersonID = face.personID
        removeFaceFromPersonIndex(face)
        face.personID = personID
        if let isManuallyCorrected {
            face.isManuallyCorrected = isManuallyCorrected
        }
        face.updatedAt = .now
        facesByID[faceID] = face
        addFaceToPersonIndex(face)
        if let oldPersonID {
            rebuildAggregateIndex(for: oldPersonID)
        }
        rebuildAggregateIndex(for: personID)
    }

    private func rebuildFaceIndexes() {
        faceIDsByPersonID = [:]
        assetIDsByPersonID = [:]
        manuallyCorrectedFaceCountsByPersonID = [:]
        representativeQualityByPersonID = [:]

        for face in facesByID.values {
            addFaceToPersonIndex(face)
        }
    }

    private func rebuildAggregateIndex(for personID: UUID) {
        guard let faceIDs = faceIDsByPersonID[personID], !faceIDs.isEmpty else {
            faceIDsByPersonID[personID] = nil
            assetIDsByPersonID[personID] = nil
            manuallyCorrectedFaceCountsByPersonID[personID] = nil
            representativeQualityByPersonID[personID] = nil
            return
        }

        var assetIDs = Set<String>()
        var manualCount = 0
        var bestQuality: Float = 0
        for faceID in faceIDs {
            guard let face = facesByID[faceID] else { continue }
            assetIDs.insert(face.assetLocalIdentifier)
            if face.isManuallyCorrected {
                manualCount += 1
            }
            bestQuality = max(bestQuality, score(face))
        }
        assetIDsByPersonID[personID] = assetIDs
        manuallyCorrectedFaceCountsByPersonID[personID] = manualCount
        representativeQualityByPersonID[personID] = bestQuality
    }

    private func rebuildAutomaticClusters(reason: String) {
        let candidates = facesByID.values
            .filter { face in
                let isEligiblePerson: Bool
                if let personID = face.personID {
                    guard let person = persons[personID] else { return false }
                    isEligiblePerson = person.isAutomaticCluster && person.isUnknown
                } else {
                    isEligiblePerson = true
                }
                return isEligiblePerson &&
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
        let oldAutomaticPersonIDs = Set(candidates.compactMap(\.personID))
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
                Diagnostics.shared.log("Face clustering rebuild using complete-link constrained clustering (\(reason)): observations \(nodes.count), threshold \(configuration.graphEdgeSimilarityThreshold).")
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

        var newAutomaticPersonIDs: [UUID] = []
        var assignmentCounts = FaceAssignmentDecisionCounts()
        for component in components {
            let componentFaces = component.nodeIDs.compactMap { facesByID[$0] }
            guard !componentFaces.isEmpty else { continue }

            let personID = UUID()
            newAutomaticPersonIDs.append(personID)
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
            assignmentCounts.newPerson += 1
        }
        assignmentDecisionCounts = assignmentCounts

        rebuildFaceIndexes()

        for personID in newAutomaticPersonIDs {
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
            clusteringAlgorithm: "complete-link constrained clustering",
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

    private func saveIndexingPerformanceDiagnostics(_ diagnostics: FaceIndexingPerformanceDiagnosticsSnapshot) {
        guard let url = Self.indexingPerformanceDiagnosticsURL() else { return }
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try JSONEncoder.pretty.encode(diagnostics)
            try data.write(to: url, options: .atomic)
            Diagnostics.shared.log("Saved face indexing performance diagnostics: photos \(diagnostics.processedImageCount)/\(diagnostics.pendingImageCount), imageRequestTotal \(diagnostics.imageRequestTotalDuration), detectionTotal \(diagnostics.detectionTotalDuration), embeddingTotal \(diagnostics.embeddingTotalDuration).")
        } catch {
            Diagnostics.shared.log("Failed to save face indexing performance diagnostics: \(error.localizedDescription)")
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
            if let face = facesByID[faceID] {
                removeFaceFromPersonIndex(face)
                facesByID[faceID] = nil
                if let personID = face.personID {
                    recompute(personID: personID)
                }
            }
        }
        faceIDsByAssetID[assetID] = []
    }

    private func deletePersonIfEmpty(_ personID: UUID) {
        if (faceIDsByPersonID[personID]?.isEmpty ?? true) {
            persons[personID] = nil
            faceIDsByPersonID[personID] = nil
            assetIDsByPersonID[personID] = nil
            manuallyCorrectedFaceCountsByPersonID[personID] = nil
            representativeQualityByPersonID[personID] = nil
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
            if (7...11).contains(snapshot.schemaVersion), FaceDatabaseSchema.currentVersion == 12 {
                persons = Dictionary(uniqueKeysWithValues: snapshot.persons.map { ($0.id, $0) })
                facesByID = Dictionary(uniqueKeysWithValues: snapshot.faces.map { ($0.id, $0) })
                faceIDsByAssetID = snapshot.faceIDsByAssetID
                indexRecords = snapshot.indexRecords
                rebuildFaceIndexes()
                rebuildAutomaticClusters(reason: "schema 12 YuNet extraction clustering migration")
                refreshPeople(persist: true, allowReorder: true)
                lastIndexingSummary = "Face detection and clustering were upgraded. Picscry rebuilt automatic unknown people while preserving named people and manual corrections."
                Diagnostics.shared.log("Migrated face database schema \(snapshot.schemaVersion) to 12 with optional unclustered observations. People \(persons.count), faces \(facesByID.count).")
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
            rebuildFaceIndexes()
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
        faceIDsByPersonID = [:]
        assetIDsByPersonID = [:]
        manuallyCorrectedFaceCountsByPersonID = [:]
        representativeQualityByPersonID = [:]
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

    private static func indexingPerformanceDiagnosticsURL() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first?
            .appendingPathComponent("FaceRecognition", isDirectory: true)
            .appendingPathComponent("face-indexing-performance.json", isDirectory: false)
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

    private func shouldPauseAfterUncleanIndexingExit(
        reason: String,
        fingerprint: String,
        forceRestart: Bool
    ) -> Bool {
        if forceRestart {
            clearActiveIndexingRunMarker()
            return false
        }
        guard indexingRunContext(for: reason) == .foregroundAutomatic else { return false }
        let defaults = UserDefaults.standard
        guard defaults.bool(forKey: Self.activeIndexingRunMarkerKey),
              defaults.string(forKey: Self.activeIndexingFingerprintKey) == fingerprint else {
            return false
        }

        let updatedAt = defaults.object(forKey: Self.activeIndexingUpdatedAtKey) as? Date ?? .distantPast
        if Date().timeIntervalSince(updatedAt) > 24 * 60 * 60 {
            clearActiveIndexingRunMarker()
            return false
        }
        return true
    }

    private func markIndexingRunActive(fingerprint: String, reason: String, processedImageCount: Int) {
        guard !fingerprint.isEmpty else { return }
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: Self.activeIndexingRunMarkerKey)
        defaults.set(fingerprint, forKey: Self.activeIndexingFingerprintKey)
        defaults.set(Date(), forKey: Self.activeIndexingUpdatedAtKey)
        defaults.set(reason, forKey: Self.activeIndexingReasonKey)
        defaults.set(processedImageCount, forKey: Self.activeIndexingProcessedCountKey)
    }

    private func clearActiveIndexingRunMarker() {
        let defaults = UserDefaults.standard
        defaults.removeObject(forKey: Self.activeIndexingRunMarkerKey)
        defaults.removeObject(forKey: Self.activeIndexingFingerprintKey)
        defaults.removeObject(forKey: Self.activeIndexingUpdatedAtKey)
        defaults.removeObject(forKey: Self.activeIndexingReasonKey)
        defaults.removeObject(forKey: Self.activeIndexingProcessedCountKey)
    }

    private func unclusteredObservationCount() -> Int {
        facesByID.values.filter { $0.personID == nil }.count
    }

    private static func backendCountsText(for faces: [DetectedFace]) -> String {
        let counts = backendCounts(for: faces)
        return FaceDetectionBackend.allCases
            .map { "\($0.rawValue)=\(counts[$0] ?? 0)" }
            .joined(separator: ", ")
    }

    private static func backendCounts(for faces: [DetectedFace]) -> [FaceDetectionBackend: Int] {
        Dictionary(grouping: faces, by: \.backend).mapValues(\.count)
    }

    private static func shortRunID(_ runID: UUID) -> String {
        String(runID.uuidString.prefix(8))
    }

    private static let activeIndexingRunMarkerKey = "FaceRecognition.activeIndexingRun"
    private static let activeIndexingFingerprintKey = "FaceRecognition.activeIndexingFingerprint"
    private static let activeIndexingUpdatedAtKey = "FaceRecognition.activeIndexingUpdatedAt"
    private static let activeIndexingReasonKey = "FaceRecognition.activeIndexingReason"
    private static let activeIndexingProcessedCountKey = "FaceRecognition.activeIndexingProcessedCount"
}

private enum FaceDatabaseSchema {
    static let currentVersion = 12
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

private struct PendingFaceIndexingAsset {
    let offset: Int
    let asset: PhotoAssetSummary
}

private struct FaceIndexingWorkerCompletion {
    let workerIndex: Int
    let offset: Int
    let asset: PhotoAssetSummary
    let result: Result<FaceAssetProcessingResult, Error>
}

private actor FaceIndexingWorker {
    private let id: Int
    private let configuration: FaceRecognitionConfiguration
    private let detectionService = FaceDetectionService()
    private let embeddingService: FaceEmbeddingService
    private let cropService = FaceCropService()

    init(id: Int, configuration: FaceRecognitionConfiguration) {
        self.id = id
        self.configuration = configuration
        embeddingService = FaceEmbeddingService(configuration: configuration)
    }

    func process(asset: PhotoAssetSummary, photoLibraryStore: PhotoLibraryStore) async throws -> FaceAssetProcessingResult {
        try Task.checkCancellation()
        let imageStartedAt = Date()
        guard let processingImage = await photoLibraryStore.imageForFaceProcessing(
            for: asset,
            maxDimension: configuration.faceProcessingMaxDimension,
            timeoutSeconds: configuration.faceImageRequestTimeoutSeconds
        ) else {
            Diagnostics.shared.log("Face worker \(id) asset \(asset.id): no processing image after \(Self.durationText(since: imageStartedAt)); recording zero faces.")
            return FaceAssetProcessingResult(
                observations: [],
                imageRequestDuration: Date().timeIntervalSince(imageStartedAt),
                detectionDuration: 0,
                embeddingDuration: 0,
                detectedFaceCount: 0,
                skippedCropCount: 0,
                imageUnavailable: true,
                detectorBackendCounts: [:],
                yunetFailureCount: 0,
                visionFallbackCount: 0
            )
        }

        let imageRequestDuration = Date().timeIntervalSince(imageStartedAt)
        let detectionStartedAt = Date()
        let detectedFaces = try await detectionService.detectFaces(
            in: processingImage.cgImage,
            orientation: processingImage.orientation
        )
        let detectionDuration = Date().timeIntervalSince(detectionStartedAt)
        Diagnostics.shared.log("Face worker \(id) asset \(asset.id): detected \(detectedFaces.count) faces in \(Self.durationText(since: detectionStartedAt)); backends \(Self.backendCountsText(for: detectedFaces)).")

        var observations: [FaceObservationInput] = []
        observations.reserveCapacity(detectedFaces.count)
        var embeddingDuration: TimeInterval = 0
        var skippedCropCount = 0

        for (index, detectedFace) in detectedFaces.enumerated() {
            try Task.checkCancellation()
            let faceStartedAt = Date()
            guard let crop = cropService.cropFace(
                from: processingImage.cgImage,
                detectedFace: detectedFace,
                configuration: configuration
            ) else {
                skippedCropCount += 1
                Diagnostics.shared.log("Face worker \(id) asset \(asset.id): face \(index + 1)/\(detectedFaces.count) crop skipped.")
                continue
            }

            let embeddingStartedAt = Date()
            let embedding = try await embeddingService.embedding(
                for: crop.modelInputImage,
                debugIdentifier: "\(asset.id)_face\(index + 1)",
                debugMetadata: FaceEmbeddingDebugMetadata(
                    detectorBackend: detectedFace.backend,
                    detectorRow: detectedFace.detectorRow,
                    alignmentMethod: crop.alignmentMethod,
                    alignmentQuality: crop.alignmentQuality
                )
            )
            embeddingDuration += Date().timeIntervalSince(embeddingStartedAt)
            Diagnostics.shared.log("Face worker \(id) crop diagnostics asset \(asset.id), face \(index + 1): modelCrop \(crop.modelInputImage.width)x\(crop.modelInputImage.height), alignment \(crop.alignmentMethod.rawValue), quality \(crop.alignmentQuality), confidence \(detectedFace.confidence), backend \(detectedFace.backend.rawValue).")
            Diagnostics.shared.log("Face worker \(id) asset \(asset.id): face \(index + 1)/\(detectedFaces.count) embedded in \(Self.durationText(since: embeddingStartedAt)); total face time \(Self.durationText(since: faceStartedAt)).")
            observations.append(FaceObservationInput(
                assetLocalIdentifier: asset.id,
                assetModificationDate: asset.modificationDate,
                assetPixelWidth: asset.pixelWidth,
                assetPixelHeight: asset.pixelHeight,
                normalizedBoundingBox: detectedFace.normalizedBoundingBox,
                leftToRightIndex: index,
                detectionConfidence: detectedFace.confidence,
                faceQuality: detectedFace.quality ?? crop.qualityScore,
                detectorBackend: detectedFace.backend,
                detectorRow: detectedFace.detectorRow,
                embedding: embedding,
                faceCropImageData: crop.avatarImageData
            ))
        }

        return FaceAssetProcessingResult(
            observations: observations,
            imageRequestDuration: imageRequestDuration,
            detectionDuration: detectionDuration,
            embeddingDuration: embeddingDuration,
            detectedFaceCount: detectedFaces.count,
            skippedCropCount: skippedCropCount,
            imageUnavailable: false,
            detectorBackendCounts: Self.backendCounts(for: detectedFaces),
            yunetFailureCount: detectionService.lastDetectionYuNetFailureCount,
            visionFallbackCount: detectionService.lastDetectionUsedVisionFallback ? 1 : 0
        )
    }

    private static func durationText(since startDate: Date) -> String {
        String(format: "%.2fs", Date().timeIntervalSince(startDate))
    }

    private static func backendCountsText(for faces: [DetectedFace]) -> String {
        let counts = backendCounts(for: faces)
        return FaceDetectionBackend.allCases
            .map { "\($0.rawValue)=\(counts[$0] ?? 0)" }
            .joined(separator: ", ")
    }

    private static func backendCounts(for faces: [DetectedFace]) -> [FaceDetectionBackend: Int] {
        Dictionary(grouping: faces, by: \.backend).mapValues(\.count)
    }
}

private struct FaceAssetProcessingResult {
    let observations: [FaceObservationInput]
    let imageRequestDuration: TimeInterval
    let detectionDuration: TimeInterval
    let embeddingDuration: TimeInterval
    let detectedFaceCount: Int
    let skippedCropCount: Int
    let imageUnavailable: Bool
    let detectorBackendCounts: [FaceDetectionBackend: Int]
    let yunetFailureCount: Int
    let visionFallbackCount: Int
}

private struct FaceIndexingPerformanceMetrics {
    let totalEligibleImageCount: Int
    let alreadyIndexedCount: Int
    let pendingImageCount: Int
    var processedImageCount = 0
    var imageUnavailableCount = 0
    var detectedFaceCount = 0
    var embeddedFaceCount = 0
    var skippedCropCount = 0
    var imageRequestTotalDuration: TimeInterval = 0
    var imageRequestMaxDuration: TimeInterval = 0
    var detectionTotalDuration: TimeInterval = 0
    var detectionMaxDuration: TimeInterval = 0
    var embeddingTotalDuration: TimeInterval = 0
    var embeddingMaxAssetDuration: TimeInterval = 0
    var detectorBackendCounts: [String: Int] = [:]
    var yunetFailureCount = 0
    var visionFallbackCount = 0
    var workerLimitHistory: [FaceIndexingWorkerLimitSample] = []

    mutating func record(_ result: FaceAssetProcessingResult) {
        processedImageCount += 1
        if result.imageUnavailable {
            imageUnavailableCount += 1
        }
        detectedFaceCount += result.detectedFaceCount
        embeddedFaceCount += result.observations.count
        skippedCropCount += result.skippedCropCount
        imageRequestTotalDuration += result.imageRequestDuration
        imageRequestMaxDuration = max(imageRequestMaxDuration, result.imageRequestDuration)
        detectionTotalDuration += result.detectionDuration
        detectionMaxDuration = max(detectionMaxDuration, result.detectionDuration)
        embeddingTotalDuration += result.embeddingDuration
        embeddingMaxAssetDuration = max(embeddingMaxAssetDuration, result.embeddingDuration)
        for (backend, count) in result.detectorBackendCounts {
            detectorBackendCounts[backend.rawValue, default: 0] += count
        }
        yunetFailureCount += result.yunetFailureCount
        visionFallbackCount += result.visionFallbackCount
    }

    mutating func recordWorkerLimit(_ workerLimit: Int, processedImageCount: Int) {
        let sample = FaceIndexingWorkerLimitSample(
            processedImageCount: processedImageCount,
            workerLimit: workerLimit,
            thermalState: FaceIndexingWorkerLimitSample.thermalStateText(ProcessInfo.processInfo.thermalState),
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
        if workerLimitHistory.last != sample {
            workerLimitHistory.append(sample)
        }
    }

    func snapshot(
        reason: String,
        completed: Bool,
        duration: TimeInterval,
        extractionDuration: TimeInterval,
        clusteringDuration: TimeInterval,
        totalFaces: Int,
        visiblePeople: Int,
        unclusteredObservationCount: Int
    ) -> FaceIndexingPerformanceDiagnosticsSnapshot {
        FaceIndexingPerformanceDiagnosticsSnapshot(
            generatedAt: Date(),
            reason: reason,
            completed: completed,
            totalDuration: duration,
            extractionDuration: extractionDuration,
            clusteringDuration: clusteringDuration,
            totalEligibleImageCount: totalEligibleImageCount,
            alreadyIndexedCount: alreadyIndexedCount,
            pendingImageCount: pendingImageCount,
            processedImageCount: processedImageCount,
            imageUnavailableCount: imageUnavailableCount,
            detectedFaceCount: detectedFaceCount,
            embeddedFaceCount: embeddedFaceCount,
            skippedCropCount: skippedCropCount,
            imageRequestTotalDuration: imageRequestTotalDuration,
            imageRequestAverageDuration: processedImageCount == 0 ? 0 : imageRequestTotalDuration / Double(processedImageCount),
            imageRequestMaxDuration: imageRequestMaxDuration,
            detectionTotalDuration: detectionTotalDuration,
            detectionAverageDuration: processedImageCount == 0 ? 0 : detectionTotalDuration / Double(processedImageCount),
            detectionMaxDuration: detectionMaxDuration,
            embeddingTotalDuration: embeddingTotalDuration,
            embeddingAverageDurationPerFace: embeddedFaceCount == 0 ? 0 : embeddingTotalDuration / Double(embeddedFaceCount),
            embeddingMaxAssetDuration: embeddingMaxAssetDuration,
            detectorBackendCounts: detectorBackendCounts,
            workerLimitHistory: workerLimitHistory,
            yunetFailureCount: yunetFailureCount,
            visionFallbackCount: visionFallbackCount,
            unclusteredObservationCount: unclusteredObservationCount,
            totalPersistedFaceCount: totalFaces,
            visiblePeopleCount: visiblePeople
        )
    }
}

private struct FaceIndexingPerformanceDiagnosticsSnapshot: Codable {
    let generatedAt: Date
    let reason: String
    let completed: Bool
    let totalDuration: TimeInterval
    let extractionDuration: TimeInterval
    let clusteringDuration: TimeInterval
    let totalEligibleImageCount: Int
    let alreadyIndexedCount: Int
    let pendingImageCount: Int
    let processedImageCount: Int
    let imageUnavailableCount: Int
    let detectedFaceCount: Int
    let embeddedFaceCount: Int
    let skippedCropCount: Int
    let imageRequestTotalDuration: TimeInterval
    let imageRequestAverageDuration: TimeInterval
    let imageRequestMaxDuration: TimeInterval
    let detectionTotalDuration: TimeInterval
    let detectionAverageDuration: TimeInterval
    let detectionMaxDuration: TimeInterval
    let embeddingTotalDuration: TimeInterval
    let embeddingAverageDurationPerFace: TimeInterval
    let embeddingMaxAssetDuration: TimeInterval
    let detectorBackendCounts: [String: Int]
    let workerLimitHistory: [FaceIndexingWorkerLimitSample]
    let yunetFailureCount: Int
    let visionFallbackCount: Int
    let unclusteredObservationCount: Int
    let totalPersistedFaceCount: Int
    let visiblePeopleCount: Int
}

private struct FaceIndexingWorkerLimitSample: Codable, Equatable {
    let processedImageCount: Int
    let workerLimit: Int
    let thermalState: String
    let lowPowerModeEnabled: Bool

    static func thermalStateText(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
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
    let clusteringAlgorithm: String
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
    let detectorBackend: FaceDetectionBackend?
    let detectorRow: [Float]?
    let embedding: [Float]
    let faceCropImageData: Data?
    var personID: UUID?
    let createdAt = Date()
    var updatedAt = Date()
    var isManuallyCorrected = false

    init(input: FaceObservationInput, personID: UUID?) {
        id = UUID()
        assetLocalIdentifier = input.assetLocalIdentifier
        assetModificationDate = input.assetModificationDate
        assetPixelWidth = input.assetPixelWidth
        assetPixelHeight = input.assetPixelHeight
        normalizedBoundingBox = input.normalizedBoundingBox
        leftToRightIndex = input.leftToRightIndex
        detectionConfidence = input.detectionConfidence
        faceQuality = input.faceQuality
        detectorBackend = input.detectorBackend
        detectorRow = input.detectorRow
        embedding = input.embedding
        faceCropImageData = input.faceCropImageData
        self.personID = personID
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case assetLocalIdentifier
        case assetModificationDate
        case assetPixelWidth
        case assetPixelHeight
        case normalizedBoundingBox
        case leftToRightIndex
        case detectionConfidence
        case faceQuality
        case detectorBackend
        case detectorRow
        case embedding
        case faceCropImageData
        case personID
        case updatedAt
        case isManuallyCorrected
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        assetLocalIdentifier = try container.decode(String.self, forKey: .assetLocalIdentifier)
        assetModificationDate = try container.decodeIfPresent(Date.self, forKey: .assetModificationDate)
        assetPixelWidth = try container.decode(Int.self, forKey: .assetPixelWidth)
        assetPixelHeight = try container.decode(Int.self, forKey: .assetPixelHeight)
        normalizedBoundingBox = try container.decode(CGRect.self, forKey: .normalizedBoundingBox)
        leftToRightIndex = try container.decode(Int.self, forKey: .leftToRightIndex)
        detectionConfidence = try container.decode(Float.self, forKey: .detectionConfidence)
        faceQuality = try container.decodeIfPresent(Float.self, forKey: .faceQuality)
        detectorBackend = try container.decodeIfPresent(FaceDetectionBackend.self, forKey: .detectorBackend)
        detectorRow = try container.decodeIfPresent([Float].self, forKey: .detectorRow)
        embedding = try container.decode([Float].self, forKey: .embedding)
        faceCropImageData = try container.decodeIfPresent(Data.self, forKey: .faceCropImageData)
        personID = try container.decodeIfPresent(UUID.self, forKey: .personID)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
        isManuallyCorrected = try container.decodeIfPresent(Bool.self, forKey: .isManuallyCorrected) ?? false
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

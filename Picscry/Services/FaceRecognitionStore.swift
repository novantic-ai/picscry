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

    private let configuration: FaceRecognitionConfiguration
    private let detectionService: FaceDetectionService
    private let embeddingService: FaceEmbeddingService
    private let cropService: FaceCropService
    private let clusteringEngine: FaceClusteringEngine

    private var persons: [UUID: StoredPerson] = [:]
    private var facesByID: [UUID: StoredFaceObservation] = [:]
    private var faceIDsByAssetID: [String: [UUID]] = [:]
    private var indexRecords: [String: AssetIndexRecord] = [:]
    private var indexingTask: Task<Void, Never>?

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        self.configuration = configuration
        detectionService = FaceDetectionService()
        embeddingService = FaceEmbeddingService(configuration: configuration)
        cropService = FaceCropService()
        clusteringEngine = FaceClusteringEngine(configuration: configuration)
        loadPersistedState()
    }

    func prepare(photoLibraryStore: PhotoLibraryStore) async {
        await startIndexing(photoLibraryStore: photoLibraryStore, reason: "foreground", waitForCompletion: false)
    }

    func runBackgroundIndexing(photoLibraryStore: PhotoLibraryStore) async {
        await startIndexing(photoLibraryStore: photoLibraryStore, reason: "background task", waitForCompletion: true)
    }

    func retry(photoLibraryStore: PhotoLibraryStore) async {
        await startIndexing(photoLibraryStore: photoLibraryStore, reason: "manual refresh", waitForCompletion: false)
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
        waitForCompletion: Bool
    ) async {
        indexingTask?.cancel()
        guard await embeddingService.isModelAvailable else {
            indexingState = .failed("Face recognition model could not be loaded.")
            currentIndexingMessage = nil
            Diagnostics.shared.log("Face indexing not started (\(reason)): model unavailable.")
            return
        }

        let assets = photoLibraryStore.assets.filter { !$0.isVideo }
        let liveIDs = Set(assets.map(\.id))
        removeDeletedAssets(liveIDs: liveIDs)

        let pending = assets.filter { assetNeedsIndex($0) }
        guard !pending.isEmpty else {
            refreshPeople()
            indexingState = .idle
            currentIndexingMessage = nil
            lastIndexingSummary = "No pending photos. \(facesByID.count) indexed faces across \(people.count) people."
            Diagnostics.shared.log("Face indexing skipped (\(reason)): no pending photos out of \(assets.count) image assets. Existing observations: \(facesByID.count).")
            scheduleBackgroundIndexing(reason: "no pending photos after \(reason)")
            return
        }

        Diagnostics.shared.log("Face indexing starting (\(reason)): \(pending.count) pending photos out of \(assets.count) image assets. Existing index records: \(indexRecords.count), people: \(persons.count), faces: \(facesByID.count).")
        scheduleBackgroundIndexing(reason: "indexing started from \(reason)")

        let task = Task { @MainActor in
            await index(pending: pending, photoLibraryStore: photoLibraryStore, reason: reason)
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
        person.updatedAt = .now
        persons[personID] = person
        refreshPeople()
        return .renamed
    }

    func confirmRenameMerge(sourcePersonID: UUID, targetPersonID: UUID, finalName: String) async {
        await mergePeople(sourcePersonID: sourcePersonID, targetPersonID: targetPersonID)
        if var target = persons[targetPersonID] {
            target.name = Self.normalizedName(finalName)
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
        persons[newPersonID] = StoredPerson(id: newPersonID, name: Self.normalizedName(name ?? ""))
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

    private func index(pending: [PhotoAssetSummary], photoLibraryStore: PhotoLibraryStore, reason: String) async {
        let indexingStartedAt = Date()
        indexingState = .indexing(processed: 0, total: pending.count)
        currentIndexingMessage = "Starting face indexing..."
        var processed = 0

        for (offset, asset) in pending.enumerated() {
            guard !Task.isCancelled else {
                indexingState = .paused
                currentIndexingMessage = "Face indexing paused at \(processed) of \(pending.count)."
                Diagnostics.shared.log("Face indexing paused (\(reason)) after \(processed) of \(pending.count) photos in \(Self.durationText(since: indexingStartedAt)).")
                return
            }

            let assetStartedAt = Date()
            currentIndexingMessage = "Processing photo \(offset + 1) of \(pending.count)"
            Diagnostics.shared.log("Face indexing asset \(offset + 1)/\(pending.count) started: \(asset.id), pixels \(asset.pixelWidth)x\(asset.pixelHeight), modified \(asset.modificationDate?.description ?? "unknown").")

            do {
                let observations = try await process(asset: asset, photoLibraryStore: photoLibraryStore)
                saveAndCluster(observations, asset: asset)
                Diagnostics.shared.log("Face indexing asset \(offset + 1)/\(pending.count) finished in \(Self.durationText(since: assetStartedAt)): \(asset.id), saved \(observations.count) face observations.")
            } catch {
                Diagnostics.shared.log("Face indexing failed for \(asset.id): \(error.localizedDescription)")
            }

            processed += 1
            indexingState = .indexing(processed: processed, total: pending.count)
            if processed == 1 || processed.isMultiple(of: 10) {
                currentIndexingMessage = "Indexed \(processed) of \(pending.count) photos"
                Diagnostics.shared.log("Face indexing progress (\(reason)): \(processed)/\(pending.count) photos in \(Self.durationText(since: indexingStartedAt)).")
            }
            if processed.isMultiple(of: configuration.indexingBatchSize) {
                refreshPeople()
                await Task.yield()
            }
        }

        refreshPeople()
        indexingState = .idle
        currentIndexingMessage = nil
        lastIndexingSummary = "Indexed \(pending.count) photos in \(Self.durationText(since: indexingStartedAt)). Found \(facesByID.count) faces across \(people.count) people."
        Diagnostics.shared.log("Face indexing finished (\(reason)) for \(pending.count) photos in \(Self.durationText(since: indexingStartedAt)) with \(facesByID.count) face observations across \(people.count) people.")
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
            let embedding = try await embeddingService.embedding(for: crop.modelInputImage)
            Diagnostics.shared.log("Face indexing asset \(asset.id): face \(index + 1)/\(detectedFaces.count) embedded in \(Self.durationText(since: embeddingStartedAt)); total face time \(Self.durationText(since: faceStartedAt)), confidence \(detectedFace.confidence).")
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

        for observation in observations {
            let clusters = persons.values.map {
                FaceCluster(personID: $0.id, centroid: $0.centroid, faceCount: faceCount(for: $0.id), name: $0.name)
            }
            let assignment = clusteringEngine.assignment(for: observation.embedding, clusters: clusters)
            let personID: UUID

            switch assignment.kind {
            case let .existingPerson(existingID, _):
                personID = existingID
            case .newPerson, .ambiguous:
                personID = UUID()
                persons[personID] = StoredPerson(id: personID, name: nil)
            }

            let face = StoredFaceObservation(input: observation, personID: personID)
            facesByID[face.id] = face
            faceIDsByAssetID[asset.id, default: []].append(face.id)
            recompute(personID: personID)
        }

        indexRecords[asset.id] = AssetIndexRecord(asset: asset, faceCount: observations.count)
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
            representativeFaceImageData: person.representativeImageData
        )
    }

    private func refreshPeople() {
        people = PeopleOrdering.sorted(persons.keys.compactMap(summary(for:)))
        savePersistedState()
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
            persons = Dictionary(uniqueKeysWithValues: snapshot.persons.map { ($0.id, $0) })
            facesByID = Dictionary(uniqueKeysWithValues: snapshot.faces.map { ($0.id, $0) })
            faceIDsByAssetID = snapshot.faceIDsByAssetID
            indexRecords = snapshot.indexRecords
            refreshPeople()
            Diagnostics.shared.log("Loaded persisted face database with \(persons.count) people and \(facesByID.count) faces.")
        } catch {
            Diagnostics.shared.log("Failed to load persisted face database: \(error.localizedDescription)")
        }
    }

    private func savePersistedState() {
        guard let url = Self.databaseURL() else { return }

        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let snapshot = FaceDatabaseSnapshot(
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
}

private struct FaceDatabaseSnapshot: Codable {
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

    var displayName: String { name ?? "Unknown" }
    var isUnknown: Bool { name == nil }
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

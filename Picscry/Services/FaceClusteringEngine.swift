import Foundation

struct ClusterCandidate: Equatable {
    let personID: UUID
    let similarity: Float
}

struct FaceClusterAssignment: Equatable {
    enum Kind: Equatable {
        case existingPerson(UUID, similarity: Float)
        case newPerson
        case ambiguous(bestPersonID: UUID, similarity: Float)
        case deferredProvisional(bestPersonID: UUID?, similarity: Float?)
    }

    let kind: Kind
}

struct FaceCluster: Equatable {
    let personID: UUID
    var centroid: [Float]
    var faceCount: Int
    var name: String?
}

struct FaceClusteringObservationNode: Equatable {
    let id: UUID
    let assetLocalIdentifier: String
    let embedding: [Float]
}

struct FaceClusteringComponent: Equatable {
    let nodeIDs: [UUID]
}

final class FaceClusteringEngine {
    let configuration: FaceRecognitionConfiguration

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        self.configuration = configuration
    }

    func bestCandidate(
        for embedding: [Float],
        clusters: [FaceCluster],
        excluding excludedPersonIDs: Set<UUID> = []
    ) -> ClusterCandidate? {
        clusters
            .filter { !excludedPersonIDs.contains($0.personID) }
            .compactMap { cluster -> ClusterCandidate? in
                guard !cluster.centroid.isEmpty else { return nil }
                return ClusterCandidate(
                    personID: cluster.personID,
                    similarity: embedding.cosineSimilarity(to: cluster.centroid)
                )
            }
            .max { $0.similarity < $1.similarity }
    }

    func rankedCandidates(
        for embedding: [Float],
        clusters: [FaceCluster],
        excluding excludedPersonIDs: Set<UUID> = []
    ) -> [ClusterCandidate] {
        clusters
            .filter { !excludedPersonIDs.contains($0.personID) }
            .compactMap { cluster -> ClusterCandidate? in
                guard !cluster.centroid.isEmpty else { return nil }
                return ClusterCandidate(
                    personID: cluster.personID,
                    similarity: embedding.cosineSimilarity(to: cluster.centroid)
                )
            }
            .sorted { $0.similarity > $1.similarity }
    }

    func assignment(
        for embedding: [Float],
        clusters: [FaceCluster],
        excluding excludedPersonIDs: Set<UUID> = []
    ) -> FaceClusterAssignment {
        let ranked = rankedCandidates(for: embedding, clusters: clusters, excluding: excludedPersonIDs)
        guard let best = ranked.first,
              let bestCluster = clusters.first(where: { $0.personID == best.personID }) else {
            return FaceClusterAssignment(kind: .newPerson)
        }
        let secondBest = ranked.dropFirst().first
        if let secondBest,
           best.similarity >= configuration.possibleMatchThreshold,
           secondBest.similarity >= configuration.possibleMatchThreshold,
           best.similarity - secondBest.similarity < configuration.minimumBestSecondBestMargin {
            return FaceClusterAssignment(kind: .deferredProvisional(
                bestPersonID: best.personID,
                similarity: best.similarity
            ))
        }

        let requiredAutoThreshold = bestCluster.faceCount <= 1
            ? configuration.singleSampleAutoMatchThreshold
            : configuration.autoMatchThreshold

        if best.similarity >= requiredAutoThreshold {
            return FaceClusterAssignment(kind: .existingPerson(best.personID, similarity: best.similarity))
        }

        if best.similarity >= configuration.possibleMatchThreshold {
            return FaceClusterAssignment(kind: .ambiguous(bestPersonID: best.personID, similarity: best.similarity))
        }

        return FaceClusterAssignment(kind: .newPerson)
    }

    func updatedCentroid(existing: [Float], existingCount: Int, adding newEmbedding: [Float]) -> [Float] {
        guard existing.count == newEmbedding.count, existingCount > 0 else {
            return newEmbedding.l2Normalized()
        }

        let weighted = zip(existing, newEmbedding).map { old, new in
            ((old * Float(existingCount)) + new) / Float(existingCount + 1)
        }
        return weighted.l2Normalized()
    }

    func mergedCentroid(clusters: [(centroid: [Float], count: Int)]) -> [Float] {
        guard let first = clusters.first, !first.centroid.isEmpty else { return [] }
        var accumulator = Array(repeating: Float(0), count: first.centroid.count)
        var totalCount = 0

        for cluster in clusters where cluster.centroid.count == accumulator.count && cluster.count > 0 {
            for index in accumulator.indices {
                accumulator[index] += cluster.centroid[index] * Float(cluster.count)
            }
            totalCount += cluster.count
        }

        guard totalCount > 0 else { return [] }
        return accumulator.map { $0 / Float(totalCount) }.l2Normalized()
    }

    func canAutomaticallyMerge(left: FaceCluster, right: FaceCluster, similarity: Float) -> Bool {
        let leftName = left.name?.trimmingCharacters(in: .whitespacesAndNewlines)
        let rightName = right.name?.trimmingCharacters(in: .whitespacesAndNewlines)

        if let leftName, !leftName.isEmpty,
           let rightName, !rightName.isEmpty,
           leftName.localizedCaseInsensitiveCompare(rightName) != .orderedSame {
            return false
        }

        if (leftName?.isEmpty == false) != (rightName?.isEmpty == false) {
            return similarity >= configuration.autoMatchThreshold
        }

        return similarity >= configuration.mergeThreshold
    }

    func constrainedComponents(
        for nodes: [FaceClusteringObservationNode],
        similarityThreshold: Float
    ) -> [FaceClusteringComponent] {
        guard !nodes.isEmpty else { return [] }
        var unionFind = CannotLinkUnionFind(nodes: nodes)

        for leftIndex in nodes.indices {
            for rightIndex in nodes.indices where rightIndex > leftIndex {
                guard nodes[leftIndex].assetLocalIdentifier != nodes[rightIndex].assetLocalIdentifier else {
                    continue
                }

                let similarity = nodes[leftIndex].embedding.cosineSimilarity(to: nodes[rightIndex].embedding)
                if similarity >= similarityThreshold {
                    unionFind.unionIfAllowed(leftIndex, rightIndex)
                }
            }
        }

        return unionFind.components()
    }

    func constrainedIncrementalComponents(
        for nodes: [FaceClusteringObservationNode],
        similarityThreshold: Float,
        singleSampleThreshold: Float
    ) -> [FaceClusteringComponent] {
        guard !nodes.isEmpty else { return [] }

        struct WorkingComponent {
            var nodeIndexes: [Int]
            var assetIDs: Set<String>
            var centroid: [Float]
        }

        var components: [WorkingComponent] = []
        for (index, node) in nodes.enumerated() {
            var ranked: [(componentIndex: Int, similarity: Float)] = []
            ranked.reserveCapacity(components.count)

            for componentIndex in components.indices where !components[componentIndex].assetIDs.contains(node.assetLocalIdentifier) {
                ranked.append((
                    componentIndex: componentIndex,
                    similarity: node.embedding.cosineSimilarity(to: components[componentIndex].centroid)
                ))
            }
            ranked.sort { $0.similarity > $1.similarity }

            let threshold: Float
            if let best = ranked.first,
               components[best.componentIndex].nodeIndexes.count <= 1 {
                threshold = singleSampleThreshold
            } else {
                threshold = similarityThreshold
            }
            let secondSimilarity = ranked.dropFirst().first?.similarity
            if let best = ranked.first,
               best.similarity >= threshold,
               isHealthyMargin(bestSimilarity: best.similarity, secondSimilarity: secondSimilarity) {
                let componentIndex = best.componentIndex
                components[componentIndex].nodeIndexes.append(index)
                components[componentIndex].assetIDs.insert(node.assetLocalIdentifier)
                components[componentIndex].centroid = updatedCentroid(
                    existing: components[componentIndex].centroid,
                    existingCount: components[componentIndex].nodeIndexes.count - 1,
                    adding: node.embedding
                )
            } else {
                components.append(WorkingComponent(
                    nodeIndexes: [index],
                    assetIDs: [node.assetLocalIdentifier],
                    centroid: node.embedding
                ))
            }
        }

        return components.map { component in
            FaceClusteringComponent(nodeIDs: component.nodeIndexes.map { nodes[$0].id })
        }
    }

    func mergedConstrainedComponents(
        from initialComponents: [FaceClusteringComponent],
        nodes: [FaceClusteringObservationNode],
        mergeThreshold: Float
    ) -> [FaceClusteringComponent] {
        guard !initialComponents.isEmpty else { return [] }
        let nodeIndexByID = Dictionary(uniqueKeysWithValues: nodes.enumerated().map { ($0.element.id, $0.offset) })

        struct WorkingComponent {
            var nodeIndexes: [Int]
            var assetIDs: Set<String>
            var centroid: [Float]
        }

        var components = initialComponents.compactMap { component -> WorkingComponent? in
            let indexes = component.nodeIDs.compactMap { nodeIndexByID[$0] }
            guard !indexes.isEmpty else { return nil }
            let clusters = indexes.map { (centroid: nodes[$0].embedding, count: 1) }
            return WorkingComponent(
                nodeIndexes: indexes,
                assetIDs: Set(indexes.map { nodes[$0].assetLocalIdentifier }),
                centroid: mergedCentroid(clusters: clusters)
            )
        }

        var didMerge = true
        while didMerge {
            didMerge = false
            var bestPair: (left: Int, right: Int, similarity: Float)?

            for leftIndex in components.indices {
                for rightIndex in components.indices where rightIndex > leftIndex {
                    guard components[leftIndex].assetIDs.isDisjoint(with: components[rightIndex].assetIDs) else {
                        continue
                    }

                    let similarity = components[leftIndex].centroid.cosineSimilarity(to: components[rightIndex].centroid)
                    guard similarity >= mergeThreshold else { continue }
                    if bestPair == nil || similarity > bestPair!.similarity {
                        bestPair = (leftIndex, rightIndex, similarity)
                    }
                }
            }

            if let bestPair {
                let left = bestPair.left
                let right = bestPair.right
                let mergedIndexes = components[left].nodeIndexes + components[right].nodeIndexes
                let mergedClusters = mergedIndexes.map { (centroid: nodes[$0].embedding, count: 1) }
                components[left].nodeIndexes = mergedIndexes
                components[left].assetIDs.formUnion(components[right].assetIDs)
                components[left].centroid = mergedCentroid(clusters: mergedClusters)
                components.remove(at: right)
                didMerge = true
            }
        }

        return components.map { component in
            FaceClusteringComponent(nodeIDs: component.nodeIndexes.sorted().map { nodes[$0].id })
        }
    }

    private func isHealthyMargin(bestSimilarity: Float, secondSimilarity: Float?) -> Bool {
        guard let secondSimilarity else { return true }
        guard secondSimilarity >= configuration.possibleMatchThreshold else { return true }
        return bestSimilarity - secondSimilarity >= configuration.minimumBestSecondBestMargin
    }
}

private struct CannotLinkUnionFind {
    private var parents: [Int]
    private var ranks: [Int]
    private var assetIDsByRoot: [Set<String>]
    private let nodes: [FaceClusteringObservationNode]

    init(nodes: [FaceClusteringObservationNode]) {
        self.nodes = nodes
        parents = Array(nodes.indices)
        ranks = Array(repeating: 0, count: nodes.count)
        assetIDsByRoot = nodes.map { [$0.assetLocalIdentifier] }
    }

    mutating func unionIfAllowed(_ leftIndex: Int, _ rightIndex: Int) {
        let leftRoot = find(leftIndex)
        let rightRoot = find(rightIndex)
        guard leftRoot != rightRoot else { return }
        guard assetIDsByRoot[leftRoot].isDisjoint(with: assetIDsByRoot[rightRoot]) else { return }

        if ranks[leftRoot] < ranks[rightRoot] {
            parents[leftRoot] = rightRoot
            assetIDsByRoot[rightRoot].formUnion(assetIDsByRoot[leftRoot])
        } else if ranks[leftRoot] > ranks[rightRoot] {
            parents[rightRoot] = leftRoot
            assetIDsByRoot[leftRoot].formUnion(assetIDsByRoot[rightRoot])
        } else {
            parents[rightRoot] = leftRoot
            ranks[leftRoot] += 1
            assetIDsByRoot[leftRoot].formUnion(assetIDsByRoot[rightRoot])
        }
    }

    mutating func components() -> [FaceClusteringComponent] {
        var indexesByRoot: [Int: [Int]] = [:]
        for index in nodes.indices {
            indexesByRoot[find(index), default: []].append(index)
        }

        return indexesByRoot.values
            .map { indexes in
                FaceClusteringComponent(nodeIDs: indexes.sorted().map { nodes[$0].id })
            }
            .sorted { lhs, rhs in
                guard let leftFirst = lhs.nodeIDs.first,
                      let rightFirst = rhs.nodeIDs.first,
                      let leftIndex = nodes.firstIndex(where: { $0.id == leftFirst }),
                      let rightIndex = nodes.firstIndex(where: { $0.id == rightFirst }) else {
                    return lhs.nodeIDs.count > rhs.nodeIDs.count
                }
                return leftIndex < rightIndex
            }
    }

    private mutating func find(_ index: Int) -> Int {
        if parents[index] != index {
            parents[index] = find(parents[index])
        }
        return parents[index]
    }
}

struct FaceEmbeddingHealthReport: Equatable {
    enum Status: Equatable {
        case warmingUp
        case healthy
        case suspiciousCollapsed
        case suspiciousNoisy
    }

    let status: Status
    let sampleCount: Int
    let pairCount: Int
    let minSimilarity: Float?
    let medianSimilarity: Float?
    let maxSimilarity: Float?
}

final class FaceEmbeddingHealthMonitor {
    private var samples: [[Float]] = []
    private let maxSamples: Int
    private let collapsedMedianThreshold: Float
    private let collapsedMinimumThreshold: Float

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        maxSamples = configuration.embeddingCalibrationSampleCount
        collapsedMedianThreshold = configuration.collapsedEmbeddingMedianSimilarityThreshold
        collapsedMinimumThreshold = configuration.collapsedEmbeddingMinimumSimilarityThreshold
    }

    func add(_ embedding: [Float]) {
        guard samples.count < maxSamples, !embedding.isEmpty else { return }
        samples.append(embedding)
    }

    func reset() {
        samples.removeAll(keepingCapacity: true)
    }

    func report() -> FaceEmbeddingHealthReport {
        guard samples.count >= 2 else {
            return FaceEmbeddingHealthReport(
                status: .warmingUp,
                sampleCount: samples.count,
                pairCount: 0,
                minSimilarity: nil,
                medianSimilarity: nil,
                maxSimilarity: nil
            )
        }

        var similarities: [Float] = []
        similarities.reserveCapacity((samples.count * (samples.count - 1)) / 2)
        for leftIndex in samples.indices {
            for rightIndex in samples.indices where rightIndex > leftIndex {
                similarities.append(samples[leftIndex].cosineSimilarity(to: samples[rightIndex]))
            }
        }
        similarities.sort()

        let minSimilarity = similarities.first
        let medianSimilarity = similarities[similarities.count / 2]
        let maxSimilarity = similarities.last
        let status: FaceEmbeddingHealthReport.Status
        if samples.count < maxSamples {
            status = .warmingUp
        } else if medianSimilarity > collapsedMedianThreshold,
                  let minSimilarity,
                  minSimilarity > collapsedMinimumThreshold {
            status = .suspiciousCollapsed
        } else if let maxSimilarity, maxSimilarity < 0.20 {
            status = .suspiciousNoisy
        } else {
            status = .healthy
        }

        return FaceEmbeddingHealthReport(
            status: status,
            sampleCount: samples.count,
            pairCount: similarities.count,
            minSimilarity: minSimilarity,
            medianSimilarity: medianSimilarity,
            maxSimilarity: maxSimilarity
        )
    }
}

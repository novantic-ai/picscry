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
    }

    let kind: Kind
}

struct FaceCluster: Equatable {
    let personID: UUID
    var centroid: [Float]
    var faceCount: Int
    var name: String?
}

final class FaceClusteringEngine {
    let configuration: FaceRecognitionConfiguration

    init(configuration: FaceRecognitionConfiguration = FaceRecognitionConfiguration()) {
        self.configuration = configuration
    }

    func bestCandidate(for embedding: [Float], clusters: [FaceCluster]) -> ClusterCandidate? {
        clusters
            .compactMap { cluster -> ClusterCandidate? in
                guard !cluster.centroid.isEmpty else { return nil }
                return ClusterCandidate(
                    personID: cluster.personID,
                    similarity: embedding.cosineSimilarity(to: cluster.centroid)
                )
            }
            .max { $0.similarity < $1.similarity }
    }

    func assignment(for embedding: [Float], clusters: [FaceCluster]) -> FaceClusterAssignment {
        guard let best = bestCandidate(for: embedding, clusters: clusters) else {
            return FaceClusterAssignment(kind: .newPerson)
        }

        if best.similarity >= configuration.autoMatchThreshold {
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
}

import CoreGraphics
import ImageIO
import Vision

struct DetectedFace {
    let normalizedBoundingBox: CGRect
    let confidence: Float
    let quality: Float?
    let landmarks: VNFaceLandmarks2D?
}

final class FaceDetectionService {
    func detectFaces(in cgImage: CGImage, orientation: CGImagePropertyOrientation) async throws -> [DetectedFace] {
        try await withCheckedThrowingContinuation { continuation in
            Task.detached(priority: .utility) {
                let rectangleRequest = VNDetectFaceRectanglesRequest()

                if #available(iOS 15.0, *) {
                    rectangleRequest.revision = VNDetectFaceRectanglesRequestRevision3
                }

                let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
                do {
                    try handler.perform([rectangleRequest])
                    let rectangles = (rectangleRequest.results ?? []).filter { $0.confidence > 0.3 }

                    let landmarkRequest = VNDetectFaceLandmarksRequest()
                    landmarkRequest.inputFaceObservations = rectangles
                    try handler.perform([landmarkRequest])

                    let landmarkResults = landmarkRequest.results ?? rectangles
                    let faces = landmarkResults
                        .map {
                            DetectedFace(
                                normalizedBoundingBox: $0.boundingBox,
                                confidence: $0.confidence,
                                quality: $0.faceCaptureQuality?.floatValue,
                                landmarks: $0.landmarks
                            )
                        }
                        .sorted { $0.normalizedBoundingBox.minX < $1.normalizedBoundingBox.minX }

                    continuation.resume(returning: faces)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

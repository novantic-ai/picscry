import CoreGraphics
import UIKit

struct FaceCropResult {
    let modelInputImage: CGImage
    let avatarImageData: Data?
    let qualityScore: Float
    let alignmentQuality: Float
    let alignmentMethod: FaceAlignmentMethod
}

enum FaceAlignmentMethod: String, Codable {
    case opencvFivePointSVD
    case eyeMouthSimilarity
    case rectangleFallback
}

final class FaceCropService {
    func cropFace(
        from cgImage: CGImage,
        detectedFace: DetectedFace,
        configuration: FaceRecognitionConfiguration
    ) -> FaceCropResult? {
        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        let pixelRect = Self.pixelRect(forVisionBoundingBox: detectedFace.normalizedBoundingBox, imageSize: imageSize)
        guard pixelRect.width >= configuration.minimumFacePixelSize,
              pixelRect.height >= configuration.minimumFacePixelSize else {
            return nil
        }

        let modelRect = Self.paddedRect(
            pixelRect,
            paddingRatio: configuration.modelInputPaddingRatio,
            imageSize: imageSize
        )
        guard let initialModelCrop = cgImage.cropping(to: modelRect.integral) else { return nil }
        let alignedCrop = Self.alignedModelCrop(
            from: cgImage,
            detectedFace: detectedFace,
            imageSize: imageSize
        )
        let modelCrop = alignedCrop?.image ?? Self.resizedSquareCrop(initialModelCrop, size: 112) ?? initialModelCrop
        let alignmentMethod = alignedCrop?.method ?? .rectangleFallback

        let avatarRect = Self.squareAvatarRect(
            around: pixelRect,
            paddingRatio: configuration.avatarPaddingRatio,
            imageSize: imageSize
        )
        let avatarCrop = cgImage.cropping(to: avatarRect.integral) ?? initialModelCrop

        let areaRatio = Float((pixelRect.width * pixelRect.height) / max(imageSize.width * imageSize.height, 1))
        let areaScore = min(max(areaRatio * 20, 0), 1)
        let quality = (detectedFace.confidence * 0.5) + (areaScore * 0.3) + ((detectedFace.quality ?? 0.5) * 0.2)

        return FaceCropResult(
            modelInputImage: modelCrop,
            avatarImageData: Self.jpegDataPreservingAspectRatio(
                from: avatarCrop,
                size: configuration.representativeThumbnailSize
            ),
            qualityScore: quality,
            alignmentQuality: alignedCrop?.quality ?? 0,
            alignmentMethod: alignmentMethod
        )
    }

    static func pixelRect(forVisionBoundingBox box: CGRect, imageSize: CGSize) -> CGRect {
        CGRect(
            x: box.minX * imageSize.width,
            y: (1 - box.maxY) * imageSize.height,
            width: box.width * imageSize.width,
            height: box.height * imageSize.height
        )
    }

    private static func paddedRect(_ rect: CGRect, paddingRatio: CGFloat, imageSize: CGSize) -> CGRect {
        let paddingX = rect.width * paddingRatio
        let paddingY = rect.height * paddingRatio
        return rect
            .insetBy(dx: -paddingX, dy: -paddingY)
            .intersection(CGRect(origin: .zero, size: imageSize))
    }

    static func squareAvatarRect(around rect: CGRect, paddingRatio: CGFloat, imageSize: CGSize) -> CGRect {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let side = max(rect.width, rect.height) * (1 + (paddingRatio * 2))
        var square = CGRect(
            x: center.x - side / 2,
            y: center.y - side / 2,
            width: side,
            height: side
        )

        if square.minX < 0 { square.origin.x = 0 }
        if square.minY < 0 { square.origin.y = 0 }
        if square.maxX > imageSize.width { square.origin.x = max(0, imageSize.width - square.width) }
        if square.maxY > imageSize.height { square.origin.y = max(0, imageSize.height - square.height) }

        square = square.intersection(CGRect(origin: .zero, size: imageSize))
        let finalSide = min(square.width, square.height)
        return CGRect(
            x: square.midX - finalSide / 2,
            y: square.midY - finalSide / 2,
            width: finalSide,
            height: finalSide
        )
        .intersection(CGRect(origin: .zero, size: imageSize))
    }

    private static func jpegDataPreservingAspectRatio(from cgImage: CGImage, size: CGFloat) -> Data? {
        let image = UIImage(cgImage: cgImage)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true

        let target = CGSize(width: size, height: size)
        let renderer = UIGraphicsImageRenderer(size: target, format: format)
        let rendered = renderer.image { context in
            UIColor.systemBackground.setFill()
            context.fill(CGRect(origin: .zero, size: target))

            let sourceSize = CGSize(width: cgImage.width, height: cgImage.height)
            guard sourceSize.width > 0, sourceSize.height > 0 else { return }

            let scale = max(target.width / sourceSize.width, target.height / sourceSize.height)
            let drawSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
            let drawRect = CGRect(
                x: (target.width - drawSize.width) / 2,
                y: (target.height - drawSize.height) / 2,
                width: drawSize.width,
                height: drawSize.height
            )

            image.draw(in: drawRect)
        }
        return rendered.jpegData(compressionQuality: 0.9)
    }

    private static func alignedModelCrop(
        from cgImage: CGImage,
        detectedFace: DetectedFace,
        imageSize: CGSize
    ) -> (image: CGImage, quality: Float, method: FaceAlignmentMethod)? {
        if let fivePointCrop = opencvFivePointSVDAlignedModelCrop(
            from: cgImage,
            detectedFace: detectedFace,
            imageSize: imageSize
        ) {
            return fivePointCrop
        }
        return eyeMouthAlignedModelCrop(
            from: cgImage,
            detectedFace: detectedFace,
            imageSize: imageSize
        )
    }

    private static func eyeMouthAlignedModelCrop(
        from cgImage: CGImage,
        detectedFace: DetectedFace,
        imageSize: CGSize
    ) -> (image: CGImage, quality: Float, method: FaceAlignmentMethod)? {
        guard let landmarks = detectedFace.landmarks else {
            return nil
        }

        let leftCenter = landmarks.leftEye
        let rightCenter = landmarks.rightEye
        let mouthCenter = CGPoint(
            x: (landmarks.leftMouth.x + landmarks.rightMouth.x) / 2,
            y: (landmarks.leftMouth.y + landmarks.rightMouth.y) / 2
        )
        let sourceEyeDelta = CGPoint(x: rightCenter.x - leftCenter.x, y: rightCenter.y - leftCenter.y)
        let sourceEyeDistance = hypot(sourceEyeDelta.x, sourceEyeDelta.y)
        guard sourceEyeDistance.isFinite, sourceEyeDistance > 1 else { return nil }

        let targetLeftEye = CGPoint(x: 38.2946, y: 51.6963)
        let targetRightEye = CGPoint(x: 73.5318, y: 51.5014)
        let targetMouthCenter = CGPoint(x: 56.1396, y: 92.2848)
        let targetEyeDelta = CGPoint(x: targetRightEye.x - targetLeftEye.x, y: targetRightEye.y - targetLeftEye.y)
        let targetEyeDistance = hypot(targetEyeDelta.x, targetEyeDelta.y)
        let sourceAngle = atan2(sourceEyeDelta.y, sourceEyeDelta.x)
        let targetAngle = atan2(targetEyeDelta.y, targetEyeDelta.x)
        let scale = targetEyeDistance / sourceEyeDistance
        guard scale.isFinite, scale > 0 else { return nil }

        let sourceEyeMid = CGPoint(x: (leftCenter.x + rightCenter.x) / 2, y: (leftCenter.y + rightCenter.y) / 2)
        let targetEyeMid = CGPoint(x: (targetLeftEye.x + targetRightEye.x) / 2, y: (targetLeftEye.y + targetRightEye.y) / 2)
        let rotation = targetAngle - sourceAngle

        var transform = CGAffineTransform.identity
        transform = transform.translatedBy(x: targetEyeMid.x, y: targetEyeMid.y)
        transform = transform.rotated(by: rotation)
        transform = transform.scaledBy(x: scale, y: scale)
        transform = transform.translatedBy(x: -sourceEyeMid.x, y: -sourceEyeMid.y)

        let width = 112
        let height = 112
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &pixels,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
              ) else {
            return nil
        }

        context.interpolationQuality = .high
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.concatenate(transform)
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        guard let rendered = context.makeImage() else { return nil }

        let transformedMouth = mouthCenter.applying(transform)
        let mouthError = hypot(transformedMouth.x - targetMouthCenter.x, transformedMouth.y - targetMouthCenter.y)
        let quality = Float(max(0, min(1, 1 - (mouthError / 56))))
        return (rendered, quality, .eyeMouthSimilarity)
    }

    private static func opencvFivePointSVDAlignedModelCrop(
        from cgImage: CGImage,
        detectedFace: DetectedFace,
        imageSize: CGSize
    ) -> (image: CGImage, quality: Float, method: FaceAlignmentMethod)? {
        guard let landmarks = detectedFace.landmarks else {
            return nil
        }

        let candidates: [(label: String, points: [CGPoint])]
        if detectedFace.backend == .visionFallback {
            candidates = FaceLandmarkFivePointSet(landmarks: landmarks).candidateSourceOrders()
        } else {
            candidates = [("yunetOpenCVOrder", landmarks.sfaceSourcePoints)]
        }
        var bestCandidate: (transform: CGAffineTransform, error: CGFloat, label: String)?

        for candidate in candidates {
            guard let transform = sfaceSimilarityTransform(
                source: candidate.points,
                destination: opencvSFaceDestinationLandmarks
            ) else {
                continue
            }

            let error = meanReprojectionError(
                source: candidate.points,
                destination: opencvSFaceDestinationLandmarks,
                transform: transform
            )
            if bestCandidate == nil || error < bestCandidate!.error {
                bestCandidate = (transform, error, candidate.label)
            }
        }

        guard let bestCandidate else { return nil }
        guard let rendered = renderAlignedImage(cgImage, transform: bestCandidate.transform, size: 112) else {
            return nil
        }

        let quality = Float(max(0, min(1, 1 - (bestCandidate.error / 56))))
        Diagnostics.shared.log("Face alignment mapping selected: \(bestCandidate.label), reprojectionError \(bestCandidate.error).")
        return (rendered, quality, .opencvFivePointSVD)
    }

    private static func renderAlignedImage(_ cgImage: CGImage, transform: CGAffineTransform, size: Int) -> CGImage? {
        let bytesPerRow = size * 4
        var pixels = [UInt8](repeating: 0, count: size * bytesPerRow)
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

        context.interpolationQuality = .high
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        context.concatenate(transform)
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        return context.makeImage()
    }

    private static func resizedSquareCrop(_ crop: CGImage, size: Int) -> CGImage? {
        let bytesPerRow = size * 4
        var pixels = [UInt8](repeating: 0, count: size * bytesPerRow)
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

        context.interpolationQuality = .high
        context.draw(crop, in: CGRect(x: 0, y: 0, width: size, height: size))
        return context.makeImage()
    }

    private static let opencvSFaceDestinationLandmarks: [CGPoint] = [
        CGPoint(x: 38.2946, y: 51.6963),
        CGPoint(x: 73.5318, y: 51.5014),
        CGPoint(x: 56.0252, y: 71.7366),
        CGPoint(x: 41.5493, y: 92.3655),
        CGPoint(x: 70.7299, y: 92.2041)
    ]

    private static func sfaceSimilarityTransform(
        source: [CGPoint],
        destination: [CGPoint]
    ) -> CGAffineTransform? {
        guard source.count == destination.count, source.count >= 2 else { return nil }

        let count = CGFloat(source.count)
        let sourceMean = source.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }.scaled(by: 1 / count)
        let destinationMean = destination.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }.scaled(by: 1 / count)

        var denominator: CGFloat = 0
        var aNumerator: CGFloat = 0
        var bNumerator: CGFloat = 0

        for (sourcePoint, destinationPoint) in zip(source, destination) {
            let sx = sourcePoint.x - sourceMean.x
            let sy = sourcePoint.y - sourceMean.y
            let dx = destinationPoint.x - destinationMean.x
            let dy = destinationPoint.y - destinationMean.y

            denominator += (sx * sx) + (sy * sy)
            aNumerator += (dx * sx) + (dy * sy)
            bNumerator += (dy * sx) - (dx * sy)
        }

        guard denominator.isFinite, denominator > .ulpOfOne else { return nil }
        let a = aNumerator / denominator
        let b = bNumerator / denominator
        guard a.isFinite, b.isFinite else { return nil }

        let tx = destinationMean.x - (a * sourceMean.x) + (b * sourceMean.y)
        let ty = destinationMean.y - (b * sourceMean.x) - (a * sourceMean.y)
        guard tx.isFinite, ty.isFinite else { return nil }

        return CGAffineTransform(a: a, b: b, c: -b, d: a, tx: tx, ty: ty)
    }

    private static func meanReprojectionError(
        source: [CGPoint],
        destination: [CGPoint],
        transform: CGAffineTransform
    ) -> CGFloat {
        guard source.count == destination.count, !source.isEmpty else { return .greatestFiniteMagnitude }
        let total = zip(source, destination).reduce(CGFloat(0)) { partial, pair in
            let transformed = pair.0.applying(transform)
            return partial + hypot(transformed.x - pair.1.x, transformed.y - pair.1.y)
        }
        return total / CGFloat(source.count)
    }
}

private struct FaceLandmarkFivePointSet {
    let rightEye: CGPoint
    let leftEye: CGPoint
    let noseTip: CGPoint
    let rightMouth: CGPoint
    let leftMouth: CGPoint

    init(landmarks: FaceLandmarkFivePoint) {
        rightEye = landmarks.rightEye
        leftEye = landmarks.leftEye
        noseTip = landmarks.noseTip
        rightMouth = landmarks.rightMouth
        leftMouth = landmarks.leftMouth
    }

    func candidateSourceOrders() -> [(label: String, points: [CGPoint])] {
        [
            ("candidateA", [rightEye, leftEye, noseTip, rightMouth, leftMouth]),
            ("candidateB", [leftEye, rightEye, noseTip, rightMouth, leftMouth]),
            ("candidateC", [rightEye, leftEye, noseTip, leftMouth, rightMouth]),
            ("candidateD", [leftEye, rightEye, noseTip, leftMouth, rightMouth])
        ]
    }
}

private extension CGPoint {
    func scaled(by scale: CGFloat) -> CGPoint {
        CGPoint(x: x * scale, y: y * scale)
    }
}

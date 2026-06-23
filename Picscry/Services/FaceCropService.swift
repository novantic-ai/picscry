import CoreGraphics
import UIKit
import Vision

struct FaceCropResult {
    let modelInputImage: CGImage
    let avatarImageData: Data?
    let qualityScore: Float
    let alignmentQuality: Float
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
            alignmentQuality: alignedCrop?.quality ?? 0
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
    ) -> (image: CGImage, quality: Float)? {
        guard let landmarks = detectedFace.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let leftCenter = landmarkCenter(leftEye, faceBox: detectedFace.normalizedBoundingBox, imageSize: imageSize),
              let rightCenter = landmarkCenter(rightEye, faceBox: detectedFace.normalizedBoundingBox, imageSize: imageSize),
              let mouthCenter = mouthCenter(from: landmarks, faceBox: detectedFace.normalizedBoundingBox, imageSize: imageSize) else {
            return nil
        }

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
        return (rendered, quality)
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

    private static func mouthCenter(
        from landmarks: VNFaceLandmarks2D,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> CGPoint? {
        if let outerLips = landmarks.outerLips, outerLips.pointCount > 0 {
            return landmarkCenter(outerLips, faceBox: faceBox, imageSize: imageSize)
        }
        if let innerLips = landmarks.innerLips, innerLips.pointCount > 0 {
            return landmarkCenter(innerLips, faceBox: faceBox, imageSize: imageSize)
        }
        return nil
    }

    private static func landmarkCenter(
        _ region: VNFaceLandmarkRegion2D,
        faceBox: CGRect,
        imageSize: CGSize
    ) -> CGPoint? {
        guard region.pointCount > 0 else { return nil }
        let points = region.normalizedPoints
        let sum = points.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + CGFloat(point.x), y: partial.y + CGFloat(point.y))
        }
        let average = CGPoint(
            x: sum.x / CGFloat(region.pointCount),
            y: sum.y / CGFloat(region.pointCount)
        )
        return CGPoint(
            x: (faceBox.minX + (average.x * faceBox.width)) * imageSize.width,
            y: (1 - (faceBox.minY + (average.y * faceBox.height))) * imageSize.height
        )
    }
}

import CoreGraphics
import UIKit
import Vision

struct FaceCropResult {
    let modelInputImage: CGImage
    let avatarImageData: Data?
    let qualityScore: Float
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

        let paddedRect = Self.paddedRect(pixelRect, paddingRatio: 0.25, imageSize: imageSize)
        guard let initialCrop = cgImage.cropping(to: paddedRect.integral) else { return nil }
        let crop = Self.eyeAlignedCrop(
            initialCrop,
            detectedFace: detectedFace,
            paddedRect: paddedRect,
            imageSize: imageSize
        ) ?? initialCrop

        let areaRatio = Float((pixelRect.width * pixelRect.height) / max(imageSize.width * imageSize.height, 1))
        let areaScore = min(max(areaRatio * 20, 0), 1)
        let quality = (detectedFace.confidence * 0.5) + (areaScore * 0.3) + ((detectedFace.quality ?? 0.5) * 0.2)

        return FaceCropResult(
            modelInputImage: crop,
            avatarImageData: Self.jpegData(from: crop, size: configuration.representativeThumbnailSize),
            qualityScore: quality
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

    private static func jpegData(from cgImage: CGImage, size: CGFloat) -> Data? {
        let image = UIImage(cgImage: cgImage)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size), format: format)
        let rendered = renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        return rendered.jpegData(compressionQuality: 0.82)
    }

    private static func eyeAlignedCrop(
        _ crop: CGImage,
        detectedFace: DetectedFace,
        paddedRect: CGRect,
        imageSize: CGSize
    ) -> CGImage? {
        guard let landmarks = detectedFace.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let leftCenter = landmarkCenter(leftEye, faceBox: detectedFace.normalizedBoundingBox, imageSize: imageSize),
              let rightCenter = landmarkCenter(rightEye, faceBox: detectedFace.normalizedBoundingBox, imageSize: imageSize) else {
            return nil
        }

        let cropLeftEye = CGPoint(x: leftCenter.x - paddedRect.minX, y: leftCenter.y - paddedRect.minY)
        let cropRightEye = CGPoint(x: rightCenter.x - paddedRect.minX, y: rightCenter.y - paddedRect.minY)
        let angle = atan2(cropRightEye.y - cropLeftEye.y, cropRightEye.x - cropLeftEye.x)
        guard angle.isFinite, abs(angle) > 0.01 else { return nil }

        let width = crop.width
        let height = crop.height
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
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            return nil
        }

        context.translateBy(x: CGFloat(width) / 2, y: CGFloat(height) / 2)
        context.rotate(by: -angle)
        context.translateBy(x: -CGFloat(width) / 2, y: -CGFloat(height) / 2)
        context.draw(crop, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
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

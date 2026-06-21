import SwiftUI

struct PhotoThumbnailView: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    let asset: PhotoAssetSummary
    @State private var thumbnail: UIImage?

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Rectangle()
                .fill(Color(.secondarySystemBackground))

            if let thumbnail {
                Image(uiImage: thumbnail)
                    .resizable()
                    .scaledToFit()
                    .aspectRatio(asset.aspectRatio, contentMode: .fit)
                    .frame(maxWidth: .infinity)
            } else {
                Image(systemName: asset.isVideo ? "video" : (asset.isScreenshot ? "rectangle.dashed" : "photo"))
                    .font(.title2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                    .aspectRatio(asset.aspectRatio, contentMode: .fit)
            }

            if asset.isVideo {
                Label(asset.durationText, systemImage: "video.fill")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white)
                    .labelStyle(.titleAndIcon)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 3)
                    .background(.black.opacity(0.62), in: Capsule())
                    .padding(5)
                    .accessibilityHidden(true)
            }
        }
        .aspectRatio(asset.aspectRatio, contentMode: .fit)
        .contentShape(Rectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(asset.accessibilitySummary)
        .task(id: asset.id) {
            thumbnail = await photoLibraryStore.thumbnail(
                for: asset,
                targetSize: asset.thumbnailTargetSize,
                deliveryMode: .highQualityFormat,
                contentMode: .aspectFit
            )
        }
    }
}

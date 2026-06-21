import SwiftUI

struct PhotoThumbnailView: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    let asset: PhotoAssetSummary
    @State private var thumbnail: UIImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                Rectangle()
                    .fill(Color(.secondarySystemBackground))

                if let thumbnail {
                    Image(uiImage: thumbnail)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipped()
                } else {
                    Image(systemName: "photo")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

            VStack(alignment: .leading, spacing: 4) {
                Text(asset.displayTitle)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(asset.dimensionsText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .task(id: asset.id) {
            thumbnail = await photoLibraryStore.thumbnail(
                for: asset,
                targetSize: CGSize(width: 440, height: 440)
            )
        }
    }
}

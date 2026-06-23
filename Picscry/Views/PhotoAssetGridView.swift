import SwiftUI

struct PhotoAssetGridView: View {
    let assets: [PhotoAssetSummary]
    var accessibilitySuffix: String?
    let onSelect: (PhotoAssetSummary) -> Void

    private let columns = Array(
        repeating: GridItem(.flexible(minimum: 0), spacing: 2),
        count: 3
    )

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 2) {
                ForEach(assets) { asset in
                    Button {
                        onSelect(asset)
                    } label: {
                        PhotoThumbnailView(asset: asset)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(accessibilityLabel(for: asset))
                    .accessibilityHint("Opens this media item")
                }
            }
            .padding(.horizontal, 2)
            .padding(.vertical, 2)
        }
    }

    private func accessibilityLabel(for asset: PhotoAssetSummary) -> String {
        if let accessibilitySuffix {
            return "\(asset.accessibilitySummary), \(accessibilitySuffix)"
        }
        return asset.accessibilitySummary
    }
}

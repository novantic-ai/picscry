import SwiftUI

struct PhotoDetailView: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @Environment(\.dismiss) private var dismiss
    let asset: PhotoAssetSummary
    @State private var thumbnail: UIImage?
    @State private var metadata: PhotoMetadata = .empty
    @State private var isLoadingMetadata = true

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ZStack {
                        Rectangle()
                            .fill(Color(.secondarySystemBackground))
                        if let thumbnail {
                            Image(uiImage: thumbnail)
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: .infinity)
                        } else {
                            ProgressView()
                        }
                    }
                    .frame(minHeight: 220)
                    .listRowInsets(EdgeInsets())
                    .accessibilityLabel(asset.accessibilitySummary)
                }

                if isLoadingMetadata {
                    Section {
                        ProgressView("Loading Metadata")
                    }
                } else {
                    ForEach(metadata.sections) { section in
                        Section(section.title) {
                            ForEach(section.items) { item in
                                MetadataRow(item: item)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Photo Metadata")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .task(id: asset.id) {
            async let loadedThumbnail = photoLibraryStore.thumbnail(
                for: asset,
                targetSize: asset.originalTargetSize,
                deliveryMode: .highQualityFormat,
                contentMode: .aspectFit
            )
            async let loadedMetadata = photoLibraryStore.metadata(for: asset)
            thumbnail = await loadedThumbnail
            metadata = await loadedMetadata
            isLoadingMetadata = false
        }
    }
}

private struct MetadataRow: View {
    let item: PhotoMetadataItem

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(item.label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(item.value)
                .font(.body)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
    }
}

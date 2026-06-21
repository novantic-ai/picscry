import AVKit
import SwiftUI

struct PhotoDetailView: View {
    let assets: [PhotoAssetSummary]
    let initialAsset: PhotoAssetSummary

    @State private var selectedAssetID: PhotoAssetSummary.ID
    @State private var isShowingMetadata = false
    @Environment(\.dismiss) private var dismiss

    init(assets: [PhotoAssetSummary], initialAsset: PhotoAssetSummary) {
        self.assets = assets
        self.initialAsset = initialAsset
        _selectedAssetID = State(initialValue: initialAsset.id)
    }

    var body: some View {
        NavigationStack {
            TabView(selection: $selectedAssetID) {
                ForEach(assets) { asset in
                    MediaDetailPage(asset: asset, isShowingMetadata: $isShowingMetadata)
                        .tag(asset.id)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .navigationTitle(currentAsset?.mediaKind.displayName ?? "Media")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button(isShowingMetadata ? "Hide Metadata" : "Show Metadata") {
                        isShowingMetadata.toggle()
                    }
                    .accessibilityLabel(isShowingMetadata ? "Hide metadata" : "Show metadata")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var currentAsset: PhotoAssetSummary? {
        assets.first { $0.id == selectedAssetID } ?? initialAsset
    }
}

private struct MediaDetailPage: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    let asset: PhotoAssetSummary
    @Binding var isShowingMetadata: Bool

    @State private var thumbnail: UIImage?
    @State private var player: AVPlayer?
    @State private var metadata: PhotoMetadata = .empty
    @State private var isLoadingMetadata = false

    var body: some View {
        List {
            Section {
                mediaView
                    .frame(minHeight: 320)
                    .listRowInsets(EdgeInsets())
                    .accessibilityElement(children: .contain)
                    .accessibilityLabel(asset.accessibilitySummary)
                    .accessibilityAction(named: isShowingMetadata ? "Hide metadata" : "Show metadata") {
                        isShowingMetadata.toggle()
                    }
            }

            if isShowingMetadata {
                if isLoadingMetadata {
                    Section { ProgressView("Loading Metadata") }
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
        }
        .task(id: asset.id) { await loadMedia() }
        .task(id: isShowingMetadata) {
            guard isShowingMetadata, metadata.sections.isEmpty else { return }
            isLoadingMetadata = true
            metadata = await photoLibraryStore.metadata(for: asset)
            isLoadingMetadata = false
        }
        .onDisappear { player?.pause() }
    }

    @ViewBuilder
    private var mediaView: some View {
        ZStack {
            Rectangle().fill(Color(.secondarySystemBackground))
            if asset.isVideo {
                if let player {
                    VideoPlayer(player: player)
                } else {
                    ProgressView("Loading Video")
                }
            } else if let thumbnail {
                Image(uiImage: thumbnail)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
            } else {
                ProgressView("Loading Image")
            }
        }
    }

    private func loadMedia() async {
        thumbnail = nil
        player = nil
        if asset.isVideo {
            if let playerItem = await photoLibraryStore.playerItem(for: asset) {
                player = AVPlayer(playerItem: playerItem)
            }
        } else {
            thumbnail = await photoLibraryStore.thumbnail(
                for: asset,
                targetSize: asset.originalTargetSize,
                deliveryMode: .highQualityFormat,
                contentMode: .aspectFit
            )
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

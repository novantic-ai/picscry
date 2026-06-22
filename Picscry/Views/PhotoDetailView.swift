import AVKit
import SwiftUI
import UIKit

struct PhotoDetailView: View {
    private let assets: [PhotoAssetSummary]
    let initialAsset: PhotoAssetSummary

    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @State private var visibleRange: Range<Int>
    @State private var selectedAssetID: PhotoAssetSummary.ID
    @State private var isShowingMetadata = false
    @Environment(\.dismiss) private var dismiss

    private var pageAssets: [PhotoAssetSummary] {
        Array(assets[visibleRange])
    }

    init(assets: [PhotoAssetSummary], initialAsset: PhotoAssetSummary) {
        self.assets = assets
        self.initialAsset = initialAsset
        let initialIndex = assets.firstIndex { $0.id == initialAsset.id }
        _visibleRange = State(initialValue: Self.pageRange(centeredOn: initialIndex, assetCount: assets.count))
        _selectedAssetID = State(initialValue: initialAsset.id)
    }

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                TabView(selection: $selectedAssetID) {
                    ForEach(pageAssets) { asset in
                        MediaDetailPage(
                            asset: asset,
                            isActive: asset.id == selectedAssetID,
                            isShowingMetadata: $isShowingMetadata
                        )
                        .tag(asset.id)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .onAppear {
                    updateVisibleRangeAndCache(for: selectedAssetID, viewportSize: geometry.size)
                }
                .onChange(of: selectedAssetID) { _, newValue in
                    updateVisibleRangeAndCache(for: newValue, viewportSize: geometry.size)
                }
                .onChange(of: geometry.size) { _, newSize in
                    cacheVisibleAssets(viewportSize: newSize)
                }
            }
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

    private func updateVisibleRangeAndCache(for assetID: PhotoAssetSummary.ID, viewportSize: CGSize) {
        guard let selectedIndex = assets.firstIndex(where: { $0.id == assetID }) else { return }
        let nextRange = Self.pageRange(centeredOn: selectedIndex, assetCount: assets.count)
        visibleRange = nextRange
        cacheAssets(in: nextRange, viewportSize: viewportSize)
    }

    private func cacheVisibleAssets(viewportSize: CGSize) {
        cacheAssets(in: visibleRange, viewportSize: viewportSize)
    }

    private func cacheAssets(in range: Range<Int>, viewportSize: CGSize) {
        photoLibraryStore.updateDetailImageCache(
            for: Array(assets[range]),
            targetSize: Self.pixelTargetSize(for: viewportSize)
        )
    }

    private static func pageRange(centeredOn initialIndex: Int?, assetCount: Int) -> Range<Int> {
        guard assetCount > 0 else { return 0..<0 }
        guard let initialIndex else { return 0..<min(assetCount, 1) }

        let pageRadius = 12
        let lowerBound = max(0, initialIndex - pageRadius)
        let upperBound = min(assetCount, initialIndex + pageRadius + 1)
        return lowerBound..<upperBound
    }

    private static func pixelTargetSize(for viewportSize: CGSize) -> CGSize {
        let scale = UIScreen.main.scale
        return CGSize(
            width: viewportSize.width * scale,
            height: viewportSize.height * scale
        )
    }
}

private struct MediaDetailPage: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore

    let asset: PhotoAssetSummary
    let isActive: Bool
    @Binding var isShowingMetadata: Bool

    @State private var image: UIImage?
    @State private var loadedImageSize: CGSize = .zero
    @State private var player: AVPlayer?
    @State private var metadata: PhotoMetadata = .empty
    @State private var isLoadingMedia = false
    @State private var isLoadingMetadata = false

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                mediaView(availableSize: CGSize(width: geometry.size.width, height: mediaHeight(in: geometry.size)))
                    .frame(maxWidth: .infinity, maxHeight: mediaHeight(in: geometry.size))
                    .accessibilityElement(children: .contain)
                    .accessibilityLabel(asset.accessibilitySummary)
                    .accessibilityHint(isShowingMetadata ? "Metadata is visible below the media." : "Swipe left or right to move through media.")
                    .accessibilityAction(named: isShowingMetadata ? "Hide metadata" : "Show metadata") {
                        isShowingMetadata.toggle()
                    }

                if isShowingMetadata {
                    metadataView
                        .frame(maxHeight: geometry.size.height * 0.38)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .center)
            .background(Color(.systemBackground))
            .animation(.easeInOut(duration: 0.2), value: isShowingMetadata)
            .task(id: isActive) {
                guard isActive else {
                    player?.pause()
                    return
                }
                await loadMedia(displaySize: CGSize(width: geometry.size.width, height: mediaHeight(in: geometry.size)))
            }
            .task(id: isShowingMetadata) {
                guard isActive, isShowingMetadata, metadata.sections.isEmpty else { return }
                isLoadingMetadata = true
                metadata = await photoLibraryStore.metadata(for: asset)
                isLoadingMetadata = false
            }
            .onDisappear { player?.pause() }
        }
    }

    @ViewBuilder
    private func mediaView(availableSize: CGSize) -> some View {
        ZStack(alignment: .center) {
            Rectangle().fill(Color(.secondarySystemBackground))

            if asset.isVideo {
                if let player {
                    VideoPlayer(player: player)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                } else if isLoadingMedia {
                    ProgressView("Loading Video")
                } else {
                    Image(systemName: "video")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                }
            } else if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: fittedImageSize(in: availableSize).width, height: fittedImageSize(in: availableSize).height)
            } else if isLoadingMedia {
                ProgressView("Loading Image")
            } else {
                Image(systemName: asset.isScreenshot ? "rectangle.dashed" : "photo")
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var metadataView: some View {
        Group {
            if isLoadingMetadata {
                ProgressView("Loading Metadata")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(metadata.sections) { section in
                        Section(section.title) {
                            ForEach(section.items) { item in
                                MetadataRow(item: item)
                            }
                        }
                    }
                }
                .listStyle(.insetGrouped)
            }
        }
    }

    private func mediaHeight(in size: CGSize) -> CGFloat {
        isShowingMetadata ? size.height * 0.62 : size.height
    }

    private func fittedImageSize(in availableSize: CGSize) -> CGSize {
        let aspectRatio = loadedImageSize == .zero ? asset.aspectRatio : loadedImageSize.width / max(loadedImageSize.height, 1)
        guard aspectRatio.isFinite, aspectRatio > 0, availableSize.width > 0, availableSize.height > 0 else {
            return availableSize
        }

        let availableRatio = availableSize.width / availableSize.height
        if aspectRatio > availableRatio {
            return CGSize(width: availableSize.width, height: availableSize.width / aspectRatio)
        }
        return CGSize(width: availableSize.height * aspectRatio, height: availableSize.height)
    }

    private func loadMedia(displaySize: CGSize) async {
        guard !isLoadingMedia, image == nil, player == nil else { return }
        isLoadingMedia = true
        defer { isLoadingMedia = false }

        if asset.isVideo {
            if let playerItem = await photoLibraryStore.playerItem(for: asset) {
                player = AVPlayer(playerItem: playerItem)
            }
            return
        }

        for await updatedImage in photoLibraryStore.displayThenFullQualityImageUpdates(
            for: asset,
            displayTargetSize: displayTargetSize(for: displaySize)
        ) {
            guard !Task.isCancelled else { return }
            image = updatedImage
            loadedImageSize = updatedImage.size
        }
    }

    private func displayTargetSize(for displaySize: CGSize) -> CGSize {
        let scale = UIScreen.main.scale
        return CGSize(width: displaySize.width * scale, height: displaySize.height * scale)
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

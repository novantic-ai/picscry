import Photos
import SwiftUI

struct LibraryView: View {
    @Environment(AuthenticationStore.self) private var authenticationStore
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @State private var selectedAsset: PhotoAssetSummary?
    @State private var isShowingDiagnostics = false
    @State private var mediaFilter: LibraryMediaFilter = .all

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Library")
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Menu {
                            ForEach(LibraryMediaFilter.allCases) { filter in
                                Button {
                                    mediaFilter = filter
                                } label: {
                                    Label(filter.displayName, systemImage: mediaFilter == filter ? "checkmark" : filter.systemImage)
                                }
                            }
                        } label: {
                            Label("Filter", systemImage: "line.3.horizontal.decrease.circle")
                                .accessibilityLabel("Filter library by media type")
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu {
                            Button("Refresh", systemImage: "arrow.clockwise") {
                                Task { await photoLibraryStore.reloadAssets() }
                            }
                            Button("Diagnostics", systemImage: "stethoscope") {
                                isShowingDiagnostics = true
                            }
                            Button("Sign Out", systemImage: "rectangle.portrait.and.arrow.right") {
                                authenticationStore.signOut()
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                                .accessibilityLabel("Library options")
                        }
                    }
                }
                .fullScreenCover(item: $selectedAsset) { asset in
                    PhotoDetailView(assets: filteredAssets, initialAsset: asset)
                }
                .sheet(isPresented: $isShowingDiagnostics) {
                    DiagnosticsView()
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch photoLibraryStore.authorizationState {
        case .unknown, .requesting:
            ProgressView("Preparing Photo Library")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .denied, .restricted:
            EmptyStateView(
                systemImage: "lock.circle",
                title: "Photo Access Needed",
                message: "Allow photo library access in Settings to show your media and metadata."
            )
        case .authorized, .limited:
            if photoLibraryStore.isLoading && photoLibraryStore.assets.isEmpty {
                loadingView
            } else if filteredAssets.isEmpty {
                EmptyStateView(
                    systemImage: "photo.stack",
                    title: "No Photos or Videos Found",
                    message: "Picscry will show your imported photo and video library here once media is available."
                )
            } else {
                visualGrid
            }
        }
    }

    private var visualGrid: some View {
        PhotoAssetGridView(assets: filteredAssets) { asset in
            selectedAsset = asset
        }
        .overlay(alignment: .bottom) {
            if photoLibraryStore.authorizationState == .limited {
                Text("Limited library access is enabled.")
                    .font(.footnote.weight(.medium))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.regularMaterial, in: Capsule())
                    .padding(.bottom, 12)
            }
        }
    }

    private var filteredAssets: [PhotoAssetSummary] {
        photoLibraryStore.assets.filter(mediaFilter.includes(_:))
    }

    private var loadingView: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text("Loading Library")
                .font(.headline)
            if photoLibraryStore.totalAssetCount > 0 {
                Text("\(photoLibraryStore.assets.count) of \(photoLibraryStore.totalAssetCount)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

}

private enum LibraryMediaFilter: String, CaseIterable, Identifiable {
    case all
    case photos
    case screenshots
    case videos

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .all: "All Media"
        case .photos: "Photos"
        case .screenshots: "Screenshots"
        case .videos: "Videos"
        }
    }

    var systemImage: String {
        switch self {
        case .all: "photo.stack"
        case .photos: "photo"
        case .screenshots: "rectangle.dashed"
        case .videos: "video"
        }
    }

    func includes(_ asset: PhotoAssetSummary) -> Bool {
        switch self {
        case .all: true
        case .photos: asset.mediaKind == .photo
        case .screenshots: asset.mediaKind == .screenshot
        case .videos: asset.mediaKind == .video
        }
    }
}

#Preview {
    LibraryView()
        .environment(AuthenticationStore())
        .environment(PhotoLibraryStore())
        .environment(FaceRecognitionStore())
}

import Photos
import SwiftUI

struct LibraryView: View {
    @Environment(AuthenticationStore.self) private var authenticationStore
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled
    @State private var selectedAsset: PhotoAssetSummary?
    @State private var isShowingDiagnostics = false

    private let columns = Array(
        repeating: GridItem(.flexible(minimum: 0), spacing: 2),
        count: 3
    )

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Library")
                .toolbar {
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
                .sheet(item: $selectedAsset) { asset in
                    PhotoDetailView(asset: asset)
                }
                .sheet(isPresented: $isShowingDiagnostics) {
                    DiagnosticsView()
                }
        }
        .task {
            await photoLibraryStore.prepareLibrary()
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
                message: "Allow photo library access in Settings to show your images and metadata."
            )
        case .authorized, .limited:
            if photoLibraryStore.isLoading && photoLibraryStore.assets.isEmpty {
                loadingView
            } else if photoLibraryStore.assets.isEmpty {
                EmptyStateView(
                    systemImage: "photo.stack",
                    title: "No Photos or Videos Found",
                    message: "Picscry will show your imported photo and video library here once media is available."
                )
            } else if voiceOverEnabled {
                accessibleList
            } else {
                visualGrid
            }
        }
    }

    private var visualGrid: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 2) {
                ForEach(photoLibraryStore.assets) { asset in
                    Button {
                        selectedAsset = asset
                    } label: {
                        PhotoThumbnailView(asset: asset)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(asset.accessibilitySummary)
                    .accessibilityHint("Opens all available metadata for this media item")
                }
            }
            .padding(.horizontal, 2)
            .padding(.vertical, 2)
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

    private var accessibleList: some View {
        List(photoLibraryStore.assets) { asset in
            Button {
                selectedAsset = asset
            } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(asset.displayTitle)
                        .font(.headline)
                    Text(asset.accessibilitySummary)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
            .accessibilityHint("Opens all available metadata for this media item")
            .accessibilityLabel(asset.accessibilitySummary)
        }
    }
}

#Preview {
    LibraryView()
        .environment(AuthenticationStore())
        .environment(PhotoLibraryStore())
}

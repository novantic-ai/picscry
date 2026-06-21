import Photos
import SwiftUI

struct LibraryView: View {
    @Environment(AuthenticationStore.self) private var authenticationStore
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @Environment(\.accessibilityVoiceOverEnabled) private var voiceOverEnabled
    @State private var selectedAsset: PhotoAssetSummary?

    private let columns = [
        GridItem(.adaptive(minimum: 156, maximum: 220), spacing: 14)
    ]

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
                ProgressView("Loading Photos")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if photoLibraryStore.assets.isEmpty {
                EmptyStateView(
                    systemImage: "photo.stack",
                    title: "No Photos Found",
                    message: "Picscry will show your imported photo library here once photos are available."
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
            LazyVGrid(columns: columns, spacing: 14) {
                ForEach(photoLibraryStore.assets) { asset in
                    Button {
                        selectedAsset = asset
                    } label: {
                        PhotoThumbnailView(asset: asset)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(asset.accessibilitySummary)
                    .accessibilityHint("Opens all available metadata for this photo")
                }
            }
            .padding(16)
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
            .accessibilityHint("Opens all available metadata for this photo")
        }
    }
}

#Preview {
    LibraryView()
        .environment(AuthenticationStore())
        .environment(PhotoLibraryStore())
}

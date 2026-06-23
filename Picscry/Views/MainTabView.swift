import SwiftUI

struct MainTabView: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @Environment(FaceRecognitionStore.self) private var faceRecognitionStore

    var body: some View {
        TabView {
            LibraryView()
                .tabItem {
                    Label("Library", systemImage: "photo.on.rectangle")
                }

            MoreView()
                .tabItem {
                    Label("More", systemImage: "ellipsis.circle")
                }
        }
        .task {
            await photoLibraryStore.prepareLibrary()
            await faceRecognitionStore.prepare(photoLibraryStore: photoLibraryStore)
        }
        .onChange(of: photoLibraryStore.libraryVersion) { _, _ in
            Task {
                await faceRecognitionStore.prepare(photoLibraryStore: photoLibraryStore)
            }
        }
    }
}

#Preview {
    MainTabView()
        .environment(AuthenticationStore())
        .environment(PhotoLibraryStore())
        .environment(FaceRecognitionStore())
}

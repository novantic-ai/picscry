import SwiftUI

@main
struct PicscryApp: App {
    @State private var authenticationStore = AuthenticationStore()
    @State private var photoLibraryStore = PhotoLibraryStore()
    @State private var faceRecognitionStore = FaceRecognitionStore()

    init() {
        Diagnostics.shared.start()
    }

    var body: some Scene {
        WindowGroup {
            AppRootView()
                .environment(authenticationStore)
                .environment(photoLibraryStore)
                .environment(faceRecognitionStore)
        }
    }
}

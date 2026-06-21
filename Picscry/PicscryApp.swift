import SwiftUI

@main
struct PicscryApp: App {
    @State private var authenticationStore = AuthenticationStore()
    @State private var photoLibraryStore = PhotoLibraryStore()

    var body: some Scene {
        WindowGroup {
            AppRootView()
                .environment(authenticationStore)
                .environment(photoLibraryStore)
        }
    }
}

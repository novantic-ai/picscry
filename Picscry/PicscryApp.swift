import SwiftUI

@main
struct PicscryApp: App {
    @State private var authenticationStore = AuthenticationStore()
    @State private var photoLibraryStore = PhotoLibraryStore()

    init() {
        Diagnostics.shared.start()
    }

    var body: some Scene {
        WindowGroup {
            AppRootView()
                .environment(authenticationStore)
                .environment(photoLibraryStore)
        }
    }
}

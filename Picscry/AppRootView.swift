import SwiftUI

struct AppRootView: View {
    @Environment(AuthenticationStore.self) private var authenticationStore

    var body: some View {
        Group {
            if authenticationStore.isSignedIn {
                LibraryView()
            } else {
                LoginView()
            }
        }
        .animation(.snappy, value: authenticationStore.isSignedIn)
        .task {
            await authenticationStore.restoreCredentialState()
        }
    }
}

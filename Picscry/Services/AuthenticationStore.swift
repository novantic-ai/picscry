import AuthenticationServices
import Foundation
import Observation

@MainActor
@Observable
final class AuthenticationStore {
    private enum DefaultsKey {
        static let appleUserIdentifier = "appleUserIdentifier"
        static let displayName = "appleDisplayName"
    }

    var isSignedIn = false
    var displayName: String?
    var errorMessage: String?

    private let provider = ASAuthorizationAppleIDProvider()
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.displayName = defaults.string(forKey: DefaultsKey.displayName)
        self.isSignedIn = defaults.string(forKey: DefaultsKey.appleUserIdentifier) != nil
    }

    func handleAuthorization(_ authorization: ASAuthorization) {
        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential else {
            errorMessage = "Apple did not return a usable credential."
            return
        }

        defaults.set(credential.user, forKey: DefaultsKey.appleUserIdentifier)

        let name = PersonNameComponentsFormatter().string(from: credential.fullName ?? PersonNameComponents())
        if !name.isEmpty {
            defaults.set(name, forKey: DefaultsKey.displayName)
            displayName = name
        }

        errorMessage = nil
        isSignedIn = true
    }

    func handleAuthorizationError(_ error: Error) {
        let authorizationError = error as? ASAuthorizationError
        if authorizationError?.code == .canceled {
            return
        }
        errorMessage = error.localizedDescription
    }

    func restoreCredentialState() async {
        guard let userIdentifier = defaults.string(forKey: DefaultsKey.appleUserIdentifier) else {
            isSignedIn = false
            return
        }

        do {
            let state = try await provider.credentialState(forUserID: userIdentifier)
            isSignedIn = state == .authorized
            if !isSignedIn {
                clearSession()
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func signOut() {
        clearSession()
    }

    private func clearSession() {
        defaults.removeObject(forKey: DefaultsKey.appleUserIdentifier)
        defaults.removeObject(forKey: DefaultsKey.displayName)
        displayName = nil
        isSignedIn = false
    }
}

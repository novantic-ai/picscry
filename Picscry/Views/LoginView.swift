import AuthenticationServices
import SwiftUI

struct LoginView: View {
    @Environment(AuthenticationStore.self) private var authenticationStore
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        VStack(spacing: 28) {
            Spacer(minLength: 24)

            VStack(spacing: 14) {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 54, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 96, height: 96)
                    .background(.black, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .accessibilityHidden(true)

                Text("Picscry")
                    .font(.largeTitle.weight(.bold))

                Text("Sign in to review your photo library and the metadata already available on your device.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 360)
            }

            SignInWithAppleButton(.signIn) { request in
                request.requestedScopes = [.fullName, .email]
            } onCompletion: { result in
                switch result {
                case .success(let authorization):
                    authenticationStore.handleAuthorization(authorization)
                case .failure(let error):
                    authenticationStore.handleAuthorizationError(error)
                }
            }
            .signInWithAppleButtonStyle(colorScheme == .dark ? .white : .black)
            .frame(height: 52)
            .frame(maxWidth: 360)
            .accessibilityLabel("Sign in with Apple")

            if let errorMessage = authenticationStore.errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 360)
                    .accessibilityLabel("Sign in error: \(errorMessage)")
            }

            Spacer(minLength: 24)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

#Preview {
    LoginView()
        .environment(AuthenticationStore())
}

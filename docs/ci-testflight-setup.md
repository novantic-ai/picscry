# CI and TestFlight Setup

This repository builds and tests the iOS app on every push and pull request with `.github/workflows/ios-ci-testflight.yml`.

Pushes to `main` run the same tests first, then archive and upload the app to TestFlight.

## Required GitHub Secrets

Add these in GitHub under `Settings > Secrets and variables > Actions > Secrets`:

- `APPLE_TEAM_ID`: Apple Developer Team ID.
- `APP_STORE_CONNECT_KEY_ID`: App Store Connect API key ID.
- `APP_STORE_CONNECT_ISSUER_ID`: App Store Connect issuer ID.
- `APP_STORE_CONNECT_API_KEY_P8`: Contents of the `.p8` private key file.
- `IOS_DISTRIBUTION_CERTIFICATE_BASE64`: Base64-encoded Apple Distribution `.p12` certificate.
- `IOS_DISTRIBUTION_CERTIFICATE_PASSWORD`: Password for the `.p12` certificate.
- `IOS_PROVISIONING_PROFILE_BASE64`: Base64-encoded App Store provisioning profile for `com.novanticai.picscry`.
- `IOS_PROVISIONING_PROFILE_NAME`: The profile name exactly as shown in Apple Developer.

On Windows, you can create the base64 values with PowerShell:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\certificate.p12")) | Set-Clipboard
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\profile.mobileprovision")) | Set-Clipboard
```

Optional:

- `APP_STORE_CONNECT_IN_HOUSE`: Set to `true` only for an in-house enterprise App Store Connect API key. Leave unset for normal TestFlight.

## Required GitHub Variable

Add this in `Settings > Secrets and variables > Actions > Variables`:

- `TESTFLIGHT_INTERNAL_GROUP`: Name of the internal TestFlight group. Defaults to `Internal Testing` if unset.

Create that group in App Store Connect and keep only your Apple ID in it.

## Apple-Side Setup

1. Register the bundle ID `com.novanticai.picscry`.
2. Enable Sign in with Apple for that App ID.
3. Create an App Store Connect app record for Picscry.
4. Create an Apple Distribution certificate and export it as `.p12`.
5. Create an App Store provisioning profile for `com.novanticai.picscry`.
6. Add yourself as an internal TestFlight tester.

Internal TestFlight testing does not require Apple Beta App Review. External testing does.

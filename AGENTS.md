# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project

Picscry is a native SwiftUI iOS app. The current implementation is an early TestFlight-only build with:

- Sign in with Apple as the first screen.
- PhotoKit authorization after sign-in.
- A library screen showing imported photo assets and metadata.
- A photo metadata detail sheet.
- In-app diagnostics logging and MetricKit diagnostic capture.

The design source of truth is `design.md`.

## Important Facts

- Bundle identifier: `com.novanticai.picscry`
- Xcode project: `Picscry.xcodeproj`
- Scheme: `Picscry`
- GitHub repo: `novantic-ai/picscry`
- Default branch: `main`
- CI workflow: `.github/workflows/ios-ci-testflight.yml`
- TestFlight upload is internal-only. Do not enable external beta review unless explicitly requested.
- GitHub Actions uses `macos-15` with `XCODE_APP=/Applications/Xcode_26.3.app`.
- The user is on Windows and does not have a local Mac. Do not rely on local `xcodebuild`, `xcrun`, or iOS Simulator availability.
- Markdown/doc-only changes are ignored by CI via workflow `paths-ignore` so they do not create new TestFlight builds.

## Build, Test, and Release

Local Windows cannot run the iOS simulator. Use GitHub Actions as the build/test loop.

On every push and PR:

```sh
xcodebuild test \
  -project Picscry.xcodeproj \
  -scheme Picscry \
  -configuration Debug \
  -destination "platform=iOS Simulator,name=iPhone 16,OS=latest" \
  -resultBundlePath TestResults/Picscry.xcresult \
  CODE_SIGNING_ALLOWED=NO
```

On pushes to `main`, the workflow:

1. Runs simulator unit tests.
2. Archives with manual App Store signing.
3. Exports an IPA.
4. Uploads the IPA artifact.
5. Uploads to internal TestFlight using fastlane `pilot`.

Use `gh` to inspect runs:

```sh
gh run list --repo novantic-ai/picscry --limit 5
gh run view <run-id> --repo novantic-ai/picscry --json status,conclusion,jobs,url
gh run view <run-id> --repo novantic-ai/picscry --job <job-id> --log
```

GitHub Actions secrets and setup notes are documented in `docs/ci-testflight-setup.md`.

The workflow ignores Markdown and docs-only changes:

```yaml
paths-ignore:
  - "**/*.md"
  - "docs/**"
```

If you edit `.github/workflows/ios-ci-testflight.yml` itself, GitHub may still run the workflow for that commit. After the path filters are in place, future changes limited to `AGENTS.md`, `README.md`, or `docs/**` should not trigger CI or TestFlight.

## TestFlight Upload Rules

Keep these fastlane flags unless requirements change:

```sh
--skip_waiting_for_build_processing false
--submit_beta_review false
--distribute_external false
--notify_external_testers false
--uses_non_exempt_encryption false
```

Why:

- Internal TestFlight does not need Beta App Review.
- Setting `submit_beta_review` to false avoids App Store Connect asking for external testing metadata such as Beta App Description.
- `ITSAppUsesNonExemptEncryption` is set to false in `Picscry/Info.plist`.

App Store Connect currently requires iOS 26 SDK or later for upload, so do not downgrade the release workflow to Xcode 16.x.

## Debugging Real-Device Issues

The first real-device crash happened immediately after Apple sign-in. Likely cause: the app enumerated the entire photo library on the main actor, fetched `PHAssetResource` metadata for every asset, and requested caching for every asset at once. That can hang the app long enough for iOS to watchdog-kill it.

Current mitigation:

- `PhotoLibraryStore` loads summaries in batches.
- It yields between batches.
- It no longer fetches per-asset resource metadata during initial library load.
- It no longer pre-caches the entire library.
- Resource metadata is deferred until opening `PhotoDetailView`.

Do not reintroduce eager full-library metadata extraction, eager image data loading, or whole-library caching.

If the app misbehaves on TestFlight:

1. Ask the user to open `Library > ... > Diagnostics`.
2. Ask them to share/export the diagnostics log.
3. Check App Store Connect/TestFlight crash reports.
4. Inspect the GitHub Actions `.xcresult` artifact for CI failures.

Diagnostics implementation:

- `Picscry/Services/Diagnostics.swift`
- `Picscry/Views/DiagnosticsView.swift`
- It writes an app-local `diagnostics.log` under Application Support.
- It logs important app milestones and MetricKit diagnostic payloads when iOS delivers them.

MetricKit crash diagnostics are not instant. They may be delivered on a later launch.

## SwiftUI and PhotoKit Practices

- Keep app-level services as `@Observable` state at the app root and inject via `.environment(...)`.
- Prefer simple SwiftUI state over heavyweight view models until a workflow truly needs one.
- Keep PhotoKit work incremental and cancellable.
- Never load original image data for every asset in the library view.
- Use thumbnails in grid/list views and defer full metadata to detail screens.
- Preserve accessibility: VoiceOver should get a list-style library experience and meaningful labels/hints.

## Git and CI Flow

When asked to ship a build:

1. Patch code.
2. Commit to `main`.
3. Push to `origin main`.
4. Watch GitHub Actions.
5. If CI fails, fetch logs, patch, commit, and push again.
6. Only report TestFlight success after the `Upload to TestFlight` job is green.

Useful commands:

```sh
git status --short
git add <files>
git commit -m "<message>"
git push origin main
```

This Windows workspace may hit Git safe-directory/ownership issues. If git commands fail because of sandbox or ownership protections, rerun with the proper escalated tool permissions.

## Known Current Limitations

- No local iOS simulator on this Windows environment.
- No UI automation tests yet.
- The app currently handles images only, not videos.
- The library UI is read-only.
- Photo detail metadata extraction can still be expensive for individual large/iCloud assets; keep loading states and cancellation in mind.
- The app icon is a generated placeholder and should be replaced before a serious release.

## Files Worth Reading First

- `design.md`
- `Picscry/PicscryApp.swift`
- `Picscry/AppRootView.swift`
- `Picscry/Services/AuthenticationStore.swift`
- `Picscry/Services/PhotoLibraryStore.swift`
- `Picscry/Services/Diagnostics.swift`
- `Picscry/Views/LibraryView.swift`
- `.github/workflows/ios-ci-testflight.yml`
- `docs/ci-testflight-setup.md`

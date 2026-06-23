import SwiftUI

struct PersonDetailView: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @Environment(FaceRecognitionStore.self) private var faceRecognitionStore

    let personID: UUID

    @State private var person: PersonSummary?
    @State private var nameText = ""
    @State private var assets: [PhotoAssetSummary] = []
    @State private var selectedAsset: PhotoAssetSummary?
    @State private var mergeCandidate: MergeCandidate?
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 18) {
            header
                .padding(.top, 18)

            if isLoading {
                ProgressView("Loading Photos")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if assets.isEmpty {
                EmptyStateView(
                    systemImage: "photo.stack",
                    title: "No Photos",
                    message: "Photos for this person will appear after face indexing finishes."
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                PhotoAssetGridView(
                    assets: assets,
                    accessibilitySuffix: "contains \(person?.displayName ?? "Unknown")"
                ) { asset in
                    selectedAsset = asset
                }
            }
        }
        .navigationTitle(person?.displayName ?? "Person")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Save") {
                    Task { await saveName() }
                }
            }
        }
        .task(id: personID) {
            await load()
        }
        .fullScreenCover(item: $selectedAsset) { asset in
            PhotoDetailView(assets: assets, initialAsset: asset)
        }
        .alert(item: $mergeCandidate) { candidate in
            Alert(
                title: Text("Merge People?"),
                message: Text("A person named \(candidate.existingName) already exists. Merge these face groups?"),
                primaryButton: .cancel(),
                secondaryButton: .default(Text("Merge")) {
                    Task {
                        await faceRecognitionStore.confirmRenameMerge(
                            sourcePersonID: personID,
                            targetPersonID: candidate.existingPersonID,
                            finalName: candidate.existingName
                        )
                        await load()
                    }
                }
            )
        }
    }

    private var header: some View {
        VStack(spacing: 12) {
            FaceAvatarView(
                imageData: person?.representativeFaceImageData,
                name: "Representative face for \(person?.displayName ?? "Unknown")",
                size: 132
            )

            TextField("Unknown", text: $nameText)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.center)
                .textInputAutocapitalization(.words)
                .submitLabel(.done)
                .onSubmit {
                    Task { await saveName() }
                }
                .accessibilityLabel("Person name")
                .padding(.horizontal)

            if let person {
                Text("\(person.photoCount) photos, \(person.faceCount) faces")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func load() async {
        isLoading = true
        defer { isLoading = false }
        person = await faceRecognitionStore.person(with: personID)
        nameText = person?.isUnknown == true ? "" : (person?.displayName ?? "")
        assets = await faceRecognitionStore.assets(
            forPersonID: personID,
            from: photoLibraryStore.assets
        )
    }

    private func saveName() async {
        let result = await faceRecognitionStore.renamePerson(personID, to: nameText)
        switch result {
        case .renamed:
            await load()
        case let .needsMergeConfirmation(existingPersonID, existingName):
            mergeCandidate = MergeCandidate(existingPersonID: existingPersonID, existingName: existingName)
        }
    }
}

private struct MergeCandidate: Identifiable {
    let existingPersonID: UUID
    let existingName: String

    var id: UUID { existingPersonID }
}

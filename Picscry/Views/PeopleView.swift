import SwiftUI

struct PeopleView: View {
    @Environment(PhotoLibraryStore.self) private var photoLibraryStore
    @Environment(FaceRecognitionStore.self) private var faceRecognitionStore

    private var sortedPeople: [PersonSummary] {
        PeopleOrdering.sorted(faceRecognitionStore.people)
    }

    var body: some View {
        List {
            statusSection

            if sortedPeople.isEmpty {
                emptySection
            } else {
                Section("People") {
                    ForEach(sortedPeople) { person in
                        NavigationLink {
                            PersonDetailView(personID: person.id)
                        } label: {
                            PeopleRow(person: person)
                        }
                        .accessibilityLabel("\(person.displayName), \(person.photoCount) photos")
                    }
                }
            }
        }
        .navigationTitle("People")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Refresh Faces", systemImage: "arrow.clockwise") {
                    Task {
                        await faceRecognitionStore.retry(photoLibraryStore: photoLibraryStore)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        switch faceRecognitionStore.indexingState {
        case let .indexing(processed, total):
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Indexing faces: \(processed) of \(total) photos")
                        .font(.headline)
                    ProgressView(value: Double(processed), total: Double(max(total, 1)))
                    Text("Face recognition happens on this device. Photos and face data are not uploaded.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
        case let .failed(message):
            Section {
                Label(message, systemImage: "exclamationmark.triangle")
            }
        default:
            Section {
                Text("Face recognition happens on this device. Photos and face data are not uploaded.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var emptySection: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text("No People Yet")
                    .font(.headline)
                Text("Picscry will show detected people after face indexing finishes.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 8)
        }
    }
}

private struct PeopleRow: View {
    let person: PersonSummary

    var body: some View {
        HStack(spacing: 12) {
            FaceAvatarView(
                imageData: person.representativeFaceImageData,
                name: person.displayName,
                size: 56
            )

            VStack(alignment: .leading, spacing: 3) {
                Text(person.displayName)
                    .font(.headline)
                Text("\(person.photoCount) photos, \(person.faceCount) faces")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
    }
}

#Preview {
    NavigationStack {
        PeopleView()
            .environment(PhotoLibraryStore())
            .environment(FaceRecognitionStore())
    }
}

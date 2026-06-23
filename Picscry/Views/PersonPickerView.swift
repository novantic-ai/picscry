import SwiftUI

struct PersonPickerView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(FaceRecognitionStore.self) private var faceRecognitionStore

    let face: PhotoFaceSummary

    @State private var searchText = ""
    @State private var newNameText = ""

    private var people: [PersonSummary] {
        PeopleOrdering.sorted(faceRecognitionStore.people)
            .filter { $0.id != face.personID }
            .filter {
                searchText.isEmpty ||
                    $0.displayName.localizedCaseInsensitiveContains(searchText)
            }
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Create") {
                    TextField("New person name", text: $newNameText)
                        .textInputAutocapitalization(.words)
                    Button("Create New Person", systemImage: "person.crop.circle.badge.plus") {
                        Task {
                            await faceRecognitionStore.moveFaceObservation(face.id, toNewPersonNamed: newNameText)
                            dismiss()
                        }
                    }
                }

                Section("Existing People") {
                    ForEach(people) { person in
                        Button {
                            Task {
                                await faceRecognitionStore.moveFaceObservation(face.id, toExistingPersonID: person.id)
                                dismiss()
                            }
                        } label: {
                            HStack(spacing: 12) {
                                FaceAvatarView(
                                    imageData: person.representativeFaceImageData,
                                    name: person.displayName,
                                    size: 44
                                )
                                VStack(alignment: .leading) {
                                    Text(person.displayName)
                                    Text("\(person.photoCount) photos")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .searchable(text: $searchText, prompt: "Search People")
            .navigationTitle("Correct Face")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}

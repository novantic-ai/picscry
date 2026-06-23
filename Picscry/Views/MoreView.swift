import SwiftUI

struct MoreView: View {
    var body: some View {
        NavigationStack {
            List {
                NavigationLink {
                    PeopleView()
                } label: {
                    Label("People", systemImage: "person.2.crop.square.stack")
                }
                .accessibilityHint("Shows detected people and unknown face groups")
            }
            .navigationTitle("More")
        }
    }
}

#Preview {
    MoreView()
        .environment(PhotoLibraryStore())
        .environment(FaceRecognitionStore())
}

import SwiftUI

struct FaceChipRowView: View {
    let faces: [PhotoFaceSummary]
    let onSelect: (PhotoFaceSummary) -> Void
    let onCorrect: (PhotoFaceSummary) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Faces")
                .font(.headline)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 14) {
                    ForEach(faces.sorted { $0.leftToRightIndex < $1.leftToRightIndex }) { face in
                        Button {
                            onSelect(face)
                        } label: {
                            VStack(spacing: 6) {
                                FaceAvatarView(
                                    imageData: face.representativeFaceImageData,
                                    name: face.displayName,
                                    size: 52
                                )
                                Text(face.displayName)
                                    .font(.caption)
                                    .lineLimit(1)
                            }
                            .frame(minWidth: 72, minHeight: 82)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button("Not This Person", systemImage: "person.crop.circle.badge.xmark") {
                                onCorrect(face)
                            }
                        }
                        .accessibilityLabel("\(face.displayName), face \(face.leftToRightIndex + 1) of \(faces.count) in this photo")
                        .accessibilityHint("Opens this person")
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }
}

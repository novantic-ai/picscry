import SwiftUI
import UIKit

struct FaceAvatarView: View {
    let imageData: Data?
    let name: String
    let size: CGFloat

    var body: some View {
        Group {
            if let imageData, let image = UIImage(data: imageData) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: "person.crop.circle.fill")
                    .resizable()
                    .scaledToFit()
                    .padding(size * 0.12)
                    .foregroundStyle(.secondary)
                    .background(Color(.secondarySystemBackground))
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
        .overlay(Circle().strokeBorder(.secondary.opacity(0.25), lineWidth: 1))
        .accessibilityLabel(name)
    }
}

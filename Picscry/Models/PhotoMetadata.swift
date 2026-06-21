import Foundation

struct PhotoMetadata {
    let sections: [PhotoMetadataSection]

    static let empty = PhotoMetadata(sections: [])
}

struct PhotoMetadataSection: Identifiable {
    let id: String
    let title: String
    let items: [PhotoMetadataItem]
}

struct PhotoMetadataItem: Identifiable {
    let id = UUID()
    let label: String
    let value: String
}

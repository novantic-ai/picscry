import SwiftUI

struct DiagnosticsView: View {
    @State private var logText = Diagnostics.shared.recentText()
    @State private var debugBundleURL: URL?
    @State private var debugBundleStatus: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    debugExportControls

                    Text(logText)
                        .font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding()
            }
            .navigationTitle("Diagnostics")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    ShareLink(item: Diagnostics.shared.exportURL()) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .accessibilityLabel("Share diagnostics log")

                    Button {
                        Diagnostics.shared.clear()
                        logText = Diagnostics.shared.recentText()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .accessibilityLabel("Clear diagnostics log")
                }
            }
            .refreshable {
                logText = Diagnostics.shared.recentText()
            }
        }
    }

    private var debugExportControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                exportFaceEmbeddingDebugBundle()
            } label: {
                Label("Export face embedding debug bundle", systemImage: "archivebox")
            }
            .buttonStyle(.bordered)
            .accessibilityHint("Shares aligned face crops and local embedding vectors for debugging face recognition.")

            if let debugBundleURL {
                ShareLink(item: debugBundleURL) {
                    Label("Share face embedding debug bundle", systemImage: "square.and.arrow.up")
                }
                .accessibilityLabel("Share face embedding debug bundle")
                .accessibilityHint("Shares the generated ZIP file with aligned face crops and local embedding vectors.")
            }

            if let debugBundleStatus {
                Text(debugBundleStatus)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel(debugBundleStatus)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func exportFaceEmbeddingDebugBundle() {
        do {
            debugBundleURL = try FaceEmbeddingDebugExportService.makeDebugBundle()
            debugBundleStatus = "Face embedding debug bundle is ready to share."
        } catch {
            debugBundleURL = nil
            debugBundleStatus = error.localizedDescription
            Diagnostics.shared.log("Face embedding debug bundle export failed: \(error.localizedDescription)")
        }
        logText = Diagnostics.shared.recentText()
    }
}

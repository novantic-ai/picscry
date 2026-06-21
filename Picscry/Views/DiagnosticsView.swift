import SwiftUI

struct DiagnosticsView: View {
    @State private var logText = Diagnostics.shared.recentText()

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(logText)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
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
}

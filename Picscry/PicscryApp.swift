import BackgroundTasks
import SwiftUI

@main
struct PicscryApp: App {
    @State private var authenticationStore: AuthenticationStore
    @State private var photoLibraryStore: PhotoLibraryStore
    @State private var faceRecognitionStore: FaceRecognitionStore

    init() {
        let authenticationStore = AuthenticationStore()
        let photoLibraryStore = PhotoLibraryStore()
        let faceRecognitionStore = FaceRecognitionStore()

        _authenticationStore = State(initialValue: authenticationStore)
        _photoLibraryStore = State(initialValue: photoLibraryStore)
        _faceRecognitionStore = State(initialValue: faceRecognitionStore)

        Diagnostics.shared.start()
        FaceIndexingBackgroundTaskCoordinator.shared.configure(
            photoLibraryStore: photoLibraryStore,
            faceRecognitionStore: faceRecognitionStore
        )
        FaceIndexingBackgroundTaskCoordinator.shared.register()
    }

    var body: some Scene {
        WindowGroup {
            AppRootView()
                .environment(authenticationStore)
                .environment(photoLibraryStore)
                .environment(faceRecognitionStore)
        }
    }
}

private final class FaceIndexingBackgroundTaskCoordinator {
    static let shared = FaceIndexingBackgroundTaskCoordinator()

    @MainActor private var photoLibraryStore: PhotoLibraryStore?
    @MainActor private var faceRecognitionStore: FaceRecognitionStore?
    private var didRegister = false

    private init() {}

    @MainActor
    func configure(photoLibraryStore: PhotoLibraryStore, faceRecognitionStore: FaceRecognitionStore) {
        self.photoLibraryStore = photoLibraryStore
        self.faceRecognitionStore = faceRecognitionStore
    }

    func register() {
        guard !didRegister else { return }
        didRegister = true

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: FaceRecognitionStore.backgroundTaskIdentifier,
            using: nil
        ) { task in
            Task { @MainActor in
                await self.handle(task)
            }
        }
    }

    @MainActor
    private func handle(_ task: BGTask) async {
        guard let task = task as? BGProcessingTask else {
            task.setTaskCompleted(success: false)
            return
        }

        guard let photoLibraryStore, let faceRecognitionStore else {
            Diagnostics.shared.log("Face indexing background task could not run: stores unavailable.")
            task.setTaskCompleted(success: false)
            return
        }

        Diagnostics.shared.log("Face indexing background task delivered by iOS.")
        let operation = Task { @MainActor in
            await photoLibraryStore.prepareLibraryIfAuthorized()
            await faceRecognitionStore.runBackgroundIndexing(photoLibraryStore: photoLibraryStore)
        }
        task.expirationHandler = {
            Diagnostics.shared.log("Face indexing background task expiration requested by iOS.")
            operation.cancel()
        }

        await operation.value
        faceRecognitionStore.scheduleBackgroundIndexing(reason: "background task completed")
        task.setTaskCompleted(success: !operation.isCancelled)
        Diagnostics.shared.log("Face indexing background task completed; success \(!operation.isCancelled).")
    }
}

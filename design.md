# Picscry - Master Functional and Technical Design Document

This document serves as the comprehensive blueprint for a unified, all-in-one photo and video management ecosystem. It bridges elite accessibility features for blind and low-vision (BLV) individuals with a high-utility, aesthetic feature set tailored for a mainstream consumer audience.

## 1. System Overview & Core Philosophy

### 1.1 Purpose

The application serves as a complete replacement for the native system photo library tracker. It utilizes an intelligent, unified cloud-and-local hybrid architecture to automate digital clutter cleanups, execute multi-layered object and frame analysis, perform deep semantic searches, offer advanced multi-modal media editing, and provide an interactive, touch-responsive spatial exploration canvas.

### 1.2 Target Audience & Dual-Mode Value Proposition

The software enforces an Inclusive Design Framework. Instead of maintaining fragmented, separate apps or codebases, a single system handles two user experience tracks:

- **The Mainstream Track:** A highly visual, gesture-driven layout emphasizing automated device storage preservation, social media copy generation, intelligent background object deletion/stickers, and sleek non-destructive video and photo filtering.
- **The Accessibility Track:** A comprehensive, VoiceOver-optimized interface that converts pixel clusters, frame changes, and image adjustments into low-latency speech, spatial audio cues, tactile haptic grids, and interactive touch-responsive spatial layouts.

### 1.3 Architectural Integrity & User Agency

The system operates as a "Scout and Propose" workflow. Due to strict operating system sandboxing and privacy restrictions, silent or fully automated deletions and global metadata modifications are prohibited in the background. The app acts as an ambient background analyst, staging groupings and captions in a local queue database, leaving the final execution block to a unified user confirmation step in the foreground.

## 2. Unified System Architecture Diagram

```text
       ┌────────────────────────────────────────────────────────┐
       │               PhotoKit Asset Extraction                │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Local Triage (Apple Vision, AVFoundation)          │
       │   • FeaturePrint Clustering (Euclidean Distance)       │
       │   • Face Vector Extraction (VNDetectFaceLandmarks)     │
       │   • Objective Score Filtering (Face Capture Quality)   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼ (Filtered Finalist Array)
       ┌────────────────────────────────────────────────────────┐
       │          Anonymized Cloud VLM REST Pipeline            │
       │     (Gemini Flash / GPT-4o mini Payload Exchange)      │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼ (Structured JSON Response)
       ┌────────────────────────────────────────────────────────┐
       │             SQLite / SwiftData Local Cache             │
       │         (Pending Actions Queue & Search Index)         │
       └───────────────────────────┬────────────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
 ┌───────────────┐                                   ┌───────────────┐
 │ Mainstream UI │                                   │ Accessibility │
 │ • Card-Swipe  │                                   │ • VoiceOver   │
 │ • Filters/FX  │                                   │ • Haptics     │
 │ • Magic Eraser│                                   │ • Touch Map   │
 └───────┬───────┘                                   └───────┬───────┘
         │                                                   │
         └─────────────────────────┬─────────────────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             User-Triggered PhotoKit Write              │
       │    • Global Title Captions   • Batch Asset Purge       │
       └────────────────────────────────────────────────────────┘
```

## 3. Detailed Functional Feature Matrix

### 3.1 Advanced Duplicate & Similar-Media Triage

- **Functional Scope:** The app groups visually identical photos, rapid burst sequences, pocket-recorded videos, and near-identical video takes (e.g., changes in lighting, angle, expressions). It flags the lower-quality duplicates for deletion while retaining the subjective best shot.
- **Mainstream Experience:** Displays a visual deck highlighting the chosen "winner" alongside a breakdown of why the others were rejected (e.g., "Photo 2 has better exposure"). Users batch-clear hundreds of megabytes with a single gesture.
- **Accessibility Experience:** VoiceOver reads an interactive summary: "Cluster found: Kitchen Table Group. 4 duplicates found. I recommend keeping photo number 2 because everyone has their eyes open and is smiling naturally. Swipe right to listen to the selection logic and scene description."

### 3.2 Automated Document & Junk Segmentation (Vault)

- **Functional Scope:** Isolates temporary media (screenshots, receipts, barcodes, Wi-Fi password images, transit tickets) from personal family memories.
- **Mainstream Experience:** Moves these items automatically into a separate "Document Wallet" and "Temporary Storage Clean-up" tab, offering a 30-day automatic auto-purge.
- **Accessibility Experience:** Proactively reads extracted text from documents out loud using Text-to-Speech (TTS), letting BLV users query their real-world paper documents or receipts via the app.

### 3.3 Deep Semantic Conversational Search ("Ask My Library")

- **Functional Scope:** Replaces simple keyword matching with natural language search queries across both images and videos.
- **User Experience:** Implements a conversational chat interface. Users type or speak queries such as: "When was the last time I wore my green winter coat?" or "Find the video clip where the dog catches a ball at sunset." The app scans the localized descriptive metadata database to display or read back the exact dates, locations, and occurrences.

### 3.4 Multi-Modal Photo & Video Editing Suite

- **Functional Scope:** Provides comprehensive, non-destructive editing parameters (Exposure, Brightness, Contrast, Saturation, Highlights, Shadows) for both static images and video clips.
- **Mainstream Experience:** Standard sliders and wheels controlling filters and parameters. Includes a "Magic Eraser" to remove background distractions and an object extractor to lift elements into custom iMessage stickers.
- **Accessibility Experience:**
  - Sliders react via the system haptic engine (CoreHaptics), increasing vibrational texture and frequency as settings intensify, clicking firmly at the balanced midpoint.
  - An AI "Describe Changes" function updates users verbally after changes are applied: "Applied Warm Cinematic preset. Skin tones are now softer, shadows are elongated, and the background lighting has a golden-hour amber hue."

### 3.5 Explore by Touch (Spatial Canvas Map)

- **Functional Scope:** Translates flat pixels into an interactive, haptic-spatial framework on the device glass.
- **Mainstream Experience:** Tapping a subject isolates its pixel mask, enabling users to instantly lift the subject out as an iMessage sticker or blur the background (Portrait effect modification).
- **Accessibility Experience:** Users glide their finger smoothly over the display. When their finger crosses an object boundary (e.g., a person's face, a chair, a car), the device emits a distinct haptic pulse, and VoiceOver pans the object's spoken identity into the corresponding left/right stereo channel using 3D Spatial Audio soundscapes.

### 3.6 Smart Album Auto-Curation & Time Capsules

- **Functional Scope:** Aggregates related photos into structural narrative-driven event books automatically (e.g., "Weekend Trip to Goa"). Generates nostalgic "On This Day" Time Capsules.
- **Mainstream Experience:** Generates high-end cinematic slideshow highlights with automatically matched audio backdrops.
- **Accessibility Experience:** Generates an interactive audio retrospective narrator that explains the visual evolution of the memory over time via voice interface readouts.

### 3.7 Interactive Voice Memo Embedding

- **Functional Scope:** Attaches personal audio footnotes directly to individual media files or entire event collections.
- **User Experience:** Users hold a microphone button to record up to 30 seconds of context (e.g., ambient party noise, a child singing). The app runs speech-to-text to index the audio file, making the photo searchable by the spoken words, while allowing the audio to autoplay whenever the file is browsed.

## 4. Technical Frameworks & Step-by-Step Specifications

### Step 4.1: PhotoKit Stream Extraction & Change Pipeline

- **Frameworks:** Photos, PhotosUI
- **Technical Flow:**
  1. Initialize explicit security parameters via the app's configuration checklist (`NSPhotoLibraryUsageDescription`).
  2. Instantiate user verification check: `PHPhotoLibrary.requestAuthorization(for: .readWrite)`.
  3. Register a continuous lifecycle background tracking handler adhering to the `PHPhotoLibraryChangeObserver` protocol.
  4. Fetch asset arrays using `PHAsset.fetchAssets(with: .image, options: options)` and process properties (GPS arrays, creation timestamps, and localized device system metadata).

### Step 4.2: Local Signature Matching & Multi-Media Clustering

- **Frameworks:** Vision, AVFoundation
- **Technical Flow:**
  1. For static photos, invoke `VNGenerateImageFeaturePrintRequest` on the asset data.
  2. For video assets, instantiate an `AVAssetImageGenerator` to sample keyframes at uniform temporal intervals (e.g., 1 frame per second).
  3. Generate an image feature print signature matrix for each sampled video frame.
  4. Save the calculated vector arrays into a lightweight relational cache mapping back to local asset unique identifiers.
  5. Calculate similarity using the Euclidean distance metric between signatures. Group assets into a single duplicate array when variance falls below a tight threshold.

### Step 4.3: Local Facial Embedding and Identity Registry

- **Frameworks:** Vision, CoreData / SwiftData
- **Technical Flow:**
  1. Loop image data through a sequential `VNDetectFaceRectanglesRequest` workflow.
  2. Pipe successful face bounding boxes into a fine-grained `VNDetectFaceLandmarksRequest` block to generate normalized face feature embedding vectors.
  3. Map recurring matching vectors into a relational schema. If a face embedding repeatedly scores high similarity metrics across separate clusters, flag the identity as a recurring person.
  4. Provide a text entry field to let the user assign a manual string label (e.g., "Sarah"). Store this string-to-vector relation inside the database cache.

### Step 4.4: Local Technical Assessment & Triage Filtering

- **Frameworks:** Vision
- **Technical Flow:**
  1. Run a `VNDetectFaceCaptureQualityRequest` across the identified face objects in the duplicate group. This analyzes sharpness, eyes-open flags, and neutral-to-happy tracking scales.
  2. Evaluate holistic photo framing using `CalculateImageAestheticsScoresRequest` to parse exposure, saturation, and blur ratios.
  3. For documents, pass the asset through a `VNRecognizeTextRequest` block. If text tokens denote a receipt or invoice structure, flag the asset as junk storage and bypass the cloud VLM pipeline.
  4. Drop obvious photographic errors (heavy motion blur, closed eyes, or extreme under-exposure) on-device. Clean the duplicate pool down to the top two "finalist" assets to minimize cloud payload sizing.

### Step 4.5: Cloud REST VLM Integration & JSON Parsing

- **Frameworks:** Foundation (`URLSession`), CoreGraphics
- **Technical Flow:**
  1. Downsample the final image representations to a web-optimized canvas size to prevent network latency.
  2. Compile a payload container holding the base64-encoded image strings and an injected textual prompt. Append extracted metadata (e.g., date, city location text, and localized custom face names).
  3. Open a secure `URLSessionDataTask` transmission block to your remote endpoint (Gemini Flash or GPT-4o mini).
  4. Force a strict JSON structural format in your payload schema. Parse the returned output into a structured data model:

```json
{
  "winning_index": 0,
  "selection_reason": "This image captures Sarah laughing naturally with open eyes, whereas the alternative is out of focus with eyes closed.",
  "description_overall": "A bright outdoor celebration during an evening sunset.",
  "description_subjects": "Sarah is in the midground, wearing a white cotton summer shirt, smiling widely with her hands resting on a table.",
  "description_environment": "In the background, a green manicured lawn sweeps back to a softly blurred wooden fence line under an orange sky.",
  "social_caption": "Golden hour smiles with Sarah! 🌅✨",
  "search_indexing_tokens": ["Sarah", "sunset", "party", "garden", "laughing", "summer"]
}
```

Use code with caution.

### Step 4.6: Spatial Touch Target Mapping Architecture

- **Frameworks:** Vision, CoreHaptics, AVFAudio
- **Technical Flow:**
  1. For every retained asset, process a local `VNGenerateAttentionBasedSaliencyImageRequest` and an object tracking block to extract multi-point bounding boxes for identifiable objects.
  2. Overlay an invisible touch gesture interrupter (`DragGesture(minimumDistance: 0)`) over the image view. Map real-time finger coordinates to a normalized scale (0.0 to 1.0).
  3. If the normalized coordinates intersect with a known object bounding box, trigger a low-latency haptic feedback texture sequence via `CHHapticEngine`. Simultaneously, output an audio notification cue panned into the corresponding audio hardware channels using `AVAudioEnvironmentNode`.

### Step 4.7: Execution Layer & Batch PhotoKit Updates

- **Frameworks:** Photos
- **Technical Flow:**
  1. Instantiation of updates or deletion arrays cannot happen seamlessly in the background. Pull staged elements into a visible master approvals dashboard layout.
  2. Apply descriptions globally to the system photo application by wrapping text targets inside a `PHAssetChangeRequest` title update parameter:

```swift
PHAssetChangeRequest(for: winningAsset).title = combinedDescriptionField
```

  3. Apply bulk purges via `PHAssetChangeRequest.deleteAssets([inferiorAssets])`.
  4. Execute changes within a single system library block transaction (`PHPhotoLibrary.shared().performChanges`). This prompts the native iOS confirmation modal just once for the entire batch.

## 5. Background Task Lifecycle & Staging Matrix

To maintain high performance and stay within operating system battery optimization limits, background task execution must strictly follow system background guidelines.

- **Framework:** BackgroundTasks
- **User-Defined Control:** The application settings layer provides a configuration slider allowing the user to select the task execution window recurrence frequency (e.g., scanning every 24 hours, 48 hours, or weekly).
- **Operational Lifecycle Rules:**
  1. Register your background execution identifier within the app lifecycle framework using `BGProcessingTaskRequest`.
  2. Background execution slots are scheduled exclusively when the iOS system registers that the hardware is locked, connected to an active AC power source, and utilizing unmetered Wi-Fi.
  3. **The Scout Processing Scope:** The background handler reads newly added photos, computes local feature prints, runs facial vector comparisons, determines duplicate clusters, and holds images ready for cloud calling.
  4. The background process must not attempt direct metadata writes or silent asset deletions. It updates a local database table named `PendingActionsQueue`. Deletions, API description calls, and system saves remain paused until the user launches the application in the foreground.

## 6. Multi-Modal Accessibility UI/UX Design

The interface follows a unified inclusive design model: VoiceOver, Dynamic Type, reduced motion, contrast, and other accessibility settings may change semantics, labels, focus order, hit targets, and interaction affordances, but they must not silently switch the user into a materially different layout for the same workflow. This prevents regressions where screen-reader users and non-screen-reader users see different home-screen structures or stale UI behavior.

### 6.1 Unified User Interface Specification Matrix

- **Single Source Layout:** Core screens use one shared SwiftUI layout for all users. The Library home screen is a media grid for both VoiceOver and non-VoiceOver users, with each item exposed as a single accessible button whose label starts with the media type followed by date and time.
- **Visual Presentation:** Media grids maintain asset aspect ratios. Three items appear per row where horizontal size class permits; row height is determined by the tallest item in that row so photos and videos are visible without cropping.
- **Accessibility Adaptation:** Accessibility-specific work should enhance the shared UI with labels, hints, traits, focus order, larger hit areas, captions, haptics, and alternate actions. Do not create a separate VoiceOver-only list or alternate screen unless a future design explicitly documents why the shared layout cannot meet WCAG and Apple Human Interface Guidelines requirements.
- **Refresh Behavior:** Library refreshes must avoid clearing visible media before replacement data is ready. Keep the current grid visible during reloads, then publish refreshed summaries in a stable update to prevent flicker and repeated layout resets.
- **Media Quality:** Detail views request high-quality, original-dimension renditions for the selected asset when feasible. Grid thumbnails request high-quality renditions sized for their visible display area and preserve the original aspect ratio.

### 6.2 VoiceOver Interaction Flow Example

```text
[Screen Reading Element Focus Container]
 ├── VoiceOver Announcement: "Cluster Group: Garden Party. 3 duplicates staged for removal. 1 winner selected."
 ├── Swipe Right Gesture ──► "Selection Logic: Kept photo 1 because it captured Sarah laughing naturally, while photo 2 had closed eyes."
 ├── Swipe Right Gesture ──► "Read Complete Scene Analysis. Button."
 │                            └── [Double Tap]: Triggers multi-paragraph playback sequence:
 │                                 ├── Paragraph 1: "Overall Setting..."
 │                                 ├── Paragraph 2: "Subject Details..."
 │                                 └── Paragraph 3: "Background Layer..."
 ├── Swipe Right Gesture ──► "Launch Explore by Touch Screen. Button."
 │                            └── [Double Tap]: Opens full-screen tactile spatial exploration layer.
 └── Swipe Right Gesture ──► "Confirm Selection and Authorize Deletions. Button."
                              └── [Double Tap]: Triggers the unified native iOS security confirmation pop-up.
```

## 7. Strategic Token Optimization & Monetization Architecture

### 7.1 Token Preservation Layer

Cloud API calling strategies are aggressively throttled via on-device filters to ensure high profit margins.

- By running face capture quality checks and aesthetic scores locally on the client machine, the app drops blurred shots and closed-eye captures before touching the network.
- For a cluster of 5 near-identical burst shots, only the top 2 finalist images are transmitted to the cloud VLM. This slashes potential hosting overhead and token expenses by over 60%, maintaining highly predictable cloud costs.

### 7.2 Mainstream Subscription Architecture

- **The Pricing Tier Model:** Feature restrictions are implemented via a tiered subscription wrapper (e.g., monthly premium access or yearly utility passes).
- **The Processing Allocation Layer:** The subscription model grants users a fixed monthly "Cloud Token Credit Pool" (e.g., up to 300 detailed photo and video scene analyses per billing cycle). This balances infrastructure expenses with high app utility, keeping the platform accessible, sustainable, and highly profitable.

## 8. Development Implementation Plan

When executing this technical blueprint, code execution blocks must be built in this strict sequence:

1. **Module 1 (Storage Connection):** Core PhotoKit implementation. Handle user library read/write authorization loops and set up the `PHPhotoLibraryChangeObserver` infrastructure.
2. **Module 2 (Local Processing):** Vision Framework integration. Build the mathematical feature print clustering logic to group visually identical photos, parse local text elements (OCR), and sample video frames.
3. **Module 3 (Identity Database):** CoreData/SwiftData persistence architecture. Build data models tracking custom facial embeddings, recognized name labels, user-defined background configuration schedules, and the local `PendingActionsQueue`.
4. **Module 4 (Triage System):** Local quality analytics. Write the objective sorting filters utilizing `VNDetectFaceCaptureQualityRequest` to isolate and drop flawed files.
5. **Module 5 (Cloud Endpoint):** Remote network management layer. Build a secure REST client using `URLSession` to pass downsampled finalist assets, process structured prompts containing local metadata, and parse returned JSON data fields safely.
6. **Module 6 (Haptic/Tactile Engineering):** Interactive Spatial Engine. Set up the `DragGesture` coordinate mapping layer, process object detection bounding boxes, and implement spatial audio pan routines and CoreHaptics patterns.
7. **Module 7 (Interface Assembly):** Unified accessible interface configuration. Assemble fluid SwiftUI component views from one shared layout per workflow, making sure all items natively support visual gestures, editing controls, and screen-reader semantics without splitting VoiceOver and non-VoiceOver users into divergent UIs.

This document is now finalized, completely exhaustive, and contains no placeholder titles. You can provide this text directly to your building LLM as context to begin generating the code blocks. Let me know if you want to start expanding on the specific SwiftUI view protocols or database schemas!

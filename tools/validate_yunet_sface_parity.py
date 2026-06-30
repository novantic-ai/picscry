import argparse
import json
import pathlib
import urllib.request

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
DEFAULT_YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_PATH = ROOT / "Picscry" / "Resources" / "face_recognition_sface_2021dec.onnx"
DEFAULT_YUNET_PATH = ROOT / "Picscry" / "Resources" / "face_detection_yunet_2023mar.onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Compare OpenCV YuNet+SFace fixture output against Picscry YuNet/SFace debug exports."
    )
    parser.add_argument("fixtures", type=pathlib.Path, help="Directory containing fixture images.")
    parser.add_argument("--picscry-debug", type=pathlib.Path, help="Exported FaceEmbeddingDebug directory from the app.")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "YuNetSFaceParity")
    parser.add_argument("--sface-model", type=pathlib.Path, default=DEFAULT_SFACE_PATH)
    parser.add_argument("--yunet-model", type=pathlib.Path, default=DEFAULT_YUNET_PATH)
    parser.add_argument("--download", action="store_true", help="Download missing OpenCV Zoo ONNX files.")
    args = parser.parse_args()

    if args.download:
        ensure_file(args.sface_model, DEFAULT_SFACE_URL)
        ensure_file(args.yunet_model, DEFAULT_YUNET_URL)
    if not args.sface_model.exists():
        raise FileNotFoundError(f"Missing SFace model: {args.sface_model}")
    if not args.yunet_model.exists():
        raise FileNotFoundError(f"Missing YuNet model: {args.yunet_model}")

    args.output.mkdir(parents=True, exist_ok=True)
    opencv_records = run_opencv(args.fixtures, args.output, args.sface_model, args.yunet_model)
    picscry_records = load_picscry_debug(args.picscry_debug) if args.picscry_debug else []
    summary = {
        "fixtureImageCount": len(list_images(args.fixtures)),
        "opencvDetectedFaceCount": sum(len(record["faces"]) for record in opencv_records),
        "opencvImages": opencv_records,
        "picscryDebugFaceCount": len(picscry_records),
        "picscryDebug": picscry_records,
        "rankingAgreement": ranking_agreement(opencv_records, picscry_records),
    }
    (args.output / "yunet_sface_parity.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote parity report to {args.output / 'yunet_sface_parity.json'}")


def run_opencv(fixtures, output, sface_model, yunet_model):
    aligned_dir = output / "opencv_aligned"
    aligned_dir.mkdir(exist_ok=True)
    recognizer = cv2.FaceRecognizerSF_create(str(sface_model), "")
    detector = cv2.FaceDetectorYN_create(str(yunet_model), "", (640, 640), 0.9, 0.3, 5000)

    records = []
    for image_index, image_path in enumerate(list_images(fixtures), start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        rows = [] if faces is None else faces.astype(np.float32)
        face_records = []
        for face_index, row in enumerate(rows, start=1):
            aligned = recognizer.alignCrop(image, row)
            feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
            normalized = l2_normalized(feature)
            stem = f"{image_index:03d}_{face_index:02d}_{safe_name(image_path.stem)}"
            aligned_path = aligned_dir / f"{stem}.png"
            cv2.imwrite(str(aligned_path), aligned)
            face_records.append(
                {
                    "detectorRow": row.tolist(),
                    "alignedCrop": str(aligned_path),
                    "normalizedEmbedding": normalized.tolist(),
                }
            )
        records.append({"image": str(image_path), "faces": face_records})
    return records


def load_picscry_debug(debug_dir):
    if not debug_dir.exists():
        raise FileNotFoundError(f"Missing Picscry debug directory: {debug_dir}")
    records = []
    for path in sorted(debug_dir.glob("aligned_*.json")):
        payload = json.loads(path.read_text())
        metadata = payload.get("debugMetadata") or {}
        records.append(
            {
                "file": str(path),
                "debugIdentifier": payload.get("debugIdentifier"),
                "detectorBackend": metadata.get("detectorBackend"),
                "detectorRow": metadata.get("detectorRow"),
                "alignmentMethod": metadata.get("alignmentMethod"),
                "normalizedEmbedding": payload.get("normalizedEmbedding") or [],
            }
        )
    return records


def ranking_agreement(opencv_records, picscry_records):
    opencv_vectors = [
        np.asarray(face["normalizedEmbedding"], dtype=np.float32)
        for record in opencv_records
        for face in record["faces"]
    ]
    picscry_vectors = [
        np.asarray(record["normalizedEmbedding"], dtype=np.float32)
        for record in picscry_records
        if record.get("normalizedEmbedding")
    ]
    if len(opencv_vectors) < 2 or len(picscry_vectors) < 2 or len(opencv_vectors) != len(picscry_vectors):
        return {
            "status": "not_comparable",
            "reason": "Need the same number of at least two OpenCV and Picscry vectors.",
        }

    opencv_pairs = pair_ranking(opencv_vectors)
    picscry_pairs = pair_ranking(picscry_vectors)
    same_top_pair = opencv_pairs[0]["pair"] == picscry_pairs[0]["pair"]
    return {
        "status": "compared",
        "sameTopPair": same_top_pair,
        "opencvTopPair": opencv_pairs[0],
        "picscryTopPair": picscry_pairs[0],
    }


def pair_ranking(vectors):
    pairs = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            pairs.append(
                {
                    "pair": [left, right],
                    "similarity": float(np.dot(vectors[left], vectors[right])),
                }
            )
    return sorted(pairs, key=lambda item: item["similarity"], reverse=True)


def list_images(fixtures):
    return sorted(
        path
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.heic")
        for path in fixtures.glob(extension)
    )


def ensure_file(path, url):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def l2_normalized(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def safe_name(value):
    return "".join(character if character.isalnum() else "_" for character in value)


if __name__ == "__main__":
    main()

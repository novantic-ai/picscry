import argparse
import json
import pathlib
import urllib.request

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ONNX_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
DEFAULT_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEFAULT_ONNX_PATH = ROOT / "Picscry" / "Resources" / "face_recognition_sface_2021dec.onnx"
DEFAULT_DETECTOR_PATH = ROOT / "tools" / "face_detection_yunet_2023mar.onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenCV SFace on fixture images and export aligned crops, features, and pairwise cosine scores."
    )
    parser.add_argument("fixtures", type=pathlib.Path, help="Directory containing fixture images.")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "SFaceOpenCVValidation")
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--detector", type=pathlib.Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--download", action="store_true", help="Download missing OpenCV Zoo ONNX files.")
    args = parser.parse_args()

    if args.download:
        ensure_file(args.model, DEFAULT_ONNX_URL)
        ensure_file(args.detector, DEFAULT_DETECTOR_URL)

    if not args.model.exists():
        raise FileNotFoundError(f"Missing SFace ONNX model: {args.model}")
    if not args.detector.exists():
        raise FileNotFoundError(f"Missing YuNet detector ONNX model: {args.detector}")

    images = sorted(
        path
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.heic")
        for path in args.fixtures.glob(extension)
    )
    if not images:
        raise ValueError(f"No fixture images found in {args.fixtures}")

    args.output.mkdir(parents=True, exist_ok=True)
    aligned_dir = args.output / "aligned"
    vector_dir = args.output / "vectors"
    aligned_dir.mkdir(exist_ok=True)
    vector_dir.mkdir(exist_ok=True)

    recognizer = cv2.FaceRecognizerSF_create(str(args.model), "")
    detector = cv2.FaceDetectorYN_create(str(args.detector), "", (320, 320))

    records = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            print(f"No face detected: {image_path}")
            continue

        face = max(faces, key=lambda row: row[2] * row[3])
        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
        normalized = l2_normalized(feature)

        stem = f"{index:03d}_{safe_name(image_path.stem)}"
        aligned_path = aligned_dir / f"{stem}.png"
        vector_path = vector_dir / f"{stem}.json"
        cv2.imwrite(str(aligned_path), aligned)
        vector_path.write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "alignedCrop": str(aligned_path),
                    "rawEmbedding": feature.tolist(),
                    "normalizedEmbedding": normalized.tolist(),
                },
                indent=2,
            )
        )
        records.append(
            {
                "image": str(image_path),
                "alignedCrop": str(aligned_path),
                "vector": str(vector_path),
                "embedding": normalized,
            }
        )

    matrix = pairwise_cosine_matrix([record["embedding"] for record in records])
    summary = {
        "imageCount": len(records),
        "images": [
            {
                "image": record["image"],
                "alignedCrop": record["alignedCrop"],
                "vector": record["vector"],
            }
            for record in records
        ],
        "pairwiseCosine": matrix.tolist(),
    }
    (args.output / "pairwise_cosine.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote OpenCV SFace validation output to {args.output}")


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


def pairwise_cosine_matrix(vectors):
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.zeros((len(vectors), len(vectors)), dtype=np.float32)
    for left, left_vector in enumerate(vectors):
        for right, right_vector in enumerate(vectors):
            matrix[left, right] = float(np.dot(left_vector, right_vector))
    return matrix


def safe_name(value):
    return "".join(character if character.isalnum() else "_" for character in value)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import pathlib
import statistics
import tempfile
import zipfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare OpenCV SFace ONNX features against Picscry Core ML features exported from the app."
    )
    parser.add_argument("--debug-bundle", required=True, type=pathlib.Path)
    parser.add_argument("--sface-onnx", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if not args.debug_bundle.exists():
        raise FileNotFoundError(f"Missing debug bundle: {args.debug_bundle}")
    if not args.sface_onnx.exists():
        raise FileNotFoundError(f"Missing SFace ONNX model: {args.sface_onnx}")

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        extracted = pathlib.Path(temp_dir)
        with zipfile.ZipFile(args.debug_bundle) as archive:
            archive.extractall(extracted)

        debug_dir = extracted / "FaceEmbeddingDebug"
        if not debug_dir.exists():
            debug_dir = extracted

        items = load_items(debug_dir)
        if not items:
            raise ValueError("No aligned PNG + JSON embedding pairs found in the debug bundle.")

        recognizer = cv2.FaceRecognizerSF_create(str(args.sface_onnx), "")
        records = []
        for item in items:
            image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read aligned crop: {item['image_path']}")

            opencv_raw = recognizer.feature(image).reshape(-1).astype(np.float32)
            opencv_norm = l2_normalized(opencv_raw)
            picscry_raw = np.asarray(item["payload"]["rawEmbedding"], dtype=np.float32)
            picscry_norm = np.asarray(item["payload"].get("normalizedEmbedding") or l2_normalized(picscry_raw), dtype=np.float32)

            if opencv_norm.shape != picscry_norm.shape:
                raise ValueError(
                    f"Feature shape mismatch for {item['image_path'].name}: "
                    f"OpenCV {opencv_norm.shape}, Picscry {picscry_norm.shape}"
                )

            records.append(
                {
                    "name": item["image_path"].stem,
                    "debugIdentifier": item["payload"].get("debugIdentifier"),
                    "opencvRaw": opencv_raw,
                    "opencvNorm": opencv_norm,
                    "picscryRaw": picscry_raw,
                    "picscryNorm": picscry_norm,
                }
            )

    opencv_matrix = pairwise_cosine_matrix([record["opencvNorm"] for record in records])
    picscry_matrix = pairwise_cosine_matrix([record["picscryNorm"] for record in records])
    same_image_rows = same_image_comparison(records)

    write_matrix_csv(args.output / "opencv_pairwise_cosine.csv", records, opencv_matrix)
    write_matrix_csv(args.output / "picscry_pairwise_cosine.csv", records, picscry_matrix)
    write_same_image_csv(args.output / "opencv_vs_picscry_same_image.csv", same_image_rows)

    summary = build_summary(records, opencv_matrix, picscry_matrix, same_image_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "summary.txt").write_text(summary_text(summary))
    print(summary_text(summary))


def load_items(debug_dir):
    items = []
    for image_path in sorted(debug_dir.glob("aligned_*.png")):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text())
        items.append(
            {
                "image_path": image_path,
                "json_path": json_path,
                "payload": payload,
            }
        )
    return items


def l2_normalized(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def pairwise_cosine_matrix(vectors):
    matrix = np.zeros((len(vectors), len(vectors)), dtype=np.float32)
    for left, left_vector in enumerate(vectors):
        for right, right_vector in enumerate(vectors):
            matrix[left, right] = float(np.dot(left_vector, right_vector))
    return matrix


def off_diagonal_values(matrix):
    return [
        float(matrix[row, column])
        for row in range(matrix.shape[0])
        for column in range(matrix.shape[1])
        if row < column
    ]


def stats(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def same_image_comparison(records):
    rows = []
    for record in records:
        feature_delta = record["opencvNorm"] - record["picscryNorm"]
        rows.append(
            {
                "name": record["name"],
                "debugIdentifier": record["debugIdentifier"],
                "cosine": float(np.dot(record["opencvNorm"], record["picscryNorm"])),
                "normalizedL2": float(np.linalg.norm(feature_delta)),
                "opencvRawNorm": float(np.linalg.norm(record["opencvRaw"])),
                "picscryRawNorm": float(np.linalg.norm(record["picscryRaw"])),
            }
        )
    return rows


def write_matrix_csv(path, records, matrix):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [record["name"] for record in records])
        for row_index, record in enumerate(records):
            writer.writerow([record["name"]] + [f"{float(value):.8f}" for value in matrix[row_index]])


def write_same_image_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "debugIdentifier",
                "cosine",
                "normalizedL2",
                "opencvRawNorm",
                "picscryRawNorm",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(records, opencv_matrix, picscry_matrix, same_image_rows):
    opencv_pairs = off_diagonal_values(opencv_matrix)
    picscry_pairs = off_diagonal_values(picscry_matrix)
    same_image_cosines = [row["cosine"] for row in same_image_rows]
    same_image_l2 = [row["normalizedL2"] for row in same_image_rows]
    return {
        "imageCount": len(records),
        "opencvPairwiseCosine": stats(opencv_pairs),
        "picscryPairwiseCosine": stats(picscry_pairs),
        "opencvVsPicscrySameImageCosine": stats(same_image_cosines),
        "opencvVsPicscrySameImageNormalizedL2": stats(same_image_l2),
        "interpretation": interpret(opencv_pairs, picscry_pairs, same_image_cosines),
    }


def interpret(opencv_pairs, picscry_pairs, same_image_cosines):
    opencv_median = statistics.median(opencv_pairs) if opencv_pairs else None
    picscry_median = statistics.median(picscry_pairs) if picscry_pairs else None
    same_image_median = statistics.median(same_image_cosines) if same_image_cosines else None

    if opencv_median is None or picscry_median is None:
        return "Not enough images for pairwise comparison."
    if opencv_median < 0.95 and picscry_median > 0.98:
        return "OpenCV features are more diverse than Picscry features. Core ML conversion/output parity is the leading suspect."
    if opencv_median > 0.98 and picscry_median > 0.98:
        return "Both OpenCV and Picscry features are highly similar on these crops. Inspect crops and test a more diverse fixture set."
    if same_image_median is not None and same_image_median > 0.98 and abs(opencv_median - picscry_median) < 0.03:
        return "OpenCV and Picscry look close on this bundle. Health sampling or fixture diversity may be the issue."
    return "Parity is inconclusive. Inspect same-image cosine and pairwise matrices."


def summary_text(summary):
    lines = [
        "SFace Parity Summary",
        f"Images: {summary['imageCount']}",
        format_stats("OpenCV pairwise cosine", summary["opencvPairwiseCosine"]),
        format_stats("Picscry pairwise cosine", summary["picscryPairwiseCosine"]),
        format_stats("OpenCV vs Picscry same-image cosine", summary["opencvVsPicscrySameImageCosine"]),
        format_stats("OpenCV vs Picscry same-image normalized L2", summary["opencvVsPicscrySameImageNormalizedL2"]),
        f"Interpretation: {summary['interpretation']}",
    ]
    return "\n".join(lines) + "\n"


def format_stats(label, values):
    return (
        f"{label}: count={values['count']}, "
        f"min={format_value(values['min'])}, "
        f"median={format_value(values['median'])}, "
        f"max={format_value(values['max'])}"
    )


def format_value(value):
    if value is None:
        return "n/a"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()

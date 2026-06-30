import pathlib
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper

from coremltools.models import MLModel, datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFERRED_DYNAMIC_ONNX_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
FIXED_ONNX_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
PREFERRED_DYNAMIC_ONNX_PATH = ROOT / "Picscry" / "Resources" / "face_detection_yunet_2026may.onnx"
ONNX_PATH = ROOT / "Picscry" / "Resources" / "face_detection_yunet_2023mar.onnx"
OUTPUT_PATH = ROOT / "Picscry" / "Resources" / "FaceDetectionModel.mlmodel"

RAW_OUTPUTS = [
    ("cls_8_raw", "258", (1, 80, 80)),
    ("cls_16_raw", "260", (1, 40, 40)),
    ("cls_32_raw", "262", (1, 20, 20)),
    ("obj_8_raw", "270", (1, 80, 80)),
    ("obj_16_raw", "272", (1, 40, 40)),
    ("obj_32_raw", "274", (1, 20, 20)),
    ("bbox_8_raw", "264", (4, 80, 80)),
    ("bbox_16_raw", "266", (4, 40, 40)),
    ("bbox_32_raw", "268", (4, 20, 20)),
    ("kps_8_raw", "276", (10, 80, 80)),
    ("kps_16_raw", "278", (10, 40, 40)),
    ("kps_32_raw", "280", (10, 20, 20)),
]


def array(initializers, name):
    return np.asarray(initializers[name], dtype=np.float32)


def attrs(node):
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def main():
    ensure_onnx()
    print(
        "Using fixed 640x640 YuNet 2023mar for this Core ML export; "
        "the 2026may dynamic-shape re-export is downloaded for provenance, "
        "but this converter intentionally preserves the fixed raw OpenCV heads "
        "that the app decodes in Swift."
    )
    model = onnx.load(str(ONNX_PATH))
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }

    builder = NeuralNetworkBuilder(
        input_features=[("input", datatypes.Array(3, 640, 640))],
        output_features=[(name, datatypes.Array(*shape)) for name, _, shape in RAW_OUTPUTS],
        mode=None,
    )

    raw_output_sources = {source for _, source, _ in RAW_OUTPUTS}
    for index, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{index}"
        node_attrs = attrs(node)
        input_name = node.input[0]
        output_name = node.output[0]

        if node.op_type == "Conv":
            weights = array(initializers, node.input[1])
            coreml_weights = np.transpose(weights, (2, 3, 1, 0))
            pads = node_attrs.get("pads", [0, 0, 0, 0])
            strides = node_attrs.get("strides", [1, 1])
            dilations = node_attrs.get("dilations", [1, 1])
            groups = int(node_attrs.get("group", 1))
            bias = array(initializers, node.input[2]) if len(node.input) > 2 and node.input[2] in initializers else None
            builder.add_convolution(
                name=name,
                kernel_channels=int(weights.shape[1]),
                output_channels=int(weights.shape[0]),
                height=int(weights.shape[2]),
                width=int(weights.shape[3]),
                stride_height=int(strides[0]),
                stride_width=int(strides[1]),
                border_mode="valid",
                groups=groups,
                W=coreml_weights,
                b=bias,
                has_bias=bias is not None,
                input_name=input_name,
                output_name=output_name,
                dilation_factors=[int(dilations[0]), int(dilations[1])],
                padding_top=int(pads[0]),
                padding_left=int(pads[1]),
                padding_bottom=int(pads[2]),
                padding_right=int(pads[3]),
            )
        elif node.op_type == "Relu":
            builder.add_activation(
                name=name,
                non_linearity="RELU",
                input_name=input_name,
                output_name=output_name,
            )
        elif node.op_type == "MaxPool":
            kernel = node_attrs.get("kernel_shape", [1, 1])
            strides = node_attrs.get("strides", [1, 1])
            pads = node_attrs.get("pads", [0, 0, 0, 0])
            builder.add_pooling(
                name=name,
                height=int(kernel[0]),
                width=int(kernel[1]),
                stride_height=int(strides[0]),
                stride_width=int(strides[1]),
                layer_type="MAX",
                padding_type="VALID",
                input_name=input_name,
                output_name=output_name,
                padding_top=int(pads[0]),
                padding_left=int(pads[1]),
                padding_bottom=int(pads[2]),
                padding_right=int(pads[3]),
            )
        elif node.op_type == "Resize":
            scales = array(initializers, node.input[2])
            builder.add_upsample(
                name=name,
                scaling_factor_h=float(scales[2]),
                scaling_factor_w=float(scales[3]),
                input_name=input_name,
                output_name=output_name,
                mode="NN",
            )
        elif node.op_type == "Add":
            builder.add_elementwise(
                name=name,
                input_names=list(node.input),
                output_name=output_name,
                mode="ADD",
            )
        elif node.op_type in {"Transpose", "Reshape", "Sigmoid"}:
            # The app decodes OpenCV-compatible YuNet detections from the raw
            # NCHW heads before ONNX transpose/reshape/sigmoid post-processing.
            continue
        else:
            raise ValueError(f"Unsupported ONNX op {node.op_type} at node {index}: {name}")

        if output_name in raw_output_sources:
            for feature_name, source_name, _ in RAW_OUTPUTS:
                if source_name == output_name:
                    builder.add_activation(
                        name=f"{feature_name}_identity",
                        non_linearity="LINEAR",
                        input_name=source_name,
                        output_name=feature_name,
                        params=[1.0, 0.0],
                    )

    spec = builder.spec
    spec.description.metadata.shortDescription = "OpenCV Zoo YuNet face detection model, fixed 640x640 Core ML export."
    spec.description.metadata.author = "OpenCV Zoo / Shiqi Yu"
    spec.description.metadata.license = "MIT"
    spec.description.metadata.versionString = "2023mar"
    spec.specificationVersion = 4
    MLModel(spec).save(str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")


def ensure_onnx():
    if not PREFERRED_DYNAMIC_ONNX_PATH.exists():
        PREFERRED_DYNAMIC_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {PREFERRED_DYNAMIC_ONNX_URL}")
        urllib.request.urlretrieve(PREFERRED_DYNAMIC_ONNX_URL, PREFERRED_DYNAMIC_ONNX_PATH)
    if ONNX_PATH.exists():
        return
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FIXED_ONNX_URL}")
    urllib.request.urlretrieve(FIXED_ONNX_URL, ONNX_PATH)


if __name__ == "__main__":
    main()

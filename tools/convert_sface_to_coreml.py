import pathlib
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper

import coremltools as ct
from coremltools.models import MLModel, datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORIGINAL_ONNX_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
ORIGINAL_ONNX_PATH = ROOT / "Picscry" / "Resources" / "face_recognition_sface_2021dec.onnx"
ONNX_PATH = ROOT / "Picscry" / "Resources" / "face_recognition_sface_2021dec_simplified.onnx"
OUTPUT_PATH = ROOT / "Picscry" / "Resources" / "FaceEmbeddingModel.mlmodel"


def array(initializers, name):
    return np.asarray(initializers[name], dtype=np.float32)


def attrs(node):
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def main():
    ensure_simplified_onnx()
    model = onnx.load(str(ONNX_PATH))
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }

    builder = NeuralNetworkBuilder(
        input_features=[("data", datatypes.Array(3, 112, 112))],
        output_features=[("fc1", datatypes.Array(128))],
        mode=None,
    )

    for index, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{index}"
        node_attrs = attrs(node)
        input_name = node.input[0]
        output_name = node.output[0]

        if node.op_type == "Sub":
            # The OpenCV model starts with (data - 127.5) * 0.0078125.
            # Fold subtraction into a scale layer bias.
            continue

        if node.op_type == "Mul":
            sub_value = float(array(initializers, node.input[0] if node.input[0] in initializers else "scalar_op1")[0]) if node.input[0] in initializers else float(array(initializers, "scalar_op1")[0])
            scale_value = float(array(initializers, "scalar_op2")[0])
            builder.add_scale(
                name=name,
                W=np.array([scale_value], dtype=np.float32),
                b=np.array([-sub_value * scale_value], dtype=np.float32),
                has_bias=True,
                input_name="data",
                output_name=output_name,
                shape_scale=[1],
                shape_bias=[1],
            )
            continue

        if node.op_type == "Conv":
            weights = array(initializers, node.input[1])
            # ONNX Conv stores kernels as OIHW. Core ML neural-network builder
            # expects HWIO, including depthwise/grouped convolutions.
            coreml_weights = np.transpose(weights, (2, 3, 1, 0))
            pads = node_attrs.get("pads", [0, 0, 0, 0])
            strides = node_attrs.get("strides", [1, 1])
            dilations = node_attrs.get("dilations", [1, 1])
            groups = int(node_attrs.get("group", 1))
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
                b=None,
                has_bias=False,
                input_name=input_name,
                output_name=output_name,
                dilation_factors=[int(dilations[0]), int(dilations[1])],
                padding_top=int(pads[0]),
                padding_left=int(pads[1]),
                padding_bottom=int(pads[2]),
                padding_right=int(pads[3]),
            )
            continue

        if node.op_type == "BatchNormalization":
            builder.add_batchnorm(
                name=name,
                channels=int(array(initializers, node.input[1]).shape[0]),
                gamma=array(initializers, node.input[1]),
                beta=array(initializers, node.input[2]),
                mean=array(initializers, node.input[3]),
                variance=array(initializers, node.input[4]),
                input_name=input_name,
                output_name=output_name,
                compute_mean_var=False,
                instance_normalization=False,
                epsilon=float(node_attrs.get("epsilon", 1e-5)),
            )
            continue

        if node.op_type == "PRelu":
            builder.add_activation(
                name=name,
                non_linearity="PRELU",
                input_name=input_name,
                output_name=output_name,
                params=array(initializers, node.input[1]),
            )
            continue

        if node.op_type == "Dropout":
            builder.add_activation(
                name=name,
                non_linearity="LINEAR",
                input_name=input_name,
                output_name=output_name,
                params=[1.0, 0.0],
            )
            continue

        if node.op_type == "Flatten":
            builder.add_flatten(
                name=name,
                mode=0,
                input_name=input_name,
                output_name=output_name,
            )
            continue

        if node.op_type == "Gemm":
            weights = array(initializers, node.input[1])
            if int(node_attrs.get("transB", 0)) != 1:
                weights = weights.T
            builder.add_inner_product(
                name=name,
                W=weights,
                b=array(initializers, node.input[2]),
                input_channels=int(weights.shape[1]),
                output_channels=int(weights.shape[0]),
                has_bias=True,
                input_name=input_name,
                output_name=output_name,
            )
            continue

        raise ValueError(f"Unsupported ONNX op {node.op_type} at node {index}: {name}")

    spec = builder.spec
    spec.description.metadata.shortDescription = "OpenCV Zoo SFace MobileFaceNet face embedding model."
    spec.description.metadata.author = "OpenCV Zoo / Yaoyao Zhong"
    spec.description.metadata.license = "Apache License 2.0"
    spec.description.metadata.versionString = "2021dec"
    spec.specificationVersion = 4
    MLModel(spec).save(str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")


def ensure_simplified_onnx():
    if ONNX_PATH.exists():
        return

    if not ORIGINAL_ONNX_PATH.exists():
        ORIGINAL_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {ORIGINAL_ONNX_URL}")
        urllib.request.urlretrieve(ORIGINAL_ONNX_URL, ORIGINAL_ONNX_PATH)

    model = onnx.load(str(ORIGINAL_ONNX_PATH))
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    kept_inputs = [
        value
        for value in model.graph.input
        if value.name == "data" or value.name not in initializer_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)
    onnx.checker.check_model(model)
    onnx.save(model, str(ONNX_PATH))


if __name__ == "__main__":
    main()

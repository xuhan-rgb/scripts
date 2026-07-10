import argparse
import os
import sys

import numpy as np


def create_onnx_model(onnx_model_path, opset_version):
    import onnx
    from onnx import TensorProto, helper

    onnx_model_path = os.path.expanduser(onnx_model_path)
    parent_dir = os.path.dirname(onnx_model_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    conv_weight = np.random.randn(1, 1, 3, 3).astype(np.float32)
    conv_bias = np.random.randn(1).astype(np.float32)
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 28, 28]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 28, 28]
    )
    conv_weights_tensor = helper.make_tensor(
        "conv_weight", TensorProto.FLOAT, conv_weight.shape, conv_weight.flatten()
    )
    conv_bias_tensor = helper.make_tensor(
        "conv_bias", TensorProto.FLOAT, conv_bias.shape, conv_bias.flatten()
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight", "conv_bias"],
        outputs=["conv_output"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
    )
    relu_node = helper.make_node("Relu", inputs=["conv_output"], outputs=["output"])
    graph = helper.make_graph(
        [conv_node, relu_node],
        "simple_cnn_model",
        [input_tensor],
        [output_tensor],
        [conv_weights_tensor, conv_bias_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx-example",
        opset_imports=[helper.make_opsetid("ai.onnx", opset_version)],
    )
    onnx.save(model, onnx_model_path)
    print(f"ONNX model has been saved to {onnx_model_path}")


def get_cuda_onnxruntime():
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Could not import ONNX Runtime: {error}") from error

    print("ONNX Runtime version:", ort.__version__)
    available_providers = ort.get_available_providers()
    print("Available execution providers:")
    print("\n".join(available_providers))
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Run get_onnx_dependencies.py "
            "to inspect missing CUDA/cuDNN libraries."
        )
    return ort


def test_onnx_model(ort, onnx_model_path):
    onnx_model_path = os.path.expanduser(onnx_model_path)
    print(f"Loading ONNX model from {onnx_model_path}")
    session = ort.InferenceSession(
        onnx_model_path, providers=["CUDAExecutionProvider"]
    )
    session_providers = session.get_providers()
    print("ONNX model compute providers:", session_providers)
    if "CUDAExecutionProvider" not in session_providers:
        raise RuntimeError("ONNX Runtime created a session without CUDAExecutionProvider.")

    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_data})
    print("Output shape:", output[0].shape)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a minimal ONNX model and verify CUDA ONNX Runtime inference."
    )
    parser.add_argument("opset_version", nargs="?", type=int, default=16)
    args = parser.parse_args(argv)
    if args.opset_version <= 0:
        parser.error("opset_version must be a positive integer")

    try:
        ort = get_cuda_onnxruntime()
        onnx_model_path = "~/.cache/simple_cnn.onnx"
        create_onnx_model(onnx_model_path, args.opset_version)
        test_onnx_model(ort, onnx_model_path)
    except (ImportError, OSError, RuntimeError) as error:
        print(f"ONNX CUDA test failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import hashlib
import os
import tempfile

import numpy as np
import pycuda.autoinit  # noqa: F401 - initializes the CUDA context used by PyCUDA.
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
import torch.onnx


def get_file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


class TensorRTModel:
    """Run a static-shape, single-input TensorRT engine with NumPy arrays."""

    def __init__(self, onnx_model_path=None, engine_model_path=None):
        if bool(onnx_model_path) == bool(engine_model_path):
            raise ValueError(
                "Provide exactly one of onnx_model_path or engine_model_path."
            )

        self.engine = None
        self.context = None
        self.runtime = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.d_input = None
        self.d_outputs = []
        self.h_input = None
        self.h_outputs = []
        self.input_name = None
        self.input_shape = None
        self.input_dtype = None
        self.output_names = []

        if engine_model_path:
            self._load_engine(engine_model_path)
        else:
            self._build_engine_from_onnx(onnx_model_path)
        print("TensorRT model initialized successfully!")

    def _build_engine_from_onnx(self, onnx_model_path):
        model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
        engine_name = "{}_{}_trt{}.engine".format(
            model_name, get_file_md5(onnx_model_path), trt.__version__
        )
        cache_dir = os.path.expanduser("~/.cache/model")
        os.makedirs(cache_dir, exist_ok=True)
        engine_path = os.path.join(cache_dir, engine_name)
        print(f"Engine file path: {engine_path}")

        if os.path.exists(engine_path):
            try:
                self._load_engine(engine_path)
                print(f"Loaded cached engine from {engine_path}.")
                return
            except RuntimeError as error:
                print(f"Cached engine is unusable; rebuilding it: {error}")
                os.remove(engine_path)

        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(onnx_model_path):
            errors = "\n".join(
                str(parser.get_error(index)) for index in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parsing failed for {onnx_model_path}:\n{errors}")

        config = builder.create_builder_config()
        print("Building TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT failed to build an engine for {onnx_model_path}.")

        temp_path = None
        try:
            descriptor, temp_path = tempfile.mkstemp(
                dir=cache_dir, prefix=".engine-", suffix=".tmp"
            )
            with os.fdopen(descriptor, "wb") as file:
                file.write(serialized_engine)
            os.replace(temp_path, engine_path)
        except OSError:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        print("Engine file generated successfully!")
        self._load_engine(engine_path)

    def _load_engine(self, engine_model_path):
        try:
            with open(engine_model_path, "rb") as file:
                self.runtime = trt.Runtime(self.logger)
                self.engine = self.runtime.deserialize_cuda_engine(file.read())

            if self.engine is None:
                raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_model_path}")

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(
                    f"Could not create an execution context for: {engine_model_path}"
                )

            self._allocate_memory()
        except Exception:
            self.release_resources()
            raise

    def _get_static_shape(self, tensor_name):
        shape = tuple(self.context.get_tensor_shape(tensor_name))
        if any(dimension < 0 for dimension in shape):
            raise ValueError(
                f"Dynamic tensor shape {shape} for '{tensor_name}' is not supported."
            )
        return shape

    def _bind_tensor_address(self, tensor_name, allocation):
        if not self.context.set_tensor_address(tensor_name, int(allocation)):
            raise RuntimeError(f"Failed to bind memory for TensorRT tensor '{tensor_name}'.")

    def _allocate_memory(self):
        tensor_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        ]
        input_names = [
            name
            for name in tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

        if len(input_names) != 1:
            raise ValueError(
                "TensorRTModel supports exactly one input tensor; found {}: {}".format(
                    len(input_names), input_names
                )
            )
        if not self.output_names:
            raise ValueError("TensorRT engine has no output tensors.")

        self.input_name = input_names[0]
        self.input_shape = self._get_static_shape(self.input_name)
        self.input_dtype = np.dtype(
            trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        )
        self.h_input = np.empty(self.input_shape, dtype=self.input_dtype)
        self.h_outputs = []

        for output_name in self.output_names:
            output_shape = self._get_static_shape(output_name)
            output_dtype = np.dtype(
                trt.nptype(self.engine.get_tensor_dtype(output_name))
            )
            self.h_outputs.append(np.empty(output_shape, dtype=output_dtype))

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_outputs = [
            cuda.mem_alloc(output.nbytes) for output in self.h_outputs
        ]
        self._bind_tensor_address(self.input_name, self.d_input)
        for output_name, allocation in zip(self.output_names, self.d_outputs):
            self._bind_tensor_address(output_name, allocation)

        print("Input tensor:", self.input_name, self.input_shape, self.input_dtype)
        print("Output tensors:", self.output_names)

    def infer(self, input_data):
        if self.context is None or self.engine is None:
            raise RuntimeError("Engine or context is not initialized.")

        input_array = np.asarray(input_data, dtype=self.input_dtype)
        if input_array.shape != self.input_shape:
            raise ValueError(
                f"Input shape {input_array.shape} does not match "
                f"engine shape {self.input_shape}."
            )

        np.copyto(self.h_input, input_array)
        cuda.memcpy_htod(self.d_input, self.h_input)
        if not self.context.execute_async_v3(0):
            raise RuntimeError("TensorRT inference execution failed.")

        for host_output, device_output in zip(self.h_outputs, self.d_outputs):
            cuda.memcpy_dtoh(host_output, device_output)
        return self.h_outputs

    def release_resources(self):
        if self.d_input is not None:
            self.d_input.free()
            self.d_input = None
        for device_output in self.d_outputs:
            device_output.free()
        self.d_outputs = []
        self.h_input = None
        self.h_outputs = []
        self.context = None
        self.engine = None
        self.runtime = None


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))
            self.fc1.bias.copy_(torch.tensor([0.1, -0.1]))

    def forward(self, inputs):
        return self.fc1(inputs)


if __name__ == "__main__":
    model = SimpleModel()
    sample_input = torch.randn(1, 3)
    onnx_path = "simple_model.onnx"
    torch.onnx.export(
        model, sample_input, onnx_path, input_names=["input"], output_names=["output"]
    )
    print(f"ONNX model saved to {onnx_path}")

    tensorrt_model = None
    try:
        tensorrt_model = TensorRTModel(onnx_model_path=onnx_path)
        outputs = tensorrt_model.infer(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        print("Inference output:", outputs)
    finally:
        if tensorrt_model is not None:
            tensorrt_model.release_resources()

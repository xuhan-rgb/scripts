import torch
import torch.nn as nn
import torch.onnx
import tensorrt as trt # 10.5
import numpy as np
import pycuda.autoinit # 这个模块不用，但是一定要引用
import pycuda.driver as cudart # 2024.1.2
import os
import hashlib

def get_file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取文件并更新MD5值
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

class TensorRTModel:
    def __init__(self, onnx_model_path=None, engine_model_path=None):
        self.onnx_model_path = onnx_model_path
        self.engine_model_path = engine_model_path
        self.engine = None
        self.context = None
        self.runtime = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        # Initialize variables for memory allocation
        self.d_input = None
        self.d_outputs = None
        self.h_input = None
        self.h_outputs = None
        self.l_tensor_name = None

        if engine_model_path:
            self._load_engine(engine_model_path)
        elif onnx_model_path:
            self._build_engine_from_onnx(onnx_model_path)
        else:
            raise ValueError("Either onnx_model_path or engine_model_path must be provided.")
        print("TensorRT model initialized successfully!")

    def _build_engine_from_onnx(self, onnx_model_path):
        md5_value = get_file_md5(onnx_model_path)
        engine_path = onnx_model_path.split('/')[-1].replace(".onnx", "_{}.engine".format(md5_value))
        os.makedirs(os.path.expanduser("~") + '/.cache/model/', exist_ok=True)
        engine_path = os.path.expanduser("~") + '/.cache/model/' + engine_path  # 缓存路径
        print(f"Engine file path: {engine_path}")
        if os.path.exists(engine_path):
            print(f"Engine file already exists at {engine_path}. Skipping engine building.")
            self._load_engine(engine_path)
            return
        # 创建构建器builder
        builder = trt.Builder(self.logger)
        # 预创建网络
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 加载onnx解析器
        parser = trt.OnnxParser(network, self.logger)
        success = parser.parse_from_file(onnx_model_path)
        if not success:
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            print("ONNX parsing failed!")
            exit(1)

        # builder配置
        config = builder.create_builder_config()
        # 序列化生成engine文件
        print("Building TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)

        # 保存engine文件
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
            print("Engine file generated successfully!")

        # 加载engine并创建执行上下文
        self._load_engine(engine_path)

    def _load_engine(self, engine_model_path):
        # 从engine文件加载TensorRT引擎
        with open(engine_model_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

            # Allocate memory for input and output tensors only once during initialization
            self._allocate_memory()

    def _allocate_memory(self):
        # 获取输入和输出tensor的名称
        n_io = self.engine.num_io_tensors
        l_tensor_name = [self.engine.get_tensor_name(ii) for ii in range(n_io)] 
        self.l_tensor_name = l_tensor_name  # 保存输入输出张量名称
        print("Number of input/output tensors:", n_io)
        print("Input tensor names:", l_tensor_name[0])
        print("Output tensor names:", l_tensor_name[1:])

        # 申请显存内存
        self.h_input = np.empty(self.context.get_tensor_shape(l_tensor_name[0]), dtype=np.float32)
        # self.h_output = np.empty(self.context.get_tensor_shape(l_tensor_name[1]), dtype=np.float32)
        self.h_outputs = []
        for output_name in l_tensor_name[1:]:  # 从索引1开始是输出张量
            shape = self.context.get_tensor_shape(output_name)
            self.h_outputs.append(np.empty(shape, dtype=np.float32))  # 为每个输出分配内存
        
        
        # Allocate device memory
        self.d_input = cudart.mem_alloc(self.h_input.nbytes)  # Device input memory
        self.d_outputs = [cudart.mem_alloc(output.nbytes) for output in self.h_outputs]  # 每个输出设备内存

        # 绑定输入输出内存到TensorRT上下文
        self.context.set_tensor_address(l_tensor_name[0], self.d_input)  # 输入张量
        for i, output_name in enumerate(l_tensor_name[1:]):  # 输出张量从索引1开始
            self.context.set_tensor_address(output_name, self.d_outputs[i])  # 绑定每个输出张量的设备内存

    def infer(self, input_data):
        if self.context is None or self.engine is None:
            raise RuntimeError("Engine or context is not initialized.")
        
        # Prepare input data and copy to device memory
        self.h_input[:] = input_data  # Copy input_data to host memory
        cudart.memcpy_htod(self.d_input, self.h_input)  # Copy from host to device
        
        # Run inference
        self.context.execute_async_v3(0)  # Perform inference
        
        # 将每个输出张量从设备内存复制到主机内存
        for i, output_name in enumerate(self.l_tensor_name[1:]):  # 输出张量从索引1开始
            cudart.memcpy_dtoh(self.h_outputs[i], self.d_outputs[i])  # 从设备到主机内存复制

        
        return self.h_outputs

    def release_resources(self):
        # Delete the execution context, engine, runtime, and other objects to free memory
        print("Releasing TensorRT resources...")
        if self.d_input:
            self.d_input.free()
        if self.d_outputs:
            for d_output in self.d_outputs:
                d_output.free()  # 释放每个设备内存
            self.d_outputs = []  # 清空列表，以避免内存泄漏
        
        del self.context  # Delete the execution context
        del self.engine   # Delete the engine
        del self.runtime  # Delete the runtime
        self.context = None
        self.engine = None
        self.runtime = None

# 定义一个简单的全连接模型 (仅用于导出ONNX模型的部分)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # 输入3个特征，输出2个特征
        # Manually setting weights and biases
        with torch.no_grad():  # We use torch.no_grad() to avoid tracking the gradient for weight initialization
            self.fc1.weight = torch.nn.Parameter(torch.tensor([[1., 2., 0.], [0., 0., 0.]]))  # Shape: (2, 3)
            self.fc1.bias = torch.nn.Parameter(torch.tensor([0.1, -0.1]))  # Shape: (2,)
    
    def forward(self, x):
        return self.fc1(x)


if __name__ == '__main__':
    # 导出一个ONNX模型
    model = SimpleModel()
    x = torch.randn(1, 3)  # 批次大小为1，输入维度为3
    onnx_path = "simple_model.onnx"
    torch.onnx.export(model, x, onnx_path, input_names=['input'], output_names=['output'])
    print(f"ONNX model saved to {onnx_path}")
    
    # 使用TensorRTModel进行推理
    tensorrt_model = TensorRTModel(onnx_model_path=onnx_path)
    
    # 创建一个示例输入数据
    test_input = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    
    # Run inference
    output = tensorrt_model.infer(test_input)
    
    # 打印推理结果
    print("Inference output:", output)
    
    # 释放资源
    tensorrt_model.release_resources()


    

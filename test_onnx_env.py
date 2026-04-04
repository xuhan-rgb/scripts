import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto
import os

try:
    import onnxruntime as ort
    print("ONNX Runtime 版本:", ort.__version__)
    # 获取所有可用的执行提供者
    all_providers = ort.get_all_providers()

    # 筛选出与 CPU 和 CUDA 相关的提供者
    cpu_cuda_providers = [provider for provider in all_providers if 'CPU' in provider or 'CUDA' in provider]

    # 打印筛选后的提供者
    print("Available CPU and CUDA providers:")
    print("\n".join(cpu_cuda_providers))
    onnx_test_flag = True
except:
    onnx_test_flag = False
    


def create_onnx_model(onnx_model_path="~/.cache/simple_cnn.onnx", opset_version=16):
    # 解析路径中的 ~ 符号
    onnx_model_path = os.path.expanduser(onnx_model_path)
    
    # 创建卷积层的权重和偏置
    conv_weight = np.random.randn(1, 1, 3, 3).astype(np.float32)  # (out_channels, in_channels, kernel_height, kernel_width)
    conv_bias = np.random.randn(1).astype(np.float32)
    
    # 创建输入的张量 (batch_size, channels, height, width)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 28, 28])
    
    # 创建卷积层的权重和偏置张量
    conv_weights_tensor = helper.make_tensor("conv_weight", TensorProto.FLOAT, conv_weight.shape, conv_weight.flatten())
    conv_bias_tensor = helper.make_tensor("conv_bias", TensorProto.FLOAT, conv_bias.shape, conv_bias.flatten())
    
    # 创建卷积节点
    conv_node = helper.make_node(
        "Conv",  # 卷积操作
        inputs=["input", "conv_weight", "conv_bias"],  # 输入
        outputs=["conv_output"],  # 输出
        kernel_shape=[3, 3],  # 卷积核大小
        strides=[1, 1],  # 步长
        pads=[1, 1, 1, 1]  # 填充
    )
    
    # 创建 ReLU 激活层
    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_output"],
        outputs=["output"]
    )

    # 创建图
    graph = helper.make_graph(
        [conv_node, relu_node],
        "simple_cnn_model",  # 图名称
        [input_tensor],  # 输入
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 28, 28])],  # 输出
        [conv_weights_tensor, conv_bias_tensor]  # 参数
    )
        
    # 创建模型并指定 opset 版本
    model = helper.make_model(
        graph, 
        producer_name="onnx-example",
        opset_imports=[helper.make_opsetid("ai.onnx", opset_version)]  # 设置 opset 版本
    )

    # 保存为文件
    onnx.save(model, onnx_model_path)

    print(f"ONNX model has been saved to {onnx_model_path}")

def test_onnx_model(onnx_model_path = "~/.cache/simple_cnn.onnx"):
    onnx_model_path = os.path.expanduser(onnx_model_path)

    # 使用 ONNX Runtime 加载模型
    print(f"Loading ONNX model from {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("ONNX model compute providers: ", session.get_providers())
    
    # 准备输入数据（随机生成一个 28x28 图像）
    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    
    # 进行推理
    output = session.run(None, {input_name: input_data})
    
    # 输出结果
    print("Output shape:", output[0].shape)
      
onnx_model_path = "~/.cache/simple_cnn.onnx"
import sys
opset_version = 16
if len(sys.argv) > 1:
    opset_version = sys.argv[1]
    try:
        opset_version = int(opset_version)
    except ValueError:
        print("opset_version must be a number")
        exit(1)

if onnx_test_flag:
    create_onnx_model(onnx_model_path, opset_version=opset_version)
    test_onnx_model(onnx_model_path)





















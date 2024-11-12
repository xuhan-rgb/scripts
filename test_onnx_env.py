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

#############################################################
'''
import torch
import torch.nn as nn
DEFAULT_ACT = nn.ReLU()  # default activation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
  
    default_act = DEFAULT_ACT  # default activation
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        print(self.conv)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # print(f"Conv(c1={c1}, c2={c2}, k={k}, s={s}, p={p}, g={g}, d={d}, act={act})", self.act)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
#        return self.act(self.bn(self.conv(x)))
        return self.conv(x)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

# 定义 PSABlock 模块
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        #self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        
        self.add = shortcut

    def forward(self, x):
        # 通过注意力和前馈网络进行计算
        #x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.attn(x)
        return x

# 定义测试和保存函数
def test_and_save_psablock():
    # 实例化模型
    model = PSABlock(c=256, attn_ratio=0.5, num_heads=4, shortcut=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    model.eval()  # 切换到评估模式

    #model = model.to(device)
    # 创建一个输入张量
    input_tensor = torch.randn(1, 256, 11, 20)  # 形状为 (1, 256, 11, 20)
    print("Input tensor shape:", input_tensor.shape)

    # 测试模型的输出
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)

    # 使用 torch.jit.trace 将模型转换为 TorchScript 格式
    scripted_model = torch.jit.trace(model, input_tensor)

    # 保存 TorchScript 模型
    scripted_model.save("psablock_scripted.pt")
    print("TorchScript model saved as psablock_scripted.pt")

# 读取并测试保存的 TorchScript 模型
def load_and_test_psablock():
    # 加载保存的 TorchScript 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    loaded_model = torch.jit.load("psablock_scripted.pt")
    
    print("TorchScript model loaded.")
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    # 创建一个相同形状的输入张量进行测试
    input_tensor = torch.randn(1, 256, 11, 20).to(device)
    output_tensor = loaded_model(input_tensor)
    print("Output tensor from loaded model shape:", output_tensor.shape)

# 测试并保存模型
test_and_save_psablock()

# 加载并测试保存的模型
load_and_test_psablock()
'''
#####################################























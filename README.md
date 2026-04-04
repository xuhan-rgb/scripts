# TensorRT / ONNX 推理工具集

ONNX 环境检测 + TensorRT 模型转换与推理的工具集，用于在 GPU 服务器上快速验证深度学习推理环境。

## 使用方法

所有脚本需在有 CUDA/TensorRT 环境的机器上运行。

### 1. 检测 CUDA/cuDNN/ONNX Runtime 环境

```bash
python get_onnx_dependencies.py
```

输出 CUDA 版本、cuDNN 版本，以及 ONNX Runtime GPU 库的依赖路径。

### 2. 测试 ONNX Runtime 推理

```bash
python test_onnx_env.py          # 默认 opset 16
python test_onnx_env.py 13       # 指定 opset 版本
```

自动创建一个简单 CNN 模型，用 CUDA Provider 推理验证环境是否正常。

### 3. TensorRT 推理（核心模块）

`tensorrt_inference.py` 提供 `TensorRTModel` 类：

```python
from tensorrt_inference import TensorRTModel

# 从 ONNX 自动转换（engine 会缓存到 ~/.cache/model/）
model = TensorRTModel(onnx_model_path='model/your_model.onnx')

# 或直接加载已有 engine
model = TensorRTModel(engine_model_path='model/your_model.engine')

output = model.infer(input_data)  # numpy array, shape 需匹配模型输入
model.release_resources()
```

### 4. 深度模型推理示例

```bash
python convert_trt.py
```

使用 `TensorRTModel` 对深度估计模型推理，并生成彩色深度图。需要准备：
- `model/depth_model.onnx`
- `images/depth_image.jpg`

## 依赖

- CUDA, cuDNN
- TensorRT, pycuda
- onnx, onnxruntime-gpu
- numpy, opencv-python, matplotlib

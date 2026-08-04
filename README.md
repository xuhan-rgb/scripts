# 个人脚本工具集

用于 Linux 桌面配置、ONNX 环境检测以及 TensorRT 模型转换与推理的个人脚本集合。

## Claude Code / Codex 网关

### 部署跟随本地 Codex 配置的桥接服务

```bash
bash claude/install-codex-bridge.sh
```

脚本会安装固定版本的 CLIProxyAPI user service。每次启动 `claudex` 时，它会读取 `~/.codex/config.toml` 当前的 `model_provider`，再从对应的 `[model_providers.<name>]` 读取 `base_url`、`wire_api` 和 `env_key`。因此 Codex 切换 provider 后，Claude 会自动使用新的连接配置；provider 的 `env_key` 必须已导出到当前 shell。模型与思考强度由本服务独立保存在 `~/.cli-proxy-api/selection.conf`，不会反向修改 Codex 配置。

```bash
claudex       # 正常权限确认
claudex-yolo  # 跳过权限确认
claudex --pick  # 启动前交互选择模型和思考强度
claudex-ui    # 打开本地可视化控制台
```

Claude 的 Opus、Sonnet、Haiku 档位分别映射到 Sol、Terra、Luna。脚本支持 Linux x86_64 和 arm64，要求所选 Codex provider 使用 Responses API，并配置可用的 `base_url` 与 `env_key`。

在 Claude Code 中输入 `/model` 可在三个 GPT 模型间选择，并在模型选择器底部用 `←/→` 调整思考强度；也可以输入 `/effort` 单独调整。三个模型共同支持 `low`、`medium`、`high`、`xhigh` 和 `max`。启动前指定选择的示例：

```bash
claudex --gpt-model gpt-5.6-terra --gpt-effort high
claudex-yolo --gpt-model gpt-5.6-luna --gpt-effort medium
```

`claudex-ui` 打开的 Codex Routing Desk 只监听 `127.0.0.1:8320`。界面可持久切换模型与 effort、查看当前 Codex provider 和网关健康状态；保存结果用于后续新开的 Claude 会话，已经运行的会话仍使用 `/model` 或 `/effort` 切换。

## 桌面工具

### 安装 Flameshot 并配置 GNOME 快捷键

```bash
bash desktop/install-flameshot-shortcuts.sh
```

脚本会检查 Flameshot、剪贴板和 GNOME 桌面组件。所有依赖均已安装时不会调用 APT；缺少依赖时会更新软件包索引，并且只安装缺少的软件包。配置完成后：

- `Alt+A`：打开 Flameshot 截图界面。
- `Alt+S`：截图保存到 `/tmp`，并将完整文件路径复制到剪贴板。

该脚本仅支持 Ubuntu/Debian 的 GNOME 桌面环境，应当以当前桌面用户运行，不要直接使用 `sudo` 启动整个脚本。

## TensorRT / ONNX 工具

`get_onnx_dependencies.py` 仅依赖 Linux 系统库和 `ldd`。ONNX Runtime 验证需要能实际加载 CUDA Execution Provider；TensorRT 推理则需要兼容的 NVIDIA 驱动、CUDA、cuDNN、TensorRT 和 PyCUDA。

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

自动创建一个简单 CNN 模型，并要求使用 CUDA Provider 推理。若 CUDA Provider 不可用或 session 回退到 CPU，脚本会以非零状态退出，而不会报告“验证成功”。

### 3. TensorRT 推理（核心模块）

`tensorrt_inference.py` 提供 `TensorRTModel` 类：

```python
from tensorrt_inference import TensorRTModel

# 从 ONNX 自动转换（engine 会缓存到 ~/.cache/model/）
model = TensorRTModel(onnx_model_path='model/your_model.onnx')
try:
    outputs = model.infer(input_data)  # list[numpy.ndarray]
    output = outputs[0]                # 单输出模型
finally:
    model.release_resources()
```

`TensorRTModel` 当前支持单输入、静态形状的 engine。它按 TensorRT 的 I/O mode 识别输入和输出，并采用 engine 声明的 dtype；`input_data` 的 shape 必须与输入 tensor 完全一致。

### 4. 深度模型推理示例

```bash
python convert_trt.py
# 需要图形窗口时显式开启
python convert_trt.py --show
```

使用 `TensorRTModel` 对深度估计模型推理，并生成彩色深度图。仓库不提供模型或示例图片，需要准备：

- `model/depth_model.onnx`
- `images/depth_image.jpg`

脚本以 RGB、`[0, 1]` 范围的 `float32` NCHW `(1, 3, 352, 640)` 输入模型，并要求第一个输出为至少含一个 batch 和通道的 NCHW tensor，随后取 `[0, 0]` 生成伪彩深度图。默认结果写入 `images/depth_colormapped.jpg`；可通过 `--model`、`--image`、`--output` 改写路径。

## 依赖

- 桌面快捷键安装：Ubuntu/Debian、GNOME、APT（缺失的软件包由脚本自动安装）
- 环境检查：Python 标准库
- ONNX Runtime 测试：`numpy`、`onnx`、`onnxruntime-gpu`
- TensorRT 推理：`numpy`、`torch`、`tensorrt`、`pycuda`
- 深度图示例：TensorRT 推理依赖，另加 `opencv-python`、`matplotlib`

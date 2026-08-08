# 个人脚本工具集

用于 Linux 桌面配置、ONNX 环境检测以及 TensorRT 模型转换与推理的个人脚本集合。

## Claude Code / Codex 网关

### 安装与使用

先安装 Codex CLI 和 Claude Code。只有需要安装 Claude-to-Codex 网关时，才需要确认 systemd user service 可用：

```bash
codex --version
claude --version
systemctl --user show-environment >/dev/null
```

获取脚本：

```bash
git clone https://github.com/xuhan-rgb/scripts.git ~/scripts
cd ~/scripts
```

不要用 `sudo` 运行安装器。

#### 1. Codex + 官方 Claude，不安装网关

不需要把 Codex 接入 Claude Code 时运行：

```bash
bash claude/setup-codex.sh
source ~/.bashrc
codex-auth status
```

该方式用于配置 `codex-yolo`、`claude-yolo` 以及 Codex 的 API/账号模式切换，不安装或启动 Claude-to-Codex 网关。`claude-yolo` 直接使用官方 Claude 账号，与 Codex provider 无关。

#### 2. Codex 接入 Claude Code

需要使用 `claudex` 或 `claudex-yolo` 时运行完整安装器：

```bash
bash claude/install-codex-bridge.sh
source ~/.bashrc
```

完整安装器会同时完成 Codex 配置，因此不需要提前单独运行 `setup-codex.sh`。

#### 3. 安装后验证

不安装网关：

```bash
codex-auth status
type codex-yolo claude-yolo
```

安装 Claude-to-Codex 网关后再检查：

```bash
type claudex claudex-yolo claudex-ui
systemctl --user is-active cli-proxy-api.service claudex-manager.service
curl --fail http://127.0.0.1:8320/healthz
```

#### 4. Codex / Claudex 使用教程

这组脚本用于切换 Codex 的 API/账号模式，并提供普通模式与跳过权限确认的快捷命令。

Codex 的 API/账号模式使用以下命令切换：

```bash
codex-auth status                 # 查看当前模式
codex-auth api                    # 使用自定义 API provider
codex-auth account                # 使用 ChatGPT 账号，浏览器登录
codex-auth account --device-auth  # 无桌面环境使用设备码登录
```

常用启动命令：

| 命令 | 功能 |
| --- | --- |
| `codex` | 使用当前 API/账号模式启动 Codex |
| `codex-yolo` | 使用当前模式启动 Codex，并跳过审批与沙箱 |
| `claude-yolo` | 使用官方 Claude 账号，并跳过权限确认 |
| `claudex` | 通过本地 GPT 网关启动 Claude Code，保留权限确认 |
| `claudex-yolo` | 通过本地 GPT 网关启动 Claude Code，并跳过权限确认 |
| `claudex-ui` | 打开本地路由控制台 |

`claudex-yolo` 的 alias 为：

```bash
alias claudex-yolo='CLAUDEX_YOLO=1 claudex'
```

直接运行 `claudex-yolo` 即可使用 GPT 网关并跳过权限确认，也可以继续传入普通命令参数：

```bash
claudex-yolo
claudex-yolo "检查当前项目并修复测试"
claudex-yolo --prompt-suggestions true
```

如果命令未找到，重新执行 `source ~/.bashrc` 或打开一个新终端。由于 `claudex-yolo` 会跳过权限确认，只应在可信项目中使用。

#### 5. Provider 日常管理

完整安装后运行 `claudex-ui`，在页面中配置、测试和切换 provider，也可以选择 GPT 模型、思考强度并查看用量。该控制台仅供安装了 Claude-to-Codex 网关的场景使用。

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

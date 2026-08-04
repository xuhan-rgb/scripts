# AGENTS.md

## 项目概览

这是一个面向 Linux 桌面与 NVIDIA GPU 服务器的轻量脚本工具集，用于：

- 部署跟随本地 Codex provider、独立管理模型和思考强度的 Claude Code 桥接服务。
- 安装 Flameshot 相关依赖并配置 GNOME 截图快捷键。
- 诊断 CUDA、cuDNN 与 ONNX Runtime GPU 动态库环境。
- 生成最小 ONNX 模型并验证 ONNX Runtime 的 CUDA Provider。
- 将静态 ONNX 模型构建或加载为 TensorRT engine，再以 NumPy 数组执行推理。
- 对一个固定输入尺寸的深度估计模型进行端到端推理和结果可视化。

除仅监听 `127.0.0.1` 的 Codex Routing Desk 外，项目没有对外 Web 服务或打包配置；脚本由当前 Python 环境直接执行，非 GPU 辅助逻辑由标准库 `unittest` 测试。

## 环境要求

运行 TensorRT 相关脚本前，主机须已具备兼容版本的 NVIDIA 驱动、CUDA、cuDNN 和 TensorRT。Python 依赖按脚本分组如下：

桌面脚本仅支持使用 APT 的 Ubuntu/Debian GNOME 环境。它会自动安装缺失的 Flameshot、剪贴板和桌面组件；所有依赖均已存在时不得刷新 APT 索引。

| 工作流 | Python 依赖 |
| --- | --- |
| 环境检查 | 标准库 |
| ONNX Runtime 测试 | `numpy`、`onnx`、`onnxruntime-gpu` |
| TensorRT 推理 | `numpy`、`torch`、`tensorrt`、`pycuda` |
| 深度图示例 | TensorRT 推理依赖，另加 `opencv-python`、`matplotlib` |

未提供锁定版本的依赖清单。安装或升级 CUDA、cuDNN、TensorRT 或 `onnxruntime-gpu` 时，必须确认它们的 ABI 兼容性；先运行环境检查，再运行 ONNX Runtime 测试。

## 常用命令

```bash
# 安装读取 ~/.codex/config.toml 的 CLIProxyAPI user service
bash claude/install-codex-bridge.sh

# 打开本地模型与思考强度控制台
claudex-ui

# Shell 语法检查；不会修改系统或用户服务
bash -n claude/install-codex-bridge.sh

# 检查并安装 Flameshot 依赖，配置 Alt+A 与 Alt+S
bash desktop/install-flameshot-shortcuts.sh

# Shell 语法检查；不会修改系统或桌面配置
bash -n desktop/install-flameshot-shortcuts.sh

# 仅做 Python 语法检查；不需要 CUDA 或第三方库
python -m py_compile test_onnx_env.py get_onnx_dependencies.py tensorrt_inference.py convert_trt.py

# 运行不依赖 TensorRT/PyCUDA 的辅助逻辑回归测试
python -m unittest discover -s tests -v

# 检查 CUDA、cuDNN 与 onnxruntime 的 CUDA provider 动态库依赖
python get_onnx_dependencies.py

# 创建临时 CNN ONNX 文件，并使用 CUDAExecutionProvider 推理
python test_onnx_env.py
python test_onnx_env.py 13

# 导出一个全连接 ONNX 示例，构建 TensorRT engine 后推理
python tensorrt_inference.py

# 深度模型示例；要求 model/depth_model.onnx 和 images/depth_image.jpg 已存在
python convert_trt.py
```

`test_onnx_env.py` 仅在 CUDA Provider 可用后才会覆盖 `~/.cache/simple_cnn.onnx`。`tensorrt_inference.py` 会在当前目录创建 `simple_model.onnx`，并将 engine 缓存到 `~/.cache/model/`。`convert_trt.py` 默认写入 `images/depth_colormapped.jpg`，且只在显式传入 `--show` 时打开 GUI 窗口。

## 结构与职责

```text
.
|- claude/install-codex-bridge.sh     # 跟随本地 Codex provider 的 Claude Code 桥接服务
|- claude/codex_bridge_manager.py     # 实时模型路由与逐请求 usage 可视化控制台
|- desktop/install-flameshot-shortcuts.sh  # Flameshot 安装与 GNOME 快捷键配置
|- get_onnx_dependencies.py  # 动态库与 onnxruntime CUDA provider 依赖诊断
|- test_onnx_env.py          # 最小 ONNX Runtime CUDA 推理验证
|- tensorrt_inference.py     # TensorRTModel 引擎构建、加载、推理与释放
|- convert_trt.py            # 固定尺寸深度估计模型示例和深度图可视化
|- README.md                 # 面向使用者的快速开始
|- tests/test_helpers.py     # 环境诊断和深度图辅助函数的无 GPU 回归测试
`- tests/test_codex_bridge_manager.py # Codex provider 解析与选择状态回归测试
```

数据流为：ONNX 文件 -> `TensorRTModel` 构建或加载 engine -> 分配主机/显存缓冲区 -> `infer()` 执行并返回 NumPy 输出列表。`convert_trt.py` 在此基础上读取图片、调整至 `(1, 3, 352, 640)`、归一化后取第一个输出的首通道生成伪彩深度图。

## TensorRTModel 使用约束

- 构造时必须传入 `onnx_model_path` 或 `engine_model_path`，不能同时省略。
- ONNX 路径会以源文件 MD5 和 TensorRT 版本命名 engine 缓存；同一组合会复用缓存。无法加载的缓存会自动重新构建，但 GPU 架构变化仍可能需要手动清理对应 `~/.cache/model/*.engine`。
- 实现按 TensorRT I/O mode 识别输入和输出，并为每个 tensor 采用 engine 声明的 NumPy dtype。当前仅支持单输入、静态形状的 engine；动态 shape 和多输入会明确报错。
- `infer(input_data)` 的形状必须与 engine 输入形状一致；返回值始终是输出数组列表，即使只有一个输出。
- 每个 `TensorRTModel` 实例用完后必须调用 `release_resources()`，释放 PyCUDA 显存缓冲区。

## 维护约定

- 修改 Codex 桥接脚本后运行 `bash -n`；不得把 provider API key、管理密钥或 usage 数据库写入仓库。
- 修改桌面安装脚本后运行 `bash -n`；已安装的依赖必须跳过，缺失依赖必须能够自动安装。
- 保持脚本直接可运行，不引入框架、命令行包装或配置层，除非需求明确要求。
- 修改 `TensorRTModel` 时，同时核对 ONNX 构建路径、已有 engine 加载路径和多输出返回行为。
- 修改 `convert_trt.py` 的预处理时，确认模型实际期望的布局、尺寸、色彩空间和归一化方式；这些参数目前是示例专用的固定值，首输出必须为 NCHW 深度图。
- 新增可复现的运行依赖时，补充版本约束文件并更新 `README.md` 与本文档；不要在文档中声称未验证的 CUDA/TensorRT 组合可用。

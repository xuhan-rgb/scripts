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

### Markdown / LaTeX 文档渲染器

首次安装或修复依赖、命令和桌面关联，统一运行：

```bash
bash ~/scripts/desktop/install-markdown-renderer.sh
```

安装脚本只在发现缺失的 APT 包时调用 `sudo`，并在终端中由 sudo 正常读取密码；不会保存密码。它会检查项目模板和锁文件，安装实时预览、X11 Chrome 窗口嵌入、Pandoc/XeLaTeX、Poppler、TikZ、Mermaid 与浏览器依赖，并配置当前用户的 `mdview` 命令、桌面入口及 Markdown/TeX MIME 默认应用。脚本可重复运行；已满足的 APT 和 npm 依赖会跳过。

直接启动原生 PyQt5 软件窗口。默认只显示渲染结果；点击工具栏的“显示原文”后，左侧显示 Markdown 原文、右侧保持实时渲染：

```bash
mdview
mdview /path/to/document.md
mdview /path/to/document.tex
```

`mdview` 是指向 `~/scripts/markdown_editor.py` 的命令入口。窗口内也可点击“打开文件”选择 Markdown 或完整 LaTeX 文档。主工具栏只保留“打开文件 / 目录 / ChatGPT / 原文 / 预览 PDF / 导出 PDF / 设置”，并提供以下功能：

- “显示原文 / 隐藏原文”：切换双栏与纯预览模式。
- “ChatGPT / 隐藏 ChatGPT”：在文档目录左侧嵌入真实的系统 Chrome 窗口，形成“Chrome ChatGPT / 目录 / 正文”三栏布局。`mdview` 查找通过 `--remote-debugging-port=9223` 运行的 Chrome，复用其 `user-data-dir` 与登录状态，再让 Chrome 原生打开 `--app=https://chatgpt.com/`；官网样式、登录、会话、模型选择、上传、语音、复制和下载均由 Chrome 本身处理，不再使用 Qt WebEngine。下载捕获按本次内嵌页面的 Chrome target/frame ID 隔离：只有该页面下载完成的 `.md`、`.markdown`、`.tex`、`.latex` 或 `.ltx` 会自动替换当前 `mdview` 文档并立即渲染；普通 Chrome 窗口的下载不会触发。该窗口嵌入依赖 X11，不支持 Wayland。
- 预览右键复制：选中右侧渲染内容后，可选择“复制为文字”或“复制为图片”；图片会保留标题、表格和公式的排版，同时写入剪贴板和唯一文件 `/tmp/mdview-selection-*.png`，完整路径显示在状态栏。
- “复制为 ChatGPT 对话…”：根据当前文件、选中原文和 PDF 物理页码自动生成定位对话，末尾保留“修改要求（由我补充）”；粘贴到 ChatGPT 后填写具体改法即可。提示词要求只修改命中位置并返回完整同格式文件。
- 左侧目录：默认显示 Markdown 的一级至三级标题；点击目录项跳转到右侧对应章节，编辑时实时更新，可用“隐藏目录 / 显示目录”切换。
- “设置”：收纳背景颜色、整体边框颜色和 Markdown 行间距；不再将低频选项平铺在主工具栏。
- 页面留白：实时预览在正文左右各保留约 `56px`，形成类似 Word 的阅读页边距；该样式只影响软件预览，不改变 PDF 的 A4 页边距。
- LaTeX 公式：支持 `$...$`、`\(...\)`、`\[...\]`、`$$...$$` 和文档中的 Pandoc `` `...`{=tex} `` 标记。
- 流程图：完整支持 Mermaid `flowchart`、`sequenceDiagram` 等标准代码块；支持 `tikz` 代码块，也会识别 `tex`/`latex` 代码块中的 `tikzpicture`。TikZ 在预览中临时编译为图像，在 PDF 中保留为 XeLaTeX 原生矢量图。可识别的 `text` ASCII 流程图也会自动转为图形。
- 报告预览：读取 YAML `title`/`subtitle` 生成封面；`numbersections: true` 会使实时预览和左侧导航使用与 Pandoc PDF 一致的章节编号。
- “导出 PDF”对话框：每次导出时临时选择输出文件、是否生成正文目录（TOC）以及导出后是否打开 PDF。默认输出位置是源文件同目录，文件名不变，只将 `.md`/`.tex` 后缀改为 `.pdf`。这些选项不写入全局设置。
- “预览 PDF”：在 `~/.cache/mdview/` 生成临时 PDF 并用系统默认阅读器打开；Markdown 预览默认不生成正文目录，完整 TeX 预览遵循源文件中的 `\tableofcontents`。
- 完整 `.tex` 预览：自动运行两遍 XeLaTeX，再把真实 PDF 页面连续显示在软件内；鼠标选择框直接使用 Poppler 提取的 PDF 单词坐标，不叠加会发生错位的透明 HTML 排版，可选择和复制正文。左侧目录读取 XeLaTeX 生成的 `.toc` 与 PDF 命名目标，点击后按物理页及页内坐标精确跳转。TikZ、页码和排版不经过 Markdown/Pandoc。
- “导出 PDF”：Markdown 使用 Pandoc → 论文风格 LaTeX 模板 → XeLaTeX；完整 `.tex` 文档直接使用 XeLaTeX。
- 底部文件路径：点击后复制当前 Markdown/LaTeX 文件的完整路径，包含文件名和扩展名。实时预览不再画虚拟分页线或页码；PDF 仍使用 LaTeX 的真实分页和页码。

如果远程调试 Chrome 尚未运行，可先启动一次；端口只监听本机：

```bash
google-chrome \
  --user-data-dir="$HOME/.config/google-chrome-shared" \
  --profile-directory=Default \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9223
```

以后点击“ChatGPT”会从这个 Chrome 进程创建并嵌入官网窗口。若使用其他端口，可在启动 `mdview` 前设置 `MDVIEW_CHROME_DEBUG_PORT`。

Python 依赖：

```bash
python3 -m pip install -r requirements-markdown.txt
```

PDF 导出必须具备 Pandoc + XeLaTeX。在 Ubuntu/Debian 安装：

```bash
sudo apt install python3-pyqt5 pandoc texlive-xetex texlive-lang-chinese texlive-latex-extra texlive-pictures fonts-noto-cjk graphviz poppler-utils x11-utils libx11-6
```

导出使用 `markdown_pdf/template.tex`，生成封面、可选三级目录、页眉页脚、论文式表格和中文数学排版。封面与页眉可通过 Markdown 开头的 YAML 设置：

```yaml
---
title: "自动驾驶世界模型：从辅助训练到在线规划"
subtitle: "基于 LAW（ICLR 2025）及相关 World Model Planning 工作"
author: "Research Notes"
date: "2026-08-12"
report-type: "研究笔记 / 技术报告"
question: "核心问题：预测出来的未来，在系统中到底拿来做什么？"
header-left: "自动驾驶世界模型：从辅助训练到在线规划"
header-right: "World Model Research Notes"
numbersections: true
toc-depth: 3
---
```

Mermaid 和可识别的 ASCII 流程图会先转成 PNG，再作为图片进入 Pandoc/XeLaTeX；公式、表格、目录和正文不经过浏览器打印。提示框支持 Pandoc fenced div，例如：

```markdown
::: {.infobox title="最重要的判别标准"}
预测出来的未来，在系统里到底拿来干什么？
:::
```

### 让 AI 生成可视化 TikZ

在 Markdown 中只写 `tikzpicture` 图形主体；不要让 AI 输出 `documentclass`、`usepackage`或 `document` 环境，字体、宏包和常用 TikZ 库由 `mdview` 统一提供。可直接给 AI 以下提示词：

```text
请生成一段可直接放入 Markdown 的 TikZ 论文架构图。

要求：
1. 使用 ```tikz 代码块，不使用 Mermaid。
2. 代码块内只输出完整的 \begin{tikzpicture}...\end{tikzpicture}。
3. 不输出 \documentclass、\usepackage、\usetikzlibrary 或 document 环境。
4. 可使用 positioning、arrows.meta、calc、fit、backgrounds、
   shapes.geometric、matrix、chains 和 decorations.pathreplacing。
5. 所有数学符号使用 LaTeX，例如 $V_t$、$W_t$、
   $\hat V_{t+\Delta}$ 和 $\mathcal L_{\mathrm{latent}}$。
6. 主数据流使用实线箭头，条件或监督使用虚线箭头。
7. 当前观测、Encoder、latent 和 Planner 使用浅蓝色节点；
8. 避免节点重叠，整体宽度适合 A4 论文正文。
9. 代码块后再输出一行 Markdown 图注。

模型结构：
[在这里写节点、输入、输出、分支和 loss 关系]

只输出 Markdown TikZ 代码块和图注，不要解释。
```

最小可运行格式：

````markdown
```tikz
\begin{tikzpicture}[
  node distance=12mm,
  block/.style={draw=blue!55, fill=blue!5, rounded corners, align=center},
  arrow/.style={-{Stealth}, semithick}
]
\node[block] (latent) {当前 latent\\$V_t$};
\node[block, right=of latent] (planner) {Planner\\$P_\phi$};
\draw[arrow] (latent) -- (planner);
\end{tikzpicture}
```

*图：当前 latent 进入 Planner。*
````

`tex` 或 `latex` 代码块只有在包含 `tikzpicture` 时才会被当作可视化图形；普通 TeX 代码仍显示为可读源码。

完整 Mermaid 渲染器固定在 `markdown_renderer_node/package.json`；首次部署或依赖目录缺失时执行：

```bash
npm ci --no-audit --no-fund --prefix markdown_renderer_node
```

Mermaid 由本机 `google-chrome` 或 `chromium` 离屏执行，不会打开浏览器窗口；若缺少浏览器，`flowchart` 会回退到 Graphviz，其他 Mermaid 类型保留为可读源码。

缺少 Pandoc 或 XeLaTeX 时，软件会显示所缺依赖和安装命令，不会静默改用 Qt/浏览器打印。

若要让 ChatGPT 点击 `.md` 文件时直接用该软件打开：

1. 在 ChatGPT 设置的“默认文件打开目标”中选择 **Default app**。
2. 将 `desktop/markdown-renderer.desktop` 安装到 `~/.local/share/applications/`。
3. 把 `text/markdown` 的系统默认应用设为 `markdown-renderer.desktop`。

本机当前用户可执行：

```bash
mkdir -p ~/.local/bin
ln -sfn ~/scripts/markdown_editor.py ~/.local/bin/mdview
install -Dm644 desktop/markdown-renderer.desktop ~/.local/share/applications/markdown-renderer.desktop
xdg-mime default markdown-renderer.desktop text/markdown
xdg-mime default markdown-renderer.desktop text/x-markdown
xdg-mime default markdown-renderer.desktop text/x-tex
xdg-mime default markdown-renderer.desktop text/x-latex
xdg-mime default markdown-renderer.desktop application/x-tex
```

以上手动命令仅用于排查；正常安装优先使用 `desktop/install-markdown-renderer.sh`。

针对性测试：

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest tests.test_markdown_editor -v
```

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

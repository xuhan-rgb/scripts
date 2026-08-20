# Linux 桌面、AI CLI 与 GPU 工具集

这个仓库包含一组可直接运行的 Linux 工具，覆盖 Codex/Claude 多账号与 API provider 管理、Markdown/LaTeX 阅读与导出、论文工作区、GNOME 桌面配置、GPU 显存管理，以及 ONNX Runtime/TensorRT 环境验证和推理。

## 工具索引

| 工具 | 入口 | 用途 | 从哪里开始 |
| --- | --- | --- | --- |
| Codex 基础配置 | `bash claude/setup-codex.sh` | 安装多账号、provider、额度和 yolo 命令 | [Codex + 官方 Claude](#1-codex--官方-claude不安装网关) |
| Claude-to-Codex 网关 | `bash claude/install-codex-bridge.sh` | 让 Claude Code 通过本地 Codex provider 工作 | [Codex 接入 Claude Code](#2-codex-接入-claude-code) |
| Codex 多账号 | `codex-auth` | 添加、选择、归档隔离的 ChatGPT 登录 | [Codex / Claudex 使用教程](#4-codex--claudex-使用教程) |
| Codex API provider 后端 | `python3 claude/codex_provider.py` | 新增、测试、切换自定义 API provider | [Provider 命令行管理](#5-provider-命令行管理) |
| Codex 账号桌面管理 | `codex-account-manager` | 用 Qt 界面管理账号、provider、额度和托盘 | [原生账号与 API 管理软件](#6-原生账号与-api-管理软件) |
| Codex 额度 | `codex-usage`、`codex-usage-widget` | 查询额度/token 活动，或显示桌面悬浮窗 | [额度命令与悬浮窗](#额度命令与悬浮窗) |
| Claude 路由控制台 | `claudex-ui` | 选择网关模型、思考强度并查看请求用量 | [网页路由控制台](#7-网页路由控制台可选) |
| mdview | `mdview [FILE]` | 预览 Markdown、HTML、完整 LaTeX 并导出 PDF | [文档渲染器](#markdown--html--latex-文档渲染器) |
| Markdown 快速转 PDF | `bash markdown_to_pdf.sh INPUT OUTPUT` | 用 Pandoc + Chromium 快速生成 PDF | [快速转换](#独立-markdown-转-pdf) |
| ScholarVault | `scholarvault` | 管理论文、PDF、TeX、Zotero 和相关代码 | [ScholarVault](#scholarvault-论文工作区c20--qt-6) |
| Flameshot 快捷键 | `bash desktop/install-flameshot-shortcuts.sh` | 配置 GNOME 区域截图和路径复制 | [Flameshot](#安装-flameshot-并配置-gnome-快捷键) |
| Zotero 输入法修复 | `bash desktop/fix-zotero-ime-candidate-position.sh` | 修复 Linux 中文候选窗位置异常 | [Zotero 修复](#修复-zotero-中文候选窗位置) |
| DEB 文件查看 | `./deb-view-file PACKAGE.deb [PATH]` | 列出或读取 `.deb` 包内文件而不安装 | [DEB 查看](#查看-deb-包内文件) |
| GPU Harbor | `python3 gpu_memory_keeper_gui.py` | 管理本机或 SSH 远程 NVIDIA GPU 显存 | [GPU Harbor](#gpu-harbor-桌面管理器) |
| GPU Memory Keeper | `python3 gpu_memory_keeper.py COMMAND` | 从终端占用、监督和释放 GPU 显存 | [显存命令行](#gpu-memory-keeper-命令行) |
| CUDA/ONNX 诊断 | `python get_onnx_dependencies.py` | 检查 CUDA、cuDNN、ONNX Runtime 动态库 | [环境诊断](#1-检测-cudacudnnonnx-runtime-环境) |
| ONNX Runtime CUDA 验证 | `python test_onnx_env.py` | 创建最小模型并验证 CUDA Provider 推理 | [ORT 验证](#2-测试-onnx-runtime-推理) |
| TensorRTModel | `tensorrt_inference.py` | 构建/缓存 TensorRT engine 并执行 NumPy 推理 | [TensorRT 核心模块](#3-tensorrt-推理核心模块) |
| 深度估计示例 | `python convert_trt.py` | 执行固定输入深度模型并输出伪彩深度图 | [深度模型示例](#4-深度模型推理示例) |

除已安装到 `~/.local/bin` 的命令外，下面的示例都默认在仓库根目录执行。

## Claude Code / Codex 网关

### 安装与使用

先安装 Codex CLI 和 Claude Code。只有需要安装 Claude-to-Codex 网关时，才需要确认 systemd user service 可用：

```bash
codex --version
claude --version
systemctl --user show-environment >/dev/null
```

原生账号管理软件使用系统 PyQt5；Ubuntu/Debian 可安装：

```bash
sudo apt install python3-pyqt5
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

这组脚本用于管理 Codex 的 API provider 和多个 ChatGPT 账号，并提供普通模式与跳过权限确认的快捷命令。推荐给每个 ChatGPT 登录创建一个命名账号：

```bash
codex-auth add user@example.com          # 浏览器登录，默认用邮箱作为账号名
codex-auth add work@example.com --device-auth # 设备码登录并保存为邮箱账号名
codex-auth add-auto                      # 普通浏览器授权，自动读取登录邮箱
codex-auth add-auto --device-auth        # 无头/远程环境的设备码授权
codex-auth list                         # 查看账号，* 表示新进程的默认账号
codex-auth use user@example.com         # 只修改以后启动的 Codex 默认账号
codex-auth run --account work@example.com -- # 本次启动临时固定为指定账号
codex-auth remove work@example.com --yes # 可恢复归档账号，不删除共享对话
codex-auth status                       # 查看当前默认模式，不显示凭据
codex-auth api                          # 以后启动的 Codex 改用上次的 API provider
codex-auth api crs_local                # 选择某个具体 API provider
codex-usage                             # 查看当前命名账号额度，不需要 ChatGPT 网页
codex-usage --account work              # 查看指定命名账号额度
codex-usage-widget                      # 桌面悬浮显示当前账号的最长额度周期
codex-account-manager                   # 打开原生 Qt 账号/API 管理软件
```

安装后，`codex` 和 `codex-yolo` 都会通过 `codex-auth run` 启动。每个进程在启动时固定自己的账号目录，因此在另一个终端执行 `codex-auth use work` 不会切换或退出已经运行的 Codex，只影响后来启动的进程。

两条命令加载相同的 5 个核心 Skills：`agent-reach`、`brainstorming`、`domain-modeling`、`grilling` 和 `tdd`。区别仅在权限策略：`codex` 保留审批流程，`codex-yolo` 跳过审批与沙箱。

命名账号的登录凭据彼此隔离；`sessions`、归档、历史和会话索引仍链接到主 `~/.codex`，所以不同账号运行 `codex resume` 时可以看到同一批对话。不要让两个进程同时修改同一个对话。

以下命令保留用于未命名的单账号/API 模式：

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
| `codex` | 使用选定的命名账号或 API 模式启动 Codex |
| `codex-yolo` | 使用选定模式启动 Codex，并跳过审批与沙箱 |
| `codex-account-manager` | 打开原生 Qt 软件，选择 Codex 账号或 API provider |
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

#### 5. Provider 命令行管理

`claude/codex_provider.py` 是 GUI 和网关共用的 provider 后端，也可以直接用于命令行管理自定义 OpenAI-compatible API。无参数运行会打开交互菜单；脚本化时使用子命令。安装器不会创建单独的 `codex-provider` 命令，所以从仓库运行如下：

```bash
python3 claude/codex_provider.py                       # 交互菜单
python3 claude/codex_provider.py list                  # 列出 provider，不显示密钥
python3 claude/codex_provider.py show myproxy          # 查看非敏感配置
python3 claude/codex_provider.py add myproxy \
  --base-url https://example.com/v1 \
  --model gpt-5.4 \
  --prompt-key                       # 隐藏输入密钥，避免进入 shell history
python3 claude/codex_provider.py test myproxy          # TCP 和模型接口
python3 claude/codex_provider.py test myproxy --model gpt-5.4 # 最小请求
python3 claude/codex_provider.py update myproxy --effort high
python3 claude/codex_provider.py switch myproxy        # 设为默认 provider
python3 claude/codex_provider.py set-key myproxy       # 安全替换密钥
python3 claude/codex_provider.py delete myproxy --yes  # 删除非当前 provider
```

provider 配置写入 `~/.codex/config.toml` 及对应 profile；密钥单独保存在权限为 `0600` 的 `~/.config/codex/secrets.env`。日常输入密钥优先使用 `--prompt-key`、`set-key` 的隐藏提示，或通过标准输入传入，不要把真实密钥直接写进命令历史、README 或仓库。

`codex-auth api myproxy` 与后端的 `switch myproxy` 都会选择 API 模式；前者适合日常账号/API 模式切换，后者还可以同步 provider 的模型、思考强度和输出设置。通常优先使用 `codex-account-manager`，只有自动化或排错时才需要直接调用后端。

#### 6. 原生账号与 API 管理软件

运行 `codex-account-manager`，或从 Linux 应用菜单打开 **Codex Account Manager**。这是独立的 PyQt5 桌面软件，不使用浏览器、WebEngine、8320 网页服务或 `claudex-manager.service`。

软件集成 `codex-auth` 和本地 provider 管理后端：添加账号时邮箱输入为可选，留空后完成普通浏览器授权，软件会从登录令牌读取邮箱并自动命名；也可以预先填写邮箱。GUI 默认不使用设备码，因此不需要开启 OpenAI 的“为 Codex 启用设备代码授权”设置；只有在无头或远程终端显式运行 `codex-auth add-auto --device-auth` 时才需要该设置。未完成的浏览器授权可用 `Cancel login` 终止，随后账号和 API 操作按钮会恢复。账号页还可选择账号、查看额度和可恢复移除命名账号，旧版 `unnamed` 登录始终保留且禁止删除。API 页可以新增或编辑名称、Base URL、环境变量名和密钥，也可测试连接、删除非当前 provider 并切换使用。API 密钥通过进程标准输入保存，不出现在命令行参数中。额度卡片中的 `Show on desktop` 会把当前额度显示在桌面工作区左上角；悬浮窗可以按住鼠标左键拖动并记住位置，拖动范围会限制在屏幕以内，再次点击变为 `Hide from desktop`。关闭主窗口后软件驻留系统托盘；托盘可以快速切换账号/API、显示额度悬浮窗或彻底退出。

Qt 软件、`codex-auth` 和 `codex-usage` 使用同一份状态。所有切换只影响新启动的 Codex，不会终止或改变已经运行的终端。额度通过本地 Codex app-server 查询，不需要打开 ChatGPT 网页。

##### 额度命令与悬浮窗

`codex-usage` 默认查询 `codex-auth use` 当前选中的命名账号，也能显式选择账号或输出 JSON：

```bash
codex-usage
codex-usage --account user@example.com
codex-usage --json
watch -n 30 codex-usage              # 每 30 秒刷新
codex-usage --browser                 # 仅旧版 Chrome 登录方式
codex-usage --browser --port 9223
```

桌面悬浮窗的控制命令为：

```bash
codex-usage-widget                    # 等同于 start，后台启动
codex-usage-widget status
codex-usage-widget stop
codex-usage-widget foreground         # 前台运行，便于查看错误
```

悬浮窗跟随当前账号选择并显示最长额度周期；切到 API provider 时显示 `No account quota`，不会继续显示上一个账号的缓存。`codex-account-manager` 已包含托盘和额度悬浮能力，启动账号管理软件时会停止独立 widget，避免重复显示。

#### 7. 网页路由控制台（可选）

完整安装后运行 `claudex-ui`。页面中的 `Accounts` 抽屉可以新增 ChatGPT 登录、查看账号状态与额度、选择以后启动的 Codex 默认账号，或切回 API provider；设备码和登录进度会直接显示在抽屉里。每个已登录账号的 `Quota` 按钮通过 Codex 本地 app-server 读取套餐、剩余百分比和重置时间，不需要打开或登录 ChatGPT 网页。`Provider config` 继续用于配置、测试和选择某个具体 API provider，主页可以选择 GPT 模型、思考强度并查看用量。

这个网页只用于 Claude-to-Codex 路由和用量控制台；原生 `codex-account-manager` 不依赖它，只运行 `setup-codex.sh` 也会安装 Qt 软件。

## 桌面工具

### Markdown / HTML / LaTeX 文档渲染器

首次安装或修复依赖、命令和桌面关联，统一运行：

```bash
bash ~/scripts/desktop/install-markdown-renderer.sh
```

安装脚本只在发现缺失的 APT 包时调用 `sudo`，并在终端中由 sudo 正常读取密码；不会保存密码。它会检查项目模板和锁文件，安装实时预览、X11 Chrome 窗口嵌入、Pandoc/XeLaTeX、Poppler、TikZ、Mermaid 与浏览器依赖，并配置当前用户的 `mdview` 命令、桌面入口及 Markdown/TeX MIME 默认应用。脚本可重复运行；已满足的 APT 和 npm 依赖会跳过。

直接启动原生 PyQt5 软件窗口。默认只显示渲染结果；点击工具栏的“显示原文”后，左侧显示 Markdown 原文、右侧保持实时渲染：

```bash
mdview
mdview /path/to/document.md
mdview /path/to/report.html
mdview file:///path/to/report.html
mdview /path/to/document.tex
```

`mdview` 是指向 `~/scripts/markdown_editor.py` 的命令入口。窗口内也可点击“打开文件”选择 Markdown、HTML 或完整 LaTeX 文档。HTML 使用 Qt WebEngine 的 Chromium 内核显示，保留页面自身的 CSS、布局、脚本和相对路径本地图片；HTML 模式只提供可视化预览，不调用 Markdown/LaTeX PDF 导出器。主工具栏提供“打开文件 / 重新加载 / 目录 / ChatGPT / 原文 / 预览 PDF / 导出 PDF / 设置”等功能：

- “显示原文 / 隐藏原文”：切换双栏与纯预览模式。
- “重新加载”（`F5`）：重新读取磁盘上的当前文件并刷新预览，适合查看其他程序刚保存的修改。
- LaTeX 加载进度：通过 `mdview /path/to/document.tex` 直接打开或按 `F5` 重新加载时，窗口中央显示分阶段进度条，完成后自动隐藏。
- “ChatGPT / 隐藏 ChatGPT”：在文档目录左侧嵌入真实的系统 Chrome 窗口，形成“Chrome ChatGPT / 目录 / 正文”三栏布局。`mdview` 查找通过 `--remote-debugging-port=9223` 运行的 Chrome，复用其 `user-data-dir` 与登录状态，再让 Chrome 原生打开 `--app=https://chatgpt.com/`；官网样式、登录、会话、模型选择、上传、语音、复制和下载均由 Chrome 本身处理，不再使用 Qt WebEngine。下载捕获按本次内嵌页面的 Chrome target/frame ID 隔离：只有该页面下载完成的 `.md`、`.markdown`、`.tex`、`.latex` 或 `.ltx` 会自动替换当前 `mdview` 文档并立即渲染；普通 Chrome 窗口的下载不会触发。隐藏面板、点击 Dock 关闭按钮或退出 `mdview` 时会同时关闭本次内嵌页面 target，避免 Chrome renderer 在后台累积并拖慢输入。该窗口嵌入依赖 X11，不支持 Wayland。
- 预览右键复制：选中右侧渲染内容后，可选择“复制为文字”或“复制为图片”；图片会保留标题、表格和公式的排版，同时写入剪贴板和唯一文件 `/tmp/mdview-selection-*.png`，完整路径显示在状态栏。
- “复制为 ChatGPT 对话…”：根据当前文件、选中原文和 PDF 物理页码自动生成定位对话，末尾保留“修改要求（由我补充）”；粘贴到 ChatGPT 后填写具体改法即可。提示词要求只修改命中位置并返回完整同格式文件。
- 左侧目录：默认隐藏，点击“显示目录”后可在“文档目录”和“收藏”两个独立 Tab 间切换，收藏列表不会占用原目录空间。收藏 Tab 可固定整份 `.tex`；点击已经打开的当前文档不会重复编译，点击其他文档收藏才会读取并完整渲染。TeX 目录章节支持右键“收藏章节”，收藏项点击后打开对应文件并跳转到保存的章节；收藏项右键可取消。收藏数据跨窗口保存在当前用户的 Qt 配置文件 `~/.config/Codex Tools/Markdown Renderer.conf` 中，不修改 TeX 源文件。
- “设置”：收纳背景颜色、整体边框颜色和 Markdown 行间距；不再将低频选项平铺在主工具栏。
- 页面留白：实时预览在正文左右各保留约 `56px`，形成类似 Word 的阅读页边距；该样式只影响软件预览，不改变 PDF 的 A4 页边距。
- LaTeX 公式：支持 `$...$`、`\(...\)`、`\[...\]`、`$$...$$` 和文档中的 Pandoc `` `...`{=tex} `` 标记。
- 流程图：完整支持 Mermaid `flowchart`、`sequenceDiagram` 等标准代码块；支持 `tikz` 代码块，也会识别 `tex`/`latex` 代码块中的 `tikzpicture`。TikZ 在预览中临时编译为图像，在 PDF 中保留为 XeLaTeX 原生矢量图。可识别的 `text` ASCII 流程图也会自动转为图形。
- 报告预览：读取 YAML `title`/`subtitle` 生成封面；`numbersections: true` 会使实时预览和左侧导航使用与 Pandoc PDF 一致的章节编号。
- “导出 PDF”对话框：每次导出时临时选择输出文件、是否生成正文目录（TOC）以及导出后是否打开 PDF；“导出后打开”默认不勾选。导出期间在窗口中央显示进度，完成提示可直接复制文件所在目录或打开生成的 PDF。默认输出位置是源文件同目录，文件名不变，只将 `.md`/`.tex` 后缀改为 `.pdf`。这些选项不写入全局设置。
- “预览 PDF”：在 `~/.cache/mdview/` 生成临时 PDF 并用系统默认阅读器打开；Markdown 预览默认不生成正文目录，完整 TeX 预览遵循源文件中的 `\tableofcontents`。
- 完整 `.tex` 预览：窗口默认使用接近文档阅读的 `1100 × 1140` 纵向比例；自动运行两遍 XeLaTeX，先显示 PDF 前 3 页，其余页面由后台进程继续栅格化，完成后原位补齐；鼠标选择框直接使用 Poppler 提取的 PDF 单词坐标，不叠加会发生错位的透明 HTML 排版，可选择和复制正文。左侧目录读取 XeLaTeX 生成的 `.toc` 与 PDF 命名目标，点击后按物理页及页内坐标精确跳转。TikZ、页码和排版不经过 Markdown/Pandoc。
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
sudo apt install python3-pyqt5 python3-pyqt5.qtwebengine pandoc texlive-xetex texlive-lang-chinese texlive-latex-extra texlive-pictures fonts-noto-cjk graphviz poppler-utils x11-utils libx11-6
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

### 独立 Markdown 转 PDF

`markdown_to_pdf.sh` 是不启动 `mdview` 的轻量转换器，适合一次性把普通 GFM Markdown 转成 A4 PDF。它使用 Pandoc 生成 HTML，再调用本机 Chrome/Chromium 的 headless 打印功能：

```bash
bash markdown_to_pdf.sh notes.md
bash markdown_to_pdf.sh notes.md output/report.pdf
```

只传输入文件时，PDF 写到输入文件旁并沿用同名文件名。输出目录必须事先存在。该工具需要 `pandoc`、Google Chrome 或 Chromium；它不使用 `markdown_pdf/template.tex`，也不提供 `mdview` 的 XeLaTeX、TikZ、目录定位和交互预览能力。需要论文式排版或完整 LaTeX 时应使用 `mdview` 的“导出 PDF”。

### 安装 Flameshot 并配置 GNOME 快捷键

```bash
bash desktop/install-flameshot-shortcuts.sh
```

脚本会检查 Flameshot、剪贴板和 GNOME 桌面组件。所有依赖均已安装时不会调用 APT；缺少依赖时会更新软件包索引，并且只安装缺少的软件包。配置完成后：

- `Alt+A`：打开 Flameshot 截图界面。
- `Alt+S`：截图保存到 `/tmp`，并将完整文件路径复制到剪贴板。

该脚本仅支持 Ubuntu/Debian 的 GNOME 桌面环境，应当以当前桌面用户运行，不要直接使用 `sudo` 启动整个脚本。

### 修复 Zotero 中文候选窗位置

当 Zotero Linux 的中文输入法候选窗固定在屏幕左下角、不跟随光标时运行：

```bash
bash desktop/fix-zotero-ime-candidate-position.sh
```

脚本会定位 Zotero profiles，删除导致焦点窗口丢失的 `focusmanager.testmode=true`，并在需要时关闭、修复后重新启动 Zotero。高级用法：

```bash
bash desktop/fix-zotero-ime-candidate-position.sh \
  --profile-root /path/to/Zotero/Profiles
bash desktop/fix-zotero-ime-candidate-position.sh \
  --no-process-control                 # 仅在 Zotero 已关闭时使用
```

完整原因、备份和手动验证方法见 [故障排查文档](docs/troubleshooting/zotero-linux-ime-candidate-position.md)。

### 查看 DEB 包内文件

`deb-view-file` 使用 `dpkg-deb` 解包查看 `.deb`，不会安装软件包：

```bash
./deb-view-file ./example.deb
./deb-view-file ./example.deb /etc/example/config.conf
./deb-view-file ./example.deb /usr/share/doc/example
```

只传 `.deb` 时，工具列出全部包内文件并保留临时解压目录，终端会打印目录路径；再传一个包内路径时，它会用 `less` 或 `cat` 显示文件，传目录则列出该目录内容，退出后自动清理临时目录。依赖 Ubuntu/Debian 的 `dpkg-deb`。

## ScholarVault 论文工作区（C++20 / Qt 6）

ScholarVault 是与 `mdview` 并列安装的本地论文工作区。一个论文项目只属于一个真实话题目录，可以从 arXiv 链接、本地 PDF 或 Zotero 文献库创建，并在项目下保存原始 TeX、相关 GitHub 代码、Markdown 笔记和标注数据。

默认 Vault 位于隐藏目录 `~/.local/share/ScholarVault/Vault`。旧默认位置 `~/Documents/ScholarVault` 会在安全条件满足时于下次启动整体迁移；手动选择的 Vault 不会自动移动。程序启动后会请求窗口置前，但不会永久保持“总在最前”。

安装并构建：

```bash
bash desktop/install-scholarvault.sh
```

安装器只在缺少依赖时调用 APT，构建完成后新增 `~/.local/bin/scholarvault` 和独立桌面入口。它会在安装前后核对现有 `mdview` 的路径和软链接目标，不会替换 `mdview`、`markdown_editor.py` 或其 MIME 关联。

运行：

```bash
scholarvault
mdview /path/to/existing-document.tex  # 原命令继续使用
```

当前实现使用 Qt PDF 的异步可见页渲染器阅读 PDF：按设备像素并至少 1.5 倍超采样，只缓存可见页附近，连续拖动分隔条时合并刷新。PDF 支持鼠标框选文字，并可用 `Ctrl+C` 或右键复制；工具栏支持缩放与恢复适合宽度。话题先加载、论文在展开话题时按需扫描。双击项目文件中的 `.tex` 才会在后台运行 XeLaTeX，预览按源码树指纹缓存在 Vault 之外，未修改的源码不会重复编译。

导入 arXiv 或同步 Zotero 时默认只保存 PDF 与可识别的 arXiv ID，不预先下载原始 LaTeX。点击当前论文的 ChatGPT 或 Codex 分析入口时才按需下载 `e-print` 源码包，安全检查后保存到项目 `source/`；没有源码或下载失败时会继续使用 PDF，不阻断分析。右侧 ChatGPT 复用 9223 端口的系统 Chrome 登录状态，通过 X11 直接嵌入并校验父窗口；Codex 通过嵌入式 xterm 在当前论文目录运行。两种嵌入窗口的尺寸更新均进行节流，避免拖动右栏时反复创建外部进程。

“新建论文”中的“从 Zotero 导入”默认读取 `~/Zotero`。Zotero 正在运行时使用其本机只读 API 获取最新题录；未运行时回退到只读 `zotero.sqlite`。选择窗口支持按标题、作者、年份和 PDF 文件名搜索。导入会复制 PDF，并保存 Zotero 条目 Key 与附件 Key；不会修改 Zotero 数据库或原附件。

顶部“同步 Zotero”会扫描所有本地 Zotero PDF，按 Zotero 分类层级建立嵌套话题，复制缺失论文，并更新已有项目的题录、分类和 PDF；未分类论文进入 `未分类`。目标 PDF 是独立副本，不是软链接。论文右键菜单可只同步当前论文。同步不会向 Zotero 写入或删除。话题右键的“删除话题…”会把完整目录移动到 Vault 的 `.trash/topics/`，不是永久删除。顶部不再放置“新建论文”和“添加相关代码”，这两个操作分别保留在话题和论文的右键菜单中。

桌面入口使用独立的 ScholarVault 图标，并安装 32、64、128、256 与 512 像素版本；不会替换 `mdview` 的图标。

开发构建与核心测试：

```bash
cmake -S scholarvault -B build/scholarvault -G Ninja
cmake --build build/scholarvault
ctest --test-dir build/scholarvault --output-on-failure
```

完整设计与持久化格式见 `docs/superpowers/specs/2026-08-13-scholarvault-design.md`。

## GPU 显存管理

GPU Memory Keeper 通过独立 CUDA 进程预留显存，方便在共享机器上提前占住空闲显存，并在真正任务启动前释放。它只预留显存，不会把 GPU 计算模式改为 exclusive，也不能阻止其他进程使用算力。

### GPU Harbor 桌面管理器

安装 `PyQt5` 后直接从仓库启动：

```bash
sudo apt install python3-pyqt5
python3 gpu_memory_keeper_gui.py
```

界面默认连接本机并显示每张 NVIDIA GPU 的显存、利用率和 Keeper 状态：

- `OCCUPY / 立即占用`：按界面中的上限启动显存预留。
- `RELEASE / 释放`：释放对应 GPU 的 Keeper 显存。
- `MONITOR / 递增监督`：让已有预留根据总显存使用量动态调整。
- `释放全部`：停止当前目标机器上的全部 Keeper 预留。
- SSH 目标：可填写 `host`、`user@host`、`ssh -p 2222 user@host` 或 `~/.ssh/config` 别名；GUI 会把 CLI 脚本部署到远端用户目录后执行。

远程连接使用非交互 `BatchMode`，因此应先配置 SSH key 并确认 `ssh TARGET` 无需输入密码。远端需要 `python3`、PyTorch CUDA、`nvidia-smi` 和可用的 NVIDIA 驱动。日志保存在 `~/.local/state/gpu-harbor/`，Keeper 工作日志保存在目标机器的 `/tmp/gpu_memory_keeper_gpu<编号>.log`。

### GPU Memory Keeper 命令行

需要 PyTorch CUDA。无子命令或使用 `menu` 会打开交互式终端菜单：

```bash
python3 gpu_memory_keeper.py
python3 gpu_memory_keeper.py menu
```

常用非交互命令：

```bash
# 后台预留：总使用量目标 99%，但 Keeper 自身最多占单卡 80%
python3 gpu_memory_keeper.py occupy --gpu 0

# 指定固定显存，或同时管理多张卡
python3 gpu_memory_keeper.py occupy --gpu 0 --memory 8G
python3 gpu_memory_keeper.py occupy --gpus 0,1 --percent 95

# 逐步增加预留；需要登录后持续守护时使用 systemd user service
python3 gpu_memory_keeper.py occupy --gpu 0 --incremental \
  --step-percent 10 --interval 5
python3 gpu_memory_keeper.py occupy --gpus 0,1 --systemd-guard

# 查看、动态监督、修改实时上限和释放
python3 gpu_memory_keeper.py status --gpu 0
python3 gpu_memory_keeper.py list
python3 gpu_memory_keeper.py list --json
python3 gpu_memory_keeper.py monitor --gpu 0 --percent 95 --step-percent 10
python3 gpu_memory_keeper.py configure --gpu 0 --keeper-percent 70
python3 gpu_memory_keeper.py release --gpu 0
python3 gpu_memory_keeper.py stop --gpu 0
python3 gpu_memory_keeper.py release-all
```

`release` 会向 Keeper 发送正常释放请求并等待进程退出；`stop` 用终止信号停止 Keeper。两者都会处理并停用对应的 systemd guard。使用 `--systemd-guard` 创建的服务会在 worker 异常退出后自动重启，因此不要只杀 worker PID。查看完整参数可运行 `python3 gpu_memory_keeper.py COMMAND --help`。

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

已有 engine 可直接加载：

```python
model = TensorRTModel(engine_model_path='model/your_model.engine')
```

运行模块本身会导出一个最小全连接 ONNX 示例、构建 engine 并执行一次推理：

```bash
python tensorrt_inference.py
```

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

完整路径示例：

```bash
python convert_trt.py \
  --model /data/models/depth.onnx \
  --image /data/images/input.jpg \
  --output /data/results/depth.jpg
```

## 配置检查与开发验证

以下脚本用于检查安装后的 Claude/Codex 配置，不会安装或切换账号：

```bash
bash claude/verify-yolo-skills.sh   # 快速列出各 yolo 命令启用的 Skills
bash claude/skill-status-report.sh  # Claude/Codex 配置、aliases 和 Skills 完整报告
bash claude/final-verification.sh   # 汇总默认命令与 yolo 配置是否存在
```

这些检查读取 `~/.claude`、`~/.codex` 和 `~/.bashrc`，部分输出依赖 `jq`。如果刚运行过安装器，先执行 `source ~/.bashrc` 再检查。

修改仓库后可运行不依赖 GPU 的回归检查：

```bash
bash -n claude/*.sh desktop/*.sh markdown_to_pdf.sh deb-view-file
python -m unittest discover -s tests -v
```

ScholarVault 使用前文的 CMake/CTest 命令。CUDA、ONNX Runtime 和 TensorRT 检查需要对应 GPU 环境，不包含在纯 CPU 单元测试中。

## 依赖

| 工作流 | 主要依赖 |
| --- | --- |
| Codex/Claude 配置 | Bash、Python 3、Codex CLI、Claude Code；桌面账号管理另需 `PyQt5` |
| mdview | `PyQt5`、Qt WebEngine、Markdown、matplotlib；PDF/图表依赖由安装器检查 |
| Markdown 快速转 PDF | Pandoc、Google Chrome 或 Chromium |
| ScholarVault | C++20、CMake、Ninja、Qt 6、Qt PDF、SQLite、X11；安装器自动检查 APT 包 |
| Flameshot 快捷键 | Ubuntu/Debian、GNOME、APT、Flameshot、Wayland/X11 剪贴板工具 |
| GPU Harbor / Keeper | NVIDIA 驱动、`nvidia-smi`、PyTorch CUDA；GUI 另需 `PyQt5`，远程模式另需 SSH key |
| 环境检查 | Python 标准库、Linux 系统库和 `ldd` |
| ONNX Runtime 测试 | `numpy`、`onnx`、`onnxruntime-gpu` |
| TensorRT 推理 | `numpy`、`torch`、`tensorrt`、`pycuda` |
| 深度图示例 | TensorRT 推理依赖，另加 `opencv-python`、`matplotlib` |

## 故障排查笔记

- [Zotero Linux 中文候选窗固定在左下角](docs/troubleshooting/zotero-linux-ime-candidate-position.md)：记录 Gecko `focusmanager.testmode` 导致焦点窗口丢失、候选窗无法跟随光标的诊断、修复和验证流程。

对应的一键修复命令见上面的“修复 Zotero 中文候选窗位置”。

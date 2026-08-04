# 个人脚本工具集

用于 Linux 桌面配置、ONNX 环境检测以及 TensorRT 模型转换与推理的个人脚本集合。

## Claude Code / Codex 网关

### 新电脑自动配置

#### 0. 前置条件

先安装官方 Claude Code 和 Codex CLI，并确认当前用户能够使用 systemd user service：

```bash
claude --version
codex --version
systemctl --user show-environment >/dev/null
```

获取脚本：

```bash
git clone https://github.com/xuhan-rgb/scripts.git ~/scripts
cd ~/scripts
```

不要用 `sudo` 运行安装器；所有配置和服务都属于当前桌面用户。

#### 1. 准备 API Key

API Key 不进入仓库，只保存在权限为 `0600` 的 `~/.config/codex/secrets.env`。有两种配置方式。

从旧电脑迁移时，先通过可信介质或 `scp` 将旧机的 `~/.config/codex/secrets.env` 复制到新机的临时路径，然后执行零交互安装：

```bash
CLAUDEX_SECRETS_FILE=/secure/path/secrets.env \
bash claude/install-codex-bridge.sh
```

没有旧密钥文件时，直接运行安装器；终端会针对当前 provider 无回显地询问 Key：

```bash
bash claude/install-codex-bridge.sh
```

也可以预先导出对应环境变量实现零交互安装，例如当前 `crs`/`crs_local` 使用 `CRS_OPENAI_KEY`。不要把真实 Key 直接写进命令历史。

#### 2. 自动部署顺序

`install-codex-bridge.sh` 严格按以下顺序执行：

1. 检查 `~/.bashrc` 中的 `codex-yolo`、`claude-yolo`、`claudex-yolo`。已有 alias 会删除旧定义并直接替换，没有则新增；重复安装始终只保留一份。
2. 安装并调用 `codex-provider`，创建或补全 `~/.codex/config.toml` 中的 provider URL、profile 和 Key 引用，并将 Key 写入私有的 `secrets.env`。
3. 读取当前 Codex `model_provider` 的 `base_url`、`wire_api` 和 `env_key`，生成 CLIProxyAPI 配置，启动 `cli-proxy-api.service` 与 `claudex-manager.service`。
4. 验证网关只暴露 `claudex-router`，然后即可使用 `codex-yolo`、`claude-yolo` 或 `claudex-yolo`。

安装器写入的 alias 为：

```bash
alias codex-yolo='codex --dangerously-bypass-approvals-and-sandbox'
alias claude-yolo="claude --dangerously-skip-permissions --disable-slash-commands --strict-mcp-config --mcp-config '{\"mcpServers\":{}}'"
alias claudex-yolo='claudex --dangerously-skip-permissions'
```

默认会提供 `crs`、`crs_local`、`sorryios`、`zskj` 四个 provider，并保留已有且有效的自定义 provider。默认项是 `crs_local`；新电脑没有本地 `127.0.0.1:3000` 服务时，可在安装时选择远程 `crs`：

```bash
CLAUDEX_DEFAULT_PROVIDER=crs \
CLAUDEX_SECRETS_FILE=/secure/path/secrets.env \
bash claude/install-codex-bridge.sh
```

#### 3. 部署后验证

安装完成后让当前终端加载 alias，并检查 provider 与服务：

```bash
source ~/.bashrc
type codex-yolo claude-yolo claudex-yolo
codex-provider list
systemctl --user is-active cli-proxy-api.service claudex-manager.service
curl --fail http://127.0.0.1:8320/healthz
```

预期两个 service 都输出 `active`，健康检查返回成功。日常启动命令：

```bash
codex-yolo        # 当前 Codex provider，跳过确认
claude-yolo       # 官方 Claude 账号，跳过确认
claudex           # GPT 中转，保留权限确认
claudex-yolo      # GPT 中转，跳过确认
claudex-ui        # 打开本地路由控制台
```

`claude-yolo` 直接调用官方 `claude`，不会修改登录状态或 `~/.claude`；只有 `claudex`/`claudex-yolo` 会为当前进程注入本地 GPT 网关环境。安装器不替换 `claude`，也不调用 `claude login/logout`。

#### 4. Provider 日常管理

```bash
codex-provider list
codex-provider show crs
codex-provider switch crs
codex-provider set-key crs
codex-provider test crs
```

切换 provider 后，新启动的 `codex-yolo` 直接使用该 provider；新启动的 `claudex`/`claudex-yolo` 会先同步连接配置再启动中转客户端。真实 Key 不会写入 `~/.codex/config.toml` 或 Git。

已有 Codex 配置不会被整体覆盖。重复执行安装器是幂等的：alias 会替换而非累加，已有 Key 和模型选择会保留，版本匹配的 CLIProxyAPI 不会重复下载。

脚本最终会安装：

- 安装 `codex-provider`、固定版本的 CLIProxyAPI user service、Claude 启动器和 Codex Routing Desk。
- 创建 `claudex`、`claudex-yolo`、`claudex-ui` 命令。
- 创建仅监听 `127.0.0.1` 的 `cli-proxy-api.service` 与 `claudex-manager.service`。
- 创建权限为 `0600` 的 Codex 配置、provider secrets、网关配置和 usage 数据库。

Claude 始终只连接一个稳定的虚拟模型 `claudex-router`，客户端 effort 固定为 `medium`。真实的 GPT 模型和 reasoning effort 由 Codex Routing Desk 在网关层逐请求覆盖，独立保存在 `~/.cli-proxy-api/selection.conf`，不会反向修改 Codex 配置。

`claudex-ui` 打开的 Codex Routing Desk 只监听 `127.0.0.1:8320`。Instant switch 默认开启；在界面点击 Sol、Terra、Luna 或 reasoning effort 后会立即应用到下一次 Claude 请求。关闭 Instant switch 后可恢复 `Apply selection` 手动确认。所有已打开的 `claudex` 会话都会跟随新路由，不需要重启。

`claudex` 默认传入 `--prompt-suggestions false`，避免 Claude Code 在主回答结束后再调用一次 GPT 生成“下一条建议”，减少每轮额外的上下文 Token 和后台等待；如确实需要该功能，可用 `claudex-yolo --prompt-suggestions true` 临时恢复。

为了降低新会话的固定上下文，安装器默认关闭 Codex 与 Claude 的自定义 skills、plugins 和 MCP，但不会卸载或删除它们：

- Codex 的每个 MCP/plugin section 会写为 `enabled = false`，已发现的 skill 会通过 `[[skills.config]]` 按绝对路径禁用；修改 `~/.codex/config.toml` 后重启 Codex 即可生效。
- Claude 已安装插件会通过官方 `claude plugin disable` 关闭；`claude-yolo` 和 `claudex` 还会使用 `--disable-slash-commands --strict-mcp-config`，避免项目级 MCP 或本地 skill 自动进入上下文。
- 临时需要 Claude GPT 中转的扩展时，使用 `CLAUDEX_EXTENSIONS=1 claudex-yolo`；官方 Claude 可先执行 `claude plugin enable <plugin@marketplace>`，再直接运行 `claude`。
- Codex 需要单个 MCP 时，可在对应 `[mcp_servers.<name>]` 下把 `enabled` 改回 `true`；需要单个 skill 时，把对应 `[[skills.config]]` 的 `enabled` 改回 `true`，然后启动新会话。

首次关闭扩展前，安装器会各保留一份固定备份：Codex 配置使用 `*.before-disabled-extensions`，Claude 设置使用 `~/.claude/settings.json.before-disabled-extensions`。重复安装不会继续堆积备份。

控制台的 Token usage 按电脑本地时区统计：Today 从当天 00:00 开始，This week 从周一 00:00 开始，This month 从每月 1 日 00:00 开始。三个周期都分别显示非缓存输入、缓存读取、输出和请求数；下方的 Last request 额外显示最后一次请求的输入、输出、缓存读取和缓存命中率。卡片每 3 秒增量更新，不会重新创建界面节点。

Request ledger 从 CLIProxyAPI usage queue 采集每次请求的真实上游模型、reasoning effort、接口、非缓存输入、输出 token、推理 token、缓存读取/创建、命中率、首 Token 时间（TTFT）、总耗时和状态，并持久化到权限为 `0600` 的 `~/.cli-proxy-api/usage.sqlite3`。其中“输入”始终按 `input_tokens - cache_read_tokens` 计算，避免与单独展示的缓存读取重复。最多保留最近 5000 条元数据，不开启完整 request log，也不保存提示词或回答正文。

延迟排查时优先比较“首 Token / 总耗时”：两者都高通常是上游网络、模型排队或上下文过长；首 Token 正常而总耗时高通常是回答生成较长。Claude 会在每轮携带会话上下文，即使缓存命中率很高，过长上下文仍可能增加模型处理时间；可使用 Claude 的 `/compact`，或在不再需要旧上下文时开启新会话。切换到 Luna 或降低 reasoning effort 可能更快，但应以该 provider 上实际记录的 TTFT 为准。

Claude Code 的系统提示词、内置工具、plugins、MCP、skills 和 `CLAUDE.md` 也属于输入上下文，因此一句 `hello` 的原始总输入可能仍有数万 Token。界面把它拆成“输入”和“缓存读取”：两者相加才是上游返回的原始输入总量。需要隔离排查时可运行 `claudex-yolo --safe-mode`：它保留认证、模型、内置工具和权限，但临时禁用自定义 plugins、MCP、skills、hooks 与 `CLAUDE.md`，不会修改现有配置或登录状态。

脚本支持 Linux x86_64 和 arm64，要求所选 Codex provider 使用 Responses API，并配置可用的 `base_url` 与 `env_key`。

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

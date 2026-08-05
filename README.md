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

没有旧密钥文件时也可以直接运行安装器。安装器先启动 8320 控制台，不在终端阻塞等待 Key；安装完成后在网页右上角打开 `Provider config` 设置：

```bash
bash claude/install-codex-bridge.sh
```

也可以预先导出对应环境变量实现零交互安装，例如当前 `crs`/`crs_local` 使用 `CRS_OPENAI_KEY`。不要把真实 Key 直接写进命令历史。

#### 2. 单脚本部署与网页配置

新电脑只使用一个安装入口：

```bash
bash claude/install-codex-bridge.sh
```

`install-codex-bridge.sh` 严格按以下顺序执行：

1. 初始化 `~/.codex/config.toml` 中的 provider；没有 Key 时只记录待配置状态，不中断安装，也不安装独立的 `codex-provider` 命令。
2. 检查 `~/.bashrc` 中的 `codex-yolo`、`claude-yolo`、`claudex-yolo`，然后安装精选 Skills 并关闭其余扩展。没有 `uv` 且 Agent Reach 尚未安装时，交互式询问是否使用 Python venv 安装；选择不使用只会关闭 Agent Reach。已有 alias 会删除旧定义并直接替换，没有则新增；重复安装始终只保留一份。
3. 无条件启动只监听 `127.0.0.1:8320` 的 `claudex-manager.service`。如果当前 provider 和 Key 已经完整，同时生成 CLIProxyAPI 配置并启动网关；否则网页显示 `Provider setup required`。
4. 在右上角 `Provider config` 中新增、修改或切换 provider。保存时由网页内置后端更新 Codex 配置和私有 Key，再运行 `claude-codex-sync` 并重启 CLIProxyAPI，使 `codex-yolo` 与 `claudex-yolo` 使用同一个当前 provider。

页面不会返回或显示已有 Key；Key 只允许设置或替换，并保存在权限为 `0600` 的 `~/.config/codex/secrets.env`。页面只监听本机回环地址，不提供删除 provider 的操作。

安装精选 Skill 时会显示正在检查或下载的 Claude plugin 来源名称；这些插件只用于复制 Skill，随后保持关闭。网络较慢时可以据此区分正在下载和 provider 配置问题。

安装器写入的 alias 为：

```bash
alias codex-yolo='codex --dangerously-bypass-approvals-and-sandbox'
alias claude-yolo='claude --dangerously-skip-permissions --strict-mcp-config'
alias claudex-yolo='CLAUDEX_YOLO=1 claudex'
```

如果新电脑的 `~/.codex/config.toml` 已经配置了有效的 `model_provider` 及对应 `[model_providers.<name>]`，安装器会直接读取并采用它，再由内部 Provider 后端创建缺失的 profile。已有 provider 的 URL、模型和选择不会被四个内置 provider 覆盖。

仅当 `config.toml` 不存在，或者当前 `model_provider` 没有对应配置段时，安装器才使用 `crs_local` 作为初始待配置项。安装完成后可直接在网页新增或切换到实际 provider；也可在安装前用 `CLAUDEX_DEFAULT_PROVIDER` 改变初始项：

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
systemctl --user is-active cli-proxy-api.service claudex-manager.service
curl --fail http://127.0.0.1:8320/healthz
```

`claudex-manager.service` 和健康检查应始终成功；完成 Provider 与 Key 配置后，`cli-proxy-api.service` 也应为 `active`。尚未配置 Key 时网关保持停止属于正常的首次安装状态。日常启动命令：

```bash
codex-yolo        # 当前 Codex provider，跳过确认
claude-yolo       # 官方 Claude 账号，跳过确认
claudex           # GPT 中转，保留权限确认
claudex-yolo      # GPT 中转，跳过确认
claudex-ui        # 打开本地路由控制台
```

`claude-yolo` 直接调用官方 `claude`，不会修改登录状态或账号凭据；它不传入 `--autocompact`，因此使用 Claude 官方默认的自动 compact 阈值和账号模型。只有 `claudex`/`claudex-yolo` 会为当前进程注入本地 GPT 网关环境。安装器不替换 `claude`，也不调用 `claude login/logout`。两个 Claude 启动命令都会保留 `~/.claude/CLAUDE.md` 和当前项目层级的 `CLAUDE.md`。

#### 4. Provider 日常管理

打开 `http://127.0.0.1:8320`，在右上角 `Provider config` 中完成新增、修改和切换。项目不再向 `~/.local/bin` 安装独立的 `codex-provider` 命令。
重复执行新版安装器时，会清理旧版安装器曾写入该路径的命令。

切换 provider 后，新启动的 `codex-yolo` 直接使用该 provider；网页会立即同步并重启网关，因此新启动的 `claudex`/`claudex-yolo` 使用同一个 provider。真实 Key 不会写入 `~/.codex/config.toml` 或 Git。

已有 Codex 配置不会被整体覆盖。重复执行安装器是幂等的：alias 会替换而非累加，已有 Key 和模型选择会保留，版本匹配的 CLIProxyAPI 不会重复下载。

脚本最终会安装：

- 安装固定版本的 CLIProxyAPI user service、Claude 启动器和带内置 Provider 后端的 Codex Routing Desk。
- 安装五个固定的精选 Skill；Agent Reach v1.5.0 从固定的官方 GitHub tag 安装，但在没有 `uv` 时可选择跳过。插件整包仅作为 Skill 来源，复制完成后保持关闭。
- 创建 `claudex`、`claudex-yolo`、`claudex-ui` 命令。
- 创建仅监听 `127.0.0.1` 的 `cli-proxy-api.service` 与 `claudex-manager.service`。
- 创建权限为 `0600` 的 Codex 配置、provider secrets、网关配置和 usage 数据库。

Claude 始终只连接一个稳定的虚拟模型 `claudex-router[1m]`，客户端 effort 固定为 `medium`。`claudex` 通过进程级 `availableModels` 把 `/model` 收敛为 `Default` 和 `GPT Router (1M)`，两项都指向本地中转；该限制不会写入 `~/.claude/settings.json`，也不会影响 `claude-yolo` 的官方账号登录。真实的 GPT 模型和 reasoning effort 仍由 Codex Routing Desk 在网关层逐请求覆盖，独立保存在 `~/.cli-proxy-api/selection.conf`，不会反向修改 Codex 配置。

当前部署使用 ChatGPT/Codex 订阅账号中转，而不是 OpenAI Platform API Key，因此按订阅产品的窗口和 usage 风险配置：Codex 的最大上下文写为 `372000`，但自动 compact 固定在 `244800`；`claudex` 的 auto-compact window 固定为 `250k`。运行 Codex `/status` 应看到 `372K` context window，运行 Claude `/context` 应看到 `claudex-router[1m]` 和 `250k` auto-compact window。

GPT-5.6 [Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)、[Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) 和 [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) 的模型 API 上限虽然是 1,050,000 Token（最大输入 922,000、最大输出 128,000），但官方同时规定：输入超过 272K 后，整次 API 请求的输入按 2 倍、输出按 1.5 倍计价。订阅账号中转不会直接产生相同形式的 API 账单，OpenAI 也没有公开完全等价的订阅 usage 换算规则；这里仍把 272K 作为保护线，避免长会话过快消耗账号或中转额度。如确实需要一次性使用更长上下文，应先手动 compact 或开启新会话，而不是长期提高默认阈值。

`claudex-ui` 打开的 Codex Routing Desk 只监听 `127.0.0.1:8320`。Instant switch 默认开启；在界面点击 Sol、Terra、Luna 或 reasoning effort 后会立即应用到下一次 Claude 请求。关闭 Instant switch 后可恢复 `Apply selection` 手动确认。所有已打开的 `claudex` 会话都会跟随新路由，不需要重启。

`claudex` 默认传入 `--prompt-suggestions false`，避免 Claude Code 在主回答结束后再调用一次 GPT 生成“下一条建议”，减少每轮额外的上下文 Token 和后台等待；如确实需要该功能，可用 `claudex-yolo --prompt-suggestions true` 临时恢复。

`claudex-yolo` 还会通过当前进程的 `skillOverrides` 关闭 Claude Code 内置的 `claude-api` Skill，避免普通问题被误判为 Claude API 开发任务后加载额外上下文。该覆盖不写入 `~/.claude/settings.json`；`claude-yolo` 仍使用官方账号环境，并保留 `claude-api`。

为了降低新会话的固定上下文，安装器固定开启五个精选 Skill：`brainstorming`、`grill-me`、`grill-with-docs`、`handoff`、`tdd`。`agent-reach` 在已安装、存在 `uv` 或用户同意使用 Python venv 时一并开启；选择跳过不会删除已有文件，但会在 Claude 和 Codex 中关闭它。`grilling` 和 `domain-modeling` 是两个 grill 命令内部调用的依赖，不作为主入口展示；其余自定义 Skill、plugins 和 MCP 默认关闭，但不会被卸载或删除。

无 `uv` 时，交互式安装会询问是否启用 Agent Reach。非交互安装默认跳过；可设置 `CLAUDEX_AGENT_REACH=1` 强制使用 Python venv（要求系统已安装 `python3-venv`），或设置 `CLAUDEX_AGENT_REACH=0` 明确关闭。

| 命令 | 可用的精选 Skill | 额外差异 |
| --- | --- | --- |
| `codex-yolo` | 五个固定主 Skill 和可选 Agent Reach；两个内部依赖也启用 | 读取 `~/.agents/skills`；Codex MCP/plugin 全部关闭；不经过 GPT 中转和 8320 usage 统计 |
| `claude-yolo` | 五个固定主 Skill 和可选 Agent Reach；两个依赖以 `name-only` 提供 | 使用官方 Claude 账号；保留 Claude 内置 `claude-api`；严格关闭 MCP |
| `claudex-yolo` | 与 `claude-yolo` 相同 | 使用 GPT 中转，并只在该进程额外关闭内置 `claude-api`；严格关闭 MCP |

- Codex 的每个 MCP/plugin section 都写为 `enabled = false`。五个固定主 Skill、按选择启用的 Agent Reach 和两个内部依赖只启用 `~/.agents/skills` 中的一份；`~/.claude/skills` 的重复副本以及其他已发现 Skill 都写为 `enabled = false`。
- Claude 的已安装插件会通过官方 `claude plugin disable` 关闭，包括用于提供 Skill 源文件的 `mattpocock-skills` 和 `superpowers`。`~/.claude/settings.json` 的 `skillOverrides` 将五个固定主 Skill 和按选择启用的 Agent Reach 设为 `on`、两个内部依赖设为 `name-only`、其余用户 Skill 设为 `off`。
- Claude 启动器使用 `--strict-mcp-config`，不会读取用户、项目或全局 MCP 配置，但不会使用 `--safe-mode`，所以 `~/.claude/CLAUDE.md` 与项目层级的 `CLAUDE.md` 仍然生效。
- 临时需要 Claude 读取已有 MCP 配置时，使用 `CLAUDEX_EXTENSIONS=1 claudex-yolo`；这会取消本地中转进程的严格 MCP 限制，Skill 仍以 `skillOverrides` 为准。
- Claude 需要临时打开其他 Skill 时，在 `~/.claude/settings.json` 的 `skillOverrides` 中将对应名称改为 `on`；Codex 则在 `~/.codex/config.toml` 中将对应 `[[skills.config]]` 的 `enabled` 改为 `true`，然后启动新会话。

首次关闭扩展前，安装器会各保留一份固定备份：Codex 配置使用 `*.before-disabled-extensions`，Claude 设置使用 `~/.claude/settings.json.before-disabled-extensions`。重复安装不会继续堆积备份。

控制台的 Token usage 按电脑本地时区统计：Today 从当天 00:00 开始，This week 从周一 00:00 开始，This month 从每月 1 日 00:00 开始。三个周期都分别显示非缓存输入、缓存读取、输出和请求数；下方的 Last request 额外显示最后一次中转请求的输入、输出、缓存读取、缓存命中率，以及“刚刚请求 / N 秒前请求”标记。请求数据每 3 秒增量更新，但后台轮询不会重置该标记；标记始终基于最后一条请求的时间戳，由浏览器每秒更新。`codex-yolo` 不经过中转，因此不会进入这份统计。

Request ledger 从 CLIProxyAPI usage queue 采集每次请求的真实上游模型、reasoning effort、接口、非缓存输入、输出 token、推理 token、缓存读取/创建、命中率、首 Token 时间（TTFT）、总耗时和状态，并持久化到权限为 `0600` 的 `~/.cli-proxy-api/usage.sqlite3`。其中“输入”始终按 `input_tokens - cache_read_tokens` 计算，避免与单独展示的缓存读取重复。最多保留最近 5000 条元数据，不开启完整 request log，也不保存提示词或回答正文。

延迟排查时优先比较“首 Token / 总耗时”：两者都高通常是上游网络、模型排队或上下文过长；首 Token 正常而总耗时高通常是回答生成较长。Claude 会在每轮携带会话上下文，即使缓存命中率很高，过长上下文仍可能增加模型处理时间；可使用 Claude 的 `/compact`，或在不再需要旧上下文时开启新会话。切换到 Luna 或降低 reasoning effort 可能更快，但应以该 provider 上实际记录的 TTFT 为准。

Claude Code 的系统提示词、内置工具、plugins、MCP、skills 和 `CLAUDE.md` 也属于输入上下文，因此一句 `hello` 的原始总输入可能仍有数万 Token。界面把它拆成“输入”和“缓存读取”：两者相加才是上游返回的原始输入总量。`claude-yolo` 与 `claudex-yolo` 默认保留用户级和项目级 `CLAUDE.md`，只按 allowlist 加载通用编程 Skill，并严格关闭 MCP；临时需要中转读取已有 MCP 配置时使用 `CLAUDEX_EXTENSIONS=1 claudex-yolo`。

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

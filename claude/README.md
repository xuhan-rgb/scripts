# Claude/Codex Skills 管理工具

管理 Claude 和 Codex 的 skill 配置，提供精简的 yolo 模式。

## 快速开始

原生账号管理软件需要 `PyQt5>=5.15,<6`；Ubuntu/Debian 推荐安装系统包 `python3-pyqt5`，版本约束也记录在 `requirements-desktop.txt`。

### 安装配置
```bash
cd /home/qwer/scripts/claude
./setup-codex.sh
source ~/.bashrc
```

### 验证配置
```bash
# 简单验证（推荐）
./verify-yolo-skills.sh

# 完整报告
./skill-status-report.sh
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `setup-codex.sh` | 主安装脚本，配置所有 yolo 命令 |
| `switch-codex-auth.sh` | Codex API provider 与多个 ChatGPT 账号管理后端 |
| `codex-usage` | 查询 Codex 剩余额度百分比和 token 使用统计 |
| `../codex-usage-widget` | 桌面显示当前 Codex 账号及最长额度周期 |
| `codex_account_manager_qt.py` | 原生 PyQt5 账号/API 管理主窗口、托盘和悬浮窗 |
| `codex_account_manager_backend.py` | Qt 软件使用的非敏感状态与额度解析层 |
| `verify-yolo-skills.sh` | 快速验证 yolo 配置是否正确 |
| `skill-status-report.sh` | 生成完整的 skill 配置报告 |
| `SKILLS_STATUS.md` | 详细的 skill 配置文档 |

## 配置文件

| 文件 | 用途 |
|------|------|
| `~/.claude/settings.json` | Claude 默认配置 |
| `~/.claude/settings.yolo.json` | Claude yolo 精简配置 |
| `~/.codex/config.toml` | Codex 默认配置 |
| `~/.codex/yolo.config.toml` | Codex yolo 精简配置 |

## Codex 与 Yolo 命令

普通 `codex` 和所有 yolo 命令统一使用 **5 个核心 skills**：

```bash
codex         # Codex + 精简 skills + 保留审批
claude-yolo   # Claude + 精简 skills + 跳过权限
codex-yolo    # Codex + 精简 skills + 跳过审批
claudex-yolo  # Claudex + 精简 skills + yolo 模式
```

## Codex 多账号

终端和 GUI 使用同一个账号目录与默认选择：

```bash
codex-auth add user@example.com
codex-auth add work@example.com --device-auth
codex-auth add-auto               # 普通浏览器授权，自动用登录邮箱命名
codex-auth add-auto --device-auth # 无头/远程环境使用设备码授权
codex-auth list
codex-auth use user@example.com
codex-auth remove work@example.com --yes # 归档凭据，共享对话不删除
codex                         # 固定使用启动时选中的账号
codex-auth run --account work --
codex-auth api crs_local      # 新进程切到指定 API provider
codex-usage                   # 当前命名账号额度，无需 ChatGPT 网页
codex-usage --account work    # 指定账号额度
codex-usage-widget            # 桌面悬浮显示当前账号额度
codex-account-manager         # 原生 Qt 软件选择账号或 API
```

命名账号的 `auth.json` 分开保存，对话目录、历史和索引与 `~/.codex` 共享。切换默认账号只影响新启动的 Codex，已经运行的终端不会自动切换或退出。

`setup-codex.sh` 会安装独立的 PyQt5 软件 **Codex Account Manager**，可从应用菜单或 `codex-account-manager` 打开。新增账号时可以不填邮箱，普通浏览器授权完成后软件会读取登录邮箱并自动命名；GUI 不要求开启设备代码授权。账号页还可选择、查看额度或可恢复移除命名账号，`unnamed` 主登录禁止删除。API 页可新增、编辑、测试、删除和选择 provider，密钥不会放进命令行参数。软件不使用网页或 8320 服务；关闭主窗口后驻留托盘，并提供当前账号额度悬浮窗。

桌面悬浮窗默认跟随 `codex-auth` 当前选择，也兼容显示为 `unnamed` 的旧版单账号登录。账号变化会触发自动刷新；自定义 API provider 模式没有账号额度，因此悬浮窗会明确显示 API 模式而不保留旧账号数据。

### 5 个核心 Skills
1. **agent-reach** - AI 代理增强
2. **brainstorming** - 头脑风暴
3. **domain-modeling** - 领域建模
4. **grilling** - 深度质询
5. **tdd** - 测试驱动开发

## 对比

| 命令 | Skills 数量 | 说明 |
|------|------------|------|
| `claude` | 43 (6+37 bundled) | 完整功能 |
| `claude-yolo` | **5** | 精简模式 |
| `codex` | **5** | 精简 Skills，保留审批 |
| `codex-yolo` | **5** | 精简模式 |

## 自定义

### 修改精简 Skills
编辑 `setup-codex.sh` 中的常量：
```bash
readonly YOLO_MINIMAL_SKILLS="agent-reach brainstorming domain-modeling grilling tdd"
```

然后重新运行安装脚本。该常量同时控制 `codex`、`codex-yolo`、`claude-yolo` 和 `claudex-yolo` 的 Skill 白名单。

### 临时修改配置
直接编辑生成的配置文件：
- `~/.claude/settings.yolo.json` (Claude)
- `~/.codex/config.toml` (Codex)
- `~/.codex/yolo.config.toml` (Codex)

## 新电脑部署

1. 复制 scripts 目录到新电脑
2. 运行安装脚本：
   ```bash
   cd /home/qwer/scripts/claude
   ./setup-codex.sh
   source ~/.bashrc
   ```
3. 验证配置：
   ```bash
   ./verify-yolo-skills.sh
   ```

## 故障排查

### Yolo 命令 skills 数量不对
```bash
# 重新生成配置
./setup-codex.sh

# 验证结果
./verify-yolo-skills.sh
```

### Aliases 不生效
```bash
source ~/.bashrc
alias | grep yolo
```

### 查看详细配置
```bash
# Claude yolo
cat ~/.claude/settings.yolo.json

# Codex yolo
grep -A 2 "enabled = true" ~/.codex/yolo.config.toml
```

## 相关文档

- `SKILLS_STATUS.md` - 详细的 skill 配置状态文档
- `ai-runtime-mcp-skill-comparison.html` - 运行时对比报告（如果存在）

## 版本信息

- Claude Code: 2.1.223
- Codex CLI: 0.146.0
- 最后更新: 2026-08-07

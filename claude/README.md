# Claude/Codex Skills 管理工具

管理 Claude 和 Codex 的 skill 配置，提供精简的 yolo 模式。

## 快速开始

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

## Yolo 命令

所有 yolo 命令统一使用 **5 个核心 skills**：

```bash
claude-yolo   # Claude + 精简 skills + 跳过权限
codex-yolo    # Codex + 精简 skills + 跳过审批
claudex-yolo  # Claudex + 精简 skills + yolo 模式
```

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
| `codex` | 8 配置 / 4 模型可见 | 默认配置 |
| `codex-yolo` | **5** | 精简模式 |

## 自定义

### 修改 yolo skills
编辑 `setup-codex.sh` 中的常量：
```bash
readonly YOLO_MINIMAL_SKILLS="agent-reach brainstorming domain-modeling grilling tdd"
```

然后重新运行安装脚本。

### 临时修改配置
直接编辑生成的配置文件：
- `~/.claude/settings.yolo.json` (Claude)
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

# Skill 配置现状总结

生成时间：2026-08-07

## 快速概览

| 命令 | Skills 数量 | 配置文件 | 说明 |
|------|------------|----------|------|
| `claude` | 6 用户 + 37 内置 | `~/.claude/settings.json` | 默认配置，包含大量 bundled skills |
| `claude-yolo` | **5 核心** | `~/.claude/settings.yolo.json` | ✨ 精简配置 |
| `codex` | 15 配置项 | `~/.codex/config.toml` | 默认配置（实际模型可见约 4-8 个） |
| `codex-yolo` | **9 配置项** | `~/.codex/yolo.config.toml` | ✨ 精简配置 |
| `claudex-yolo` | **9 配置项** | `~/.codex/yolo.config.toml` | 同 codex-yolo |

---

## 1. Claude 默认配置 (`claude`)

**配置文件**: `~/.claude/settings.json`

### 启用的用户 skills (on): 6 个
1. agent-reach
2. brainstorming
3. grill-me
4. grill-with-docs
5. handoff
6. tdd

### 名称可见 skills (name-only): 2 个
1. domain-modeling
2. grilling

### Bundled Skills: 37 个
Claude Code 官方内置，无法禁用。包括：
- 文件操作、搜索、编辑
- Git 操作
- 终端命令
- 工作流管理
- 等等...

### 统计
- **总计**: 6 + 37 = 43 个 skills 可用
- **禁用**: 31 个用户自定义 skills

---

## 2. Claude-yolo 配置 (`claude-yolo`)

**配置文件**: `~/.claude/settings.yolo.json`

### 启用的 skills: 5 个 ✨
1. **agent-reach** - AI 代理增强
2. **brainstorming** - 头脑风暴
3. **domain-modeling** - 领域建模
4. **grilling** - 深度质询
5. **tdd** - 测试驱动开发

### 特点
- ✅ 只加载 5 个核心 skills
- ✅ 不加载 bundled skills（使用 `skillOverrides` 机制）
- ✅ 跳过权限检查
- ✅ 保留内置命令（`/clear` 等）

---

## 3. Codex 默认配置 (`codex`)

**配置文件**: `~/.codex/config.toml`

### 启用的 skills (enabled = true): 8 个
1. agent-reach
2. brainstorming
3. domain-modeling
4. grilling
5. grill-me *(disable-model-invocation)*
6. grill-with-docs *(disable-model-invocation)*
7. handoff *(disable-model-invocation)*
8. tdd

### 统计
- **配置项总数**: 259 个
- **启用**: 15 个配置项（包括重复路径）
- **禁用**: 244 个配置项
- **实际模型可见**: 约 4 个（根据之前的 HTML 报告）

### 注意
- `grill-me`, `grill-with-docs`, `handoff` 设置了 `disable-model-invocation: true`，需要手动调用
- 实际送入模型的 skill 数量比配置少

---

## 4. Codex-yolo 配置 (`codex-yolo`)

**配置文件**: `~/.codex/yolo.config.toml`

### 启用的 skills: 5 个（去重后）✨
1. **agent-reach** - AI 代理增强
2. **brainstorming** - 头脑风暴
3. **domain-modeling** - 领域建模
4. **grilling** - 深度质询
5. **tdd** - 测试驱动开发

### 统计
- **配置项**: 9 个（因为 `~/.agents/skills` 和 `~/.claude/skills` 有重复）
- **实际 skills**: 5 个（去重后）

### 特点
- ✅ 只加载 5 个核心 skills
- ✅ 跳过审批和沙箱
- ✅ 从主配置中过滤生成，保留其他设置

---

## 5. Claudex-yolo 配置 (`claudex-yolo`)

使用与 `codex-yolo` 相同的配置文件：`~/.codex/yolo.config.toml`

通过环境变量指定：`CLAUDEX_CODEX_CONFIG=~/.codex/yolo.config.toml`

---

## 用户自定义 Skills 目录

### ~/.claude/skills/
**总数**: 40 个

包括：
- agent-reach, arxiv-search, auto-commit, brainstorm-ideas-new, brainstorming
- cmd-flow-viz, cmd-notes, codex-cli, codex-skill, desktop-app-design
- dev-plan, document-converter-suite, document-project, domain-modeling
- fastapi-full-stack, file-organizer, find-skills, grill-me, grill-with-docs
- grilling, handoff, hugging-face-datasets, json-canvas, kaggle
- knowledge-query, lit-review-assistant, my-skills, network-pipeline-walkthrough
- obsidian-bases, obsidian-markdown, paper-notes, project-audit
- save-knowledge, tailwindcss, tdd, ui-design-brain, ui-mockup
- url-summarize, web-scraping, xhs-topic-analysis

### ~/.agents/skills/
**总数**: 68 个（包含重复和额外的 Codex 专用 skills）

---

## Yolo 命令的 5 个核心 Skills

所有 yolo 命令（`claude-yolo`, `codex-yolo`, `claudex-yolo`）统一使用：

1. **agent-reach** - AI 代理增强
   - 跨 agent 协作
   - 任务分解
   
2. **brainstorming** - 头脑风暴
   - 创意生成
   - 方案探索

3. **domain-modeling** - 领域建模
   - 系统架构设计
   - 数据模型设计

4. **grilling** - 深度质询
   - 方案审查
   - 漏洞挖掘

5. **tdd** - 测试驱动开发
   - 测试编写
   - 测试优先开发

---

## Bashrc Aliases

```bash
# 在 ~/.bashrc 中自动配置
alias codex-yolo='codex --dangerously-bypass-approvals-and-sandbox -p yolo'
alias claude-yolo='claude --dangerously-skip-permissions --settings ~/.claude/settings.yolo.json'
alias claudex-yolo='CLAUDEX_YOLO=1 CLAUDEX_CODEX_CONFIG=~/.codex/yolo.config.toml claudex'
```

---

## 管理脚本

### 安装/更新配置
```bash
cd /home/qwer/scripts/claude
./setup-codex.sh
```

### 验证配置
```bash
/home/qwer/scripts/claude/verify-yolo-skills.sh
```

### 查看完整报告
```bash
/home/qwer/scripts/claude/skill-status-report.sh
```

---

## 设计理念

### 默认命令 (`claude`, `codex`)
- **目标**: 功能全面，满足各种使用场景
- **策略**: 保留大部分可用 skills
- **适用**: 日常开发、探索新功能

### Yolo 命令 (`*-yolo`)
- **目标**: 快速响应，减少 token 开销
- **策略**: 只保留最核心的 5 个 skills
- **适用**: 高频任务、快速迭代、token 预算有限

---

## 对比：Claude vs Codex

| 维度 | Claude | Codex |
|------|--------|-------|
| 默认 Skills | 43 个（6+37） | 8 个配置（4 个模型可见） |
| Yolo Skills | 5 个 | 5 个 |
| 配置复杂度 | 简单（JSON） | 复杂（TOML + 重复路径） |
| Bundled Skills | 37 个（无法完全禁用） | 0 个 |
| 模型可见性 | 高（大部分加载） | 低（精确控制） |

---

## 注意事项

1. **Bundled Skills**: Claude 的 37 个 bundled skills 是内置的，即使使用 `skillOverrides` 也无法完全移除它们的定义，但可以通过 yolo 配置避免将它们的完整内容加载到 system prompt。

2. **配置优先级**: 
   - `--settings` 参数 > `settings.json`
   - `-c` 参数 > `config.toml`

3. **路径重复**: Codex 会扫描多个目录，同一个 skill 可能有多个路径配置项。

4. **MCP 服务器**: 当前所有运行时的 MCP 都是 0（未启用任何 MCP 服务器）。

---

## 更新记录

- **2026-08-07**: 初始版本，统一 yolo 命令使用 5 个核心 skills

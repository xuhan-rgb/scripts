# Skills 配置最终方案

## 🎯 配置策略

### 默认 Claude 命令（无限制）
- `claude` - 使用所有已安装的 skills（40个用户 + 37个内置）

### Codex 与 Yolo 命令（精简配置）
- `codex` - 只使用 5 个核心 skills，保留审批
- `claude-yolo` - 只使用 5 个核心 skills
- `codex-yolo` - 只使用 5 个核心 skills  
- `claudex-yolo` - 只使用 5 个核心 skills
- **目的**: 快速响应，减少 token 开销

---

## ✅ 当前状态

### 1. Claude 默认配置
**文件**: `~/.claude/settings.json`
- ✅ 无 `skillOverrides` 限制
- ✅ 可使用所有 40 个用户自定义 skills
- ✅ 可使用所有 37 个 bundled skills
- **总计**: 77 个 skills 可用

### 2. Claude-yolo 配置
**文件**: `~/.claude/settings.yolo.json`
```json
{
  "skillOverrides": {
    "agent-reach": "on",
    "brainstorming": "on",
    "domain-modeling": "on",
    "grilling": "on",
    "tdd": "on"
  }
}
```
- **总计**: 5 个核心 skills

### 3. Codex 默认配置
**文件**: `~/.codex/config.toml`
- ✅ 与 `codex-yolo` 使用相同的 Skill 白名单
- ✅ 其他 Skills 和重复路径配置为 `enabled = false`
- **总计**: 5 个核心 skills

### 4. Codex-yolo 配置
**文件**: `~/.codex/yolo.config.toml`
- ✅ 只启用 5 个核心 skills
- 其他所有 skills 不包含在配置中
- **总计**: 5 个核心 skills

---

## 🔧 安装脚本修改

### 修改的函数

#### 1. `configure_claude_skill_overrides()`
**修改前**: 为所有用户 skills 设置 on/off/name-only  
**修改后**: 不设置 skillOverrides，保持默认（无限制）

#### 2. `configure_codex_skills()`
**修改后**: 使用与 `codex-yolo` 相同的 `YOLO_MINIMAL_SKILLS` 白名单

#### 3. 新增函数
- `create_claude_yolo_settings()` - 创建 Claude yolo 精简配置
- `create_codex_yolo_config()` - 创建 Codex yolo 精简配置

---

## 📋 5 个核心 Skills

`codex` 与所有 yolo 命令统一使用：

1. **agent-reach** - AI 代理增强
   - 跨 agent 协作
   - 任务分解与并行处理

2. **brainstorming** - 头脑风暴
   - 创意生成
   - 方案探索

3. **domain-modeling** - 领域建模
   - 系统架构设计
   - 数据模型设计

4. **grilling** - 深度质询
   - 方案审查
   - 漏洞挖掘
   - 边界条件测试

5. **tdd** - 测试驱动开发
   - 测试编写
   - 测试优先开发流程

---

## 📦 新电脑部署

```bash
# 1. 运行安装脚本
cd /home/qwer/scripts/claude
./setup-codex.sh

# 2. 重新加载 bashrc
source ~/.bashrc

# 3. 验证配置
./final-verification.sh
```

### 预期结果
- `~/.claude/settings.json` - 不含 skillOverrides（或为空对象）
- `~/.claude/settings.yolo.json` - 包含 5 个 skills
- `~/.codex/config.toml` - 只启用 5 个核心 skills
- `~/.codex/yolo.config.toml` - 只包含 5 个 skills
- `~/.bashrc` - 包含 3 个 yolo aliases

---

## 🔍 验证工具

### 快速验证
```bash
/home/qwer/scripts/claude/final-verification.sh
```

### 详细报告
```bash
/home/qwer/scripts/claude/skill-status-report.sh
```

### Yolo 配置验证
```bash
/home/qwer/scripts/claude/verify-yolo-skills.sh
```

---

## 📊 对比表

| 命令 | Skills 数量 | MCP | 权限 | 适用场景 |
|------|------------|-----|------|---------|
| `claude` | 77 (40+37) | 所有 | 需确认 | 全功能开发 |
| `claude-yolo` | 5 | 所有 | 跳过 | 快速迭代 |
| `codex` | 5 | 所有 | 需确认 | 日常开发 |
| `codex-yolo` | 5 | 所有 | 跳过 | 快速迭代 |
| `claudex-yolo` | 5 | 所有 | 跳过 | 快速迭代 |

---

## 💡 设计理念

### 默认命令的哲学
- `claude` 保留可用 Skills
- `codex` 聚焦 5 个核心 Skills，同时保留审批

### Yolo 命令的哲学
- **精简**: 只保留最核心、最常用的功能
- **高效**: 减少 token 开销，加快响应速度
- **聚焦**: 专注于编码核心任务（建模、测试、质询、协作）

### 平衡点
- 通过**两套配置**实现灵活切换
- 用户根据**任务性质**选择合适的工具
- 默认安全（有权限检查），yolo 快速（跳过检查）

---

## 🗂️ 文件清单

### 脚本文件
- `setup-codex.sh` (21K) - 主安装脚本
- `verify-yolo-skills.sh` (1.5K) - Yolo 配置验证
- `skill-status-report.sh` (6.9K) - 完整状态报告
- `final-verification.sh` - 最终配置验证

### 文档文件
- `README.md` (2.8K) - 快速开始指南
- `SKILLS_STATUS.md` (6.4K) - 详细配置文档
- `FINAL_CONFIGURATION.md` (本文件) - 最终方案说明

### 配置文件
- `~/.claude/settings.json` - Claude 默认（无限制）
- `~/.claude/settings.yolo.json` - Claude yolo（5个）
- `~/.codex/config.toml` - Codex 默认（5个）
- `~/.codex/yolo.config.toml` - Codex yolo（5个）

---

## ⚠️ 注意事项

1. **MCP 服务器**: 当前配置了 1 个 MCP server，默认和 yolo 命令都可以使用

2. **Bundled Skills**: Claude 的 37 个内置 skills 无法完全移除，但可以通过配置减少其在 system prompt 中的内容

3. **Token 开销**: 
   - `claude`: ~高（77个 skills 描述）
   - `claude-yolo`: ~低（5个 skills 描述）
   - 差异可能达到数千 tokens

4. **Codex 权限差异**: `codex` 与 `codex-yolo` 的 Skill 集合相同，前者保留审批，后者跳过审批与沙箱

---

## 🔄 更新历史

- **2026-08-07 初版**: 创建 yolo 精简配置
- **2026-08-07 v2**: 修改安装脚本，默认命令不限制 skills
- **2026-08-07 v3**: 添加 brainstorming 到核心 skills（共5个）
- **2026-08-13 v4**: 普通 `codex` 使用与 `codex-yolo` 相同的 5 个核心 Skills

---

## 📞 使用建议

### 日常开发
```bash
claude
```
- 功能全面
- 可以探索各种 skills
- 适合复杂任务

普通 `codex` 仍保留审批，但使用与 `codex-yolo` 相同的 5 个核心 Skills。

### 快速迭代
```bash
claude-yolo  # 或 codex-yolo
```
- 响应快速
- Token 开销小
- 适合高频编码任务

### 选择原则
- 不确定需要什么 → 用默认命令
- 明确只需要核心功能 → 用 yolo 命令
- Token 预算有限 → 用 yolo 命令
- 需要跳过权限检查 → 用 yolo 命令

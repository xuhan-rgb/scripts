#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "    Skill 配置现状全面报告"
echo "=========================================="
echo

# 1. Claude 默认配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Claude 默认配置 (claude)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -f ~/.claude/settings.json ]]; then
  echo "配置文件: ~/.claude/settings.json"
  echo
  echo "启用的用户 skills (on):"
  jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.json | sort | nl
  echo
  echo "名称可见 skills (name-only):"
  jq -r '.skillOverrides | to_entries[] | select(.value == "name-only") | .key' ~/.claude/settings.json | sort | nl
  echo
  echo "统计:"
  echo "  - 启用 (on): $(jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.json | wc -l) 个"
  echo "  - 名称可见 (name-only): $(jq -r '.skillOverrides | to_entries[] | select(.value == "name-only") | .key' ~/.claude/settings.json | wc -l) 个"
  echo "  - 禁用 (off): $(jq -r '.skillOverrides | to_entries[] | select(.value == "off") | .key' ~/.claude/settings.json | wc -l) 个"
  echo "  - Bundled skills: 37 个 (Claude Code 内置)"
else
  echo "❌ 配置文件不存在"
fi
echo

# 2. Claude-yolo 配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Claude-yolo 配置 (claude-yolo)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -f ~/.claude/settings.yolo.json ]]; then
  echo "配置文件: ~/.claude/settings.yolo.json"
  echo
  echo "启用的 skills:"
  jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.yolo.json | sort | nl
  echo
  echo "统计: $(jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.yolo.json | wc -l) 个 skills"
  echo "Bundled skills: 不启用"
else
  echo "❌ 配置文件不存在"
fi
echo

# 3. Codex 默认配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Codex 默认配置 (codex)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -f ~/.codex/config.toml ]]; then
  echo "配置文件: ~/.codex/config.toml"
  echo
  echo "启用的 skills (enabled = true):"
  grep -B 1 "enabled = true" ~/.codex/config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|\1|' | sort -u | nl
  echo
  echo "统计:"
  echo "  - 启用: $(grep "enabled = true" ~/.codex/config.toml | wc -l) 个配置项"
  echo "  - 禁用: $(grep "enabled = false" ~/.codex/config.toml | wc -l) 个配置项"
  echo
  echo "注: 实际模型可见 skills 可能更少 (如 name-only 或 disable-model-invocation)"
else
  echo "❌ 配置文件不存在"
fi
echo

# 4. Codex-yolo 配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Codex-yolo 配置 (codex-yolo)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -f ~/.codex/yolo.config.toml ]]; then
  echo "配置文件: ~/.codex/yolo.config.toml"
  echo
  echo "启用的 skills (enabled = true):"
  grep -B 1 "enabled = true" ~/.codex/yolo.config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|\1|' | sort -u | nl
  echo
  echo "统计: $(grep -B 1 "enabled = true" ~/.codex/yolo.config.toml | grep "path =" | wc -l) 个配置项"
else
  echo "❌ 配置文件不存在"
fi
echo

# 5. 用户自定义 skills 目录
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. 用户自定义 Skills 目录"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Claude skills 目录: ~/.claude/skills/"
if [[ -d ~/.claude/skills ]]; then
  echo "总数: $(find ~/.claude/skills -name "SKILL.md" | wc -l) 个"
  echo
  find ~/.claude/skills -name "SKILL.md" -exec dirname {} \; | xargs -I {} basename {} | sort | nl
else
  echo "❌ 目录不存在"
fi
echo

echo "Codex skills 目录: ~/.agents/skills/"
if [[ -d ~/.agents/skills ]]; then
  echo "总数: $(find ~/.agents/skills -name "SKILL.md" | wc -l) 个"
else
  echo "❌ 目录不存在"
fi
echo

# 6. Bashrc aliases
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Bashrc Aliases"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
awk '/# >>> scripts AI yolo aliases >>>/{flag=1;next}/# <<< scripts AI yolo aliases <<</{flag=0}flag' ~/.bashrc | grep "^alias"
echo

# 7. 对比总结
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. 对比总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "命令             | Skills 数量 | 配置文件"
echo "----------------|------------|----------"
printf "%-15s | %-10s | %s\n" "claude" "$(jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.json 2>/dev/null | wc -l)+37 bundled" "~/.claude/settings.json"
printf "%-15s | %-10s | %s\n" "claude-yolo" "$(jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.yolo.json 2>/dev/null | wc -l)" "~/.claude/settings.yolo.json"
printf "%-15s | %-10s | %s\n" "codex" "$(grep -c "enabled = true" ~/.codex/config.toml 2>/dev/null || echo 0)" "~/.codex/config.toml"
printf "%-15s | %-10s | %s\n" "codex-yolo" "$(grep -c "enabled = true" ~/.codex/yolo.config.toml 2>/dev/null || echo 0)" "~/.codex/yolo.config.toml"
printf "%-15s | %-10s | %s\n" "claudex-yolo" "$(grep -c "enabled = true" ~/.codex/yolo.config.toml 2>/dev/null || echo 0)" "~/.codex/yolo.config.toml"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Yolo 命令统一的 5 个核心 skills:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. agent-reach      - AI 代理增强"
echo "  2. brainstorming    - 头脑风暴"
echo "  3. domain-modeling  - 领域建模"
echo "  4. grilling         - 深度质询"
echo "  5. tdd              - 测试驱动开发"
echo "=========================================="

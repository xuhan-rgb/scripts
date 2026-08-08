#!/usr/bin/env bash
# 验证所有 yolo 命令使用的 skills

set -euo pipefail

echo "======================================"
echo "  Yolo Commands Skill Verification"
echo "======================================"
echo

# 1. Claude yolo skills
echo "=== Claude-yolo skills ==="
if [[ -f ~/.claude/settings.yolo.json ]]; then
  jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.yolo.json | sort
  echo
  echo "Total: $(jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.yolo.json | wc -l) skills"
else
  echo "❌ File not found: ~/.claude/settings.yolo.json"
fi
echo

# 2. Codex yolo skills
echo "=== Codex-yolo skills ==="
if [[ -f ~/.codex/yolo.config.toml ]]; then
  grep -B 1 "enabled = true" ~/.codex/yolo.config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|\1|' | sort -u
  echo
  echo "Total: $(grep -B 1 "enabled = true" ~/.codex/yolo.config.toml | grep "path =" | wc -l) skills"
else
  echo "❌ File not found: ~/.codex/yolo.config.toml"
fi
echo

# 3. 检查 bashrc aliases
echo "=== Bashrc aliases ==="
awk '/# >>> scripts AI yolo aliases >>>/{flag=1;next}/# <<< scripts AI yolo aliases <<</{flag=0}flag' ~/.bashrc | grep "^alias.*-yolo="
echo

echo "======================================"
echo "Expected skills for all yolo commands:"
echo "  1. agent-reach"
echo "  2. brainstorming"
echo "  3. domain-modeling"
echo "  4. grilling"
echo "  5. tdd"
echo "======================================"

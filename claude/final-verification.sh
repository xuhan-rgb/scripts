#!/usr/bin/env bash

echo "=========================================="
echo "  最终配置验证"
echo "=========================================="
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 默认命令配置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Claude 默认
echo "【claude】"
if [[ -f ~/.claude/settings.json ]]; then
  has_overrides=$(jq 'has("skillOverrides")' ~/.claude/settings.json 2>/dev/null)
  if [[ "$has_overrides" == "true" ]]; then
    override_count=$(jq '.skillOverrides | length' ~/.claude/settings.json 2>/dev/null)
    if [[ "$override_count" -gt 0 ]]; then
      echo "  ⚠️  有 skillOverrides 配置 ($override_count 个)"
    else
      echo "  ✅ skillOverrides 为空，使用所有 skills"
    fi
  else
    echo "  ✅ 无 skillOverrides，使用所有 skills"
  fi
  echo "  可用: 所有已安装用户 skills (40个) + 37 个 bundled skills"
else
  echo "  ⚠️  配置文件不存在"
fi
echo

# Codex 默认
echo "【codex】"
if [[ -f ~/.codex/config.toml ]]; then
  skill_count=$(grep -B 1 "enabled = true" ~/.codex/config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|\1|' | sort -u | wc -l)
  echo "  ✅ 精简配置: $skill_count 个核心 skills（保留审批）"
  grep -B 1 "enabled = true" ~/.codex/config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|     - \1|' | sort -u
else
  echo "  ⚠️  配置文件不存在"
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Yolo 命令配置 (精简限制)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Claude yolo
echo "【claude-yolo】"
if [[ -f ~/.claude/settings.yolo.json ]]; then
  skill_count=$(jq -r '.skillOverrides | to_entries[] | select(.value == "on") | .key' ~/.claude/settings.yolo.json | wc -l)
  echo "  ✅ 精简配置: $skill_count 个核心 skills"
  jq -r '.skillOverrides | to_entries[] | select(.value == "on") | "     - " + .key' ~/.claude/settings.yolo.json
else
  echo "  ❌ 配置文件不存在"
fi
echo

# Codex yolo
echo "【codex-yolo / claudex-yolo】"
if [[ -f ~/.codex/yolo.config.toml ]]; then
  skill_count=$(grep -B 1 "enabled = true" ~/.codex/yolo.config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|\1|' | sort -u | wc -l)
  echo "  ✅ 精简配置: $skill_count 个核心 skills"
  grep -B 1 "enabled = true" ~/.codex/yolo.config.toml | grep "path =" | sed 's|.*/\([^/]*\)/SKILL.md.*|     - \1|' | sort -u
else
  echo "  ❌ 配置文件不存在"
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "✅ 默认 Claude 命令:"
echo "   - 无 skill 限制"
echo "   - 可使用所有已安装的 skills"
echo
echo "✅ Codex 命令 (codex, codex-yolo):"
echo "   - 使用相同的 5 个核心 skills"
echo "   - codex 保留审批；codex-yolo 跳过审批与沙箱"
echo
echo "✅ 其他 Yolo 命令 (claude-yolo, claudex-yolo):"
echo "   - 只使用 5 个核心 skills"
echo "   - 精简高效，减少 token 开销"
echo "   - 适合快速迭代和高频任务"
echo
echo "=========================================="

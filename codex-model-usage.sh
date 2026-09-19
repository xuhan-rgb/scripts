#!/usr/bin/env bash
set -euo pipefail

# Summarize local token usage by model, not actual subscription quota charges.
# Additional arguments are forwarded to ccusage, e.g. --since 2026-09-01.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: codex-model-usage.sh [--last-session] [--full-text] [--since YYYY-MM-DD] [--until YYYY-MM-DD]

按模型汇总本地 Codex token 用量，输出表格；不代表实际套餐扣除额度。
默认统计本机日期的今天，按总 token 从高到低排序。
传入 --since 或 --until 时使用指定范围，其他参数直接传给 ccusage codex daily。
--last-session：查询最后有活动的会话，统计整个会话已记录的累计用量。
  不加今天的日期限制；会话可能仍在进行中，不合并独立的子代理会话。
  表格后显示最近一轮已有最终回答的问答预览（问题 300、回答 400 字符）。
--full-text：启用最近会话模式，并展开该轮问答原文；不显示工具或中间消息。
依赖：ccusage、jq。
字段：input=非缓存输入，cached=缓存输入，output=输出，total=总 token。
Cache Hit = cached / (input + cached)，无输入时显示 N/A。
EOF
  exit 0
fi

has_date_filter=false
last_session=false
full_text=false
args=()
for arg in "$@"; do
  case "$arg" in
    --last-session) last_session=true; continue ;;
    --full-text) last_session=true; full_text=true; continue ;;
    --since|--since=*|--until|--until=*|-s|-s?*|-u|-u?*) has_date_filter=true ;;
  esac
  args+=("$arg")
done
set -- "${args[@]}"
if [[ "$has_date_filter" == false && "$last_session" == false ]]; then
  today=$(date +%F)
  set -- --since "$today" --until "$today" "$@"
fi

if [[ "$last_session" == true ]]; then
  report=$(ccusage codex session --offline "$@" --json)
  session=$(jq '.sessions | max_by(.lastActivity)' <<< "$report")
  jq -r 'if . == null then "没有找到会话记录。" else
    "Session: \(.sessionId)\nLast activity: \(.lastActivity)\n用量范围：整个会话累计" end' <<< "$session"
  report=$(jq '{daily: (if . == null then [] else [.] end)}' <<< "$session")
else
  report=$(ccusage codex daily --offline "$@" --json)
fi
jq -r '
  [.daily[].models | to_entries[]]
  | group_by(.key)
  | map({
      model: .[0].key,
      input: (map(.value.inputTokens) | add),
      cached: (map(.value.cachedInputTokens) | add),
      output: (map(.value.outputTokens) | add),
      total: (map(.value.totalTokens) | add)
    })
  | sort_by(-.total)
  | .[]
  | (if (.input + .cached) == 0 then "N/A"
     else (((.cached / (.input + .cached) * 10000 | round) / 100) | tostring) + "%"
     end) as $hit
  | [.model, .input, .cached, .output, .total, $hit] | @tsv
' <<< "$report" | {
  printf '%-28s %18s %18s %18s %18s %10s\n' 'Model' 'Input' 'Cached' 'Output' 'Total' 'Cache Hit'
  printf '%-28s %18s %18s %18s %18s %10s\n' '----------------------------' '------------------' '------------------' '------------------' '------------------' '----------'
  while IFS=$'\t' read -r model input cached output total hit; do
    printf '%-28s %18s %18s %18s %18s %10s\n' "$model" "$input" "$cached" "$output" "$total" "$hit"
  done
}

if [[ "$last_session" == true && "$session" != null ]]; then
  session_id=$(jq -r '.sessionId' <<< "$session")
  session_log="${CODEX_HOME:-$HOME/.codex}/sessions/$session_id.jsonl"
  if [[ ! -r "$session_log" ]]; then
    printf '\n无法读取会话原文：%s\n' "$session_log" >&2
    exit 1
  fi
  jq -nr --argjson full "$full_text" '
    def message_text: [.content[]? | select(.type == "input_text" or .type == "output_text") | .text] | join("\n");
    def preview($limit):
      if $full then . else gsub("\\s+"; " ")
      | if length > $limit then .[:$limit] + "… [已截断，使用 --full-text 展开]" else . end end;
    reduce inputs as $event ({question: null, pair: null};
      $event.payload as $p
      | if $event.type == "event_msg" and $p.type == "task_started" then .question = null
        elif $event.type == "response_item" and $p.type == "message" and $p.role == "user" then
          .question = ($p | message_text)
        elif $event.type == "response_item" and $p.type == "message" and $p.role == "assistant"
          and ($p.phase == "final_answer" or $p.phase == "final") and .question != null then
          .pair = {question: .question, answer: ($p | message_text)} | .question = null
        elif $event.type == "event_msg" and $p.type == "task_complete"
          and ($p.last_agent_message // "") != "" and .question != null then
          .pair = {question: .question, answer: $p.last_agent_message} | .question = null
        else . end)
    | if .pair == null then "\n暂无可配对的最终问答。"
      else "\n最近一轮已有最终回答的问答（表格为整个会话用量）：",
        ("你问：" + (.pair.question | preview(300))),
        ("模型答：" + (.pair.answer | preview(400))) end
  ' "$session_log"
fi

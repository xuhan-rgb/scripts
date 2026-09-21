#!/usr/bin/env python3
"""统计最近有活动的 Codex 会话中，最新一轮问答的 token 用量。"""

import argparse
import json
import os
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path


FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")


def call_action(call):
    name = call.get("name") or call["type"]
    if name == "exec":
        match = re.search(r"\btools\.(\w+)\s*\(", call.get("input", ""))
        if match:
            name = match.group(1)
    return {"exec_command": "执行本地命令", "apply_patch": "修改文件",
            "view_image": "查看图片", "web__run": "查询网页",
            "web_search_call": "搜索网页", "write_stdin": "继续终端任务"}.get(name, "调用 " + name)


def summarize(path):
    models = {}
    records = []
    calls = []
    pending_actions = defaultdict(list)
    seen_action_calls = set()
    seen_action_requests = set()
    latest_turn = None
    question = "（未找到用户问题）"
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # 正在写入的最后一行可能尚未完整。
                if not line.endswith("\n"):
                    break
                raise ValueError(f"日志第 {number} 行不是有效 JSON") from None
            payload = event.get("payload", {})
            if event.get("type") == "turn_context":
                latest_turn = payload.get("turn_id")
                models[payload.get("turn_id")] = payload.get("model") or "unknown"
            elif event.get("type") == "token_usage_record":
                identity = payload.get("response_id")
                if identity and identity in seen_action_requests:
                    continue
                if identity:
                    seen_action_requests.add(identity)
                turn = payload.get("turn_id")
                actions = list(dict.fromkeys(pending_actions.pop(turn, [])))
                records.append((payload, models.get(turn, "unknown"), " / ".join(actions) or "模型处理 / 回复"))
            elif event.get("type") == "response_item" and payload.get("type") in (
                    "function_call", "custom_tool_call", "web_search_call", "local_shell_call"):
                turn = payload.get("internal_chat_message_metadata_passthrough", {}).get("turn_id") or latest_turn
                calls.append((turn, payload))
                identity = payload.get("call_id") or payload.get("id")
                if not identity or identity not in seen_action_calls:
                    pending_actions[turn].append(call_action(payload))
                if identity:
                    seen_action_calls.add(identity)
            elif event.get("type") == "response_item" and payload.get("role") == "user":
                latest_turn = payload.get("internal_chat_message_metadata_passthrough", {}).get("turn_id")
                texts = [part.get("text", "") for part in payload.get("content", [])
                         if part.get("type") == "input_text"]
                question = "\n".join(texts).strip() or "（图片或其他非文本消息）"
            elif event.get("type") == "event_msg" and payload.get("type") == "user_message":
                latest_turn = payload.get("turn_id")
                question = payload.get("message") or "（图片或其他非文本消息）"

    totals = defaultdict(lambda: dict.fromkeys(FIELDS, 0))
    seen = set()
    requests = []
    for record, model, action in records:
        if latest_turn is None or record.get("turn_id") != latest_turn:
            continue
        response_id = record.get("response_id")
        if response_id and response_id in seen:
            continue
        if response_id:
            seen.add(response_id)
        usage = record["usage"]
        requests.append({"response_id": response_id, "model": model, "action": action,
                         "usage": {field: usage.get(field, 0) for field in FIELDS}})
        for field in FIELDS:
            # 只累计逐请求 usage，不能累加 turn/thread 的累计值。
            totals[model][field] += usage.get(field, 0)
    tools = []
    seen_calls = set()
    for turn, call in calls:
        if latest_turn is None or turn != latest_turn:
            continue
        identity = call.get("call_id") or call.get("id")
        if identity and identity in seen_calls:
            continue
        if identity:
            seen_calls.add(identity)
        arguments = call.get("arguments", call.get("input", call.get("action", "")))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        content = (json.dumps(arguments, ensure_ascii=False, indent=2)
                   if not isinstance(arguments, str) else arguments)
        tools.append({"id": identity, "name": call.get("name") or call["type"], "content": content})
    return totals, question, tools, requests


def display_width(text):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in "WF" else 1
               for char in text)


def print_question(question):
    width = max(20, shutil.get_terminal_size((88, 24)).columns - 4)
    print("最近问题：")
    for paragraph in question.expandtabs(4).splitlines():
        line = ""
        for char in paragraph:
            if unicodedata.category(char) == "Cc":
                continue
            if display_width(line + char) > width:
                print("  " + line)
                line = ""
            line += char
        print("  " + line)


def print_table(rows):
    widths = [max(display_width(row[i]) for row in rows) for i in range(len(rows[0]))]

    def border(left, middle, right):
        print(left + middle.join("─" * (width + 2) for width in widths) + right)

    border("┌", "┬", "┐")
    for index, row in enumerate(rows):
        cells = []
        for column, cell in enumerate(row):
            padding = " " * (widths[column] - display_width(cell))
            cells.append(" " + (cell + padding if column == 0 else padding + cell) + " ")
        print("│" + "│".join(cells) + "│")
        if index == 0 or index == len(rows) - 2:
            border("├", "┼", "┤")
    border("└", "┴", "┘")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="可选：指定会话 JSONL 文件")
    parser.add_argument("--json", action="store_true", help="输出供程序读取的 JSON")
    args = parser.parse_args()
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"
    try:
        path = args.file
        if path is None:
            path = max(root.rglob("*.jsonl"), key=lambda item: item.stat().st_mtime, default=None)
        if path is None:
            parser.error(f"没有找到会话日志：{root}")
        totals, question, tool_calls, requests = summarize(path.expanduser())
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps({"path": str(path), "question": question, "models": totals,
                          "tools": tool_calls, "requests": requests}, ensure_ascii=False))
        return

    print("\nCodex 最近一轮 Token 统计\n")
    print_question(question)
    print(f"\n本轮工具调用：{len(tool_calls)} 次（按日志调用记录去重）")
    print("\n用量范围：上方问题这一轮的已记录请求，不含此前问答或子代理独立日志。\n")
    if not totals:
        print("没有找到 token_usage_record；可能尚未记录用量，或日志版本不支持。")
        return
    def hit(usage):
        return f'{usage["cached_input_tokens"] / usage["input_tokens"]:.1%}' if usage["input_tokens"] else "N/A"

    rows = [["模型", "输入", "其中缓存输入", "输出", "总 token", "Cache hit"]]
    for model, usage in sorted(totals.items(), key=lambda item: item[1]["total_tokens"], reverse=True):
        rows.append([model] + [f"{usage[field]:,}" for field in FIELDS] + [hit(usage)])
    total = {field: sum(usage[field] for usage in totals.values()) for field in FIELDS}
    rows.append(["合计"] + [f"{total[field]:,}" for field in FIELDS] + [hit(total)])
    print_table(rows)
    print(f"\n本轮模型请求：{len(requests)} 次（按发生顺序）")
    rows = [["请求 / 模型", "输入", "其中缓存输入", "输出", "总 token", "Cache hit"]]
    for index, request in enumerate(requests, 1):
        usage = request["usage"]
        rows.append([f'{index} / {request["model"]}'] + [f"{usage[field]:,}" for field in FIELDS] + [hit(usage)])
    print_table(rows)
    print("\n缓存输入包含在输入中；总量 = 输入 + 输出。用量不代表套餐额度或费用。")
    print(f"日志：{path}\n")


if __name__ == "__main__":
    main()

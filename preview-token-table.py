#!/usr/bin/env python3
"""只读本地 Codex 日志的独立执行过程预览，不依赖或修改现有统计实现。"""

import json
import os
import re
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QProcess, QTimer
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)


CALL_TYPES = {"function_call", "custom_tool_call", "web_search_call", "local_shell_call"}


def call_title(call):
    name = call.get("name") or call["type"]
    if name == "exec":
        match = re.search(r"\btools\.(\w+)\s*\(", call.get("input", ""))
        if match:
            name = match.group(1)
    return {"exec_command": "执行本地命令", "apply_patch": "修改文件",
            "view_image": "查看图片", "web__run": "查询网页",
            "web_search_call": "搜索网页", "write_stdin": "读取或继续终端任务"}.get(name, "调用 " + name)


def read_report(path=None):
    if path is None:
        root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"
        path = max(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, default=None)
    if path is None:
        raise ValueError("没有找到会话日志")
    events = []
    turn = None
    question = "（未找到用户问题）"
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), 1):
        try:
            event = json.loads(line)
        except ValueError:
            if not line.endswith("\n"):
                break
            raise ValueError(f"日志第 {number} 行无效") from None
        payload = event.get("payload", {})
        kind = event.get("type")
        meta = payload.get("internal_chat_message_metadata_passthrough", {})
        if kind == "turn_context":
            turn = payload.get("turn_id")
        elif kind == "response_item" and payload.get("role") == "user":
            turn = meta.get("turn_id")
            question = "\n".join(part.get("text", "") for part in payload.get("content", [])
                                 if part.get("type") == "input_text") or "（非文本问题）"
        elif kind == "event_msg" and payload.get("type") == "user_message":
            turn = payload.get("turn_id")
            question = payload.get("message") or "（非文本问题）"
        event_turn = payload.get("turn_id") or meta.get("turn_id") or turn
        events.append((event_turn, kind, payload))

    steps, pending, messages = [], [], []
    outputs, seen_calls, seen_requests = {}, set(), set()
    model = "unknown"
    for event_turn, kind, payload in events:
        if turn is None or event_turn != turn:
            continue
        if kind == "turn_context":
            model = payload.get("model", "unknown")
        elif kind == "response_item" and payload.get("type") in CALL_TYPES:
            identity = payload.get("call_id") or payload.get("id")
            if identity and identity in seen_calls:
                continue
            seen_calls.add(identity)
            pending.append(payload)
        elif kind == "response_item" and payload.get("type") in ("function_call_output", "custom_tool_call_output"):
            outputs[payload.get("call_id")] = payload.get("output", "")
        elif kind == "response_item" and payload.get("role") == "assistant":
            messages.append(payload)
        elif kind == "token_usage_record":
            identity = payload.get("response_id")
            if identity and identity in seen_requests:
                continue
            seen_requests.add(identity)
            usage = payload["usage"]
            steps.append({"id": identity or str(len(steps)), "calls": pending, "messages": messages,
                          "model": model, "usage": usage})
            pending, messages = [], []
    if pending:
        steps.append({"id": "pending", "calls": pending, "messages": messages, "model": model, "usage": None})
    for step in steps:
        calls = step.pop("calls")
        messages = step.pop("messages")
        titles = list(dict.fromkeys(call_title(call) for call in calls))
        step["title"] = " / ".join(titles) if titles else (
            "回复最终结果" if any(m.get("phase") == "final_answer" for m in messages) else "模型继续处理或回复")
        returned = sum(call.get("call_id") in outputs for call in calls)
        step["result"] = (f"{len(calls)} 次工具调用 · 已收到 {returned} 次返回"
                          if calls else "本次没有记录到工具调用")
        details = [f"模型：{step['model']}"]
        for call in calls:
            args = call.get("arguments", call.get("input", call.get("action", "")))
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False, indent=2)
            details.append(f"工具：{call.get('name') or call['type']}\n{args}")
            if call.get("call_id") in outputs:
                output = outputs[call["call_id"]]
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False, indent=2)
                details.append("返回内容：\n" + output[:12000] + ("\n（返回内容仅预览前 12000 字符）" if len(output) > 12000 else ""))
        for message in messages:
            details.extend(part["text"] for part in message.get("content", []) if "text" in part)
        step["details"] = "\n\n".join(details)
    return {"path": str(path), "turn": turn, "question": question, "steps": steps,
            "tool_count": len(seen_calls)}



class Preview(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("本轮用量表格预览 · 真实日志 · 只读")
        self.resize(1080, 650)
        self.setMinimumWidth(650)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        title = QLabel("本轮用量 · 操作与 Token 对照")
        title.setObjectName("heading")
        layout.addWidget(title)
        note = QLabel("真实日志 · 每 2 秒刷新 · 独立只读预览")
        note.setObjectName("muted")
        layout.addWidget(note)
        question = QLabel("正在读取最近会话…")
        question.setTextFormat(Qt.PlainText)
        self.question = question
        question.setObjectName("question")
        question.setWordWrap(True)
        layout.addWidget(question)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["请求 / 模型", "本次操作", "输入", "其中缓存", "输出", "总 Token", "Cache hit", "详情"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFixedHeight(180)
        self.details.hide()
        self.expanded_id = None
        layout.addWidget(self.details)
        self.summary = QLabel("等待用量记录")
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        footer = QLabel("Token 来自逐请求记录；工具与请求按日志顺序关联。\n"
                        "输入含上下文；Cache hit = 缓存输入 ÷ 输入。仅当前轮，不含子代理独立日志。")
        footer.setObjectName("muted")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.setStyleSheet("""
            QWidget { background: #111827; color: #e5e7eb; font: 13px 'Sans Serif'; }
            QLabel#heading { font-size: 22px; font-weight: bold; }
            QLabel#muted { color: #94a3b8; font-size: 11px; }
            QLabel#question { background: #1e293b; padding: 12px; border-radius: 8px; }
            QTableWidget { background: #182234; gridline-color: #334155; selection-background-color: #26364d; }
            QHeaderView::section { background: #1e293b; color: #bfdbfe; padding: 8px; border: 1px solid #334155; }
            QLabel#summary { color: #93c5fd; padding: 12px; background: #1e293b; border-radius: 8px; }
            QPushButton { color: #93c5fd; background: transparent; border: none; padding: 5px; }
            QPushButton:hover { color: #dbeafe; background: #26364d; border-radius: 4px; }
            QPlainTextEdit { color: #cbd5e1; background: #0f172a; border: 1px solid #334155; }
        """)
        self.report = None
        self.reader = QProcess(self)
        self.reader.finished.connect(self.loaded)
        self.reader.errorOccurred.connect(lambda _: self.summary.setText("读取进程启动失败：" + self.reader.errorString()))
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        QTimer.singleShot(0, self.refresh)

    def refresh(self):
        if self.reader.state() == QProcess.NotRunning:
            self.reader.start(sys.executable, [str(Path(__file__).resolve()), "--report"])

    def loaded(self, code, status):
        data = bytes(self.reader.readAllStandardOutput()).decode("utf-8")
        error = bytes(self.reader.readAllStandardError()).decode("utf-8")
        if code or status != QProcess.NormalExit:
            self.summary.setText("读取失败：" + error.strip())
            return
        try:
            self.render(json.loads(data))
        except (ValueError, KeyError, TypeError) as error:
            self.summary.setText("日志解析失败：" + str(error))

    def render(self, report):
        if report == self.report:
            return
        same_turn = self.report and (report["path"], report["turn"]) == (self.report["path"], self.report["turn"])
        if not same_turn:
            self.expanded_id = None
        scroll = self.table.verticalScrollBar().value() if same_turn else 0
        self.table.setRowCount(len(report["steps"]))
        self.question.setText("你的问题\n" + report["question"])
        self.question.setToolTip(report["path"])
        for index, step in enumerate(report["steps"], 1):
            usage = step["usage"]
            values = [f"#{index} {step['model']}", step["title"]]
            if usage is None:
                values += ["等待记录"] + ["—"] * 4
            else:
                values += [f"{usage.get(field, 0):,}" for field in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")]
                values.append(f"{usage.get('cached_input_tokens', 0) / usage['input_tokens']:.1%}" if usage.get("input_tokens") else "N/A")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(step["title"] + "\n" + step["result"])
                item.setTextAlignment((Qt.AlignRight if column >= 2 else Qt.AlignLeft) | Qt.AlignVCenter)
                self.table.setItem(index - 1, column, item)
            button = QPushButton("详情")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, identity=step["id"]: self.toggle_details(identity))
            self.table.setCellWidget(index - 1, 7, button)
        usage = [step["usage"] for step in report["steps"] if step["usage"] is not None]
        inputs = sum(item.get("input_tokens", 0) for item in usage)
        cached = sum(item.get("cached_input_tokens", 0) for item in usage)
        total = sum(item.get("total_tokens", 0) for item in usage)
        hit = f"{cached / inputs:.1%}" if inputs else "N/A"
        self.summary.setText(f"本轮合计   {total:,} token    ·    Cache hit {hit}\n"
                             f"{len(usage)} 次已记录模型请求 · {report['tool_count']} 次工具调用")
        self.report = report
        self.update_details()
        self.table.resizeRowsToContents()
        self.table.verticalScrollBar().setValue(scroll)

    def toggle_details(self, identity):
        self.expanded_id = None if self.expanded_id == identity else identity
        self.update_details()

    def update_details(self):
        selected = None
        for index, step in enumerate(self.report["steps"]):
            active = step["id"] == self.expanded_id
            self.table.cellWidget(index, 7).setText("收起" if active else "详情")
            if active:
                selected = step
        self.details.setVisible(selected is not None)
        if selected:
            content = selected["title"] + "\n" + selected["result"] + "\n\n" + selected["details"]
            if content != self.details.toPlainText():
                position = self.details.verticalScrollBar().value()
                self.details.setPlainText(content)
                self.details.verticalScrollBar().setValue(position)

    def closeEvent(self, event):
        self.timer.stop()
        if self.reader.state() != QProcess.NotRunning:
            self.reader.kill()
            self.reader.waitForFinished(1000)
        super().closeEvent(event)


if __name__ == "__main__":
    if "--report" in sys.argv:
        try:
            print(json.dumps(read_report(), ensure_ascii=False))
        except (OSError, ValueError, KeyError) as error:
            print(str(error), file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    app = QApplication(sys.argv)
    preview = Preview()
    preview.show()
    sys.exit(app.exec_())

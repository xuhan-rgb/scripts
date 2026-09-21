"""Token totals must belong to the displayed question only."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "codex_tokens", Path(__file__).resolve().parents[1] / "codex-tokens.py"
)
tokens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokens)


class LatestTurnTests(unittest.TestCase):
    def test_actions_match_request_order_without_changing_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        events = [{"type": "turn_context", "payload": {"turn_id": "t", "model": "a"}}]
        for index, source in enumerate(('text(await tools.exec_command({}));', 'text(await tools.apply_patch("x"));', None)):
            if source:
                call = {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec",
                    "call_id": str(index), "input": source}}
                events.extend([call, call])
            record = {"type": "token_usage_record", "payload": {"turn_id": "t", "response_id": str(index), "usage": usage}}
            events.extend([record, record])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            totals, _, _, requests = tokens.summarize(path)
        self.assertEqual([r["action"] for r in requests], ["执行本地命令", "修改文件", "模型处理 / 回复"])
        self.assertEqual(totals["a"]["total_tokens"], 330)

    def test_tool_calls_are_current_turn_deduplicated_and_keep_queries(self):
        events = [
            {"type": "turn_context", "payload": {"turn_id": "old", "model": "model-a"}},
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "old-call",
                "name": "search", "arguments": '{"query":"旧查询"}'}},
            {"type": "turn_context", "payload": {"turn_id": "new", "model": "model-a"}},
        ]
        search = {"type": "response_item", "payload": {"type": "function_call", "call_id": "search-1",
            "name": "search", "arguments": '{"query":"中文查询 <test>"}'}}
        events.extend([search, search, {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "search-1", "output": "结果"}},
            {"type": "response_item", "payload": {"type": "custom_tool_call", "call_id": "exec-1",
                "name": "exec", "input": 'await tools.one(); await tools.two();'}}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            _, _, calls, _ = tokens.summarize(path)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "search")
        self.assertIn("中文查询 <test>", calls[0]["content"])
        self.assertEqual(calls[1]["name"], "exec")

    def test_latest_question_excludes_previous_turn_and_sums_requests(self):
        events = []
        for turn, question, amounts in (("old", "旧问题", [9000]), ("new", "新问题", [100, 200])):
            events.append({"type": "response_item", "payload": {
                "role": "user", "content": [{"type": "input_text", "text": question}],
                "internal_chat_message_metadata_passthrough": {"turn_id": turn}}})
            events.append({"type": "turn_context", "payload": {"turn_id": turn, "model": "model-a"}})
            for index, amount in enumerate(amounts):
                record = {"type": "token_usage_record", "payload": {
                    "turn_id": turn, "response_id": f"{turn}-{index}",
                    "usage": {"input_tokens": amount, "output_tokens": 10, "total_tokens": amount + 10}}}
                events.extend([record, record])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            totals, question, calls, requests = tokens.summarize(path)
            self.assertEqual(question, "新问题")
            self.assertEqual(totals["model-a"]["total_tokens"], 320)
            self.assertEqual([item["usage"]["total_tokens"] for item in requests], [110, 210])
            self.assertEqual([item["response_id"] for item in requests], ["new-0", "new-1"])
            events.append({"type": "response_item", "payload": {
                "role": "user", "content": [{"type": "input_text", "text": "尚未回复"}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "pending"}}})
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            totals, question, calls, requests = tokens.summarize(path)
            self.assertEqual(question, "尚未回复")
            self.assertEqual(totals, {})
            self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()

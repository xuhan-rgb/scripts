"""Request timing reads existing logs without modifying the request transport."""
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("codex_tokens_timing", Path(__file__).resolve().parents[1] / "codex-tokens.py")
tokens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokens)


class RequestTimingTests(unittest.TestCase):
    def test_readonly_logs_filter_thread_turn_and_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "logs.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, ts INTEGER, ts_nanos INTEGER, "
                                   "thread_id TEXT, target TEXT, feedback_log_body TEXT)")
                for index, (thread, turn, warmup, second) in enumerate([
                        ("thread", "turn", "false", 101), ("other", "turn", "false", 102),
                        ("thread", "other", "false", 103), ("thread", "turn", "true", 104)]):
                    body = ('run_sampling_request{turn_id=' + turn + '}:model_client.stream_responses_websocket{'
                            'websocket.warmup=' + warmup + '}:responses_websocket.stream_request{transport="responses_websocket"}: '
                            'endpoint="/responses" auth_header_attached=true')
                    connection.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
                                       (index, second, 250000000, thread, "feedback_tags", body))
            before = database.read_bytes()
            self.assertEqual(tokens.request_starts(database, "thread", "turn", 100), [101.25])
            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(tokens.request_starts(database, "thread", "turn", 102), [])

    def test_request_windows_exclude_tools_and_keep_pending(self):
        requests = [{"usage_recorded_at": 105.5}, {"usage_recorded_at": 125.0}]
        pending = tokens.attach_request_timing(requests, [100.0, 120.0, 130.0])
        self.assertEqual(requests[0]["request_duration_ms"], 5500)
        self.assertEqual(requests[1]["request_duration_ms"], 5000)
        self.assertEqual(requests[1]["request_started_at"], 120)
        self.assertEqual(pending, 130)

    def test_ambiguous_retry_and_missing_timestamp_are_not_guessed(self):
        requests = [{"usage_recorded_at": 105}, {"usage_recorded_at": None}, {"usage_recorded_at": 120}]
        tokens.attach_request_timing(requests, [100, 102, 110])
        self.assertTrue(all(item["request_started_at"] is None for item in requests))
        self.assertTrue(all(item["request_duration_ms"] is None for item in requests))

    def test_missing_or_incompatible_database_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "absent.sqlite"
            self.assertEqual(tokens.request_starts(database, "thread", "turn", 0), [])
            self.assertFalse(database.exists())
            with sqlite3.connect(database):
                pass
            self.assertEqual(tokens.request_starts(database, "thread", "turn", 0), [])

    def test_jsonl_metrics_join_database_without_changing_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            events = [
                {"type": "session_meta", "payload": {"id": "thread"}},
                {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn", "started_at": 100}},
                {"type": "turn_context", "payload": {"turn_id": "turn", "model": "test"}},
                {"timestamp": "1970-01-01T00:01:45.500Z", "type": "token_usage_record", "payload": {
                    "turn_id": "turn", "response_id": "response", "usage": {"input_tokens": 123}}},
            ]
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            from unittest.mock import patch
            timing = {}
            with patch.object(tokens, "request_starts", return_value=[101.25]):
                totals, _, _, requests = tokens.summarize(path, timing=timing)
            self.assertEqual(totals["test"]["input_tokens"], 123)
            self.assertEqual(requests[0]["usage_recorded_at"], 105.5)
            self.assertEqual(requests[0]["request_started_at"], 101.25)
            self.assertEqual(requests[0]["request_duration_ms"], 4250)
            self.assertIsNone(timing["upload_duration_ms"])


if __name__ == "__main__":
    unittest.main()

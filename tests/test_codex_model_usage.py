import json
import os
import stat
import subprocess
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).parents[1] / "codex-model-usage.sh"
SPEC = importlib.util.spec_from_file_location("codex_usage_window", SCRIPT.with_name("codex_usage_window.py"))
WINDOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WINDOW)


class CodexModelUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls"
        self.mode = self.root / "mode"
        mock = self.bin / "ccusage"
        mock.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            f"pathlib.Path({str(self.calls)!r}).write_text(' '.join(sys.argv[1:]))\n"
            f"mode = pathlib.Path({str(self.mode)!r}).read_text().strip() if pathlib.Path({str(self.mode)!r}).exists() else 'daily'\n"
            "if mode == 'fail':\n"
            "    print('ccusage failed', file=sys.stderr); raise SystemExit(7)\n"
            "if mode == 'session':\n"
            "    print(json.dumps({'sessions': json.loads(os.environ['SESSION_JSON'])}))\n"
            "else:\n"
            "    print(os.environ['DAILY_JSON'])\n"
        )
        mock.chmod(mock.stat().st_mode | stat.S_IEXEC)
        self.env = os.environ.copy()
        self.env.update({
            "PATH": f"{self.bin}:{self.env['PATH']}",
            "HOME": str(self.root / "home"),
            "CODEX_HOME": str(self.root / ".codex"),
        })
        Path(self.env["CODEX_HOME"]).joinpath("sessions").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=self.env, text=True, capture_output=True, check=False,
        )

    def write_session(self, path, thread_id, *, parent_thread_id=None,
                      old_parent_thread_id=None, question=None, answer=None):
        log = self.root / ".codex" / "sessions" / f"{path}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        payload = {"id": thread_id, "session_id": "root-thread"}
        if parent_thread_id is not None:
            payload["parent_thread_id"] = parent_thread_id
        if old_parent_thread_id is not None:
            payload["source"] = {"subagent": {"thread_spawn": {
                "parent_thread_id": old_parent_thread_id,
            }}}
        events = [{"type": "session_meta", "payload": payload}]
        if question is not None:
            events.extend([
                {"type": "response_item", "payload": {
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": question}],
                }},
                {"type": "response_item", "payload": {
                    "type": "message", "role": "assistant", "phase": "final_answer",
                    "content": [{"type": "output_text", "text": answer}],
                }},
            ])
        log.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    def daily(self, models):
        self.env["DAILY_JSON"] = json.dumps({"daily": [{"models": models}]})
        self.mode.write_text("daily")

    def test_rolling_window_boundaries_duplicates_models_and_child(self):
        end = datetime(2026, 9, 21, 0, 30, tzinfo=timezone.utc)
        start = end - timedelta(hours=1)

        def event(when, input_tokens, cached, output, last=None):
            info = {"total_token_usage": dict(zip(
                ("input_tokens", "cached_input_tokens", "output_tokens"),
                (input_tokens, cached, output)))}
            if last is not None:
                info["last_token_usage"] = dict(zip(
                    ("input_tokens", "cached_input_tokens", "output_tokens"), last))
            return {"type": "event_msg", "timestamp": when.isoformat(),
                    "payload": {"type": "token_count", "info": info}}

        events = [
            {"type": "turn_context", "payload": {"model": "astra"}},
            event(start - timedelta(seconds=1), 100, 80, 10),
            event(start, 120, 90, 15),
            event(start + timedelta(seconds=1), 120, 90, 15),
            {"type": "turn_context", "payload": {"model": "luna"}},
            event(end, 150, 110, 19, (30, 20, 4)),
            event(end + timedelta(seconds=1), 200, 140, 25),
        ]
        root = Path(self.env["CODEX_HOME"]) / "sessions"
        (root / "parent.jsonl").write_text("\n".join(map(json.dumps, events)) + '\n{"partial":')
        (root / "child.jsonl").write_text("\n".join(map(json.dumps, [
            {"type": "turn_context", "payload": {"model": "luna"}},
            event(end, 8, 5, 2, (8, 5, 2)),
        ])))
        rows = WINDOW.summarize(root, start, end)["daily"][0]["models"]
        self.assertEqual(rows, {
            "astra": {"inputTokens": 10, "cachedInputTokens": 10,
                      "outputTokens": 5, "totalTokens": 25},
            "luna": {"inputTokens": 13, "cachedInputTokens": 25,
                     "outputTokens": 6, "totalTokens": 44},
        })

    def test_last_hour_and_minutes_do_not_call_ccusage(self):
        root = Path(self.env["CODEX_HOME"]) / "sessions"
        (root / "active.jsonl").write_text("\n".join(map(json.dumps, [
            {"type": "turn_context", "payload": {"model": "gpt-6-astra"}},
            {"type": "event_msg", "timestamp": datetime.now(timezone.utc).isoformat(),
             "payload": {"type": "token_count", "info": {"total_token_usage": {
                 "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 5}}}},
        ])))
        for duration in ("1h", "30m"):
            result = self.run_script("--last", duration)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("用量范围", result.stdout)
            self.assertRegex(result.stdout, r"gpt-6-astra\s+20\s+80\s+5\s+105\s+80%")
            self.assertFalse(self.calls.exists())

    def test_last_invalid_and_conflicting_arguments(self):
        for args in [("--last",), ("--last", "0h"), ("--last", "bad"),
                     ("--last", "1h", "--last-session"),
                     ("--last", "1h", "--since", "2026-09-21")]:
            result = self.run_script(*args)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(self.calls.exists())

    def test_default_range_sorting_cache_hit_and_zero_input(self):
        self.daily({
            "small": {"inputTokens": 0, "cachedInputTokens": 0,
                      "outputTokens": 1, "totalTokens": 1},
            "big": {"inputTokens": 3, "cachedInputTokens": 1,
                    "outputTokens": 4, "totalTokens": 8},
        })
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("big", lines[2])
        self.assertIn("25%", lines[2])
        self.assertIn("small", lines[3])
        self.assertIn("N/A", lines[3])
        self.assertLess(lines.index(next(line for line in lines if "big" in line)),
                        lines.index(next(line for line in lines if "small" in line)))
        self.assertEqual(self.calls.read_text().split(),
                         ["codex", "daily", "--offline", "--since", date.today().isoformat(),
                          "--until", date.today().isoformat(), "--json"])

    def test_last_session_uses_latest_activity_without_default_dates(self):
        sessions = [
            {"sessionId": "new", "lastActivity": "2026-09-19T09:00:00Z",
             "models": {"new-model": {"inputTokens": 2, "cachedInputTokens": 2,
                                        "outputTokens": 1, "totalTokens": 5}}},
            {"sessionId": "old", "lastActivity": "2026-09-18T09:00:00Z",
             "models": {"old-model": {"inputTokens": 1, "cachedInputTokens": 0,
                                        "outputTokens": 1, "totalTokens": 2}}},
        ]
        self.env["SESSION_JSON"] = json.dumps(sessions)
        self.mode.write_text("session")
        (Path(self.env["CODEX_HOME"]) / "sessions" / "new.jsonl").write_text(
            '{"type":"response_item","payload":{"type":"message","role":"user","phase":"commentary","content":[{"type":"input_text","text":"question"}]}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"ignore this"}]}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"actual question"}]}}\n'
            '{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"actual answer"}]}}\n'
        )
        result = self.run_script("--last-session")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Session: new", result.stdout)
        self.assertIn("actual question", result.stdout)
        self.assertIn("actual answer", result.stdout)
        self.assertNotIn("ignore this", result.stdout)
        self.assertNotIn("你问：question", result.stdout)
        self.assertNotIn("--since", self.calls.read_text())
        self.assertNotIn("--until", self.calls.read_text())

    def test_last_session_preview_filters_unanswered_and_full_text_expands(self):
        long_question = "Q" * 350
        long_answer = "A" * 450
        self.env["SESSION_JSON"] = json.dumps([{
            "sessionId": "long", "lastActivity": "2026-09-19T10:00:00Z",
            "models": {"m": {"inputTokens": 1, "cachedInputTokens": 0,
                               "outputTokens": 1, "totalTokens": 2}},
        }])
        self.mode.write_text("session")
        (Path(self.env["CODEX_HOME"]) / "sessions" / "long.jsonl").write_text(
            json.dumps({"type": "response_item", "payload": {"type": "message",
                "role": "user", "content": [{"type": "input_text", "text": long_question}]}}) + "\n" +
            json.dumps({"type": "response_item", "payload": {"type": "message",
                "role": "assistant", "phase": "final", "content": [{"type": "output_text", "text": long_answer}]}}) + "\n"
            + json.dumps({"type": "event_msg", "payload": {"type": "task_started"}}) + "\n"
            + json.dumps({"type": "response_item", "payload": {"type": "message",
                "role": "user", "content": [{"type": "input_text", "text": "unanswered"}]}}) + "\n"
        )
        preview = self.run_script("--last-session")
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertNotIn("unanswered", preview.stdout)
        self.assertIn("已截断", preview.stdout)
        full = self.run_script("--full-text")
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertIn(long_question, full.stdout)
        self.assertIn(long_answer, full.stdout)
        self.assertNotIn("已截断", full.stdout)

    def test_last_session_aggregates_recursive_root_tree_and_excludes_independent(self):
        sessions = [
            {"sessionId": "root/log", "lastActivity": "2026-09-19T12:06:00Z",
             "models": {"shared": {"inputTokens": 10, "cachedInputTokens": 5,
                                       "outputTokens": 2, "totalTokens": 17}}},
            {"sessionId": "child-a/log", "lastActivity": "2026-09-19T12:01:00Z",
             "models": {"shared": {"inputTokens": 20, "cachedInputTokens": 10,
                                       "outputTokens": 3, "totalTokens": 33}}},
            {"sessionId": "child-b/log", "lastActivity": "2026-09-19T12:02:00Z",
             "models": {"shared": {"inputTokens": 30, "cachedInputTokens": 0,
                                       "outputTokens": 4, "totalTokens": 34}}},
            {"sessionId": "grandchild/log", "lastActivity": "2026-09-19T12:03:00Z",
             "models": {"shared": {"inputTokens": 40, "cachedInputTokens": 20,
                                       "outputTokens": 5, "totalTokens": 65}}},
            {"sessionId": "independent/log", "lastActivity": "2026-09-19T12:04:00Z",
             "models": {"shared": {"inputTokens": 900, "cachedInputTokens": 0,
                                       "outputTokens": 1, "totalTokens": 901}}},
        ]
        self.env["SESSION_JSON"] = json.dumps(sessions)
        self.mode.write_text("session")
        self.write_session("root/log", "root-thread", question="root question",
                           answer="root answer")
        self.write_session("child-a/log", "child-a-thread",
                           parent_thread_id="root-thread")
        self.write_session("child-b/log", "child-b-thread",
                           parent_thread_id="root-thread")
        self.write_session("grandchild/log", "grandchild-thread",
                           parent_thread_id="child-a-thread")
        self.write_session("independent/log", "independent-thread")

        result = self.run_script("--last-session")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Session: root/log", result.stdout)
        self.assertIn("shared", result.stdout)
        self.assertRegex(result.stdout, r"\b100\s+35\s+14\s+149\s+25\.93%")
        self.assertIn("3 个子代理", result.stdout)
        self.assertIn("root question", result.stdout)
        self.assertIn("root answer", result.stdout)
        self.assertNotIn("independent", result.stdout)

    def test_last_session_child_activity_still_uses_root_and_old_parent_metadata(self):
        sessions = [
            {"sessionId": "root/log", "lastActivity": "2026-09-19T12:00:00Z",
             "models": {"root-model": {"inputTokens": 4, "cachedInputTokens": 1,
                                          "outputTokens": 2, "totalTokens": 7}}},
            {"sessionId": "legacy-child/log", "lastActivity": "2026-09-19T12:05:00Z",
             "models": {"child-model": {"inputTokens": 8, "cachedInputTokens": 2,
                                           "outputTokens": 3, "totalTokens": 13}}},
            {"sessionId": "sibling/log", "lastActivity": "2026-09-19T12:04:00Z",
             "models": {"child-model": {"inputTokens": 6, "cachedInputTokens": 1,
                                           "outputTokens": 2, "totalTokens": 9}}},
            {"sessionId": "legacy-grandchild/log", "lastActivity": "2026-09-19T12:03:00Z",
             "models": {"child-model": {"inputTokens": 5, "cachedInputTokens": 1,
                                           "outputTokens": 2, "totalTokens": 8}}},
        ]
        self.env["SESSION_JSON"] = json.dumps(sessions)
        self.mode.write_text("session")
        self.write_session("root/log", "root-thread", question="root identity",
                           answer="root response")
        self.write_session("legacy-child/log", "legacy-child-thread",
                           old_parent_thread_id="root-thread")
        self.write_session("sibling/log", "sibling-thread",
                           parent_thread_id="root-thread")
        self.write_session("legacy-grandchild/log", "legacy-grandchild-thread",
                           old_parent_thread_id="legacy-child-thread")

        result = self.run_script("--last-session")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Session: root/log", result.stdout)
        self.assertIn("root-model", result.stdout)
        self.assertIn("child-model", result.stdout)
        self.assertIn("3 个子代理", result.stdout)
        self.assertIn("root identity", result.stdout)
        self.assertIn("root response", result.stdout)

    def test_last_session_root_without_ccusage_row_still_keeps_child_usage(self):
        sessions = [
            {"sessionId": "child/log", "lastActivity": "2026-09-19T12:05:00Z",
             "models": {"child-model": {"inputTokens": 8, "cachedInputTokens": 2,
                                           "outputTokens": 3, "totalTokens": 13}}},
            {"sessionId": "grandchild/log", "lastActivity": "2026-09-19T12:06:00Z",
             "models": {"child-model": {"inputTokens": 12, "cachedInputTokens": 4,
                                           "outputTokens": 5, "totalTokens": 21}}},
        ]
        self.env["SESSION_JSON"] = json.dumps(sessions)
        self.mode.write_text("session")
        self.write_session("root/log", "root-thread", question="root-only question",
                           answer="root-only answer")
        self.write_session("child/log", "child-thread",
                           parent_thread_id="root-thread")
        self.write_session("grandchild/log", "grandchild-thread",
                           parent_thread_id="child-thread")

        result = self.run_script("--last-session")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Session: root/log", result.stdout)
        self.assertRegex(result.stdout, r"\b20\s+6\s+8\s+34\s+23\.08%")
        self.assertIn("root-only question", result.stdout)
        self.assertIn("root-only answer", result.stdout)

    def test_last_session_with_no_sessions_reports_empty_result(self):
        self.env["SESSION_JSON"] = "[]"
        self.mode.write_text("session")

        result = self.run_script("--last-session")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("没有找到会话记录", result.stdout)
        self.assertRegex(result.stdout, r"Model\s+Input\s+Cached\s+Output\s+Total")

    def test_explicit_date_range_is_forwarded_unchanged(self):
        self.daily({"m": {"inputTokens": 1, "cachedInputTokens": 0,
                           "outputTokens": 1, "totalTokens": 2}})
        result = self.run_script("--since", "2026-01-02", "--until=2026-01-03")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls.read_text().split(),
                         ["codex", "daily", "--offline", "--since", "2026-01-02",
                          "--until=2026-01-03", "--json"])

    def test_ccusage_failure_is_nonzero(self):
        self.mode.write_text("fail")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ccusage failed", result.stderr)

    def test_last_session_rejects_date_filters_before_querying(self):
        self.mode.write_text("session")
        self.env["SESSION_JSON"] = "[]"
        for args in [("--last-session", "--since", "2026-01-02"),
                     ("--full-text", "--until=2026-01-03")]:
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("不能与日期筛选同时使用", result.stderr)
                self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()

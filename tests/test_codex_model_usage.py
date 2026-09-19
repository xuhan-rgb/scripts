import json
import os
import stat
import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "codex-model-usage.sh"


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

    def daily(self, models):
        self.env["DAILY_JSON"] = json.dumps({"daily": [{"models": models}]})
        self.mode.write_text("daily")

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


if __name__ == "__main__":
    unittest.main()

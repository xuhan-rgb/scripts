import importlib.util
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "claude" / "codex_bridge_manager.py"
SPEC = importlib.util.spec_from_file_location("codex_bridge_manager", MODULE_PATH)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


SAMPLE_CONFIG = '''model_provider = "provider_b"
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"

[model_providers.provider_a]
base_url = "https://a.example/openai"
wire_api = "responses"
env_key = "A_KEY"

[model_providers.provider_b]
base_url = "http://127.0.0.1:3000/openai"
wire_api = "responses"
env_key = "B_KEY"
'''

SAMPLE_GATEWAY_CONFIG = '''codex-api-key:
  - models:
      - name: "gpt-5.6-sol" # claudex-route-model
        alias: "claudex-router"
payload:
  override:
    - models:
        - name: "gpt-5.6-sol" # claudex-effort-model
          protocol: "codex"
          from-protocol: "claude"
      params:
        "reasoning.effort": "xhigh" # claudex-route-effort
'''


class CodexBridgeManagerTests(unittest.TestCase):
    def test_dashboard_preserves_unchanged_live_sections_between_polls(self):
        self.assertIn("if (app.rendered.models !== modelCatalog)", manager.HTML)
        self.assertIn("if (app.rendered.efforts !== effortCatalog)", manager.HTML)
        self.assertIn("if (app.rendered.requests === signature) return", manager.HTML)

    def test_dashboard_enables_instant_switching_by_default(self):
        self.assertIn("id=\"instant\"", manager.HTML)
        self.assertIn("localStorage.getItem('claudex-instant-switch') !== 'false'", manager.HTML)
        self.assertIn("if (app.autoApply) scheduleSave()", manager.HTML)
        self.assertIn("$('save').hidden = app.autoApply", manager.HTML)
        self.assertNotIn("Claude keeps one stable client model", manager.HTML)

    def test_dashboard_renders_usage_periods_and_ttft_incrementally(self):
        self.assertIn('id="usage-day-total"', manager.HTML)
        self.assertIn('id="usage-week-total"', manager.HTML)
        self.assertIn('id="usage-month-total"', manager.HTML)
        self.assertIn('id="last-request-input"', manager.HTML)
        self.assertIn('id="last-request-cache-hit"', manager.HTML)
        self.assertIn('id="last-request-age"', manager.HTML)
        self.assertIn("$(`usage-${name}-total`).textContent", manager.HTML)
        self.assertIn("Math.max(totalInput - cacheRead, 0)", manager.HTML)
        self.assertIn("setInterval(refreshLastRequestAge, 1000)", manager.HTML)
        self.assertIn("`${seconds} 秒前`", manager.HTML)
        self.assertIn("Number(row.ttft_ms || 0)", manager.HTML)
        self.assertIn("首 Token / 总耗时", manager.HTML)
        self.assertIn("<th>输入</th><th>输出</th>", manager.HTML)
        self.assertNotIn("<th>上下文</th>", manager.HTML)
        self.assertNotIn("<th>新增</th>", manager.HTML)

    def test_reads_the_dynamically_selected_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text(SAMPLE_CONFIG, encoding="utf-8")

            state = manager.read_codex_state(config)

        self.assertEqual(state["provider"], "provider_b")
        self.assertEqual(state["base_url"], "http://127.0.0.1:3000/openai")
        self.assertEqual(state["env_key"], "B_KEY")

    def test_selection_is_independent_from_codex_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.conf"
            lock = Path(directory) / "selection.lock"
            manager.write_selection("gpt-5.6-terra", "high", selection, lock)

            result = manager.read_selection(
                selection,
                {"codex_model": "gpt-5.6-sol", "codex_effort": "xhigh"},
            )

        self.assertEqual(result, {"model": "gpt-5.6-terra", "effort": "high"})

    def test_applies_live_model_and_effort_to_gateway(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.conf"
            config = Path(directory) / "config.yaml"
            lock = Path(directory) / "selection.lock"
            config.write_text(SAMPLE_GATEWAY_CONFIG, encoding="utf-8")

            result = manager.apply_selection(
                "gpt-5.6-luna",
                "medium",
                selection,
                config,
                lock,
            )
            updated = config.read_text(encoding="utf-8")

        self.assertEqual(result, {"model": "gpt-5.6-luna", "effort": "medium"})
        self.assertEqual(updated.count('name: "gpt-5.6-luna"'), 2)
        self.assertIn('"reasoning.effort": "medium"', updated)

    def test_persists_request_usage_without_prompt_content(self):
        record = {
            "request_id": "request-1",
            "timestamp": "2026-08-04T17:59:48+08:00",
            "api_key": "claudex-local",
            "provider": "codex",
            "auth_index": "provider-key-1",
            "model": "gpt-5.6-sol",
            "alias": "claudex-router",
            "reasoning_effort": "xhigh",
            "endpoint": "/openai/responses",
            "latency_ms": 8000,
            "ttft_ms": 720,
            "tokens": {
                "input_tokens": 88100,
                "output_tokens": 82,
                "reasoning_tokens": 20,
                "cache_read_tokens": 71400,
                "cache_creation_tokens": 0,
                "total_tokens": 88202,
            },
            "failed": False,
            "fail": {},
            "prompt": "must not be stored",
        }
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "usage.sqlite3"
            manager.store_usage_records([record, record], database)
            rows = manager.list_usage_records(10, database)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "gpt-5.6-sol")
        self.assertEqual(rows[0]["cache_read_tokens"], 71400)
        self.assertNotIn("prompt", rows[0])

    def test_summarizes_tokens_by_local_day_week_and_month(self):
        now = datetime(2026, 8, 4, 21, 0, tzinfo=timezone.utc)

        def record(request_id, timestamp, total_tokens, cache_read_tokens=0):
            return {
                "request_id": request_id,
                "timestamp": timestamp,
                "tokens": {
                    "input_tokens": total_tokens - 10,
                    "output_tokens": 10,
                    "reasoning_tokens": 3,
                    "cache_read_tokens": cache_read_tokens,
                    "total_tokens": total_tokens,
                },
            }

        records = [
            record("today", "2026-08-04T08:00:00.123456789+00:00", 100, 60),
            record("week", "2026-08-03T08:00:00+00:00", 200),
            record("month", "2026-08-01T08:00:00+00:00", 300),
            record("previous-month", "2026-07-31T08:00:00+00:00", 400),
        ]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "usage.sqlite3"
            manager.store_usage_records(records, database)
            summary = manager.usage_summary(database, now=now)

        self.assertEqual(summary["day"]["total_tokens"], 100)
        self.assertEqual(summary["day"]["input_tokens"], 30)
        self.assertEqual(summary["day"]["cache_read_tokens"], 60)
        self.assertEqual(summary["week"]["total_tokens"], 300)
        self.assertEqual(summary["month"]["total_tokens"], 600)
        self.assertEqual(summary["day"]["requests"], 1)
        self.assertEqual(summary["week"]["requests"], 2)
        self.assertEqual(summary["month"]["requests"], 3)

    def test_rejects_unknown_model_and_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.conf"
            lock = Path(directory) / "selection.lock"
            with self.assertRaisesRegex(ValueError, "unsupported model"):
                manager.write_selection("gpt-image-2", "high", selection, lock)
            with self.assertRaisesRegex(ValueError, "unsupported effort"):
                manager.write_selection("gpt-5.6-sol", "ultra", selection, lock)


if __name__ == "__main__":
    unittest.main()

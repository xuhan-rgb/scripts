import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from unittest import mock


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

    def test_dashboard_exposes_provider_config_without_rendering_keys(self):
        self.assertIn('id="provider-config-open"', manager.HTML)
        self.assertIn('id="provider-config-layer"', manager.HTML)
        self.assertIn('id="provider-edit-key"', manager.HTML)
        self.assertIn("/api/providers/save", manager.HTML)
        self.assertIn("/api/providers/switch", manager.HTML)
        self.assertIn("/api/providers/test", manager.HTML)
        self.assertIn("/api/providers/delete", manager.HTML)
        self.assertIn('id="provider-test"', manager.HTML)
        self.assertIn('id="provider-delete"', manager.HTML)
        self.assertIn("The Key is stored locally with mode 0600", manager.HTML)
        self.assertIn("built-in provider backend", manager.HTML)
        self.assertIn("$('provider-edit-key').required = !provider?.key_set", manager.HTML)
        self.assertIn("provider?.active", manager.HTML)

    def test_dashboard_renders_usage_periods_and_ttft_incrementally(self):
        self.assertIn('id="usage-day-total"', manager.HTML)
        self.assertIn('id="usage-week-total"', manager.HTML)
        self.assertIn('id="usage-month-total"', manager.HTML)
        self.assertIn('id="last-request-input"', manager.HTML)
        self.assertIn('id="last-request-cache-hit"', manager.HTML)
        self.assertIn('id="refresh-age"', manager.HTML)
        self.assertIn("$(`usage-${name}-total`).textContent", manager.HTML)
        self.assertIn("Math.max(totalInput - cacheRead, 0)", manager.HTML)
        self.assertNotIn("lastRefreshAt", manager.HTML)
        self.assertIn("const timestamp = app.requests[0]?.timestamp", manager.HTML)
        self.assertIn("setInterval(refreshRequestAge, 1000)", manager.HTML)
        self.assertIn("`${seconds} 秒前请求`", manager.HTML)
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

    def test_provider_catalog_reports_key_state_without_exposing_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            secrets = Path(directory) / "secrets.env"
            config.write_text(SAMPLE_CONFIG, encoding="utf-8")
            secrets.write_text("export B_KEY='secret-never-return'\n", encoding="utf-8")

            providers = manager.read_provider_catalog(config, secrets)

        provider_b = next(provider for provider in providers if provider["name"] == "provider_b")
        self.assertTrue(provider_b["active"])
        self.assertTrue(provider_b["key_set"])
        self.assertNotIn("secret-never-return", json.dumps(providers))

    def test_saves_provider_without_activating_or_putting_key_in_process_arguments(self):
        commands = []

        def run(arguments, input_text=None):
            commands.append((arguments, input_text))
            return ""

        with (
            mock.patch.object(manager, "read_provider_catalog", return_value=[]),
            mock.patch.object(manager, "_run_provider_command", side_effect=run),
            mock.patch.object(manager, "_sync_provider_gateway") as sync,
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            name = manager.save_provider(
                {
                    "name": "new_route",
                    "base_url": "https://gateway.example/openai",
                    "env_key": "NEW_ROUTE_KEY",
                    "api_key": " secret-never-in-argv ",
                }
            )

        self.assertEqual(name, "new_route")
        self.assertEqual(commands[0][0][0], "add")
        self.assertEqual(commands[1], (["set-key", "new_route", "--stdin"], " secret-never-in-argv "))
        self.assertEqual(len(commands), 2)
        self.assertFalse(any("secret-never-in-argv" in argument for command, _ in commands for argument in command))
        sync.assert_not_called()

    def test_activates_and_syncs_a_saved_provider(self):
        provider = {
            "name": "saved_route",
            "base_url": "https://gateway.example/openai",
            "wire_api": "responses",
            "env_key": "SAVED_ROUTE_KEY",
            "key_set": True,
            "active": True,
        }
        with (
            mock.patch.object(manager, "read_provider_catalog", return_value=[provider]),
            mock.patch.object(manager, "_run_provider_command") as run,
            mock.patch.object(manager, "_sync_provider_gateway") as sync,
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            name = manager.switch_provider({"name": "saved_route"})

        self.assertEqual(name, "saved_route")
        run.assert_called_once_with(["switch", "saved_route"])
        sync.assert_called_once_with()

    def test_tests_unsaved_provider_fields_without_putting_key_in_process_arguments(self):
        with (
            mock.patch.object(manager, "_run_provider_command", return_value="connection test:\nresult: ok\n") as run,
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            result = manager.test_provider(
                {
                    "name": "candidate",
                    "base_url": "https://candidate.example/openai",
                    "env_key": "CANDIDATE_KEY",
                    "api_key": "candidate-secret",
                }
            )

        self.assertEqual(result, "connection test:\nresult: ok")
        run.assert_called_once_with(
            [
                "test",
                "candidate",
                "--base-url",
                "https://candidate.example/openai",
                "--env-key",
                "CANDIDATE_KEY",
                "--stdin",
            ],
            input_text="candidate-secret",
        )

    def test_deletes_only_an_inactive_provider(self):
        providers = [
            {"name": "active_route", "active": True, "key_set": True},
            {"name": "old_route", "active": False, "key_set": True},
        ]
        with (
            mock.patch.object(manager, "read_provider_catalog", return_value=providers),
            mock.patch.object(manager, "_run_provider_command") as run,
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            name = manager.delete_provider({"name": "old_route", "confirm_name": "old_route"})

        self.assertEqual(name, "old_route")
        run.assert_called_once_with(["delete", "old_route", "--yes"])

        with (
            mock.patch.object(manager, "read_provider_catalog", return_value=providers),
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            with self.assertRaisesRegex(ValueError, "active provider"):
                manager.delete_provider({"name": "active_route", "confirm_name": "active_route"})

    def test_internal_provider_backend_updates_config_without_installing_a_command(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            codex_home = home / ".codex"
            environment = {"HOME": str(home), "CODEX_HOME": str(codex_home)}
            secret = "internal-backend-secret"
            with mock.patch.dict(os.environ, environment):
                manager._run_provider_command(
                    [
                        "add",
                        "web_route",
                        "--base-url",
                        "https://gateway.example/openai",
                        "--env-key",
                        "WEB_ROUTE_KEY",
                        "--wire-api",
                        "responses",
                        "--skip-test",
                    ]
                )
                output = manager._run_provider_command(
                    ["set-key", "web_route", "--stdin"],
                    input_text=secret,
                )
                manager._run_provider_command(["switch", "web_route"])

            config_text = (codex_home / "config.toml").read_text(encoding="utf-8")
            secrets_text = (home / ".config" / "codex" / "secrets.env").read_text(encoding="utf-8")
            self.assertIn('model_provider = "web_route"', config_text)
            self.assertNotIn(secret, config_text)
            self.assertIn(secret, secrets_text)
            self.assertNotIn(secret, output)
            self.assertFalse((home / ".local" / "bin" / "codex-provider").exists())

    def test_internal_provider_backend_deletes_only_an_inactive_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            codex_home = home / ".codex"
            environment = {"HOME": str(home), "CODEX_HOME": str(codex_home)}
            with mock.patch.dict(os.environ, environment):
                for name, env_key in (("active_route", "ACTIVE_KEY"), ("old_route", "OLD_KEY")):
                    manager._run_provider_command(
                        [
                            "add",
                            name,
                            "--base-url",
                            "https://gateway.example/openai",
                            "--env-key",
                            env_key,
                            "--skip-test",
                        ]
                    )
                    manager._run_provider_command(["set-key", name, "--stdin"], input_text=f"{name}-secret")
                manager._run_provider_command(["switch", "active_route"])
                manager._run_provider_command(["delete", "old_route", "--yes"])

                with self.assertRaisesRegex(ValueError, "cannot delete the active provider"):
                    manager._run_provider_command(["delete", "active_route", "--yes"])

            config_text = (codex_home / "config.toml").read_text(encoding="utf-8")
            secrets_text = (home / ".config" / "codex" / "secrets.env").read_text(encoding="utf-8")
            self.assertIn('model_provider = "active_route"', config_text)
            self.assertNotIn("[model_providers.old_route]", config_text)
            self.assertFalse((codex_home / "old_route.config.toml").exists())
            self.assertNotIn("OLD_KEY", secrets_text)
            self.assertIn("ACTIVE_KEY", secrets_text)

    def test_refuses_to_save_an_existing_provider_without_key(self):
        providers = [
            {
                "name": "missing_key",
                "base_url": "https://gateway.example/openai",
                "wire_api": "responses",
                "env_key": "MISSING_KEY",
                "key_set": False,
                "active": True,
            }
        ]
        with (
            mock.patch.object(manager, "read_provider_catalog", return_value=providers),
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            with self.assertRaisesRegex(ValueError, "provider has no stored Key"):
                manager.save_provider(
                    {
                        "name": "missing_key",
                        "base_url": "https://gateway.example/openai",
                        "env_key": "MISSING_KEY",
                        "api_key": "",
                    }
                )

    def test_derives_an_environment_key_for_an_older_provider_without_one(self):
        providers = [
            {
                "name": "older-route",
                "base_url": "https://gateway.example/openai",
                "wire_api": "responses",
                "env_key": "",
                "key_set": False,
                "active": True,
            }
        ]
        commands = []

        def run(arguments, input_text=None):
            commands.append((arguments, input_text))
            return ""

        with (
            mock.patch.object(manager, "read_provider_catalog", return_value=providers),
            mock.patch.object(manager, "_run_provider_command", side_effect=run),
            mock.patch.object(manager, "_provider_lock", return_value=nullcontext()),
        ):
            manager.save_provider(
                {
                    "name": "older-route",
                    "base_url": "https://gateway.example/openai",
                    "env_key": "",
                    "api_key": "secret",
                }
            )

        self.assertIn("OLDER_ROUTE_OPENAI_KEY", commands[0][0])

    def test_refuses_to_switch_to_provider_without_key(self):
        providers = [
            {
                "name": "missing_key",
                "base_url": "https://gateway.example/openai",
                "wire_api": "responses",
                "env_key": "MISSING_KEY",
                "key_set": False,
                "active": False,
            }
        ]
        with mock.patch.object(manager, "read_provider_catalog", return_value=providers):
            with self.assertRaisesRegex(ValueError, "has no API key"):
                manager.switch_provider({"name": "missing_key"})

    def test_provider_posts_require_same_origin_json(self):
        handler = object.__new__(manager.ManagerHandler)
        handler.rfile = io.BytesIO(b'{"name":"provider_a"}')
        handler.headers = Message()
        handler.headers["Origin"] = "https://attacker.example"
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = "21"
        with self.assertRaisesRegex(PermissionError, "origin or content type"):
            handler._read_json()

        handler.rfile = io.BytesIO(b'{"name":"provider_a"}')
        handler.headers.replace_header("Origin", "http://127.0.0.1:8320")
        handler.headers.replace_header("Content-Type", "text/plain")
        with self.assertRaisesRegex(PermissionError, "origin or content type"):
            handler._read_json()

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

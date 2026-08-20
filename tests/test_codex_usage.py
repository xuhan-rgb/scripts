import importlib.machinery
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "claude" / "codex-usage"
LOADER = importlib.machinery.SourceFileLoader("codex_usage", str(MODULE_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
usage = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(usage)


class CodexUsageTests(unittest.TestCase):
    def test_resolves_the_selected_named_account_without_reading_its_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / ".codex"
            account_home = root / "accounts" / "work"
            active_file = root / "active-account"
            codex_home.mkdir()
            account_home.mkdir(parents=True)
            (account_home / "auth.json").write_text("secret-never-return", encoding="utf-8")
            active_file.write_text("work\n", encoding="utf-8")

            selected = usage.resolve_account_home(
                None,
                codex_home=codex_home,
                accounts_dir=account_home.parent,
                active_file=active_file,
            )

        self.assertEqual(selected, ("work", account_home, codex_home))
        self.assertNotIn("secret-never-return", repr(selected))

    def test_resolves_the_current_unnamed_chatgpt_account(self):
        with tempfile.TemporaryDirectory() as directory:
            codex_home = Path(directory) / ".codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                'model_provider = "openai"\n\n[features]\nweb_search = true\n',
                encoding="utf-8",
            )
            (codex_home / "auth.json").write_text(
                '{"auth_mode":"chatgpt","tokens":{"access_token":"secret-never-return"}}\n',
                encoding="utf-8",
            )

            selected = usage.resolve_account_home(
                None,
                codex_home=codex_home,
                accounts_dir=Path(directory) / "accounts",
                active_file=Path(directory) / "missing-active-account",
            )

        self.assertEqual(selected, ("unnamed", codex_home, codex_home))
        self.assertNotIn("secret-never-return", repr(selected))

    def test_does_not_treat_a_saved_chatgpt_login_as_current_in_api_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            codex_home = Path(directory) / ".codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                'model_provider = "crs_local"\n', encoding="utf-8"
            )
            (codex_home / "auth.json").write_text(
                '{"auth_mode":"chatgpt","tokens":{}}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(usage.UsageError, "No named account"):
                usage.resolve_account_home(
                    None,
                    codex_home=codex_home,
                    accounts_dir=Path(directory) / "accounts",
                    active_file=Path(directory) / "missing-active-account",
                )

    def test_builds_the_existing_usage_shape_from_codex_app_server(self):
        rate_limits = {
            "rateLimits": {
                "limitId": "codex",
                "planType": "plus",
                "primary": {"usedPercent": 25, "windowDurationMins": 300, "resetsAt": 2000},
                "secondary": {"usedPercent": 60, "windowDurationMins": 10080, "resetsAt": 3000},
                "credits": {"hasCredits": True, "unlimited": False, "balance": "12.5"},
                "spendControlReached": False,
            },
            "rateLimitsByLimitId": None,
        }
        token_usage = {
            "summary": {"lifetimeTokens": 1234, "peakDailyTokens": 456},
            "dailyUsageBuckets": [{"startDate": "2026-08-10", "tokens": 99}],
        }

        snapshot = usage.build_app_server_snapshot(
            "work", rate_limits, token_usage, fetched_at="2026-08-11T10:00:00+08:00"
        )

        self.assertEqual(snapshot["source"], "codex_app_server")
        self.assertEqual(snapshot["account"], "work")
        self.assertEqual(snapshot["plan_type"], "plus")
        self.assertEqual(snapshot["rate_limits"][0]["name"], "Codex")
        self.assertEqual(snapshot["rate_limits"][0]["windows"][0]["remaining_percent"], 75)
        self.assertEqual(snapshot["rate_limits"][0]["windows"][1]["window_seconds"], 604800)
        self.assertEqual(snapshot["token_usage"]["lifetime_tokens"], 1234)
        self.assertEqual(snapshot["token_usage"]["daily_usage_buckets"][0]["start_date"], "2026-08-10")

    def test_fetches_usage_through_the_named_accounts_codex_app_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / ".codex"
            account_home = root / "accounts" / "work"
            fake_bin = root / "bin"
            codex_home.mkdir()
            account_home.mkdir(parents=True)
            fake_bin.mkdir()
            (account_home / "auth.json").write_text('{"fake":true}\n', encoding="utf-8")
            active_file = root / "active-account"
            active_file.write_text("work\n", encoding="utf-8")
            fake_codex = fake_bin / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "for line in sys.stdin:\n"
                "    request = json.loads(line)\n"
                "    request_id = request.get('id')\n"
                "    if request_id == 1:\n"
                "        result = {'codexHome': 'ok'}\n"
                "    elif request_id == 2:\n"
                "        result = {'rateLimits': {'planType': 'plus', 'primary': {'usedPercent': 10}}}\n"
                "    elif request_id == 3:\n"
                "        result = {'summary': {'lifetimeTokens': 50}, 'dailyUsageBuckets': []}\n"
                "    else:\n"
                "        continue\n"
                "    print(json.dumps({'id': request_id, 'result': result}), flush=True)\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            environment = {
                "HOME": str(root),
                "CODEX_HOME": str(codex_home),
                "CODEX_ACCOUNTS_DIR": str(account_home.parent),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
            }
            with (
                mock.patch.dict(os.environ, environment),
                mock.patch.object(Path, "home", return_value=root),
            ):
                account, limits, token_usage, token_error = usage.fetch_account_usage("work", 2)

        self.assertEqual(account, "work")
        self.assertEqual(limits["rateLimits"]["primary"]["usedPercent"], 10)
        self.assertEqual(token_usage["summary"]["lifetimeTokens"], 50)
        self.assertIsNone(token_error)


if __name__ == "__main__":
    unittest.main()

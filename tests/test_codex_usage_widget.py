import importlib.machinery
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "codex-usage-widget"
LOADER = importlib.machinery.SourceFileLoader("codex_usage_widget", str(MODULE_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
widget = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(widget)


class CodexUsageWidgetTests(unittest.TestCase):
    def test_displays_the_selected_account_and_its_longest_quota_window(self):
        output = json.dumps(
            {
                "account": "work",
                "plan_type": "plus",
                "rate_limits": [
                    {
                        "name": "Codex",
                        "windows": [
                            {
                                "name": "primary",
                                "remaining_percent": 75,
                                "window_seconds": 18000,
                                "resets_at": 2000,
                            },
                            {
                                "name": "secondary",
                                "remaining_percent": 40,
                                "window_seconds": 604800,
                                "resets_at": 3000,
                            },
                        ],
                    }
                ],
            }
        )

        parsed = widget.parse_usage(output)

        self.assertEqual(parsed, ("work", "plus", "7d", "40", 3000.0))

    def test_rejects_usage_without_a_named_account(self):
        output = json.dumps(
            {
                "rate_limits": [
                    {
                        "name": "Codex",
                        "windows": [
                            {
                                "name": "primary",
                                "remaining_percent": 75,
                                "window_seconds": 18000,
                                "resets_at": 2000,
                            }
                        ],
                    }
                ]
            }
        )

        with self.assertRaisesRegex(RuntimeError, "命名账号"):
            widget.parse_usage(output)

    def test_reads_the_account_selected_by_codex_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            active_file = Path(directory) / "active-account"
            config_file = Path(directory) / "config.toml"
            self.assertIsNone(widget.selected_account(active_file, config_file))

            active_file.write_text("personal\n", encoding="utf-8")
            self.assertEqual(widget.selected_account(active_file, config_file), "personal")

            active_file.unlink()
            self.assertIsNone(widget.selected_account(active_file, config_file))

    def test_recognizes_the_current_unnamed_chatgpt_account(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text('model_provider = "openai"\n', encoding="utf-8")

            account = widget.selected_account(root / "missing-active-account", config_file)

        self.assertEqual(account, "unnamed")


if __name__ == "__main__":
    unittest.main()

import base64
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[1]
BACKEND_PATH = REPOSITORY / "claude" / "codex_account_manager_backend.py"
QT_APP_PATH = REPOSITORY / "claude" / "codex_account_manager_qt.py"
ICON_PATH = REPOSITORY / "claude" / "codex-account-manager.svg"
SPEC = importlib.util.spec_from_file_location("codex_account_manager_backend", BACKEND_PATH)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


class CodexAccountManagerBackendTests(unittest.TestCase):
    def test_extracts_device_login_url_without_terminal_color_codes(self):
        output = (
            "\x1b[1mOpen this URL in your browser:\x1b[0m\n"
            "https://auth.openai.com/codex/device\x1b[0m\n"
        )

        self.assertEqual(
            backend.extract_login_url(output),
            "https://auth.openai.com/codex/device",
        )

    def test_account_names_accept_email_addresses(self):
        self.assertIsNotNone(
            backend.ACCOUNT_NAME_PATTERN.fullmatch("user+codex@example.com")
        )

    def test_reads_named_accounts_and_api_providers_without_exposing_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / ".codex"
            accounts_dir = root / "accounts"
            active_file = root / "active-account"
            secrets = root / "secrets.env"
            account_home = accounts_dir / "work"
            codex_home.mkdir()
            account_home.mkdir(parents=True)
            (account_home / "auth.json").write_text("account-secret", encoding="utf-8")
            active_file.write_text("work\n", encoding="utf-8")
            codex_home.joinpath("config.toml").write_text(
                '''model_provider = "crs_local"

[model_providers.crs_local]
base_url = "http://127.0.0.1:3000/openai"
wire_api = "responses"
env_key = "CRS_OPENAI_KEY"
''',
                encoding="utf-8",
            )
            secrets.write_text("export CRS_OPENAI_KEY='provider-secret'\n", encoding="utf-8")

            state = backend.read_state(
                codex_home=codex_home,
                accounts_dir=accounts_dir,
                active_file=active_file,
                secrets_file=secrets,
            )

        self.assertEqual(state["mode"], "account")
        self.assertEqual(state["active_account"], "work")
        self.assertEqual(state["accounts"], [{"name": "work", "active": True, "logged_in": True}])
        self.assertEqual(state["providers"][0]["name"], "crs_local")
        self.assertTrue(state["providers"][0]["key_set"])
        self.assertNotIn("account-secret", repr(state))
        self.assertNotIn("provider-secret", repr(state))

    def test_recognizes_legacy_unnamed_account_and_custom_api_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / ".codex"
            codex_home.mkdir()
            config = codex_home / "config.toml"
            config.write_text('model_provider = "openai"\n', encoding="utf-8")
            codex_home.joinpath("auth.json").write_text(
                '{"auth_mode":"chatgpt","tokens":{}}\n', encoding="utf-8"
            )

            unnamed = backend.read_state(
                codex_home=codex_home,
                accounts_dir=root / "accounts",
                active_file=root / "missing-active",
                secrets_file=root / "missing-secrets",
            )
            config.write_text('model_provider = "crs_local"\n', encoding="utf-8")
            api = backend.read_state(
                codex_home=codex_home,
                accounts_dir=root / "accounts",
                active_file=root / "missing-active",
                secrets_file=root / "missing-secrets",
            )

        self.assertEqual(unnamed["mode"], "account")
        self.assertEqual(unnamed["active_account"], "unnamed")
        self.assertEqual(
            unnamed["accounts"],
            [{"name": "unnamed", "active": True, "logged_in": True, "legacy": True}],
        )
        self.assertEqual(api["mode"], "api")
        self.assertEqual(api["active_provider"], "crs_local")
        self.assertEqual(
            api["accounts"],
            [{"name": "unnamed", "active": False, "logged_in": True, "legacy": True}],
        )

    def test_uses_the_login_email_as_the_legacy_account_display_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home = root / ".codex"
            codex_home.mkdir()
            codex_home.joinpath("config.toml").write_text(
                'model_provider = "openai"\n', encoding="utf-8"
            )
            payload = base64.urlsafe_b64encode(
                json.dumps(
                    {"https://api.openai.com/profile": {"email": "user@example.com"}}
                ).encode()
            ).decode().rstrip("=")
            codex_home.joinpath("auth.json").write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "tokens": {"id_token": f"header.{payload}.signature"},
                    }
                ),
                encoding="utf-8",
            )

            state = backend.read_state(
                codex_home=codex_home,
                accounts_dir=root / "accounts",
                active_file=root / "missing-active",
                secrets_file=root / "missing-secrets",
            )

        self.assertEqual(state["accounts"][0]["email"], "user@example.com")

    def test_parses_all_quota_windows_and_selects_the_longest_for_overlay(self):
        output = json.dumps(
            {
                "account": "work",
                "plan_type": "plus",
                "rate_limits": [
                    {
                        "name": "Codex",
                        "windows": [
                            {"name": "primary", "remaining_percent": 75, "window_seconds": 18000, "resets_at": 2000},
                            {"name": "secondary", "remaining_percent": 40, "window_seconds": 604800, "resets_at": 3000},
                        ],
                    }
                ],
            }
        )

        quota = backend.parse_quota(output)

        self.assertEqual(quota["account"], "work")
        self.assertEqual([window["label"] for window in quota["windows"]], ["5h", "7d"])
        self.assertEqual(quota["overlay_window"]["remaining_percent"], 40.0)

    def test_formats_active_account_row_with_longest_quota_window(self):
        account = {
            "name": "unnamed",
            "email": "user@example.com",
            "active": True,
            "logged_in": True,
            "legacy": True,
        }
        quota = {
            "account": "unnamed",
            "overlay_window": {
                "label": "7d",
                "remaining_percent": 75.0,
                "resets_at": 91_000,
            },
        }

        self.assertEqual(
            backend.format_account_row(account, quota, now=1_000),
            "● user@example.com · logged in · legacy · "
            "7d: 75% left · resets in 1d 1h",
        )
        account["active"] = False
        self.assertEqual(
            backend.format_account_row(account, quota, now=1_000),
            "○ user@example.com · logged in · legacy",
        )

    def test_qt_application_is_native_and_does_not_use_the_web_console(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertIn("QMainWindow", source)
        self.assertIn("QSystemTrayIcon", source)
        self.assertIn("class QuotaOverlay", source)
        self.assertIn('["use", name]', source)
        self.assertIn('["account"]', source)
        self.assertIn('["api", name]', source)
        self.assertIn('["add", name]', source)
        self.assertIn('["add-auto"]', source)
        self.assertNotIn('["add", name, "--device-auth"]', source)
        self.assertNotIn('["add-auto", "--device-auth"]', source)
        self.assertIn('QPushButton("Add browser login")', source)
        self.assertIn('QPushButton("Cancel login")', source)
        self.assertIn("def cancel_account_login", source)
        self.assertIn("os.killpg", source)
        self.assertIn('process.setProperty("separateProcessGroup", True)', source)
        self.assertIn('["remove", name, "--yes"]', source)
        self.assertIn('command_path("codex-usage")', source)
        self.assertIn('QLineEdit.Password', source)
        self.assertIn('["set-key", name, "--stdin"]', source)
        self.assertIn('["test", name, "--base-url", base_url', source)
        self.assertIn('["delete", name, "--yes"]', source)
        self.assertNotIn('"--api-key"', source)
        self.assertIn("Optional email", source)
        self.assertIn('account.get("email")', source)
        self.assertNotIn("QWebEngine", source)
        self.assertNotIn("127.0.0.1:8320", source)
        self.assertNotIn("claudex-manager.service", source)

    def test_quota_card_can_show_a_draggable_desktop_overlay(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertIn('QPushButton("Show on desktop")', source)
        self.assertIn("self.overlay_button.clicked.connect(self.toggle_overlay)", source)
        self.assertIn("def toggle_overlay", source)
        self.assertIn('"Hide from desktop" if visible else "Show on desktop"', source)
        self.assertIn("Qt.X11BypassWindowManagerHint", source)
        self.assertIn('"overlayBypassPositioned", False, type=bool', source)
        self.assertIn('settings.setValue("overlayBypassPositioned", True)', source)
        self.assertIn("layout.setContentsMargins(12, 5, 12, 5)", source)
        self.assertIn("screen.geometry().top() + 4", source)
        self.assertIn("target.installEventFilter(self)", source)
        self.assertIn("def eventFilter", source)
        self.assertIn("self.grabMouse()", source)
        self.assertIn("self.releaseMouse()", source)
        self.assertIn("def bounded_position", source)
        self.assertIn("max(geometry.left(), min(position.x(), maximum_x))", source)
        self.assertIn("max(geometry.top(), min(position.y(), maximum_y))", source)
        self.assertIn("self.move(self.bounded_position", source)
        self.assertIn("def mousePressEvent", source)
        self.assertIn("def mouseMoveEvent", source)
        self.assertIn('self.settings.setValue("overlayPosition", self.pos())', source)

    def test_qt_application_uses_a_professional_desktop_visual_hierarchy(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertIn('central.setObjectName("appRoot")', source)
        self.assertIn('setProperty("role", "primary")', source)
        self.assertIn('setProperty("role", "danger")', source)
        self.assertIn('QFrame#statusBar', source)
        self.assertIn('QWidget#appRoot { background: #f4f7fb;', source)
        self.assertIn('QPushButton[role="primary"]', source)
        self.assertNotIn("#f4f1e8", source)

    def test_qt_application_handles_terminal_shutdown_signals(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertIn("signal.signal(signal.SIGINT, request_shutdown)", source)
        self.assertIn("signal.signal(signal.SIGTERM, request_shutdown)", source)
        self.assertIn("signal_timer = QTimer(app)", source)
        self.assertIn("signal_timer.start(200)", source)

    def test_api_provider_page_keeps_provider_details_and_actions_readable(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertIn("setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)", source)
        self.assertIn('QPushButton("Use provider")', source)
        self.assertIn('QPushButton("New")', source)
        self.assertIn('QPushButton("Delete")', source)
        self.assertIn("item.setSizeHint(QSize(0, 58))", source)
        self.assertIn("self.provider_list.setItemWidget(item, provider_widget)", source)
        self.assertIn('setObjectName("providerName")', source)
        self.assertIn('setObjectName("providerUrl")', source)
        self.assertIn('setObjectName("activeBadge")', source)
        self.assertIn("editor_layout.setAlignment(Qt.AlignTop)", source)
        self.assertNotIn(
            "provider_layout.addWidget(self.provider_delete, 0, Qt.AlignRight)",
            source,
        )

    def test_account_manager_uses_its_own_scalable_application_icon(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertTrue(ICON_PATH.is_file())
        icon = ICON_PATH.read_text(encoding="utf-8")
        self.assertIn("<linearGradient", icon)
        self.assertIn("<path", icon)
        self.assertIn('"codex-account-manager.svg"', source)

    def test_provider_test_log_expands_before_enabling_scrollbars(self):
        source = QT_APP_PATH.read_text(encoding="utf-8")

        self.assertIn("def _resize_provider_output", source)
        self.assertIn("documentSizeChanged.connect", source)
        self.assertIn("PROVIDER_OUTPUT_MAX_HEIGHT = 180", source)
        self.assertIn("Qt.ScrollBarAlwaysOff", source)
        self.assertIn("def _fit_provider_output", source)
        self.assertIn("self.minimumSizeHint().height()", source)
        self.assertNotIn("self.provider_output.setFixedHeight(90)", source)


if __name__ == "__main__":
    unittest.main()

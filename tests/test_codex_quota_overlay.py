"""Quota overlay component checks without showing desktop windows."""

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude"))

try:
    from PyQt5.QtCore import QSettings, QProcess, QPoint, QRect, QUrl
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from codex_account_manager_qt import MainWindow, QuotaOverlay
except ImportError:
    QuotaOverlay = None


@unittest.skipIf(QuotaOverlay is None, "PyQt5 is not installed")
class QuotaOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.directory.name) / "settings.ini"), QSettings.IniFormat)
        self.overlay = QuotaOverlay(settings)
        self.clock = patch("codex_account_manager_backend.time.time", return_value=1_000_000)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.overlay.close()
        self.overlay.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def quota(self, remaining=98, hours=166, window_seconds=604800):
        window = {
            "label": "7d", "remaining_percent": remaining,
            "window_seconds": window_seconds, "resets_at": 1_000_000 + hours * 3600,
        }
        return {"account": "demo", "plan_type": "pro", "windows": [window], "overlay_window": window}

    def test_budget_details_and_pace_update(self):
        self.overlay.set_quota(self.quota())
        self.assertFalse(self.overlay.budget_panel.isHidden())
        self.assertEqual(self.overlay.budget_label.text(), "Daily budget: 14.2% · On track · Pause 1h 22m to get on pace")
        self.assertAlmostEqual(self.overlay.pace_bar.target, 98.8095238095)
        self.assertTrue(self.overlay.budget_details.isHidden())
        self.overlay.details_button.click()
        self.assertFalse(self.overlay.budget_details.isHidden())
        self.assertIn("Percentage of your total weekly quota.", self.overlay.budget_details.text())
        self.overlay.set_quota(self.quota(30, 96))
        self.assertEqual(self.overlay.budget_label.text(), "Daily budget: 7.5% · Over pace by 27.1% · Pause 1d 21h 36m to get on pace")
        self.assertEqual(self.overlay.pace_bar.color, "#ffbf69")
        self.overlay.details_button.click()
        self.assertTrue(self.overlay.budget_details.isHidden())

    def wait_for_tokens(self):
        for _ in range(200):
            self.app.processEvents()
            if self.overlay.token_process.state() == QProcess.NotRunning:
                return
            QTest.qWait(10)
        self.fail("Token reader did not finish")

    def test_token_expansion_reads_latest_session_and_refreshes(self):
        root = Path(self.directory.name)
        sessions = root / "sessions"
        sessions.mkdir()
        old = sessions / "old.jsonl"
        old.write_text("")
        os.utime(old, (1, 1))
        path = sessions / "latest.jsonl"
        events = [
            {"type": "response_item", "payload": {"role": "user", "content": [
                {"type": "input_text", "text": "问题 <b>原文</b>"}]}},
            {"type": "turn_context", "payload": {"turn_id": "t", "model": "model-a"}},
            {"type": "token_usage_record", "payload": {"turn_id": "t", "response_id": "r", "usage": {
                "input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 20, "total_tokens": 1020}}},
        ]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        with patch.dict(os.environ, {"CODEX_HOME": str(root)}):
            self.assertTrue(self.overlay.token_details.isHidden())
            self.overlay.tokens_button.click()
            self.assertFalse(self.overlay.token_details.isHidden())
            self.assertTrue(self.overlay.token_refresh_timer.isActive())
            self.wait_for_tokens()
            text = self.overlay.token_details.toPlainText()
            for expected in ("问题 <b>原文</b>", "model-a", "1,020", "合计", "仅上方问题这一轮"):
                self.assertIn(expected, text)
            self.assertEqual(self.overlay.token_details.toolTip(), str(path))
            self.overlay.tokens_button.click()
            self.assertTrue(self.overlay.token_details.isHidden())
            self.assertFalse(self.overlay.token_refresh_timer.isActive())
            events[0]["payload"]["content"][0]["text"] = "新问题"
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            self.overlay.tokens_button.click()
            self.wait_for_tokens()
            self.assertIn("新问题", self.overlay.token_details.toPlainText())

    def test_open_token_panel_refreshes_tool_calls_without_reopening(self):
        root = Path(self.directory.name)
        (root / "sessions").mkdir()
        path = root / "sessions" / "latest.jsonl"
        events = [{"type": "turn_context", "payload": {"turn_id": "turn", "model": "model-a"}}]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        self.overlay.token_refresh_timer.setInterval(50)
        with patch.dict(os.environ, {"CODEX_HOME": str(root)}):
            self.overlay.tokens_button.click()
            self.wait_for_tokens()
            self.assertIn("本轮工具调用：0 次", self.overlay.token_details.toPlainText())
            with path.open("a") as stream:
                stream.write(json.dumps({"type": "response_item", "payload": {
                    "type": "function_call", "call_id": "call-1", "name": "search",
                    "arguments": '{"query":"查询 <原文>"}'}}) + "\n")
            for _ in range(100):
                QTest.qWait(10)
                if "本轮工具调用：1 次" in self.overlay.token_details.toPlainText():
                    break
            text = self.overlay.token_details.toPlainText()
            self.assertIn("本轮工具调用：1 次", text)
            self.assertIn("查询 <原文>", text)
            self.assertIn("search", text)
            self.overlay.tokens_button.click()
            self.wait_for_tokens()

    def test_token_refresh_does_not_overlap_processes(self):
        with patch.object(self.overlay.token_process, "start") as start:
            self.overlay.tokens_button.setChecked(True)
            start.reset_mock()
            with patch.object(self.overlay.token_process, "state", return_value=QProcess.Running):
                self.overlay.refresh_token_usage()
            start.assert_not_called()

    def test_request_token_rows_and_weighted_cache_hit(self):
        first = {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10, "total_tokens": 110}
        second = {"input_tokens": 900, "cached_input_tokens": 900, "output_tokens": 20, "total_tokens": 920}
        empty = dict.fromkeys(first, 0)
        report = {"path": "test.jsonl", "question": "测试问题", "models": {
            "model-a": first, "model-b": second}, "tools": [], "requests": [
                {"model": "model-a", "action": "执行本地命令", "usage": first}, {"model": "model-b", "usage": second},
                {"model": "model-b", "usage": empty}]}
        with patch.object(self.overlay.token_process, "readAllStandardOutput", return_value=json.dumps(report).encode()), \
                patch.object(self.overlay.token_process, "readAllStandardError", return_value=b""):
            self.overlay.token_usage_finished(0, QProcess.NormalExit)
        text = self.overlay.token_details.toPlainText()
        for expected in ("Cache hit", "90.0%", "100.0%", "0.0%", "N/A", "本轮模型请求：3 次", "#1 model-a", "#2 model-b", "#3 model-b"):
            self.assertIn(expected, text)
        self.assertNotIn("50.0%", text)
        self.assertIn("本次操作", text)
        self.assertIn("执行本地命令", text)

    def test_tool_content_previews_two_lines_and_preserves_expansion(self):
        report = {"path": "test.jsonl", "question": "问题", "models": {}, "tools": [
            {"id": "call-a", "name": "exec", "content": "第一行\n第二行\n第三行完整内容"}]}
        self.overlay.render_token_report(report)
        text = self.overlay.token_details.toPlainText()
        self.assertIn("第一行\n第二行…", text)
        self.assertNotIn("第三行完整内容", text)
        self.overlay.token_details.anchorClicked.emit(QUrl("tool:0"))
        self.assertIn("第三行完整内容", self.overlay.token_details.toPlainText())
        refreshed = {**report, "tools": report["tools"] + [
            {"id": "call-b", "name": "search", "content": "A\nB\n默认隐藏"}]}
        self.overlay.render_token_report(refreshed)
        text = self.overlay.token_details.toPlainText()
        self.assertIn("第三行完整内容", text)
        self.assertNotIn("默认隐藏", text)
        self.overlay.token_details.anchorClicked.emit(QUrl("tool:0"))
        self.assertNotIn("第三行完整内容", self.overlay.token_details.toPlainText())
        preview, truncated = self.overlay.tool_preview("很长的单行查询" * 100)
        self.assertEqual(len(preview.splitlines()), 2)
        self.assertTrue(truncated)

    def test_short_tool_content_has_no_expand_link(self):
        report = {"path": "test.jsonl", "question": "问题", "models": {}, "tools": [
            {"id": "one", "name": "exec", "content": "一行"},
            {"id": "two", "name": "exec", "content": "第一行\n第二行"},
            {"id": "three", "name": "exec", "content": "第一行\n第二行\n第三行"}]}
        self.overlay.render_token_report(report)
        markup = self.overlay.token_details.toHtml()
        self.assertNotIn('href="tool:0"', markup)
        self.assertNotIn('href="tool:1"', markup)
        self.assertIn('href="tool:2"', markup)

    def test_token_expansion_without_logs(self):
        with patch.dict(os.environ, {"CODEX_HOME": self.directory.name}):
            self.overlay.tokens_button.click()
            self.wait_for_tokens()
        self.assertIn("没有找到会话日志", self.overlay.token_details.toPlainText())

    def test_overlay_preserves_position_at_screen_top(self):
        screen = Mock()
        screen.geometry.return_value = QRect(0, 0, 1920, 1080)
        screen.availableGeometry.return_value = QRect(0, 32, 1920, 1048)
        with patch("codex_account_manager_qt.QApplication.screenAt", return_value=screen):
            self.assertEqual(self.overlay.bounded_position(QPoint(100, 0)).y(), 0)

    def test_panel_click_expands_once_and_ignores_release_outside(self):
        self.overlay.show()
        self.app.processEvents()
        button = self.overlay.tokens_button
        position = button.mapToGlobal(button.rect().center())
        with patch.object(self.overlay.token_process, "start"):
            self.overlay.handle_panel_click(position, True)
            self.overlay.handle_panel_click(position, True)
            self.assertFalse(button.isChecked())
            self.overlay.handle_panel_click(position, False)
            self.assertTrue(button.isChecked())
            position = button.mapToGlobal(button.rect().center())
            self.overlay.handle_panel_click(position, True)
            self.overlay.handle_panel_click(QPoint(-100, -100), False)
            self.assertTrue(button.isChecked())
            self.overlay.panel_native_click = True
            self.overlay.handle_panel_click(position, True)
            self.overlay.handle_panel_click(position, False)
            self.assertTrue(button.isChecked())

    def test_token_collapse_restores_compact_height(self):
        self.overlay.show()
        self.app.processEvents()
        with patch.object(self.overlay.token_process, "start"):
            self.overlay.tokens_button.click()
            self.app.processEvents()
            self.assertGreater(self.overlay.height(), 200)
            self.overlay.tokens_button.click()
            self.app.processEvents()
            self.assertLessEqual(self.overlay.height(), 48)

    def test_pause_countdown_updates_and_clears_at_baseline(self):
        self.overlay.set_quota(self.quota(75, 144))
        self.assertIn("Pause 18h 0m to get on pace", self.overlay.budget_label.text())
        with patch("codex_account_manager_backend.time.time", return_value=1_000_060):
            self.overlay.set_quota(self.quota(75, 144))
            self.assertIn("Pause 17h 59m to get on pace", self.overlay.budget_label.text())
        with patch("codex_account_manager_backend.time.time", return_value=1_064_800):
            self.overlay.set_quota(self.quota(75, 144))
            self.assertNotIn("Pause", self.overlay.budget_label.text())
        self.overlay.set_quota(self.quota(0, 24))
        self.assertIn("Wait 1d 0h 0m for reset", self.overlay.budget_label.text())

    def test_exhausted_short_remaining_and_reset_states(self):
        self.overlay.set_quota(self.quota(0, 24))
        self.assertIn("Quota exhausted", self.overlay.budget_label.text())
        self.assertEqual(self.overlay.pace_bar.color, "#ff8c95")
        self.overlay.set_quota(self.quota(10, 12))
        self.assertIn("Until reset: 10.0%", self.overlay.budget_label.text())
        self.overlay.set_quota(self.quota(10, 0))
        self.assertEqual(self.overlay.budget_label.text(), "Reset due · Awaiting refresh")
        self.assertTrue(self.overlay.pace_bar.isHidden())

    def test_api_error_and_nonweekly_quota_hide_budget(self):
        for clear in (
            lambda: self.overlay.set_api_mode("demo"),
            lambda: self.overlay.set_error("demo"),
            lambda: self.overlay.set_quota(self.quota(80, 4, 18000)),
        ):
            self.overlay.set_quota(self.quota())
            self.overlay.details_button.setChecked(True)
            clear()
            self.assertTrue(self.overlay.budget_panel.isHidden())
            self.assertTrue(self.overlay.budget_details.isHidden())

    def test_collapsed_overlay_has_two_compact_columns(self):
        self.overlay.set_quota(self.quota(30, 96))
        self.overlay.ensurePolished()
        self.overlay.adjustSize()
        self.overlay.layout().activate()
        frame = self.overlay.account_label.parentWidget()
        frame.resize(frame.sizeHint())
        frame.layout().activate()
        self.assertLessEqual(self.overlay.sizeHint().height(), 48)
        for label in (self.overlay.account_label, self.overlay.quota_label):
            self.assertLess(label.geometry().right(), self.overlay.budget_panel.geometry().left())
        self.assertEqual(
            self.overlay.account_label.geometry().left(),
            self.overlay.quota_label.geometry().left(),
        )
        self.assertGreater(
            self.overlay.quota_label.geometry().top(),
            self.overlay.account_label.geometry().bottom(),
        )

    def test_main_quota_card_shows_weekly_budget(self):
        window = Mock()
        window.state = {"mode": "account", "active_account": "demo"}
        window.account_display_name.side_effect = lambda name: name
        window.quota = self.quota()
        MainWindow.render_quota(window)
        self.assertIn(
            "Daily budget: 14.2% of total weekly quota · On track",
            window.quota_details.setText.call_args.args[0],
        )
        window.quota = self.quota(30, 96)
        MainWindow.render_quota(window)
        self.assertIn(
            "Daily budget: 7.5% of total weekly quota · Over pace by 27.1%",
            window.quota_details.setText.call_args.args[0],
        )
        window.quota = self.quota(10, 0)
        MainWindow.render_quota(window)
        self.assertIn("Reset due · Awaiting refresh", window.quota_details.setText.call_args.args[0])


if __name__ == "__main__":
    unittest.main()

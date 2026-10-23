"""Offscreen geometry checks; no services, credentials, or desktop windows."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude"))

try:
    from PyQt5.QtCore import QPoint, QRect
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication, QLabel, QMainWindow, QPushButton, QScrollArea,
        QStyle, QStyleOptionButton, QStyleOptionFrame,
    )
    import codex_account_manager_qt as manager
except ImportError:
    manager = None


@unittest.skipIf(manager is None, "PyQt5 is not installed")
class AccountManagerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setFont(QFont("DejaVu Sans", 12))

    def setUp(self):
        self.window = manager.MainWindow.__new__(manager.MainWindow)
        QMainWindow.__init__(self.window)
        self.window.icon = manager.application_icon()
        self.window._build_ui()
        self.window._apply_style()
        self.window.resize(980, 820)

    def tearDown(self):
        self.window.deleteLater()
        self.app.processEvents()

    def test_buttons_have_room_for_their_rendered_text(self):
        self.window.overlay_button.setText("Hide from desktop")
        self.window.show()
        for page in (0, 1):
            self.window.tabs.setCurrentIndex(page)
            self.app.processEvents()
            for button in self.window.findChildren(QPushButton):
                if not button.isVisible():
                    continue
                with self.subTest(page=page, text=button.text()):
                    option = QStyleOptionButton()
                    button.initStyleOption(option)
                    rect = button.style().subElementRect(
                        QStyle.SE_PushButtonContents, option, button
                    )
                    self.assertGreaterEqual(
                        rect.width(), button.fontMetrics().horizontalAdvance(button.text())
                    )

    def test_provider_row_reserves_height_for_all_labels(self):
        self.window.tabs.setCurrentIndex(1)
        provider = {
            "name": "example_" * 12, "base_url": "https://example.com/" + "long-path/" * 20,
            "key_set": True, "active": True, "env_key": "EXAMPLE_KEY",
        }
        listing = self.window.provider_list
        listing.blockSignals(True)
        self.window.state = {"mode": "api", "active_provider": provider["name"], "providers": [provider]}
        self.window.tray = Mock()
        with patch.object(self.window, "update_action_buttons"), patch.object(
            self.window, "_rebuild_tray_menu"
        ), patch.object(self.window, "render_quota"):
            self.window.render_state()
        row = listing.itemWidget(listing.item(0))
        self.window.show()
        self.app.processEvents()
        for label in row.findChildren(QLabel):
            with self.subTest(label=label.objectName()):
                self.assertGreaterEqual(label.height(), label.sizeHint().height())
                self.assertGreaterEqual(label.width(), label.sizeHint().width())
        label = row.findChild(QLabel, "providerName")
        self.assertNotEqual(label.text(), provider["name"])
        self.assertIn(provider["base_url"], row.toolTip())
        self.assertIn("Key ready", row.toolTip())
        self.assertEqual(
            {label.objectName() for label in row.findChildren(QLabel)},
            {"providerDot", "providerName"},
        )

    def test_form_labels_and_actions_fit_at_minimum_window_size(self):
        self.window.tabs.setCurrentIndex(1)
        self.window.resize(980, 760)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.width(), 980)
        self.assertEqual(self.window.height(), 760)
        self.window.resize(820, 640)
        self.app.processEvents()
        self.assertEqual(self.window.width(), 820)
        self.assertLessEqual(self.window.height(), 760)
        self.assertFalse(self.window.findChildren(QScrollArea))
        page = self.window.tabs.currentWidget()
        for widget in (
            self.window.provider_name, self.window.provider_base_url,
            self.window.provider_env_key, self.window.provider_key,
            self.window.provider_save, self.window.provider_test,
            self.window.provider_activate, self.window.provider_new,
            self.window.provider_delete,
        ):
            self.assertTrue(page.rect().contains(QRect(
                widget.mapTo(page, QPoint()), widget.size()
            )))
        for label in self.window.findChildren(QLabel):
            if label.isVisible() and not label.wordWrap():
                with self.subTest(text=label.text()):
                    self.assertGreaterEqual(label.width(), label.sizeHint().width())
                    self.assertGreaterEqual(label.height(), label.sizeHint().height())

    def test_provider_inputs_keep_text_height_when_window_shrinks(self):
        self.window.tabs.setCurrentIndex(1)
        self.window.resize(820, 640)
        self.window.show()
        self.app.processEvents()
        for field in (
            self.window.provider_name, self.window.provider_base_url,
            self.window.provider_env_key, self.window.provider_key,
        ):
            for focused in (False, True):
                with self.subTest(field=field.placeholderText(), focused=focused):
                    if focused:
                        field.setFocus()
                    else:
                        self.window.provider_save.setFocus()
                    self.app.processEvents()
                    option = QStyleOptionFrame()
                    field.initStyleOption(option)
                    content = field.style().subElementRect(
                        QStyle.SE_LineEditContents, option, field
                    )
                    self.assertGreaterEqual(content.height(), field.fontMetrics().height() + 2)


if __name__ == "__main__":
    unittest.main()

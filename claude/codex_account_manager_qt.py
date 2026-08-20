#!/usr/bin/env python3
"""Native PyQt5 desktop application for Codex account and API selection."""

from __future__ import annotations

import os
import re
import signal
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from PyQt5.QtCore import (
    QProcess,
    QProcessEnvironment,
    QSettings,
    QSize,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPixmap, QTextCursor
from PyQt5.QtNetwork import QLocalServer, QLocalSocket
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from codex_account_manager_backend import (
    ACCOUNT_NAME_PATTERN,
    extract_login_url,
    format_account_row,
    format_countdown,
    parse_quota,
    read_state,
)


APP_NAME = "Codex Account Manager"
APP_ID = "com.codex.account-manager"
STATE_REFRESH_MS = 2500
QUOTA_REFRESH_MS = 300_000
PROVIDER_OUTPUT_MIN_HEIGHT = 92
PROVIDER_OUTPUT_MAX_HEIGHT = 180
PROXY_ENV_FILE = Path.home() / ".cli-proxy-api/proxy.env"


def application_icon() -> QIcon:
    icon_paths = (
        Path(__file__).with_name("codex-account-manager.svg"),
        Path.home()
        / ".local/share/icons/hicolor/scalable/apps/codex-account-manager.svg",
    )
    for icon_path in icon_paths:
        if icon_path.is_file():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon

    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#2563eb"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 14, 14)
    painter.setPen(QColor("#ffffff"))
    font = QFont("DejaVu Sans", 30, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "C")
    painter.end()
    return QIcon(pixmap)


def command_path(name: str) -> str | None:
    user_command = Path.home() / ".local/bin" / name
    if user_command.is_file() and os.access(user_command, os.X_OK):
        return str(user_command)
    installed = shutil.which(name)
    if installed:
        return installed
    source_fallbacks = {
        "codex-auth": Path(__file__).with_name("switch-codex-auth.sh"),
        "codex-usage": Path(__file__).with_name("codex-usage"),
    }
    fallback = source_fallbacks.get(name)
    if fallback and fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    return None


def process_environment() -> QProcessEnvironment:
    environment = QProcessEnvironment.systemEnvironment()
    path = environment.value("PATH")
    user_bin = str(Path.home() / ".local/bin")
    if user_bin not in path.split(":"):
        environment.insert("PATH", f"{user_bin}:{path}" if path else user_bin)
    try:
        lines = PROXY_ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if "\n" not in value and "\r" not in value:
            environment.insert(name, value)
    return environment


class QuotaOverlay(QWidget):
    open_manager = pyqtSignal()
    refresh_requested = pyqtSignal()
    hide_requested = pyqtSignal()

    def __init__(self, settings: QSettings) -> None:
        super().__init__(
            None,
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint,
        )
        self.settings = settings
        self.drag_offset = None
        self.setObjectName("quotaOverlay")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("Codex quota")
        self.setMinimumWidth(255)

        frame = QFrame(self)
        frame.setObjectName("overlayFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(3)
        self.account_label = QLabel("Codex · checking")
        self.account_label.setObjectName("overlayAccount")
        self.quota_label = QLabel("Quota: --")
        self.quota_label.setObjectName("overlayQuota")
        layout.addWidget(self.account_label)
        layout.addWidget(self.quota_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        self.setStyleSheet(
            """
            QFrame#overlayFrame { background: #111827; border: 1px solid #334155; border-radius: 12px; }
            QLabel#overlayAccount { color: #93c5fd; font: 700 11px 'Ubuntu Sans'; }
            QLabel#overlayQuota { color: #e2e8f0; font: 10px 'Ubuntu Sans'; }
            """
        )
        position = settings.value("overlayPosition")
        if position is not None:
            self.move(position)
        else:
            self.move(280, 8)

    def set_quota(self, quota: dict[str, Any]) -> None:
        plan = f" · {quota['plan_type']}" if quota.get("plan_type") else ""
        window = quota["overlay_window"]
        self.account_label.setText(f"Codex · {quota['account']}{plan}")
        self.quota_label.setText(
            f"{window['label']}: {window['remaining_percent']:g}% left · "
            f"resets in {format_countdown(window['resets_at'])}"
        )
        self.adjustSize()

    def set_api_mode(self, provider: str) -> None:
        self.account_label.setText(f"Codex · API · {provider or 'provider'}")
        self.quota_label.setText("No account quota")
        self.adjustSize()

    def set_error(self, account: str, message: str = "Quota unavailable") -> None:
        self.account_label.setText(f"Codex · {account or 'account'}")
        self.quota_label.setText(message)
        self.adjustSize()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton:
            self.drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self.drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton:
            self.drag_offset = None
            self.settings.setValue("overlayPosition", self.pos())
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton:
            self.open_manager.emit()

    def contextMenuEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        menu = QMenu(self)
        menu.addAction("Open manager", self.open_manager.emit)
        menu.addAction("Refresh quota", self.refresh_requested.emit)
        menu.addSeparator()
        menu.addAction("Hide overlay", self.hide_requested.emit)
        menu.exec_(event.globalPos())


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication, settings: QSettings) -> None:
        super().__init__()
        self.app = app
        self.settings = settings
        self.state: dict[str, Any] = {}
        self.state_signature = ""
        self.quota: dict[str, Any] | None = None
        self.action_process: QProcess | None = None
        self.action_buffer = ""
        self.action_output_widget: QPlainTextEdit | None = None
        self.action_success_message = ""
        self.action_success_callback: Callable[[], None] | None = None
        self.action_clear_account_name = False
        self.action_command_name = "command"
        self.action_cancel_requested = False
        self.quota_process: QProcess | None = None
        self.quota_pending = False
        self.quitting = False
        self.last_login_url = ""
        self.provider_editing_existing = False
        self.icon = application_icon()
        self.setWindowIcon(self.icon)
        self.setWindowTitle(APP_NAME)
        self.resize(980, 760)
        self.setMinimumSize(820, 640)

        self.overlay = QuotaOverlay(settings)
        self.overlay.open_manager.connect(self.show_manager)
        self.overlay.refresh_requested.connect(self.refresh_quota)
        self.overlay.hide_requested.connect(lambda: self.set_overlay_visible(False))
        self._build_ui()
        self._build_tray()
        self._apply_style()

        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.refresh_state)
        self.state_timer.start(STATE_REFRESH_MS)
        self.quota_timer = QTimer(self)
        self.quota_timer.timeout.connect(self.refresh_quota)
        self.quota_timer.start(QUOTA_REFRESH_MS)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.render_quota)
        self.countdown_timer.start(60_000)

        self.refresh_state(force=True)
        overlay_visible = settings.value("overlayVisible", True, type=bool)
        self.set_overlay_visible(overlay_visible)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(18)

        header = QHBoxLayout()
        header.setSpacing(14)
        brand = QLabel()
        brand.setObjectName("brandMark")
        brand.setAlignment(Qt.AlignCenter)
        brand.setFixedSize(48, 48)
        brand.setPixmap(self.icon.pixmap(48, 48))
        header.addWidget(brand, 0, Qt.AlignTop)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel("Manage account profiles, API providers, and usage from one desktop app")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        mode_box = QVBoxLayout()
        mode_box.setSpacing(5)
        mode_caption = QLabel("ACTIVE FOR NEW PROCESSES")
        mode_caption.setObjectName("modeCaption")
        mode_caption.setAlignment(Qt.AlignRight)
        self.mode_label = QLabel("Checking current mode…")
        self.mode_label.setObjectName("modePill")
        mode_box.addWidget(mode_caption)
        mode_box.addWidget(self.mode_label, 0, Qt.AlignRight)
        header.addLayout(mode_box)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_accounts_tab(), "Accounts")
        self.tabs.addTab(self._build_api_tab(), "API Providers")
        root.addWidget(self.tabs, 1)

        quota_group = QGroupBox("USAGE & QUOTA")
        quota_group.setObjectName("quotaCard")
        quota_layout = QVBoxLayout(quota_group)
        quota_layout.setContentsMargins(18, 24, 18, 16)
        quota_layout.setSpacing(7)
        quota_head = QHBoxLayout()
        self.quota_summary = QLabel("Waiting for account state…")
        self.quota_summary.setObjectName("quotaSummary")
        refresh_button = QPushButton("Refresh")
        refresh_button.setProperty("role", "quiet")
        refresh_button.clicked.connect(self.refresh_quota)
        quota_head.addWidget(self.quota_summary, 1)
        quota_head.addWidget(refresh_button)
        self.quota_details = QLabel("")
        self.quota_details.setObjectName("quotaDetails")
        self.quota_details.setWordWrap(True)
        self.quota_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        quota_layout.addLayout(quota_head)
        quota_layout.addWidget(self.quota_details)
        root.addWidget(quota_group)

        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        status_layout.addWidget(self.status_label, 1)
        local_label = QLabel("LOCAL DESKTOP")
        local_label.setObjectName("localStatus")
        status_layout.addWidget(local_label)
        root.addWidget(status_bar)
        self.setCentralWidget(central)

    def _build_accounts_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        section_title = QLabel("ChatGPT account profiles")
        section_title.setObjectName("sectionTitle")
        section_note = QLabel(
            "Choose the identity used by future Codex processes. Existing terminals keep their current account."
        )
        section_note.setObjectName("sectionNote")
        section_note.setWordWrap(True)
        layout.addWidget(section_title)
        layout.addWidget(section_note)

        self.account_list = QListWidget()
        self.account_list.setObjectName("resourceList")
        self.account_list.setSpacing(2)
        self.account_list.itemDoubleClicked.connect(lambda _: self.activate_selected_account())
        self.account_list.currentItemChanged.connect(self.update_action_buttons)
        layout.addWidget(self.account_list, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.account_activate = QPushButton("Use selected account")
        self.account_activate.setProperty("role", "primary")
        self.account_activate.clicked.connect(self.activate_selected_account)
        self.account_quota = QPushButton("View quota")
        self.account_quota.setProperty("role", "quiet")
        self.account_quota.clicked.connect(self.refresh_quota)
        self.account_remove = QPushButton("Remove account")
        self.account_remove.setProperty("role", "danger")
        self.account_remove.clicked.connect(self.remove_selected_account)
        actions.addWidget(self.account_activate)
        actions.addWidget(self.account_quota)
        actions.addStretch(1)
        actions.addWidget(self.account_remove)
        layout.addLayout(actions)

        add_group = QGroupBox("ADD ACCOUNT")
        add_group.setObjectName("actionCard")
        add_layout = QVBoxLayout(add_group)
        add_layout.setContentsMargins(16, 24, 16, 14)
        add_layout.setSpacing(9)
        add_note = QLabel(
            "Sign in with your browser. Leave the email blank to name the profile from the authenticated account."
        )
        add_note.setObjectName("cardNote")
        add_note.setWordWrap(True)
        add_layout.addWidget(add_note)
        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        self.account_name = QLineEdit()
        self.account_name.setPlaceholderText(
            "Optional email address"
        )
        self.account_name.setMaxLength(128)
        self.account_add = QPushButton("Add browser login")
        self.account_add.setProperty("role", "primary")
        self.account_add.clicked.connect(self.start_account_login)
        self.account_cancel = QPushButton("Cancel login")
        self.account_cancel.setProperty("role", "danger")
        self.account_cancel.clicked.connect(self.cancel_account_login)
        self.account_cancel.setEnabled(False)
        add_row.addWidget(self.account_name, 1)
        add_row.addWidget(self.account_add)
        add_row.addWidget(self.account_cancel)
        self.login_link = QLabel("")
        self.login_link.setObjectName("actionLink")
        self.login_link.setOpenExternalLinks(False)
        self.login_link.linkActivated.connect(
            lambda value: QDesktopServices.openUrl(QUrl(value))
        )
        self.login_output = QPlainTextEdit()
        self.login_output.setReadOnly(True)
        self.login_output.setPlaceholderText("Browser login progress will appear here.")
        self.login_output.setMaximumBlockCount(300)
        self.login_output.setFixedHeight(92)
        self.login_output.setVisible(False)
        add_layout.addLayout(add_row)
        add_layout.addWidget(self.login_link)
        add_layout.addWidget(self.login_output)
        layout.addWidget(add_group)
        return page

    def _build_api_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("tabPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        section_title = QLabel("API provider profiles")
        section_title.setObjectName("sectionTitle")
        note = QLabel(
            "Select a compatible API endpoint for future Codex processes, or securely manage a provider profile."
        )
        note.setObjectName("sectionNote")
        note.setWordWrap(True)
        layout.addWidget(section_title)
        layout.addWidget(note)

        content = QHBoxLayout()
        content.setSpacing(14)
        provider_panel = QFrame()
        provider_panel.setObjectName("innerPanel")
        provider_panel.setMinimumWidth(310)
        provider_layout = QVBoxLayout(provider_panel)
        provider_layout.setContentsMargins(14, 14, 14, 14)
        provider_layout.setSpacing(10)
        provider_heading = QLabel("Configured providers")
        provider_heading.setObjectName("panelTitle")
        provider_layout.addWidget(provider_heading)
        self.provider_list = QListWidget()
        self.provider_list.setObjectName("resourceList")
        self.provider_list.setSpacing(2)
        self.provider_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.provider_list.setTextElideMode(Qt.ElideRight)
        self.provider_list.itemDoubleClicked.connect(lambda _: self.activate_selected_provider())
        self.provider_list.currentItemChanged.connect(self.load_selected_provider)
        provider_layout.addWidget(self.provider_list, 1)

        provider_actions = QHBoxLayout()
        provider_actions.setSpacing(8)
        self.provider_activate = QPushButton("Use provider")
        self.provider_activate.setProperty("role", "primary")
        self.provider_activate.setToolTip("Use this provider for future Codex processes")
        self.provider_activate.clicked.connect(self.activate_selected_provider)
        self.provider_new = QPushButton("New")
        self.provider_new.setProperty("role", "quiet")
        self.provider_new.setToolTip("Create a new API provider")
        self.provider_new.clicked.connect(self.clear_provider_form)
        self.provider_delete = QPushButton("Delete")
        self.provider_delete.setProperty("role", "danger")
        self.provider_delete.setToolTip("Delete the selected API provider")
        self.provider_delete.clicked.connect(self.delete_selected_provider)
        provider_actions.addWidget(self.provider_activate, 1)
        provider_actions.addWidget(self.provider_new)
        provider_actions.addWidget(self.provider_delete)
        provider_layout.addLayout(provider_actions)

        editor = QGroupBox("PROVIDER SETTINGS")
        editor.setObjectName("editorCard")
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(18, 26, 18, 16)
        editor_layout.setSpacing(12)
        editor_layout.setAlignment(Qt.AlignTop)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(11)
        self.provider_name = QLineEdit()
        self.provider_name.setPlaceholderText("provider_name")
        self.provider_name.setMaxLength(64)
        self.provider_base_url = QLineEdit()
        self.provider_base_url.setPlaceholderText("https://api.example.com/v1")
        self.provider_env_key = QLineEdit()
        self.provider_env_key.setPlaceholderText("PROVIDER_OPENAI_KEY")
        self.provider_key = QLineEdit()
        self.provider_key.setEchoMode(QLineEdit.Password)
        self.provider_key.setPlaceholderText("Leave blank to keep the stored key")
        form.addRow("Name", self.provider_name)
        form.addRow("Base URL", self.provider_base_url)
        form.addRow("Environment key", self.provider_env_key)
        form.addRow("API key", self.provider_key)
        editor_layout.addLayout(form)

        security_note = QLabel("API keys are stored locally and never shown in the provider list.")
        security_note.setObjectName("cardNote")
        security_note.setWordWrap(True)
        editor_layout.addWidget(security_note)

        editor_actions = QHBoxLayout()
        editor_actions.setSpacing(8)
        self.provider_save = QPushButton("Save API")
        self.provider_save.setProperty("role", "primary")
        self.provider_save.clicked.connect(self.save_provider)
        self.provider_test = QPushButton("Test connection")
        self.provider_test.setProperty("role", "quiet")
        self.provider_test.clicked.connect(self.test_provider)
        editor_actions.addWidget(self.provider_save)
        editor_actions.addWidget(self.provider_test)
        editor_actions.addStretch(1)
        editor_layout.addLayout(editor_actions)

        self.provider_output = QPlainTextEdit()
        self.provider_output.setReadOnly(True)
        self.provider_output.setPlaceholderText("Save and connection-test results will appear here.")
        self.provider_output.setMaximumBlockCount(200)
        self.provider_output.setFixedHeight(PROVIDER_OUTPUT_MIN_HEIGHT)
        self.provider_output.document().documentLayout().documentSizeChanged.connect(
            lambda _size: QTimer.singleShot(0, self._resize_provider_output)
        )
        self.provider_output.setVisible(False)
        editor_layout.addWidget(self.provider_output)
        content.addWidget(provider_panel, 5)
        content.addWidget(editor, 7)
        layout.addLayout(content, 1)
        return page

    def _resize_provider_output(self) -> None:
        document = self.provider_output.document()
        visual_lines = 0
        block = document.firstBlock()
        while block.isValid():
            block_layout = block.layout()
            visual_lines += max(1, block_layout.lineCount() if block_layout else 1)
            block = block.next()
        visual_lines = max(document.blockCount(), visual_lines)
        content_height = visual_lines * self.provider_output.fontMetrics().lineSpacing() + 35
        overflow = content_height > PROVIDER_OUTPUT_MAX_HEIGHT
        scrollbar_policy = Qt.ScrollBarAsNeeded if overflow else Qt.ScrollBarAlwaysOff
        if self.provider_output.verticalScrollBarPolicy() != scrollbar_policy:
            self.provider_output.setVerticalScrollBarPolicy(scrollbar_policy)
        target_height = max(
            PROVIDER_OUTPUT_MIN_HEIGHT,
            min(PROVIDER_OUTPUT_MAX_HEIGHT, content_height),
        )
        if self.provider_output.height() != target_height:
            self.provider_output.setFixedHeight(target_height)
        if self.provider_output.isVisible():
            QTimer.singleShot(0, self._fit_provider_output)

    def _fit_provider_output(self) -> None:
        required_height = self.minimumSizeHint().height()
        if self.height() < required_height:
            self.resize(self.width(), required_height)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(self.icon, self)
        self.tray.setToolTip(APP_NAME)
        self.tray.activated.connect(self._tray_activated)
        self.tray_menu = QMenu()
        self.tray.setContextMenu(self.tray_menu)
        self._rebuild_tray_menu()
        self.tray.show()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f7fb; }
            QWidget#appRoot { background: #f4f7fb; color: #172033; font: 10pt 'Ubuntu Sans'; }
            QWidget#tabPage { background: transparent; }
            QLabel { background: transparent; }
            QLabel#brandMark { background: transparent; }
            QLabel#title { color: #0f172a; font-size: 20pt; font-weight: 800; }
            QLabel#subtitle { color: #64748b; font-size: 9.5pt; }
            QLabel#modeCaption {
                color: #94a3b8;
                font-size: 7.5pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#modePill {
                padding: 7px 12px;
                color: #1d4ed8;
                background: #dbeafe;
                border: 1px solid #bfdbfe;
                border-radius: 11px;
                font-weight: 700;
            }
            QLabel#sectionTitle { color: #0f172a; font-size: 13pt; font-weight: 800; }
            QLabel#sectionNote, QLabel#cardNote { color: #64748b; font-size: 9pt; }
            QLabel#panelTitle { color: #334155; font-size: 10pt; font-weight: 800; }
            QLabel#providerDot { color: #94a3b8; font-size: 12pt; }
            QLabel#providerDot[active="true"] { color: #2563eb; }
            QLabel#providerName { color: #1e293b; font-size: 10pt; font-weight: 800; }
            QLabel#providerUrl { color: #64748b; font-size: 8.5pt; }
            QLabel#keyBadge {
                padding: 2px 5px;
                color: #047857;
                background: #d1fae5;
                border-radius: 5px;
                font-size: 7pt;
                font-weight: 800;
            }
            QLabel#keyBadge[ready="false"] { color: #b45309; background: #fef3c7; }
            QLabel#activeBadge {
                padding: 2px 5px;
                color: #1d4ed8;
                background: #dbeafe;
                border-radius: 5px;
                font-size: 7pt;
                font-weight: 800;
            }
            QLabel#quotaSummary { color: #0f172a; font-size: 10.5pt; font-weight: 800; }
            QLabel#quotaDetails { color: #475569; font-size: 9pt; }
            QLabel#status { color: #475569; font-size: 9pt; }
            QLabel#localStatus {
                color: #64748b;
                font-size: 7.5pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#actionLink { color: #2563eb; font-weight: 700; }

            QTabWidget::pane {
                top: -1px;
                border: 1px solid #dbe3ee;
                border-radius: 12px;
                background: #ffffff;
            }
            QTabBar::tab {
                min-width: 130px;
                padding: 10px 18px;
                margin-right: 4px;
                color: #64748b;
                background: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                font-weight: 700;
            }
            QTabBar::tab:hover { color: #1e40af; background: #eff6ff; }
            QTabBar::tab:selected { color: #1d4ed8; border-bottom-color: #2563eb; }

            QGroupBox {
                margin-top: 11px;
                color: #64748b;
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 10px;
                font-size: 8pt;
                font-weight: 800;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                background: #ffffff;
            }
            QGroupBox#actionCard { background: #f8fafc; }
            QGroupBox#actionCard::title { background: #f8fafc; }
            QGroupBox#editorCard, QGroupBox#quotaCard { background: #ffffff; }
            QFrame#innerPanel {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
            QFrame#statusBar {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }

            QListWidget {
                padding: 5px;
                color: #334155;
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 9px;
                outline: 0;
            }
            QListWidget::item {
                min-height: 34px;
                padding: 9px 11px;
                margin: 2px;
                border: 1px solid transparent;
                border-radius: 7px;
            }
            QListWidget::item:hover { background: #f1f5f9; }
            QListWidget::item:selected {
                color: #1e3a8a;
                background: #eaf2ff;
                border-color: #bfdbfe;
            }

            QLineEdit, QPlainTextEdit {
                padding: 8px 10px;
                color: #172033;
                selection-color: white;
                selection-background-color: #2563eb;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
            }
            QLineEdit { min-height: 20px; }
            QLineEdit:hover, QPlainTextEdit:hover { border-color: #94a3b8; }
            QLineEdit:focus, QPlainTextEdit:focus { border: 2px solid #3b82f6; }
            QLineEdit:read-only { color: #64748b; background: #f8fafc; }
            QPlainTextEdit { font: 9pt 'Ubuntu Mono'; }

            QPushButton {
                min-height: 20px;
                padding: 8px 13px;
                color: #334155;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                font-weight: 700;
            }
            QPushButton:hover { color: #1d4ed8; background: #f8fafc; border-color: #93c5fd; }
            QPushButton:pressed { background: #eff6ff; }
            QPushButton[role="primary"] {
                color: #ffffff;
                background: #2563eb;
                border-color: #2563eb;
            }
            QPushButton[role="primary"]:hover { background: #1d4ed8; border-color: #1d4ed8; }
            QPushButton[role="primary"]:pressed { background: #1e40af; }
            QPushButton[role="danger"] { color: #b42318; background: #ffffff; border-color: #fecaca; }
            QPushButton[role="danger"]:hover { color: #991b1b; background: #fef2f2; border-color: #fca5a5; }
            QPushButton[role="quiet"] { color: #475569; background: #f8fafc; }
            QPushButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
                border-color: #e2e8f0;
            }

            QMenu {
                padding: 5px;
                color: #1e293b;
                background: #ffffff;
                border: 1px solid #cbd5e1;
            }
            QMenu::item { padding: 7px 24px 7px 10px; border-radius: 5px; }
            QMenu::item:selected { color: #1d4ed8; background: #eff6ff; }
            """
        )

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_manager()

    def _rebuild_tray_menu(self) -> None:
        self.tray_menu.clear()
        self.tray_menu.addAction("Open Codex Account Manager", self.show_manager)
        mode = self.current_mode_text()
        mode_action = self.tray_menu.addAction(mode)
        mode_action.setEnabled(False)
        self.tray_menu.addSeparator()

        account_menu = self.tray_menu.addMenu("Use account")
        for account in self.state.get("accounts", []):
            prefix = "✓ " if account["active"] else ""
            label = account.get("email") or account["name"]
            action = account_menu.addAction(f"{prefix}{label}")
            action.setEnabled(account["logged_in"] and not account["active"])
            if account.get("legacy"):
                action.triggered.connect(
                    lambda checked=False: self.use_unnamed_account()
                )
            else:
                action.triggered.connect(
                    lambda checked=False, name=account["name"]: self.use_account(name)
                )
        if account_menu.isEmpty():
            empty = account_menu.addAction("No named accounts")
            empty.setEnabled(False)

        api_menu = self.tray_menu.addMenu("Use API provider")
        for provider in self.state.get("providers", []):
            active = self.state.get("mode") == "api" and provider["active"]
            prefix = "✓ " if active else ""
            action = api_menu.addAction(f"{prefix}{provider['name']}")
            action.setEnabled(provider["key_set"] and not active)
            action.triggered.connect(
                lambda checked=False, name=provider["name"]: self.use_provider(name)
            )
        if api_menu.isEmpty():
            empty = api_menu.addAction("No API providers")
            empty.setEnabled(False)

        self.tray_menu.addSeparator()
        overlay_action = self.tray_menu.addAction("Show quota overlay")
        overlay_action.setCheckable(True)
        overlay_action.setChecked(self.overlay.isVisible())
        overlay_action.toggled.connect(self.set_overlay_visible)
        self.tray_menu.addAction("Refresh quota", self.refresh_quota)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction("Quit", self.quit_application)

    def current_mode_text(self) -> str:
        if self.state.get("mode") == "account":
            account = self.state.get("active_account") or "unknown"
            return f"Account · {self.account_display_name(account)}"
        return f"API · {self.state.get('active_provider') or 'unknown'}"

    def account_display_name(self, name: str) -> str:
        for account in self.state.get("accounts", []):
            if account["name"] == name:
                return str(account.get("email") or name)
        return name

    def _build_provider_row(self, provider: dict[str, Any], active: bool) -> QWidget:
        row = QWidget()
        row.setObjectName("providerItem")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(7, 5, 7, 5)
        layout.setSpacing(2)

        heading = QHBoxLayout()
        heading.setSpacing(6)
        dot = QLabel("●" if active else "○")
        dot.setObjectName("providerDot")
        dot.setProperty("active", active)
        name = QLabel(str(provider["name"]))
        name.setObjectName("providerName")
        heading.addWidget(dot)
        heading.addWidget(name, 1)
        if active:
            active_badge = QLabel("ACTIVE")
            active_badge.setObjectName("activeBadge")
            heading.addWidget(active_badge)
        key_badge = QLabel("KEY READY" if provider["key_set"] else "KEY MISSING")
        key_badge.setObjectName("keyBadge")
        key_badge.setProperty("ready", provider["key_set"])
        heading.addWidget(key_badge)

        url = QLabel(str(provider["base_url"]))
        url.setObjectName("providerUrl")
        url.setToolTip(str(provider["base_url"]))
        layout.addLayout(heading)
        layout.addWidget(url)
        return row

    def refresh_state(self, force: bool = False) -> None:
        try:
            state = read_state()
        except OSError as error:
            self.status_label.setText(f"Cannot read Codex state: {error}")
            return
        signature = repr(state)
        changed = signature != self.state_signature
        if not force and not changed:
            return
        previous_mode = (self.state.get("mode"), self.state.get("active_account"), self.state.get("active_provider"))
        self.state = state
        self.state_signature = signature
        self.render_state()
        current_mode = (state.get("mode"), state.get("active_account"), state.get("active_provider"))
        if force or current_mode != previous_mode:
            if state["mode"] == "account":
                self.refresh_quota()
            else:
                self.quota = None
                self.render_quota()

    def render_state(self) -> None:
        mode_text = self.current_mode_text()
        self.mode_label.setText(mode_text)
        self.tray.setToolTip(f"{APP_NAME}\n{mode_text}")

        selected_account = self.account_list.currentItem()
        selected_account_name = selected_account.data(Qt.UserRole) if selected_account else None
        self.account_list.clear()
        for account in self.state.get("accounts", []):
            item = QListWidgetItem(format_account_row(account, self.quota))
            item.setData(Qt.UserRole, account["name"])
            item.setData(Qt.UserRole + 1, account["logged_in"])
            item.setData(Qt.UserRole + 2, account["active"])
            item.setData(Qt.UserRole + 3, bool(account.get("legacy")))
            self.account_list.addItem(item)
            if account["name"] == selected_account_name or account["active"]:
                self.account_list.setCurrentItem(item)
        if self.account_list.currentRow() < 0 and self.account_list.count():
            self.account_list.setCurrentRow(0)

        selected_provider = self.provider_list.currentItem()
        selected_provider_name = selected_provider.data(Qt.UserRole) if selected_provider else None
        self.provider_list.clear()
        for provider in self.state.get("providers", []):
            active = self.state.get("mode") == "api" and provider["active"]
            key_state = "Key ready" if provider["key_set"] else "Key missing"
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 58))
            item.setToolTip(
                f"{provider['name']}\n{key_state}\n{provider['base_url']}"
            )
            item.setData(Qt.UserRole, provider["name"])
            item.setData(Qt.UserRole + 1, provider["key_set"])
            item.setData(Qt.UserRole + 2, active)
            item.setData(Qt.UserRole + 3, provider["base_url"])
            item.setData(Qt.UserRole + 4, provider["env_key"])
            self.provider_list.addItem(item)
            provider_widget = self._build_provider_row(provider, active)
            self.provider_list.setItemWidget(item, provider_widget)
            if provider["name"] == selected_provider_name or active:
                self.provider_list.setCurrentItem(item)
        if self.provider_list.currentRow() < 0 and self.provider_list.count():
            self.provider_list.setCurrentRow(0)

        self.update_action_buttons()
        self._rebuild_tray_menu()
        self.render_quota()

    def update_account_row_texts(self) -> None:
        accounts = {
            account["name"]: account for account in self.state.get("accounts", [])
        }
        for row in range(self.account_list.count()):
            item = self.account_list.item(row)
            account = accounts.get(item.data(Qt.UserRole))
            if account:
                item.setText(format_account_row(account, self.quota))

    def update_action_buttons(self, *_args) -> None:  # type: ignore[no-untyped-def]
        busy = self.action_process is not None
        login_busy = bool(
            self.action_process and self.action_process.property("streamLogin")
        )
        account = self.account_list.currentItem()
        account_ready = bool(account and account.data(Qt.UserRole + 1))
        account_active = bool(account and account.data(Qt.UserRole + 2))
        account_legacy = bool(account and account.data(Qt.UserRole + 3))
        provider = self.provider_list.currentItem()
        provider_ready = bool(provider and provider.data(Qt.UserRole + 1))
        provider_active = bool(provider and provider.data(Qt.UserRole + 2))
        self.account_activate.setEnabled(not busy and account_ready and not account_active)
        self.account_quota.setEnabled(
            not busy
            and account_ready
            and account_active
            and self.state.get("mode") == "account"
        )
        self.account_add.setEnabled(not busy)
        self.account_cancel.setEnabled(login_busy and not self.action_cancel_requested)
        self.account_remove.setEnabled(not busy and bool(account) and not account_legacy)
        self.provider_activate.setEnabled(not busy and provider_ready and not provider_active)
        self.provider_new.setEnabled(not busy)
        self.provider_save.setEnabled(not busy)
        self.provider_test.setEnabled(not busy)
        self.provider_delete.setEnabled(
            not busy and bool(provider) and self.provider_editing_existing and not provider_active
        )

    def render_quota(self) -> None:
        self.update_account_row_texts()
        if self.state.get("mode") == "api":
            provider = self.state.get("active_provider") or "provider"
            self.quota_summary.setText(f"API mode · {provider}")
            self.quota_details.setText("API providers do not expose ChatGPT account quota.")
            self.overlay.set_api_mode(provider)
            return
        account_name = self.state.get("active_account") or "account"
        account = self.account_display_name(account_name)
        if self.quota is None:
            self.quota_summary.setText(f"{account} · quota not loaded")
            self.quota_details.setText("")
            return
        plan = self.quota.get("plan_type") or "unknown plan"
        quota_account = self.account_display_name(str(self.quota["account"]))
        self.quota_summary.setText(f"{quota_account} · {plan}")
        lines = []
        for window in self.quota["windows"]:
            lines.append(
                f"{window['label']}: {window['remaining_percent']:g}% left · "
                f"resets in {format_countdown(window['resets_at'])}"
            )
        self.quota_details.setText("\n".join(lines))
        overlay_quota = dict(self.quota)
        overlay_quota["account"] = quota_account
        self.overlay.set_quota(overlay_quota)

    def selected_account_name(self) -> str | None:
        item = self.account_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def selected_provider_name(self) -> str | None:
        item = self.provider_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def load_selected_provider(self, item: QListWidgetItem | None, *_args) -> None:  # type: ignore[no-untyped-def]
        if item is None:
            self.update_action_buttons()
            return
        self.provider_editing_existing = True
        self.provider_name.setText(str(item.data(Qt.UserRole)))
        self.provider_name.setReadOnly(True)
        self.provider_base_url.setText(str(item.data(Qt.UserRole + 3) or ""))
        self.provider_env_key.setText(str(item.data(Qt.UserRole + 4) or ""))
        self.provider_key.clear()
        self.update_action_buttons()

    def clear_provider_form(self) -> None:
        self.provider_list.setCurrentRow(-1)
        self.provider_editing_existing = False
        self.provider_name.setReadOnly(False)
        self.provider_name.clear()
        self.provider_base_url.clear()
        self.provider_env_key.clear()
        self.provider_key.clear()
        self.provider_output.clear()
        self.provider_output.setVisible(False)
        self.provider_name.setFocus()
        self.update_action_buttons()

    def activate_selected_account(self) -> None:
        item = self.account_list.currentItem()
        name = self.selected_account_name()
        if not item or not name:
            return
        if not item.data(Qt.UserRole + 1):
            QMessageBox.warning(self, APP_NAME, f"Account {name} is not logged in.")
            return
        if item.data(Qt.UserRole + 3):
            self.use_unnamed_account()
            return
        self.use_account(name)

    def activate_selected_provider(self) -> None:
        name = self.selected_provider_name()
        if name:
            self.use_provider(name)

    def use_account(self, name: str) -> None:
        self.run_action(["use", name], f"Switching future Codex processes to {name}…")

    def use_unnamed_account(self) -> None:
        self.run_action(["account"], "Switching future Codex processes to the legacy account…")

    def use_provider(self, name: str) -> None:
        self.run_action(["api", name], f"Switching future Codex processes to API {name}…")

    def remove_selected_account(self) -> None:
        item = self.account_list.currentItem()
        name = self.selected_account_name()
        if not item or not name or item.data(Qt.UserRole + 3):
            return
        answer = QMessageBox.question(
            self,
            "Remove named account",
            f"Move {name} to the recoverable account archive?\n\n"
            "Shared Codex conversations will remain available.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self.run_action(
            ["remove", name, "--yes"],
            f"Archiving account {name}…",
            success_message=f"Account {name} was moved to the recoverable archive.",
        )

    def start_account_login(self) -> None:
        name = self.account_name.text().strip()
        if name and not ACCOUNT_NAME_PATTERN.fullmatch(name):
            QMessageBox.warning(
                self,
                "Invalid account name",
                "Enter a lowercase email address, for example name@example.com.",
            )
            return
        if name and not re.fullmatch(r"[a-z0-9][a-z0-9._+-]*@[a-z0-9.-]+\.[a-z]{2,}", name):
            QMessageBox.warning(
                self,
                "Email required",
                "The Qt account manager uses the login email as the default account name.",
            )
            return
        self.login_output.clear()
        self.login_link.clear()
        self.last_login_url = ""
        if not name:
            self.run_action(
                ["add-auto"],
                "Starting browser login for the current browser account…",
                stream_login=True,
                success_message="Account added using its authenticated email address.",
            )
            return
        self.run_action(
            ["add", name],
            f"Starting browser login for {name}…",
            stream_login=True,
            clear_account_name=True,
            success_message=f"Account {name} is ready for future Codex processes.",
        )

    def cancel_account_login(self) -> None:
        process = self.action_process
        if process is None or not process.property("streamLogin"):
            return
        self.action_cancel_requested = True
        self.status_label.setText("Cancelling account login…")
        self.update_action_buttons()
        process_id = int(process.processId())
        if process_id > 0 and process.property("separateProcessGroup"):
            try:
                os.killpg(process_id, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        else:
            process.terminate()
        QTimer.singleShot(
            3000,
            lambda process=process: self._force_cancelled_login(process),
        )

    def _force_cancelled_login(self, process: QProcess) -> None:
        if self.action_process is not process or not self.action_cancel_requested:
            return
        process_id = int(process.processId())
        if process_id > 0 and process.property("separateProcessGroup"):
            try:
                os.killpg(process_id, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError):
                pass
        process.kill()

    def provider_form_values(self) -> tuple[str, str, str] | None:
        name = self.provider_name.text().strip()
        base_url = self.provider_base_url.text().strip()
        env_key = self.provider_env_key.text().strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in {
            "openai",
            "ollama",
            "lmstudio",
        }:
            QMessageBox.warning(
                self,
                "Invalid provider name",
                "Use a name beginning with a letter and containing only letters, numbers, _ or -.",
            )
            return None
        if not re.fullmatch(r"https?://[^\s]+", base_url):
            QMessageBox.warning(
                self,
                "Invalid Base URL",
                "Enter a complete http:// or https:// API base URL.",
            )
            return None
        if not env_key:
            env_key = re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_OPENAI_KEY"
            self.provider_env_key.setText(env_key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_key):
            QMessageBox.warning(
                self,
                "Invalid environment key",
                "Use an environment-variable name such as PROVIDER_OPENAI_KEY.",
            )
            return None
        return name, base_url.rstrip("/"), env_key

    def save_provider(self) -> None:
        values = self.provider_form_values()
        if values is None:
            return
        name, base_url, env_key = values
        existing_names = {provider["name"] for provider in self.state.get("providers", [])}
        if not self.provider_editing_existing and name in existing_names:
            QMessageBox.warning(
                self,
                APP_NAME,
                f"API provider {name} already exists. Select it before editing.",
            )
            return
        command = "update" if self.provider_editing_existing else "add"
        arguments = [
            command,
            name,
            "--base-url",
            base_url,
            "--env-key",
            env_key,
            "--wire-api",
            "responses",
            "--skip-test",
        ]
        api_key = self.provider_key.text()
        self.provider_key.clear()
        self.provider_output.clear()

        def saved_provider() -> None:
            if api_key:
                self.run_provider_action(
                    ["set-key", name, "--stdin"],
                    f"Saving API key for {name}…",
                    stdin_text=api_key,
                    success_message=f"API provider {name} and its key were saved.",
                )

        self.run_provider_action(
            arguments,
            f"Saving API provider {name}…",
            success_message=f"API provider {name} was saved.",
            on_success=saved_provider if api_key else None,
        )

    def test_provider(self) -> None:
        values = self.provider_form_values()
        if values is None:
            return
        name, base_url, env_key = values
        arguments = ["test", name, "--base-url", base_url, "--env-key", env_key, "--timeout", "20"]
        api_key = self.provider_key.text()
        if api_key:
            arguments.append("--stdin")
        self.provider_output.clear()
        self.run_provider_action(
            arguments,
            f"Testing API provider {name}…",
            stdin_text=api_key or None,
            success_message=f"API provider {name} passed the connection test.",
        )

    def delete_selected_provider(self) -> None:
        item = self.provider_list.currentItem()
        name = self.selected_provider_name()
        if not item or not name or item.data(Qt.UserRole + 2):
            return
        answer = QMessageBox.question(
            self,
            "Delete API provider",
            f"Delete API provider {name} and its unshared stored key?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self.provider_output.clear()
        self.run_provider_action(
            ["delete", name, "--yes"],
            f"Deleting API provider {name}…",
            success_message=f"API provider {name} was deleted.",
            on_success=self.clear_provider_form,
        )

    def run_action(
        self,
        arguments: list[str],
        message: str,
        stream_login: bool = False,
        clear_account_name: bool = False,
        success_message: str = "Updated. Existing Codex processes keep their current account or API.",
    ) -> None:
        if self.action_process is not None:
            return
        command = command_path("codex-auth")
        if not command:
            QMessageBox.critical(self, APP_NAME, "codex-auth is not installed.")
            return
        self.start_process(
            command,
            arguments,
            message,
            output_widget=self.login_output if stream_login else None,
            stream_login=stream_login,
            clear_account_name=clear_account_name,
            success_message=success_message,
            command_name="codex-auth",
        )

    def run_provider_action(
        self,
        arguments: list[str],
        message: str,
        stdin_text: str | None = None,
        success_message: str = "API provider updated.",
        on_success: Callable[[], None] | None = None,
    ) -> None:
        provider_script = Path(__file__).with_name("codex_provider.py")
        if not provider_script.is_file():
            QMessageBox.critical(self, APP_NAME, "The API provider manager is not installed.")
            return
        self.start_process(
            sys.executable,
            [str(provider_script), *arguments],
            message,
            output_widget=self.provider_output,
            stdin_text=stdin_text,
            success_message=success_message,
            on_success=on_success,
            command_name="API provider manager",
        )

    def start_process(
        self,
        program: str,
        arguments: list[str],
        message: str,
        output_widget: QPlainTextEdit | None = None,
        stdin_text: str | None = None,
        stream_login: bool = False,
        clear_account_name: bool = False,
        success_message: str = "Updated.",
        on_success: Callable[[], None] | None = None,
        command_name: str = "command",
    ) -> None:
        if self.action_process is not None:
            return
        process = QProcess(self)
        process.setProcessEnvironment(process_environment())
        process.setProcessChannelMode(QProcess.MergedChannels)
        process_group_launcher = shutil.which("setsid") if stream_login else None
        if process_group_launcher:
            process.setProgram(process_group_launcher)
            process.setArguments([program, *arguments])
            process.setProperty("separateProcessGroup", True)
        else:
            process.setProgram(program)
            process.setArguments(arguments)
            process.setProperty("separateProcessGroup", False)
        process.setProperty("streamLogin", stream_login)
        process.readyReadStandardOutput.connect(self._read_action_output)
        process.finished.connect(self._action_finished)
        process.errorOccurred.connect(self._action_error)
        self.action_process = process
        self.action_buffer = ""
        self.action_output_widget = output_widget
        self.action_success_message = success_message
        self.action_success_callback = on_success
        self.action_clear_account_name = clear_account_name
        self.action_command_name = command_name
        self.action_cancel_requested = False
        if output_widget is not None:
            output_widget.setVisible(True)
            if output_widget is self.provider_output:
                QTimer.singleShot(0, self._fit_provider_output)
        self.status_label.setText(message)
        self.update_action_buttons()
        if stdin_text is not None:
            secret_bytes = stdin_text.encode()

            def send_stdin() -> None:
                process.write(secret_bytes)
                process.closeWriteChannel()

            process.started.connect(send_stdin)
        process.start()

    def _read_action_output(self) -> None:
        if self.action_process is None:
            return
        output = bytes(self.action_process.readAllStandardOutput()).decode(errors="replace")
        if not output:
            return
        self.action_buffer += output
        if self.action_output_widget is not None:
            self.action_output_widget.moveCursor(QTextCursor.End)
            self.action_output_widget.insertPlainText(output)
            self.action_output_widget.moveCursor(QTextCursor.End)
        if self.action_process.property("streamLogin"):
            if not self.last_login_url:
                login_url = extract_login_url(output)
                if login_url:
                    self.last_login_url = login_url
                    self.login_link.setText(
                        f'<a href="{self.last_login_url}">Open browser login page</a>'
                    )
        else:
            self.status_label.setText(output.strip().splitlines()[-1])

    def _action_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process = self.action_process
        if process is None:
            return
        self._read_action_output()
        output = self.action_buffer.strip()
        success_message = self.action_success_message
        callback = self.action_success_callback
        clear_account_name = self.action_clear_account_name
        cancelled = self.action_cancel_requested
        self.action_process = None
        self.action_buffer = ""
        self.action_output_widget = None
        self.action_success_callback = None
        self.action_cancel_requested = False
        process.deleteLater()
        if cancelled:
            self.login_link.clear()
            self.last_login_url = ""
            self.status_label.setText("Account login cancelled.")
        elif exit_code == 0:
            self.status_label.setText(success_message)
            if clear_account_name:
                self.account_name.clear()
            if callback is None:
                self.tray.showMessage(APP_NAME, success_message)
        else:
            tail = output.splitlines()
            detail = tail[-1] if tail else "Command failed."
            self.status_label.setText(detail)
            self.tray.showMessage(APP_NAME, detail, QSystemTrayIcon.Warning)
        self.refresh_state(force=True)
        if not cancelled and exit_code == 0 and callback is not None:
            QTimer.singleShot(0, callback)

    def _action_error(self, _error: QProcess.ProcessError) -> None:
        process = self.action_process
        if process is None:
            return
        if self.action_cancel_requested:
            return
        message = process.errorString()
        self.action_process = None
        self.action_buffer = ""
        self.action_output_widget = None
        self.action_success_callback = None
        self.action_cancel_requested = False
        process.deleteLater()
        self.status_label.setText(f"Cannot run {self.action_command_name}: {message}")
        self.update_action_buttons()

    def refresh_quota(self) -> None:
        if self.state.get("mode") != "account":
            self.quota = None
            self.render_quota()
            return
        if self.quota_process is not None:
            self.quota_pending = True
            return
        command = command_path("codex-usage")
        if not command:
            self.quota_summary.setText("codex-usage is not installed")
            account = self.state.get("active_account") or "account"
            self.overlay.set_error(self.account_display_name(account))
            return
        process = QProcess(self)
        process.setProcessEnvironment(process_environment())
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.setProgram(command)
        process.setArguments(["--json", "--timeout", "15"])
        process.finished.connect(self._quota_finished)
        process.errorOccurred.connect(self._quota_error)
        self.quota_process = process
        account = self.state.get("active_account") or "account"
        self.quota_summary.setText(
            f"{self.account_display_name(account)} · refreshing quota…"
        )
        process.start()

    def _quota_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process = self.quota_process
        self.quota_process = None
        account_name = self.state.get("active_account") or "account"
        account = self.account_display_name(account_name)
        if process is None:
            return
        stdout = bytes(process.readAllStandardOutput()).decode(errors="replace")
        stderr = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        try:
            if exit_code != 0:
                raise ValueError(stderr or "Quota request failed")
            self.quota = parse_quota(stdout)
            self.status_label.setText("Quota refreshed without using a browser.")
        except ValueError as error:
            self.quota = None
            self.quota_summary.setText(f"{account} · quota unavailable")
            self.quota_details.setText(str(error))
            self.overlay.set_error(account)
        self.render_quota()
        if self.quota_pending:
            self.quota_pending = False
            QTimer.singleShot(500, self.refresh_quota)

    def _quota_error(self, _error: QProcess.ProcessError) -> None:
        if self.quota_process is None:
            return
        message = self.quota_process.errorString()
        self.quota_process = None
        self.quota = None
        self.update_account_row_texts()
        account_name = self.state.get("active_account") or "account"
        account = self.account_display_name(account_name)
        self.quota_summary.setText(f"{account} · quota unavailable")
        self.quota_details.setText(message)
        self.overlay.set_error(account)

    def set_overlay_visible(self, visible: bool) -> None:
        self.settings.setValue("overlayVisible", visible)
        self.overlay.setVisible(visible)
        self._rebuild_tray_menu()

    def show_manager(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.refresh_state(force=True)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.quitting or not self.tray.isVisible():
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self.settings.value("closeHintShown", False, type=bool):
            self.tray.showMessage(APP_NAME, "The application is still running in the system tray.")
            self.settings.setValue("closeHintShown", True)

    def quit_application(self) -> None:
        self.quitting = True
        self.settings.setValue("overlayVisible", self.overlay.isVisible())
        self.settings.setValue("overlayPosition", self.overlay.pos())
        self.overlay.close()
        self.tray.hide()
        self.app.quit()


class SingleInstance:
    def __init__(self, on_message: Callable[[], None] | None = None) -> None:
        self.name = f"codex-account-manager-{os.getuid()}"
        self.server: QLocalServer | None = None
        self.on_message = on_message

    def notify_existing(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self.name)
        if not socket.waitForConnected(180):
            return False
        socket.write(b"show")
        socket.flush()
        socket.waitForBytesWritten(180)
        socket.disconnectFromServer()
        return True

    def listen(self, on_message: Callable[[], None]) -> None:
        self.on_message = on_message
        QLocalServer.removeServer(self.name)
        self.server = QLocalServer()
        self.server.newConnection.connect(self._receive)
        if not self.server.listen(self.name):
            raise RuntimeError(self.server.errorString())

    def _receive(self) -> None:
        if self.server is None:
            return
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            socket.waitForReadyRead(100)
            socket.readAll()
            socket.disconnectFromServer()
        if self.on_message:
            self.on_message()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName("Codex")
    app.setDesktopFileName("codex-account-manager")
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(application_icon())

    instance = SingleInstance()
    if instance.notify_existing():
        return 0

    settings = QSettings(APP_ID, "desktop")
    window = MainWindow(app, settings)
    instance.listen(window.show_manager)

    def request_shutdown(_signum, _frame) -> None:  # type: ignore[no-untyped-def]
        if not window.quitting:
            window.quit_application()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    signal_timer = QTimer(app)
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)

    background = "--background" in sys.argv[1:]
    if not background:
        window.show_manager()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

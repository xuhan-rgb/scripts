#!/usr/bin/env python3
"""GPU Harbor 桌面管理器，支持本机和 SSH 远程 NVIDIA GPU。"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PyQt5.QtCore import QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_DIR = Path(__file__).resolve().parent
KEEPER_SCRIPT = APP_DIR / "gpu_memory_keeper.py"
REMOTE_DIR = ".local/share/gpu-memory-keeper"
REMOTE_SCRIPT = f"{REMOTE_DIR}/gpu_memory_keeper.py"
STATE_HOME = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
LOCAL_LOG_DIR = STATE_HOME / "gpu-harbor"
WORKER_LOG_DIR = LOCAL_LOG_DIR / "workers"
APP_LOG_PATH = LOCAL_LOG_DIR / "gpu-harbor.log"
APP_ICON_PATH = APP_DIR / "gpu-harbor.svg"
CHEVRON_ICON_PATH = APP_DIR / "gpu-harbor-chevron.svg"
RESERVATION_PERCENT = 99
KEEPER_LIMIT_PERCENT = 80


def parse_ssh_entry(value: str) -> tuple[str, str | None, int | None, str | None]:
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise RuntimeError(f"SSH 命令格式无效: {exc} / Invalid SSH command: {exc}") from exc
    if parts and Path(parts[0]).name == "ssh":
        parts = parts[1:]
    if not parts:
        raise RuntimeError("请输入 SSH 命令、主机或 ~/.ssh/config 别名 / Enter an SSH target")

    user = None
    port = None
    identity = None
    target = None
    index = 0
    while index < len(parts):
        token = parts[index]
        if token in {"-p", "-i", "-l"}:
            if index + 1 >= len(parts):
                raise RuntimeError(
                    f"SSH 参数 {token} 缺少值 / SSH option {token} requires a value"
                )
            option_value = parts[index + 1]
            if token == "-p":
                try:
                    port = int(option_value)
                except ValueError as exc:
                    raise RuntimeError("SSH 端口必须是数字 / Port must be a number") from exc
            elif token == "-i":
                identity = option_value
            else:
                user = option_value
            index += 2
            continue
        if token.startswith("-p") and len(token) > 2:
            try:
                port = int(token[2:])
            except ValueError as exc:
                raise RuntimeError("SSH 端口必须是数字 / Port must be a number") from exc
            index += 1
            continue
        if token.startswith("-i") and len(token) > 2:
            identity = token[2:]
            index += 1
            continue
        if token.startswith("-l") and len(token) > 2:
            user = token[2:]
            index += 1
            continue
        if token.startswith("-"):
            raise RuntimeError(
                f"GUI 暂不支持 SSH 参数 {token}；高级参数请写入 ~/.ssh/config / "
                f"SSH option {token} is not supported here; add advanced options to ~/.ssh/config"
            )
        if target is not None:
            raise RuntimeError(
                "SSH 输入框只接受连接命令，不接受远程 shell 命令 / "
                "Enter an SSH connection only, without a remote shell command"
            )
        target = token
        index += 1

    if target is None:
        raise RuntimeError("SSH 命令中缺少主机 / SSH command has no host")
    if "@" in target:
        target_user, target = target.rsplit("@", 1)
        if target_user:
            user = target_user
    return target, user, port, identity


@dataclass(frozen=True)
class TargetConfig:
    remote: bool
    host: str = ""
    user: str = ""
    port: int | None = None
    identity: str = ""
    python: str = "python3"

    @property
    def label(self) -> str:
        if not self.remote:
            return "本机 / Local"
        target = f"{self.user}@{self.host}" if self.user else self.host
        return f"{target}:{self.port}" if self.port else target

    @property
    def key(self) -> tuple[Any, ...]:
        return self.remote, self.host, self.user, self.port, self.identity, self.python


class GpuClient:
    def __init__(self, config: TargetConfig) -> None:
        self.config = config

    def _ssh_base(self) -> list[str]:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
        if self.config.port:
            command.extend(("-p", str(self.config.port)))
        if self.config.identity:
            command.extend(("-i", os.path.expanduser(self.config.identity)))
        if self.config.user:
            command.extend(("-l", self.config.user))
        command.append(self.config.host)
        return command

    def _scp_base(self) -> list[str]:
        command = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
        if self.config.port:
            command.extend(("-P", str(self.config.port)))
        if self.config.identity:
            command.extend(("-i", os.path.expanduser(self.config.identity)))
        return command

    def _target(self) -> str:
        return f"{self.config.user}@{self.config.host}" if self.config.user else self.config.host

    @staticmethod
    def _run(command: list[str], timeout: int = 45, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"找不到命令: {command[0]} / Command not found: {command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"命令在 {timeout} 秒后超时 / Command timed out after {timeout} seconds"
            ) from exc
        if result.returncode != 0 and not allow_failure:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(detail)
        return result

    def validate(self) -> None:
        if not self.config.remote:
            return
        if not self.config.host or self.config.host.startswith("-") or any(char.isspace() for char in self.config.host):
            raise RuntimeError(
                "请输入有效的 SSH 主机或 ~/.ssh/config 别名 / "
                "Enter a valid SSH host or ~/.ssh/config alias"
            )
        if self.config.port is not None and not 1 <= self.config.port <= 65535:
            raise RuntimeError(
                "SSH 端口必须在 1 到 65535 之间 / SSH port must be between 1 and 65535"
            )
        if not self.config.python.strip():
            raise RuntimeError("必须填写远端 Python 命令 / Enter the remote Python command")

    def deploy(self) -> str:
        self.validate()
        if not self.config.remote:
            return "本机 worker 已就绪 / Local worker ready"
        self._run(self._ssh_base() + [f"mkdir -p {shlex.quote(REMOTE_DIR)}"])
        destination = f"{self._target()}:{REMOTE_SCRIPT}"
        self._run(self._scp_base() + [str(KEEPER_SCRIPT), destination])
        self._run(self._ssh_base() + [f"chmod 755 {shlex.quote(REMOTE_SCRIPT)}"])
        target = f"{self.config.label}:{REMOTE_SCRIPT}"
        return f"远端 worker 已部署: {target} / Remote worker deployed: {target}"

    def _cli(self, arguments: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
        if not self.config.remote:
            command = [sys.executable, str(KEEPER_SCRIPT), *arguments]
        else:
            remote_parts = [*shlex.split(self.config.python), REMOTE_SCRIPT, *arguments]
            remote_command = " ".join(shlex.quote(part) for part in remote_parts)
            command = self._ssh_base() + [remote_command]
        return self._run(command, allow_failure=allow_failure)

    def snapshot(self) -> dict[str, Any]:
        result = self._cli(["list", "--json"], allow_failure=True)
        for line in reversed(result.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("gpus"), list):
                if result.returncode != 0 and not payload.get("errors"):
                    payload["errors"] = [result.stderr.strip() or f"exit code {result.returncode}"]
                return payload
        detail = result.stderr.strip() or result.stdout.strip() or "未返回 GPU 状态 / No status data"
        raise RuntimeError(detail)

    def occupy(
        self,
        gpu_indexes: list[int],
        incremental: bool = False,
        step_percent: float = 10.0,
        interval: float = 5.0,
        systemd_guard: bool = False,
        total_percent: float = RESERVATION_PERCENT,
        keeper_percent: float = KEEPER_LIMIT_PERCENT,
    ) -> str:
        indexes = ",".join(map(str, gpu_indexes))
        arguments = [
            "occupy",
            "--gpus",
            indexes,
            "--percent",
            str(total_percent),
            "--keeper-percent",
            str(keeper_percent),
        ]
        if incremental:
            arguments.extend(
                ("--incremental", "--step-percent", str(step_percent), "--interval", str(interval))
            )
        if systemd_guard:
            arguments.append("--systemd-guard")
        self._cli(arguments)
        protection = "systemd 后台防护" if systemd_guard else "watchdog 后台监督"
        action = "开始动态递增监督" if incremental else "立即占用"
        english_protection = "systemd guard" if systemd_guard else "watchdog guard"
        english_action = "dynamic monitoring started" if incremental else "reservation started"
        return (
            f"GPU {indexes} {action}（{protection}） / "
            f"GPU {indexes} {english_action} ({english_protection})"
        )

    def monitor(
        self,
        gpu: int,
        step_percent: float,
        interval: float,
        total_limit_percent: float,
        keeper_limit_percent: float,
    ) -> str:
        self._cli(
            [
                "monitor",
                "--gpu",
                str(gpu),
                "--percent",
                str(total_limit_percent),
                "--keeper-percent",
                str(keeper_limit_percent),
                "--step-percent",
                str(step_percent),
                "--interval",
                str(interval),
            ]
        )
        return f"GPU {gpu} 已开始动态递增监督 / Dynamic monitoring started on GPU {gpu}"

    def configure(
        self,
        gpu: int,
        step_percent: float,
        interval: float,
        total_limit_percent: float,
        keeper_limit_percent: float,
    ) -> str:
        self._cli(
            [
                "configure",
                "--gpu",
                str(gpu),
                "--percent",
                str(total_limit_percent),
                "--keeper-percent",
                str(keeper_limit_percent),
                "--step-percent",
                str(step_percent),
                "--interval",
                str(interval),
            ]
        )
        return f"GPU {gpu} 阈值已实时更新 / GPU {gpu} limits updated"

    def release(self, gpu_indexes: list[int]) -> str:
        for gpu in gpu_indexes:
            self._cli(["release", "--gpu", str(gpu)])
        indexes = ",".join(map(str, gpu_indexes))
        return f"GPU {indexes} 已释放 / GPU {indexes} released"

    def release_all(self) -> str:
        self._cli(["release-all"])
        return "全部 GPU 占用已释放 / All reservations released"

    def sync_logs(self, gpu_indexes: list[int]) -> list[Path]:
        WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        target_name = "local" if not self.config.remote else self.config.label
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_name).strip("_") or "target"
        saved: list[Path] = []
        for gpu in gpu_indexes:
            remote_log = f"/tmp/gpu_memory_keeper_gpu{gpu}.log"
            if self.config.remote:
                result = self._run(self._ssh_base() + [f"cat {shlex.quote(remote_log)}"], allow_failure=True)
                if result.returncode != 0:
                    continue
                content = result.stdout
            else:
                try:
                    content = Path(remote_log).read_text(encoding="utf-8")
                except OSError:
                    continue
            destination = WORKER_LOG_DIR / f"{slug}_gpu{gpu}.log"
            destination.write_text(content, encoding="utf-8")
            saved.append(destination)
        return saved


class TaskThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.succeeded.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))


class SummaryCard(QFrame):
    def __init__(self, caption: str, value: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("SummaryCard")
        self.setStyleSheet(f"QFrame#SummaryCard {{ border-top: 4px solid {accent}; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        self.caption_label = QLabel(caption.upper())
        self.caption_label.setObjectName("CardCaption")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        layout.addWidget(self.caption_label)
        layout.addWidget(self.value_label)

    def set_caption(self, caption: str) -> None:
        self.caption_label.setText(caption.upper())

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class GpuHarborWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("CodexTools", "GPUHarbor")
        saved_language = str(self.settings.value("language", "zh"))
        self.language = saved_language if saved_language in {"zh", "en"} else "zh"
        self.active_tasks: dict[TaskThread, dict[str, Any]] = {}
        self.busy_gpus: dict[int, str] = {}
        self.refresh_busy = False
        self.release_all_busy = False
        self.connected_key: tuple[Any, ...] | None = None
        self.loading_profile = False
        self.profiles: dict[str, dict[str, str]] = {}
        self.gpu_action_buttons: list[QPushButton] = []
        self.gpu_progress_bars: dict[int, QProgressBar] = {}
        self.snapshot_data: dict[str, Any] = {"gpus": [], "errors": []}
        self.threshold_timer = QTimer(self)
        self.threshold_timer.setSingleShot(True)
        self.threshold_timer.setInterval(600)
        self.threshold_timer.timeout.connect(self._apply_live_thresholds)
        self.setWindowTitle(self._t("GPU 管理中心", "GPU Harbor"))
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setMinimumSize(1060, 700)
        self.resize(1380, 900)
        self._build_ui()
        self._apply_style()
        self._load_settings()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(5000)
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start()
        QTimer.singleShot(150, lambda: self.refresh(deploy=self.mode_combo.currentIndex() == 1))

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        self.sidebar = sidebar
        sidebar.setFixedWidth(320)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(22, 20, 22, 18)
        side.setSpacing(7)

        self.brand_label = QLabel(self._t("GPU 管理中心", "GPU HARBOR"))
        self.brand_label.setObjectName("Brand")
        self.tagline_label = QLabel(self._t("管理本机与远程显存", "LOCAL & REMOTE VRAM CONTROL"))
        self.tagline_label.setObjectName("Tagline")
        side.addWidget(self.brand_label)
        side.addWidget(self.tagline_label)
        side.addSpacing(12)

        language_row = QWidget()
        language_layout = QHBoxLayout(language_row)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.setSpacing(8)
        self.language_label = QLabel(self._t("语言", "LANGUAGE"))
        self.language_label.setObjectName("SectionLabel")
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.setCurrentIndex(0 if self.language == "zh" else 1)
        language_layout.addWidget(self.language_label)
        language_layout.addStretch()
        language_layout.addWidget(self.language_combo)
        side.addWidget(language_row)

        self.target_label = QLabel(self._t("目标", "TARGET"))
        self.target_label.setObjectName("SectionLabel")
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("TargetCombo")
        self.mode_combo.addItems((self._t("本机", "Local"), self._t("远程 SSH", "Remote SSH")))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        side.addWidget(self.target_label)
        side.addWidget(self.mode_combo)

        self.remote_panel = QFrame()
        self.remote_panel.setObjectName("RemotePanel")
        remote_form = QFormLayout(self.remote_panel)
        remote_form.setContentsMargins(0, 8, 0, 4)
        remote_form.setSpacing(6)
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("ProfileCombo")
        self.profile_combo.currentIndexChanged.connect(self._profile_selected)
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText(self._t("例如 ssh user@host", "For example: ssh user@host"))
        self.python_edit = QLineEdit("python3")
        profile_buttons = QWidget()
        profile_actions = QHBoxLayout(profile_buttons)
        profile_actions.setContentsMargins(0, 0, 0, 0)
        profile_actions.setSpacing(6)
        self.save_profile_button = QPushButton(self._t("保存", "Save"))
        self.delete_profile_button = QPushButton(self._t("删除", "Delete"))
        self.save_profile_button.setObjectName("ProfileButton")
        self.delete_profile_button.setObjectName("ProfileButton")
        self.save_profile_button.clicked.connect(self._save_profile)
        self.delete_profile_button.clicked.connect(self._delete_profile)
        profile_actions.addWidget(self.save_profile_button)
        profile_actions.addWidget(self.delete_profile_button)
        self.profile_field_label = QLabel(self._t("连接", "Connection"))
        self.ssh_field_label = QLabel(self._t("SSH 命令", "SSH command"))
        self.python_field_label = QLabel("Python")
        remote_form.addRow(self.profile_field_label, self.profile_combo)
        remote_form.addRow(self.ssh_field_label, self.host_edit)
        remote_form.addRow(self.python_field_label, self.python_edit)
        remote_form.addRow("", profile_buttons)
        self.profile_hint = QLabel(
            self._t("关闭界面后后台任务继续运行", "Jobs persist after the GUI closes")
        )
        self.profile_hint.setObjectName("ProfileHint")
        self.profile_hint.setWordWrap(True)
        remote_form.addRow(self.profile_hint)
        side.addWidget(self.remote_panel)

        self.connect_button = QPushButton(self._t("连接 / 刷新", "CONNECT / REFRESH"))
        self.connect_button.setObjectName("SecondaryButton")
        self.connect_button.clicked.connect(lambda: self.refresh(deploy=True))
        self.connection_label = QLabel(self._t("本机 worker 已就绪", "Local worker ready"))
        self.connection_label.setObjectName("ConnectionStatus")
        self.connection_label.setWordWrap(True)
        side.addWidget(self.connect_button)
        side.addWidget(self.connection_label)
        side.addSpacing(8)

        self.protection_label = QLabel(self._t("防护", "PROTECTION"))
        self.protection_label.setObjectName("SectionLabel")
        side.addWidget(self.protection_label)
        self.systemd_guard_check = QCheckBox(self._t("系统级防护", "Systemd guard"))
        self.systemd_guard_check.setChecked(True)
        self.systemd_guard_check.setToolTip(
            self._t(
                "由 systemd --user 后台托管；进程异常退出后 3 秒自动重启。",
                "Managed by systemd --user and restarted 3 seconds after an abnormal exit.",
            )
        )
        self.systemd_guard_check.toggled.connect(self._save_settings)
        side.addWidget(self.systemd_guard_check)
        self.monitor_settings_label = QLabel(self._t("监督参数", "MONITOR SETTINGS"))
        self.monitor_settings_label.setObjectName("SectionLabel")
        side.addWidget(self.monitor_settings_label)
        growth_row = QWidget()
        growth_row.setObjectName("GrowthRow")
        growth_layout = QHBoxLayout(growth_row)
        growth_layout.setContentsMargins(0, 0, 0, 0)
        growth_layout.setSpacing(6)
        self.step_label = QLabel(self._t("步长", "Step"))
        growth_layout.addWidget(self.step_label)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 98.0)
        self.step_spin.setDecimals(1)
        self.step_spin.setValue(10.0)
        self.step_spin.setSuffix("%")
        growth_layout.addWidget(self.step_spin)
        self.interval_label = QLabel(self._t("间隔", "Interval"))
        growth_layout.addWidget(self.interval_label)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.2, 3600.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(5.0)
        self.interval_spin.setSuffix("s")
        growth_layout.addWidget(self.interval_spin)
        side.addWidget(growth_row)
        self.growth_row = growth_row
        total_limit_row = QWidget()
        total_limit_row.setObjectName("ThresholdRow")
        total_limit_layout = QHBoxLayout(total_limit_row)
        total_limit_layout.setContentsMargins(0, 0, 0, 0)
        self.total_limit_label = QLabel(self._t("整卡总占用上限", "Total usage cap"))
        total_limit_layout.addWidget(self.total_limit_label)
        total_limit_layout.addStretch()
        self.total_limit_spin = QDoubleSpinBox()
        self.total_limit_spin.setRange(1.0, 99.9)
        self.total_limit_spin.setDecimals(1)
        self.total_limit_spin.setValue(float(RESERVATION_PERCENT))
        self.total_limit_spin.setSuffix("%")
        self.total_limit_spin.setMinimumWidth(94)
        self.total_limit_spin.setToolTip(
            self._t("其他进程与 Keeper 的显存总占用上限", "Combined VRAM cap for all processes")
        )
        total_limit_layout.addWidget(self.total_limit_spin)
        side.addWidget(total_limit_row)

        keeper_limit_row = QWidget()
        keeper_limit_row.setObjectName("ThresholdRow")
        keeper_limit_layout = QHBoxLayout(keeper_limit_row)
        keeper_limit_layout.setContentsMargins(0, 0, 0, 0)
        self.keeper_limit_label = QLabel(self._t("Keeper 自身上限", "Keeper cap"))
        keeper_limit_layout.addWidget(self.keeper_limit_label)
        keeper_limit_layout.addStretch()
        self.keeper_limit_spin = QDoubleSpinBox()
        self.keeper_limit_spin.setRange(1.0, 99.9)
        self.keeper_limit_spin.setDecimals(1)
        self.keeper_limit_spin.setValue(float(KEEPER_LIMIT_PERCENT))
        self.keeper_limit_spin.setSuffix("%")
        self.keeper_limit_spin.setMinimumWidth(94)
        self.keeper_limit_spin.setToolTip(
            self._t("Keeper 在单张 GPU 上最多预留的显存比例", "Maximum VRAM reserved by Keeper")
        )
        keeper_limit_layout.addWidget(self.keeper_limit_spin)
        side.addWidget(keeper_limit_row)
        self.threshold_rows = (total_limit_row, keeper_limit_row)
        side.addStretch()

        self.auto_refresh = QCheckBox(self._t("每 5 秒刷新", "Refresh every 5 seconds"))
        self.auto_refresh.setChecked(True)
        side.addWidget(self.auto_refresh)
        root_layout.addWidget(sidebar)

        content = QWidget()
        content.setObjectName("Content")
        self.content = content
        main = QVBoxLayout(content)
        main.setContentsMargins(34, 28, 34, 26)
        main.setSpacing(18)

        heading_row = QHBoxLayout()
        heading_box = QVBoxLayout()
        self.heading_label = QLabel(self._t("GPU 资源总览", "GPU Fleet Overview"))
        self.heading_label.setObjectName("Heading")
        self.target_heading = QLabel(self._t("本机", "Local"))
        self.target_heading.setObjectName("Subheading")
        heading_box.addWidget(self.heading_label)
        heading_box.addWidget(self.target_heading)
        heading_row.addLayout(heading_box)
        heading_row.addStretch()
        self.release_all_button = QPushButton(self._t("全部释放", "RELEASE ALL"))
        self.release_all_button.setObjectName("ReleaseAllButton")
        self.release_all_button.clicked.connect(self.release_all)
        heading_row.addWidget(self.release_all_button)
        self.refresh_button = QPushButton(self._t("刷新", "REFRESH"))
        self.refresh_button.setObjectName("SmallButton")
        self.refresh_button.clicked.connect(lambda: self.refresh(deploy=False))
        heading_row.addWidget(self.refresh_button)
        main.addLayout(heading_row)

        summaries = QHBoxLayout()
        summaries.setSpacing(12)
        self.gpu_card = SummaryCard(self._t("可见 GPU", "VISIBLE GPUS"), "-", "#0B6E69")
        self.util_card = SummaryCard(self._t("平均负载", "AVERAGE LOAD"), "-", "#E39A2D")
        self.reserved_card = SummaryCard(self._t("已预留", "RESERVED"), "-", "#E4572E")
        summaries.addWidget(self.gpu_card)
        summaries.addWidget(self.util_card)
        summaries.addWidget(self.reserved_card)
        main.addLayout(summaries)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            (
                "GPU",
                self._t("设备", "DEVICE"),
                self._t("显存", "VRAM"),
                self._t("占用", "MEMORY"),
                self._t("算力", "LOAD"),
                self._t("监督", "KEEPER"),
                self._t("操作", "ACTION"),
            )
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 48)
        self.table.setColumnWidth(1, 155)
        self.table.setColumnWidth(2, 140)
        self.table.setColumnWidth(3, 112)
        self.table.setColumnWidth(4, 90)
        self.table.setColumnWidth(6, 240)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main.addWidget(self.table, 1)

        log_header = QHBoxLayout()
        self.log_title_label = QLabel(self._t("活动记录", "ACTIVITY"))
        self.log_title_label.setObjectName("SectionLabel")
        self.clear_log_button = QPushButton(self._t("清空", "CLEAR"))
        self.clear_log_button.setObjectName("TextButton")
        self.clear_log_button.clicked.connect(self._clear_log)
        log_header.addWidget(self.log_title_label)
        log_header.addStretch()
        log_header.addWidget(self.clear_log_button)
        main.addLayout(log_header)
        self.activity = QTextEdit()
        self.activity.setObjectName("Activity")
        self.activity.setReadOnly(True)
        self.activity.setMaximumHeight(135)
        main.addWidget(self.activity)
        root_layout.addWidget(content, 1)

        for widget in (self.host_edit, self.python_edit):
            widget.textChanged.connect(self._profile_changed)
            widget.textChanged.connect(widget.setToolTip)
        for spin in (self.step_spin, self.interval_spin, self.total_limit_spin, self.keeper_limit_spin):
            spin.valueChanged.connect(lambda _value: self._save_settings())
        self.total_limit_spin.valueChanged.connect(self._schedule_live_thresholds)
        self.keeper_limit_spin.valueChanged.connect(self._schedule_live_thresholds)
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        self._retranslate_ui()

    def _t(self, chinese: str, english: str) -> str:
        return chinese if self.language == "zh" else english

    def _target_label(self, config: TargetConfig) -> str:
        return self._t("本机", "Local") if not config.remote else config.label

    def _update_connection_label(self) -> None:
        if self.active_tasks:
            self.connection_label.setText(self._t("正在执行...", "Working..."))
            return
        try:
            config = self._config()
        except RuntimeError:
            self.connection_label.setText(self._t("尚未连接", "Not connected"))
            return
        if self.connected_key == config.key:
            self.connection_label.setText(
                self._t(
                    f"已连接: {self._target_label(config)}",
                    f"Connected: {self._target_label(config)}",
                )
            )
        elif config.remote:
            self.connection_label.setText(self._t("尚未连接", "Not connected"))
        else:
            self.connection_label.setText(self._t("本机 worker 已就绪", "Local worker ready"))

    def _language_changed(self) -> None:
        selected_profile = self.profile_combo.currentText()
        language = self.language_combo.currentData()
        if language not in {"zh", "en"} or language == self.language:
            return
        self.language = language
        self.settings.setValue("language", self.language)
        self._retranslate_ui()
        self._refresh_profile_combo(selected_profile if selected_profile in self.profiles else "")
        self.target_heading.setText(self._safe_target_label())
        self._update_connection_label()
        self._render_snapshot(self.snapshot_data)
        self.activity.clear()
        self._log(self._t("语言已切换为中文", "Language changed to English"))

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(self._t("GPU 管理中心", "GPU Harbor"))
        self.brand_label.setText(self._t("GPU 管理中心", "GPU HARBOR"))
        self.tagline_label.setText(self._t("管理本机与远程显存", "LOCAL & REMOTE VRAM CONTROL"))
        self.language_label.setText(self._t("语言", "LANGUAGE"))
        self.target_label.setText(self._t("目标", "TARGET"))
        self.mode_combo.setItemText(0, self._t("本机", "Local"))
        self.mode_combo.setItemText(1, self._t("远程 SSH", "Remote SSH"))
        self.host_edit.setPlaceholderText(self._t("例如 ssh user@host", "For example: ssh user@host"))
        self.profile_field_label.setText(self._t("连接", "Connection"))
        self.ssh_field_label.setText(self._t("SSH 命令", "SSH command"))
        self.save_profile_button.setText(self._t("保存", "Save"))
        self.delete_profile_button.setText(self._t("删除", "Delete"))
        self.profile_hint.setText(
            self._t("关闭界面后后台任务继续运行", "Jobs persist after the GUI closes")
        )
        self.connect_button.setText(self._t("连接 / 刷新", "CONNECT / REFRESH"))
        self.protection_label.setText(self._t("防护", "PROTECTION"))
        self.systemd_guard_check.setText(self._t("系统级防护", "Systemd guard"))
        self.systemd_guard_check.setToolTip(
            self._t(
                "由 systemd --user 后台托管；进程异常退出后 3 秒自动重启。",
                "Managed by systemd --user and restarted 3 seconds after an abnormal exit.",
            )
        )
        self.monitor_settings_label.setText(self._t("监督参数", "MONITOR SETTINGS"))
        self.step_label.setText(self._t("步长", "Step"))
        self.interval_label.setText(self._t("间隔", "Interval"))
        self.total_limit_label.setText(self._t("整卡总占用上限", "Total usage cap"))
        self.total_limit_spin.setToolTip(
            self._t("其他进程与 Keeper 的显存总占用上限", "Combined VRAM cap for all processes")
        )
        self.keeper_limit_label.setText(self._t("Keeper 自身上限", "Keeper cap"))
        self.keeper_limit_spin.setToolTip(
            self._t("Keeper 在单张 GPU 上最多预留的显存比例", "Maximum VRAM reserved by Keeper")
        )
        self.auto_refresh.setText(self._t("每 5 秒刷新", "Refresh every 5 seconds"))
        self.heading_label.setText(self._t("GPU 资源总览", "GPU Fleet Overview"))
        self.release_all_button.setText(self._t("全部释放", "RELEASE ALL"))
        self.refresh_button.setText(self._t("刷新", "REFRESH"))
        self.gpu_card.set_caption(self._t("可见 GPU", "VISIBLE GPUS"))
        self.util_card.set_caption(self._t("平均负载", "AVERAGE LOAD"))
        self.reserved_card.set_caption(self._t("已预留", "RESERVED"))
        self.table.setHorizontalHeaderLabels(
            (
                "GPU",
                self._t("设备", "DEVICE"),
                self._t("显存", "VRAM"),
                self._t("占用", "MEMORY"),
                self._t("算力", "LOAD"),
                self._t("监督", "KEEPER"),
                self._t("操作", "ACTION"),
            )
        )
        self.log_title_label.setText(self._t("活动记录", "ACTIVITY"))
        self.clear_log_button.setText(self._t("清空", "CLEAR"))
        self._update_task_controls()

    def _apply_style(self) -> None:
        base_style = (
            """
            QMainWindow, QWidget#Content { background: #F2ECDD; color: #17252A; }
            QFrame#Sidebar { background: #183238; color: #FFF8EA; }
            QLabel#Brand { color: #FFF8EA; font: 700 25px 'Ubuntu'; letter-spacing: 2px; }
            QLabel#Tagline { color: #AFC6C4; font: 14px 'Ubuntu'; line-height: 1.4; }
            QLabel#SectionLabel { color: #70807D; font: 700 11px 'Ubuntu'; letter-spacing: 1px; }
            QFrame#Sidebar QLabel#SectionLabel { color: #87A6A2; }
            QComboBox, QLineEdit, QDoubleSpinBox { background: #FAF5E9; color: #17252A; border: 1px solid #D7CDBB;
                border-radius: 6px; padding: 8px; min-height: 19px; }
            QFrame#Sidebar QComboBox, QFrame#Sidebar QLineEdit, QFrame#Sidebar QDoubleSpinBox { background: #24464C; color: #FFF8EA;
                border-color: #3C6065; selection-background-color: #0B6E69; }
            QFrame#RemotePanel QLabel { color: #B9CCCA; }
            QLabel#ProfileHint { color: #789995; font: 10px 'Ubuntu'; padding-top: 2px; }
            QWidget#GrowthRow QLabel, QWidget#ThresholdRow QLabel { color: #B9CCCA; }
            QLabel#ConnectionStatus { color: #AFC6C4; font: 12px 'Ubuntu'; padding: 4px 2px; }
            QPushButton { border: 0; border-radius: 6px; padding: 10px 12px; font: 700 12px 'Ubuntu'; }
            QPushButton:hover { margin-top: -1px; }
            QPushButton:disabled { color: #879592; background: #D4D7CF; }
            QPushButton#OccupyRowButton { background: #E4572E; color: white; padding: 7px 6px; }
            QPushButton#OccupyRowButton:hover { background: #F0643A; }
            QPushButton#ReleaseRowButton { background: #0B6E69; color: white; padding: 7px 6px; }
            QPushButton#ReleaseRowButton:hover { background: #11847D; }
            QPushButton#MonitorButton { background: #D58C2F; color: #17252A; padding: 7px 6px; }
            QPushButton#MonitorButton:hover { background: #E8A33C; }
            QPushButton#MonitorButton:disabled { background: #D8D4C8; color: #78827B; }
            QPushButton#MonitorWaitingButton:disabled { background: #E7D8C8; color: #86694D;
                border: 1px solid #D1BDA8; padding: 7px 6px; }
            QPushButton#MonitorRunningButton:disabled { background: #B9DAD4; color: #075C58;
                border: 1px solid #78B5AC; padding: 7px 6px; }
            QPushButton#MonitorStartingButton:disabled { background: #F0D49B; color: #715016;
                border: 1px solid #D7B66E; padding: 7px 6px; }
            QPushButton#SecondaryButton { background: transparent; color: #D9E5E2;
                border: 1px solid #507176; }
            QPushButton#ProfileButton { background: #24464C; color: #D9E5E2; border: 1px solid #507176;
                padding: 7px 5px; }
            QPushButton#SmallButton { background: #183238; color: #FFF8EA; padding: 9px 18px; }
            QPushButton#ReleaseAllButton { background: #E4572E; color: white; padding: 9px 18px; }
            QPushButton#ReleaseAllButton:hover { background: #F0643A; }
            QPushButton#TextButton { background: transparent; color: #0B6E69; padding: 2px 6px; }
            QLabel#Heading { color: #17252A; font: 700 30px 'Ubuntu'; }
            QLabel#Subheading { color: #6E7D78; font: 14px 'Ubuntu'; }
            QFrame#SummaryCard { background: #FFF9EE; border: 1px solid #DED4C1; border-radius: 8px; }
            QLabel#CardCaption { color: #78827B; font: 700 10px 'Ubuntu'; letter-spacing: 1px; }
            QLabel#CardValue { color: #17252A; font: 700 24px 'Ubuntu'; }
            QTableWidget { background: #FFF9EE; border: 1px solid #DED4C1; border-radius: 8px;
                color: #24363A; outline: 0; }
            QHeaderView::section { background: #E8E0D0; color: #68736F; border: 0;
                border-bottom: 1px solid #D5CCBA; padding: 9px; font: 700 10px 'Ubuntu'; }
            QTableWidget::item { border-bottom: 1px solid #E9E0D0; padding: 7px; }
            QProgressBar { background: #E7DECE; border: 0; border-radius: 5px; height: 20px;
                text-align: center; color: #17252A; font: 700 10px 'Ubuntu'; }
            QProgressBar#MemoryBar::chunk { background: #E39A2D; border-radius: 5px; }
            QProgressBar#ComputeBar::chunk { background: #0B6E69; border-radius: 5px; }
            QProgressBar#ActionProgress { background: #DDD4C4; border: 0; border-radius: 2px;
                min-height: 4px; max-height: 4px; }
            QProgressBar#ActionProgress::chunk { background: #E4572E; border-radius: 2px; }
            QTextEdit#Activity { background: #142B30; color: #CFE1DD; border: 0; border-radius: 8px;
                padding: 9px; font: 12px 'Ubuntu Mono'; }
            QCheckBox { color: #B9CCCA; font: 12px 'Ubuntu'; spacing: 7px; }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QCheckBox::indicator:checked { background: #E4572E; border: 2px solid #FFF8EA; }
            """
        )
        chevron = str(CHEVRON_ICON_PATH).replace("\\", "/")
        combo_style = f"""
            QFrame#Sidebar QComboBox {{
                padding: 9px 38px 9px 12px;
                min-height: 24px;
                border: 1px solid #456A6F;
                border-radius: 7px;
                background: #23464C;
                color: #FFF8EA;
                font: 600 13px 'Ubuntu';
            }}
            QFrame#Sidebar QComboBox:hover {{
                border-color: #7AA09C;
                background: #294F55;
            }}
            QFrame#Sidebar QComboBox:focus {{ border: 1px solid #E2A33B; }}
            QFrame#Sidebar QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 34px;
                border: 0;
                border-left: 1px solid #456A6F;
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
                background: #1C3B40;
            }}
            QFrame#Sidebar QComboBox::down-arrow {{
                image: url({chevron});
                width: 12px;
                height: 7px;
            }}
            QFrame#Sidebar QComboBox QAbstractItemView {{
                background: #1F4046;
                color: #FFF8EA;
                border: 1px solid #55787B;
                border-radius: 7px;
                outline: 0;
                padding: 5px;
                selection-background-color: #D87635;
                selection-color: white;
            }}
            QComboBox#TargetCombo {{ font: 700 14px 'Ubuntu'; }}
        """
        self.setStyleSheet(base_style + combo_style)

    def _refresh_profile_combo(self, selected: str = "") -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem(self._t("新连接", "New connection"))
        for name in sorted(self.profiles):
            self.profile_combo.addItem(name)
        if selected in self.profiles:
            self.profile_combo.setCurrentText(selected)
        self.profile_combo.blockSignals(False)

    def _profile_selected(self) -> None:
        name = self.profile_combo.currentText()
        if name not in self.profiles:
            return
        profile = self.profiles[name]
        self.loading_profile = True
        self.host_edit.setText(profile.get("host", ""))
        self.python_edit.setText(profile.get("python", "python3"))
        self.host_edit.setCursorPosition(0)
        self.python_edit.setCursorPosition(0)
        self.host_edit.setToolTip(self.host_edit.text())
        self.python_edit.setToolTip(self.python_edit.text())
        self.loading_profile = False
        self.connected_key = None
        self.settings.setValue("current_profile", name)
        self.connection_label.setText(self._t("配置已载入，请连接", "Profile loaded; connect to continue"))
        self.target_heading.setText(self._safe_target_label())

    def _save_profile(self) -> None:
        try:
            config = self._config()
            GpuClient(config).validate()
        except RuntimeError as exc:
            self._log(str(exc), error=True)
            return
        current = self.profile_combo.currentText()
        suggested = current if current in self.profiles else self._target_label(config)
        name, accepted = QInputDialog.getText(
            self,
            self._t("保存连接", "Save connection"),
            self._t("连接名称:", "Profile name:"),
            text=suggested,
        )
        name = name.strip()
        if not accepted or not name:
            return
        self.profiles[name] = {
            "host": self.host_edit.text().strip(),
            "python": self.python_edit.text().strip() or "python3",
        }
        self.settings.setValue("profiles", json.dumps(self.profiles, ensure_ascii=False))
        self.settings.setValue("current_profile", name)
        self._refresh_profile_combo(name)
        self._log(self._t(f"连接配置已保存: {name}", f"Profile saved: {name}"))

    def _delete_profile(self) -> None:
        name = self.profile_combo.currentText()
        if name not in self.profiles:
            self._log(self._t("请选择已保存的连接", "Select a saved profile"), error=True)
            return
        answer = QMessageBox.question(
            self,
            self._t("删除连接", "Delete connection"),
            self._t(f"确定删除“{name}”吗？", f'Delete profile "{name}"?'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        del self.profiles[name]
        self.settings.setValue("profiles", json.dumps(self.profiles, ensure_ascii=False))
        self.settings.remove("current_profile")
        self._refresh_profile_combo()
        self._log(self._t(f"连接配置已删除: {name}", f"Profile deleted: {name}"))

    def _load_settings(self) -> None:
        try:
            profiles = json.loads(self.settings.value("profiles", "{}"))
            if isinstance(profiles, dict):
                self.profiles = profiles
        except (TypeError, json.JSONDecodeError):
            self.profiles = {}
        current_profile = self.settings.value("current_profile", "")
        self.loading_profile = True
        self._refresh_profile_combo(current_profile)
        mode = self.settings.value("mode", "local")
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1 if mode == "remote" else 0)
        self.mode_combo.blockSignals(False)
        self.host_edit.setText(self.settings.value("host", ""))
        self.python_edit.setText(self.settings.value("python", "python3"))
        self.systemd_guard_check.blockSignals(True)
        self.systemd_guard_check.setChecked(self.settings.value("systemd_guard", True, type=bool))
        self.systemd_guard_check.blockSignals(False)
        self.step_spin.setValue(float(self.settings.value("step_percent", 10.0)))
        self.interval_spin.setValue(float(self.settings.value("interval", 5.0)))
        self.total_limit_spin.setValue(
            float(self.settings.value("total_limit_percent", RESERVATION_PERCENT))
        )
        self.keeper_limit_spin.setValue(
            float(self.settings.value("keeper_limit_percent", KEEPER_LIMIT_PERCENT))
        )
        self.loading_profile = False
        if current_profile in self.profiles:
            self._profile_selected()
        self._mode_changed()

    def _save_settings(self) -> None:
        self.settings.setValue("language", self.language)
        self.settings.setValue("mode", "remote" if self.mode_combo.currentIndex() == 1 else "local")
        self.settings.setValue("host", self.host_edit.text().strip())
        self.settings.setValue("python", self.python_edit.text().strip())
        self.settings.setValue("systemd_guard", self.systemd_guard_check.isChecked())
        self.settings.setValue("step_percent", self.step_spin.value())
        self.settings.setValue("interval", self.interval_spin.value())
        self.settings.setValue("total_limit_percent", self.total_limit_spin.value())
        self.settings.setValue("keeper_limit_percent", self.keeper_limit_spin.value())

    def _config(self) -> TargetConfig:
        remote = self.mode_combo.currentIndex() == 1
        parsed_host = self.host_edit.text().strip()
        parsed_user = None
        parsed_port = None
        parsed_identity = None
        if remote:
            parsed_host, parsed_user, parsed_port, parsed_identity = parse_ssh_entry(parsed_host)
        return TargetConfig(
            remote=remote,
            host=parsed_host if remote else "",
            user=parsed_user or "",
            port=parsed_port,
            identity=parsed_identity or "",
            python=self.python_edit.text().strip() or "python3",
        )

    def _mode_changed(self) -> None:
        remote = self.mode_combo.currentIndex() == 1
        self.remote_panel.setVisible(remote)
        self.connection_label.setText(
            self._t("尚未连接", "Not connected")
            if remote
            else self._t("本机 worker 已就绪", "Local worker ready")
        )
        self.connected_key = None
        self._save_settings()
        self.target_heading.setText(self._safe_target_label())

    def _profile_changed(self) -> None:
        if self.loading_profile:
            return
        self.connected_key = None
        if self.mode_combo.currentIndex() == 1:
            self.connection_label.setText(
                self._t("配置已更改，请重新连接", "Configuration changed; reconnect")
            )
        self._save_settings()
        self.target_heading.setText(self._safe_target_label())

    def _schedule_live_thresholds(self, _value: float = 0.0) -> None:
        if not self.loading_profile:
            self.threshold_timer.start()

    def _apply_live_thresholds(self) -> None:
        active_gpus = [
            gpu["index"]
            for gpu in self.snapshot_data.get("gpus", [])
            if gpu.get("reservation") and gpu["reservation"].get("active")
        ]
        if not active_gpus:
            return
        try:
            config, client, needs_deploy = self._client_for_action()
        except RuntimeError as exc:
            self._log(str(exc), error=True)
            return
        if config.remote and self.connected_key != config.key:
            return

        step_percent = self.step_spin.value()
        interval = self.interval_spin.value()
        total_limit_percent = self.total_limit_spin.value()
        keeper_limit_percent = self.keeper_limit_spin.value()

        def operation() -> dict[str, Any]:
            deployment = client.deploy() if needs_deploy else ""
            for gpu in active_gpus:
                client.configure(
                    gpu,
                    step_percent,
                    interval,
                    total_limit_percent,
                    keeper_limit_percent,
                )
            return {"deployment": deployment, "snapshot": client.snapshot()}

        def success(result: dict[str, Any]) -> None:
            self.connected_key = config.key
            if result["deployment"]:
                self._log(result["deployment"])
            self._log(
                self._t(
                    f"阈值已实时应用到 GPU {active_gpus}",
                    f"Live limits applied to GPU {active_gpus}",
                )
            )
            self._render_snapshot(result["snapshot"])

        started = self._start_task(
            operation,
            success,
            self._t(
                f"正在实时更新阈值: GPU {active_gpus}",
                f"Applying live limits: GPU {active_gpus}",
            ),
            gpu_indexes=active_gpus,
            action="configure",
        )
        if not started:
            self.threshold_timer.start(1000)

    def _safe_target_label(self) -> str:
        try:
            return self._target_label(self._config())
        except RuntimeError:
            return self._t("远程 SSH", "Remote SSH")

    def _log(self, message: str, error: bool = False) -> None:
        if " / " in message:
            chinese, english = message.split(" / ", 1)
            message = chinese if self.language == "zh" else english
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = "#FF9B7D" if error else "#CFE1DD"
        escaped = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self.activity.append(f'<span style="color:#789692">{timestamp}</span> <span style="color:{color}">{escaped}</span>')
        try:
            LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with APP_LOG_PATH.open("a", encoding="utf-8") as handle:
                level = "ERROR" if error else "INFO"
                handle.write(f"{datetime.now().isoformat(timespec='seconds')} {level} {message}\n")
        except OSError:
            pass

    def _clear_log(self) -> None:
        self.activity.clear()

    def _update_task_controls(self) -> None:
        self.refresh_button.setEnabled(not self.refresh_busy)
        self.refresh_button.setText(
            self._t("刷新中...", "REFRESHING...")
            if self.refresh_busy
            else self._t("刷新", "REFRESH")
        )
        self.release_all_button.setEnabled(not self.release_all_busy)
        self.release_all_button.setText(
            self._t("释放中...", "RELEASING...")
            if self.release_all_busy
            else self._t("全部释放", "RELEASE ALL")
        )

    def _start_task(
        self,
        operation: Callable[[], Any],
        success_handler: Callable[[Any], None],
        announce: str | None = None,
        gpu_indexes: list[int] | None = None,
        action: str = "",
        task_kind: str = "action",
    ) -> bool:
        requested_gpus = set(gpu_indexes or [])
        conflicts = requested_gpus.intersection(self.busy_gpus)
        if conflicts:
            indexes = ", ".join(map(str, sorted(conflicts)))
            self._log(
                self._t(
                    f"GPU {indexes} 正在执行其他操作",
                    f"GPU {indexes} already has an operation in progress",
                ),
                error=True,
            )
            return False
        if task_kind == "refresh" and self.refresh_busy:
            return False
        if task_kind == "release_all" and self.release_all_busy:
            return False
        if announce:
            self._log(announce)
        worker = TaskThread(operation)
        self.active_tasks[worker] = {
            "success": success_handler,
            "gpus": requested_gpus,
            "kind": task_kind,
        }
        for gpu in requested_gpus:
            self.busy_gpus[gpu] = action or task_kind
        if task_kind == "refresh":
            self.refresh_busy = True
        elif task_kind == "release_all":
            self.release_all_busy = True
        self._update_task_controls()
        if requested_gpus:
            self._render_snapshot(self.snapshot_data)
        self._update_connection_label()
        worker.succeeded.connect(self._task_succeeded)
        worker.failed.connect(self._task_failed)
        worker.finished.connect(self._task_finished)
        worker.start()
        return True

    def _task_succeeded(self, result: Any) -> None:
        worker = self.sender()
        task = self.active_tasks.get(worker) if isinstance(worker, TaskThread) else None
        if task:
            task["success"](result)

    def _task_failed(self, message: str) -> None:
        self.connection_label.setText(self._t("操作失败", "Operation failed"))
        self._log(message, error=True)

    def _task_finished(self) -> None:
        finished_worker = self.sender()
        if not isinstance(finished_worker, TaskThread):
            return
        task = self.active_tasks.pop(finished_worker, None)
        if task:
            for gpu in task["gpus"]:
                self.busy_gpus.pop(gpu, None)
            if task["kind"] == "refresh":
                self.refresh_busy = False
            elif task["kind"] == "release_all":
                self.release_all_busy = False
        self._update_task_controls()
        self._render_snapshot(self.snapshot_data)
        self._update_connection_label()
        finished_worker.deleteLater()

    def refresh(self, deploy: bool = False, quiet: bool = False) -> None:
        try:
            config = self._config()
        except RuntimeError as exc:
            self._log(str(exc), error=True)
            return
        client = GpuClient(config)

        def operation() -> dict[str, Any]:
            deployment = client.deploy() if deploy else ""
            return {"snapshot": client.snapshot(), "deployment": deployment, "config": config}

        def success(result: dict[str, Any]) -> None:
            self.connected_key = config.key
            target_label = self._target_label(config)
            self.connection_label.setText(
                self._t(f"已连接: {target_label}", f"Connected: {target_label}")
            )
            self.target_heading.setText(target_label)
            if result["deployment"]:
                self._log(result["deployment"])
            self._render_snapshot(result["snapshot"])

        target_label = self._target_label(config)
        self._start_task(
            operation,
            success,
            None
            if quiet
            else self._t(f"正在刷新: {target_label}", f"Refreshing: {target_label}"),
            task_kind="refresh",
        )

    def _auto_refresh(self) -> None:
        if not self.auto_refresh.isChecked() or self.refresh_busy:
            return
        try:
            config = self._config()
        except RuntimeError:
            return
        if config.remote and self.connected_key != config.key:
            return
        self.refresh(deploy=False, quiet=True)

    def _client_for_action(self) -> tuple[TargetConfig, GpuClient, bool]:
        config = self._config()
        needs_deploy = config.remote and self.connected_key != config.key
        return config, GpuClient(config), needs_deploy

    def _occupy_gpus(self, gpu_indexes: list[int], incremental: bool) -> None:
        try:
            config, client, needs_deploy = self._client_for_action()
        except RuntimeError as exc:
            self._log(str(exc), error=True)
            return

        systemd_guard = self.systemd_guard_check.isChecked()
        step_percent = self.step_spin.value()
        interval = self.interval_spin.value()
        total_limit_percent = self.total_limit_spin.value()
        keeper_limit_percent = self.keeper_limit_spin.value()

        def operation() -> dict[str, Any]:
            deployment = client.deploy() if needs_deploy else ""
            if incremental:
                if len(gpu_indexes) != 1:
                    raise RuntimeError(
                        self._t(
                            "递增监督必须通过单个 GPU 行内按钮启动",
                            "Incremental monitoring must start from one GPU row",
                        )
                    )
                message = client.monitor(
                    gpu_indexes[0],
                    step_percent,
                    interval,
                    total_limit_percent,
                    keeper_limit_percent,
                )
            else:
                message = client.occupy(
                    gpu_indexes,
                    False,
                    step_percent,
                    interval,
                    systemd_guard,
                    total_limit_percent,
                    keeper_limit_percent,
                )
            snapshot = client.snapshot()
            logs = client.sync_logs(gpu_indexes)
            return {"deployment": deployment, "message": message, "snapshot": snapshot, "logs": logs}

        def success(result: dict[str, Any]) -> None:
            self.connected_key = config.key
            target_label = self._target_label(config)
            self.connection_label.setText(
                self._t(f"已连接: {target_label}", f"Connected: {target_label}")
            )
            if result["deployment"]:
                self._log(result["deployment"])
            self._log(result["message"])
            if result["logs"]:
                self._log(
                    self._t(
                        f"worker 日志已保存到本地: {WORKER_LOG_DIR}",
                        f"Worker logs saved: {WORKER_LOG_DIR}",
                    )
                )
            self._render_snapshot(result["snapshot"])

        mode = self._t("递增监督", "Incremental monitoring") if incremental else self._t("立即占用", "Immediate reservation")
        protection = (
            self._t("沿用现有后台防护", "existing background guard")
            if incremental
            else (
                self._t("systemd 防护", "systemd guard")
                if systemd_guard
                else self._t("watchdog 监督", "watchdog guard")
            )
        )
        details = (
            self._t(
                f"动态等待释放并补占，整卡上限 {total_limit_percent:g}%，Keeper 上限 {keeper_limit_percent:g}%",
                f"claiming newly freed memory; total cap {total_limit_percent:g}%, Keeper cap {keeper_limit_percent:g}%",
            )
            if incremental
            else self._t(
                f"整卡上限 {total_limit_percent:g}%，Keeper 上限 {keeper_limit_percent:g}%",
                f"total cap {total_limit_percent:g}%, Keeper cap {keeper_limit_percent:g}%",
            )
        )
        self._start_task(
            operation,
            success,
            f"{mode}, {protection}: GPU {gpu_indexes}, {details}",
            gpu_indexes=gpu_indexes,
            action="monitor" if incremental else "occupy",
        )

    def occupy_gpu(self, gpu: int) -> None:
        self._occupy_gpus([gpu], incremental=False)

    def monitor_gpu(self, gpu: int) -> None:
        self._occupy_gpus([gpu], incremental=True)

    def release_gpu(self, gpu: int) -> None:
        gpu_indexes = [gpu]
        try:
            config, client, needs_deploy = self._client_for_action()
        except RuntimeError as exc:
            self._log(str(exc), error=True)
            return

        def operation() -> dict[str, Any]:
            deployment = client.deploy() if needs_deploy else ""
            message = client.release(gpu_indexes)
            logs = client.sync_logs(gpu_indexes)
            snapshot = client.snapshot()
            return {
                "deployment": deployment,
                "message": message,
                "snapshot": snapshot,
                "logs": logs,
            }

        def success(result: dict[str, Any]) -> None:
            self.connected_key = config.key
            target_label = self._target_label(config)
            self.connection_label.setText(
                self._t(f"已连接: {target_label}", f"Connected: {target_label}")
            )
            if result["deployment"]:
                self._log(result["deployment"])
            self._log(result["message"])
            if result["logs"]:
                self._log(
                    self._t(
                        f"worker 日志已保存到本地: {WORKER_LOG_DIR}",
                        f"Worker logs saved: {WORKER_LOG_DIR}",
                    )
                )
            self._render_snapshot(result["snapshot"])

        self._start_task(
            operation,
            success,
            self._t(f"正在释放 GPU: {gpu_indexes}", f"Releasing GPU: {gpu_indexes}"),
            gpu_indexes=gpu_indexes,
            action="release",
        )

    def release_all(self) -> None:
        try:
            config, client, needs_deploy = self._client_for_action()
        except RuntimeError as exc:
            self._log(str(exc), error=True)
            return

        known_gpus = [gpu["index"] for gpu in self.snapshot_data.get("gpus", [])]

        def operation() -> dict[str, Any]:
            deployment = client.deploy() if needs_deploy else ""
            message = client.release_all()
            logs = client.sync_logs(known_gpus)
            return {
                "deployment": deployment,
                "message": message,
                "snapshot": client.snapshot(),
                "logs": logs,
            }

        def success(result: dict[str, Any]) -> None:
            self.connected_key = config.key
            target_label = self._target_label(config)
            self.connection_label.setText(
                self._t(f"已连接: {target_label}", f"Connected: {target_label}")
            )
            if result["deployment"]:
                self._log(result["deployment"])
            self._log(result["message"])
            if result["logs"]:
                self._log(
                    self._t(
                        f"worker 日志已保存到本地: {WORKER_LOG_DIR}",
                        f"Worker logs saved: {WORKER_LOG_DIR}",
                    )
                )
            self._render_snapshot(result["snapshot"])

        self._start_task(
            operation,
            success,
            self._t("正在释放全部占用", "Releasing all reservations"),
            gpu_indexes=known_gpus,
            action="release_all",
            task_kind="release_all",
        )

    def _render_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot_data = snapshot
        gpus = snapshot.get("gpus", [])
        self.gpu_action_buttons = []
        self.gpu_progress_bars = {}
        self.table.setRowCount(len(gpus))
        total_utilization = 0
        utilization_count = 0
        reserved_bytes = 0

        for row, gpu in enumerate(gpus):
            index_item = QTableWidgetItem(str(gpu["index"]))
            index_item.setFlags(Qt.ItemIsEnabled)
            index_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, index_item)
            unavailable = self._t("不可用", "Unavailable")
            device_item = QTableWidgetItem(gpu.get("name") or unavailable)
            device_item.setToolTip(gpu.get("name") or unavailable)
            self.table.setItem(row, 1, device_item)

            used = gpu.get("memory_used_mib")
            total = gpu.get("memory_total_mib")
            utilization = gpu.get("utilization_percent")
            memory_ratio = used / total * 100 if used is not None and total else 0
            vram_text = (
                f"{used / 1024:.1f}/{total / 1024:.1f} GiB\n"
                f"{self._t('占用', 'Used')} {memory_ratio:.1f}%"
                if total
                else "N/A"
            )
            vram_item = QTableWidgetItem(vram_text)
            vram_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.table.setItem(row, 2, vram_item)

            memory_bar = QProgressBar()
            memory_bar.setObjectName("MemoryBar")
            memory_percent = round(memory_ratio)
            memory_bar.setValue(memory_percent)
            memory_bar.setFormat(f"{memory_percent}%")
            self.table.setCellWidget(row, 3, memory_bar)

            compute_bar = QProgressBar()
            compute_bar.setObjectName("ComputeBar")
            compute_value = utilization if isinstance(utilization, int) else 0
            compute_bar.setValue(compute_value)
            compute_bar.setFormat(f"{compute_value}%" if utilization is not None else "N/A")
            self.table.setCellWidget(row, 4, compute_bar)
            if utilization is not None:
                total_utilization += utilization
                utilization_count += 1

            reservation = gpu.get("reservation")
            if reservation and reservation.get("active"):
                gib = reservation["requested_bytes"] / 1024**3
                total_gib = total / 1024 if total else 0
                reserved_ratio = gib / total_gib * 100 if total_gib else 0
                target_gib = reservation.get("target_bytes", reservation["requested_bytes"]) / 1024**3
                guard = "systemd" if reservation.get("systemd_guard") else "watchdog"
                if reservation.get("mode") == "incremental":
                    if reservation.get("dynamic_target"):
                        total_limit = reservation.get("limit_percent") or RESERVATION_PERCENT
                        keeper_limit = reservation.get("keeper_limit_percent") or KEEPER_LIMIT_PERCENT
                        keeper_text = self._t(
                            f"动态 {gib:.2f}/{total_gib:.1f} GiB ({reserved_ratio:.1f}%)\n"
                            f"目标 {target_gib:.2f} GiB（整卡 {total_limit:g}% / 自身 {keeper_limit:g}%） | "
                            f"{guard} | PID {reservation['pid']}",
                            f"Dynamic {gib:.2f}/{total_gib:.1f} GiB ({reserved_ratio:.1f}%)\n"
                            f"Target {target_gib:.2f} GiB (total {total_limit:g}% / Keeper {keeper_limit:g}%) | "
                            f"{guard} | PID {reservation['pid']}",
                        )
                    else:
                        keeper_text = self._t(
                            f"递增 {gib:.2f}/{total_gib:.1f} GiB ({reserved_ratio:.1f}%)\n"
                            f"目标 {target_gib:.2f} GiB | {guard} | PID {reservation['pid']}",
                            f"Incremental {gib:.2f}/{total_gib:.1f} GiB ({reserved_ratio:.1f}%)\n"
                            f"Target {target_gib:.2f} GiB | {guard} | PID {reservation['pid']}",
                        )
                else:
                    keeper_text = (
                        f"{gib:.2f}/{total_gib:.1f} GiB ({reserved_ratio:.1f}%)\n"
                        f"{guard} | PID {reservation['pid']}"
                    )
                reserved_bytes += reservation["requested_bytes"]
            elif reservation:
                keeper_text = self._t(
                    f"正在重启 PID {reservation['pid']}",
                    f"Restarting PID {reservation['pid']}",
                )
            else:
                keeper_text = self._t("可用", "Available")
            self.table.setItem(row, 5, QTableWidgetItem(keeper_text))

            action_widget = QWidget()
            action_layout = QVBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 3, 4, 3)
            action_layout.setSpacing(3)
            button_row = QWidget()
            button_layout = QHBoxLayout(button_row)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.setSpacing(6)
            button_row.setFixedHeight(40)
            row_busy = gpu["index"] in self.busy_gpus

            if reservation:
                primary_button = QPushButton(self._t("释放", "RELEASE"))
                primary_button.setObjectName("ReleaseRowButton")
                primary_button.clicked.connect(
                    lambda _checked=False, gpu_index=gpu["index"]: self.release_gpu(gpu_index)
                )
            else:
                primary_button = QPushButton(self._t("立即占用", "OCCUPY"))
                primary_button.setObjectName("OccupyRowButton")
                primary_button.clicked.connect(
                    lambda _checked=False, gpu_index=gpu["index"]: self.occupy_gpu(gpu_index)
                )
            primary_button.setMinimumHeight(40)
            primary_button.setProperty("action_available", not row_busy)
            primary_button.setEnabled(not row_busy)
            button_layout.addWidget(primary_button)

            monitor_button = QPushButton(self._t("递增监督", "MONITOR"))
            monitor_button.setObjectName("MonitorButton")
            monitor_button.setMinimumHeight(40)
            monitor_available = bool(
                reservation
                and reservation.get("active")
                and not reservation.get("dynamic_target")
            )
            monitor_button.setProperty("action_available", monitor_available and not row_busy)
            if monitor_available:
                monitor_button.clicked.connect(
                    lambda _checked=False, gpu_index=gpu["index"]: self.monitor_gpu(gpu_index)
                )
            elif reservation and reservation.get("dynamic_target"):
                monitor_button.setObjectName("MonitorRunningButton")
                monitor_button.setText(self._t("监督中", "RUNNING"))
                monitor_button.setEnabled(False)
            elif reservation:
                monitor_button.setObjectName("MonitorStartingButton")
                monitor_button.setText(self._t("恢复中", "STARTING"))
                monitor_button.setEnabled(False)
            else:
                monitor_button.setObjectName("MonitorWaitingButton")
                monitor_button.setText(self._t("先占用", "OCCUPY FIRST"))
                monitor_button.setEnabled(False)
            if row_busy:
                monitor_button.setEnabled(False)
            button_layout.addWidget(monitor_button)
            action_layout.addWidget(button_row)

            progress = QProgressBar()
            progress.setObjectName("ActionProgress")
            progress.setRange(0, 0)
            progress.setTextVisible(False)
            progress.setFixedHeight(5)
            progress.setVisible(row_busy)
            action_layout.addWidget(progress)
            self.gpu_progress_bars[gpu["index"]] = progress
            self.gpu_action_buttons.extend((primary_button, monitor_button))
            self.table.setCellWidget(row, 6, action_widget)
            self.table.setRowHeight(row, 66)

        average = total_utilization / utilization_count if utilization_count else 0
        self.gpu_card.set_value(str(len(gpus)))
        self.util_card.set_value(f"{average:.0f}%")
        self.reserved_card.set_value(f"{reserved_bytes / 1024**3:.2f} GiB")
        for issue in snapshot.get("errors", []):
            self._log(issue, error=True)

    def closeEvent(self, event: Any) -> None:
        self._save_settings()
        if self.active_tasks:
            QMessageBox.information(
                self,
                self._t("GPU 管理中心", "GPU Harbor"),
                self._t("请等待当前操作完成后再关闭。", "Wait for the current operation to finish."),
            )
            event.ignore()
            return
        event.accept()


def main() -> int:
    screenshot_path = None
    if len(sys.argv) == 3 and sys.argv[1] == "--screenshot":
        screenshot_path = Path(sys.argv[2]).expanduser()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("GPU Harbor")
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    app.setStyle("Fusion")
    app.setFont(QFont("Noto Sans CJK SC", 10))
    window = GpuHarborWindow()
    window.show()
    # Restore terminal semantics: Ctrl+C terminates the GUI process immediately.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if screenshot_path:
        def capture() -> None:
            window.grab().save(str(screenshot_path))
            # Quit directly so a still-running SSH refresh cannot block closeEvent.
            app.quit()

        QTimer.singleShot(1800, capture)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

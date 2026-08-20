#!/usr/bin/env python3
"""Non-Qt state parsing for the native Codex account manager."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any


ACCOUNT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._+@-]{0,127}$")
PROVIDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
DEFAULT_CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
DEFAULT_ACCOUNTS_DIR = Path(
    os.environ.get("CODEX_ACCOUNTS_DIR", Path.home() / ".local/share/codex/accounts")
)
DEFAULT_STATE_DIR = Path.home() / ".config/codex"


def extract_login_url(output: str) -> str | None:
    plain_text = ANSI_ESCAPE_PATTERN.sub("", output)
    match = re.search(r"https://[^\s<>\"']+", plain_text)
    return match.group(0).rstrip(".,)") if match else None


def parse_top_level_strings(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("["):
            break
        match = re.match(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"', line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def _section_strings(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"', line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def _configured_secret_names(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    names = set()
    for line in text.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match:
            names.add(match.group(1))
    return names


def read_provider_catalog(
    config_file: Path, secrets_file: Path, active_provider: str | None
) -> list[dict[str, Any]]:
    try:
        text = config_file.read_text(encoding="utf-8")
    except OSError:
        return []
    secret_names = _configured_secret_names(secrets_file)
    providers = []
    pattern = re.compile(
        r"(?ms)^\[model_providers\.([^]]+)]\s*\n(.*?)(?=^\[[^\n]+]\s*$|\Z)"
    )
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        if not PROVIDER_NAME_PATTERN.fullmatch(name):
            continue
        values = _section_strings(match.group(2))
        env_key = values.get("env_key", "")
        providers.append(
            {
                "name": name,
                "base_url": values.get("base_url", ""),
                "env_key": env_key,
                "key_set": bool(env_key and env_key in secret_names),
                "active": name == active_provider,
            }
        )
    return sorted(providers, key=lambda item: (not item["active"], item["name"]))


def read_account_catalog(accounts_dir: Path, active_account: str | None) -> list[dict[str, Any]]:
    accounts = []
    if not accounts_dir.is_dir():
        return accounts
    for account_dir in sorted(accounts_dir.iterdir(), key=lambda path: path.name):
        if account_dir.is_symlink() or not account_dir.is_dir():
            continue
        if account_dir.name.startswith("pending-login-"):
            continue
        if not ACCOUNT_NAME_PATTERN.fullmatch(account_dir.name):
            continue
        auth_file = account_dir / "auth.json"
        logged_in = (
            auth_file.is_file()
            and not auth_file.is_symlink()
            and auth_file.stat().st_size > 0
        )
        account = {
            "name": account_dir.name,
            "active": account_dir.name == active_account,
            "logged_in": logged_in,
        }
        email = _auth_email(auth_file) if logged_in else None
        if email:
            account["email"] = email
        accounts.append(account)
    return accounts


def _read_valid_name(path: Path, pattern: re.Pattern[str]) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if pattern.fullmatch(value) else None


def _legacy_chatgpt_login_available(codex_home: Path) -> bool:
    try:
        auth = json.loads((codex_home / "auth.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(auth, dict) and auth.get("auth_mode") == "chatgpt" and isinstance(
        auth.get("tokens"), dict
    )


def _auth_email(auth_file: Path) -> str | None:
    try:
        auth = json.loads(auth_file.read_text(encoding="utf-8"))
        token = auth.get("tokens", {}).get("id_token")
        payload_segment = token.split(".", 2)[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_segment + padding))
    except (AttributeError, IndexError, OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None

    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() == "email" and isinstance(child, str):
                    email = child.strip().lower()
                    if len(email) <= 254 and EMAIL_PATTERN.fullmatch(email):
                        return email
                if isinstance(child, (dict, list)):
                    pending.append(child)
        elif isinstance(value, list):
            pending.extend(child for child in value if isinstance(child, (dict, list)))
    return None


def read_state(
    *,
    codex_home: Path = DEFAULT_CODEX_HOME,
    accounts_dir: Path = DEFAULT_ACCOUNTS_DIR,
    active_file: Path = DEFAULT_STATE_DIR / "active-account",
    secrets_file: Path = DEFAULT_STATE_DIR / "secrets.env",
    api_provider_file: Path = DEFAULT_STATE_DIR / "api-provider",
) -> dict[str, Any]:
    codex_home = Path(codex_home)
    config_file = codex_home / "config.toml"
    try:
        config_text = config_file.read_text(encoding="utf-8")
    except OSError:
        config_text = ""
    configured_provider = parse_top_level_strings(config_text).get("model_provider", "")
    active_account = _read_valid_name(Path(active_file), ACCOUNT_NAME_PATTERN)
    accounts = read_account_catalog(Path(accounts_dir), active_account)
    if active_account and not any(item["active"] for item in accounts):
        active_account = None

    if active_account:
        mode = "account"
    elif configured_provider == "openai":
        mode = "account"
        active_account = "unnamed"
    else:
        mode = "api"

    if _legacy_chatgpt_login_available(codex_home):
        legacy_account = {
            "name": "unnamed",
            "active": active_account == "unnamed",
            "logged_in": True,
            "legacy": True,
        }
        email = _auth_email(codex_home / "auth.json")
        if email:
            legacy_account["email"] = email
        accounts.insert(0, legacy_account)

    saved_api = _read_valid_name(Path(api_provider_file), PROVIDER_NAME_PATTERN)
    selected_api = configured_provider if mode == "api" else saved_api
    providers = read_provider_catalog(config_file, Path(secrets_file), selected_api)
    return {
        "mode": mode,
        "active_account": active_account,
        "active_provider": configured_provider if mode == "api" else None,
        "saved_api_provider": saved_api,
        "accounts": accounts,
        "providers": providers,
    }


def format_window(window_seconds: float | int | None, fallback: str) -> str:
    if isinstance(window_seconds, (int, float)) and not isinstance(window_seconds, bool):
        seconds = int(window_seconds)
        if seconds > 0 and seconds % 86400 == 0:
            return f"{seconds // 86400}d"
        if seconds > 0 and seconds % 3600 == 0:
            return f"{seconds // 3600}h"
        if seconds > 0 and seconds % 60 == 0:
            return f"{seconds // 60}m"
    return fallback


def parse_quota(output: str) -> dict[str, Any]:
    try:
        snapshot = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError("codex-usage returned invalid JSON") from error
    if not isinstance(snapshot, dict):
        raise ValueError("codex-usage returned an invalid payload")
    account = snapshot.get("account")
    if not isinstance(account, str) or not ACCOUNT_NAME_PATTERN.fullmatch(account):
        raise ValueError("codex-usage did not return an account")
    plan = snapshot.get("plan_type")
    windows = []
    for limit in snapshot.get("rate_limits") or []:
        if not isinstance(limit, dict) or limit.get("name") != "Codex":
            continue
        for window in limit.get("windows") or []:
            if not isinstance(window, dict):
                continue
            remaining = window.get("remaining_percent")
            reset_at = window.get("resets_at")
            seconds = window.get("window_seconds")
            if (
                not isinstance(remaining, (int, float))
                or isinstance(remaining, bool)
                or not isinstance(reset_at, (int, float))
                or isinstance(reset_at, bool)
            ):
                continue
            windows.append(
                {
                    "name": str(window.get("name") or "quota"),
                    "label": format_window(seconds, str(window.get("name") or "quota")),
                    "remaining_percent": float(remaining),
                    "window_seconds": float(seconds) if isinstance(seconds, (int, float)) else 0.0,
                    "resets_at": float(reset_at),
                }
            )
    if not windows:
        raise ValueError("codex-usage did not return Codex quota windows")
    return {
        "account": account,
        "plan_type": plan if isinstance(plan, str) else "",
        "windows": windows,
        "overlay_window": max(windows, key=lambda item: item["window_seconds"]),
    }


def format_countdown(reset_at: float, now: float | None = None) -> str:
    remaining = int(reset_at - (time.time() if now is None else now))
    if remaining <= 0:
        return "now"
    days, remainder = divmod(remaining, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{max(1, minutes)}m"


def format_account_row(
    account: dict[str, Any],
    quota: dict[str, Any] | None,
    now: float | None = None,
) -> str:
    marker = "●" if account["active"] else "○"
    login = "logged in" if account["logged_in"] else "login needed"
    legacy = " · legacy" if account.get("legacy") else ""
    label = account.get("email") or account["name"]
    quota_suffix = ""
    if account["active"] and quota and quota.get("account") == account["name"]:
        window = quota["overlay_window"]
        quota_suffix = (
            f" · {window['label']}: {window['remaining_percent']:g}% left · "
            f"resets in {format_countdown(window['resets_at'], now)}"
        )
    return f"{marker} {label} · {login}{legacy}{quota_suffix}"

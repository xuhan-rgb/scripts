#!/usr/bin/env python3
import fcntl
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8320
MODELS = (
    {"id": "gpt-5.6-sol", "name": "Sol", "role": "Frontier", "description": "Hard problems and long-horizon work"},
    {"id": "gpt-5.6-terra", "name": "Terra", "role": "Balanced", "description": "Everyday engineering with strong judgment"},
    {"id": "gpt-5.6-luna", "name": "Luna", "role": "Fast", "description": "Quick edits, searches, and iterations"},
)
MODEL_IDS = tuple(model["id"] for model in MODELS)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
CLIENT_MODEL = "claudex-router"
CLIENT_EFFORT = "medium"
STATE_DIR = Path.home() / ".cli-proxy-api"
SELECTION_FILE = STATE_DIR / "selection.conf"
LOCK_FILE = STATE_DIR / "selection.lock"
GATEWAY_CONFIG = STATE_DIR / "config.yaml"
USAGE_DB = STATE_DIR / "usage.sqlite3"
MANAGEMENT_PASSWORD = os.environ.get("MANAGEMENT_PASSWORD", "")
CODEX_DIR = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
CODEX_CONFIG = CODEX_DIR / "config.toml"
CODEX_PROVIDER_BACKEND = Path(__file__).with_name("codex_provider.py")
CODEX_SYNC = Path.home() / ".local" / "bin" / "claude-codex-sync"
CODEX_SECRETS = Path.home() / ".config" / "codex" / "secrets.env"
PROVIDER_LOCK = STATE_DIR / "provider.lock"


def parse_top_level_strings(text):
    values = {}
    for line in text.splitlines():
        if line.lstrip().startswith("["):
            break
        match = re.match(r'^\s*([A-Za-z0-9_.-]+)\s*=\s*"([^"]*)"', line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_provider_strings(text, provider):
    target = f"model_providers.{provider}"
    active = False
    values = {}
    for line in text.splitlines():
        section = re.match(r"^\s*\[([^]]+)]\s*$", line)
        if section:
            active = section.group(1) == target
            continue
        if not active:
            continue
        match = re.match(r'^\s*([A-Za-z0-9_.-]+)\s*=\s*"([^"]*)"', line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def read_codex_state(path=CODEX_CONFIG):
    text = Path(path).read_text(encoding="utf-8")
    top_level = parse_top_level_strings(text)
    provider = top_level.get("model_provider", "")
    provider_values = parse_provider_strings(text, provider) if provider else {}
    return {
        "config_path": str(path),
        "provider": provider,
        "base_url": provider_values.get("base_url", ""),
        "wire_api": provider_values.get("wire_api", ""),
        "env_key": provider_values.get("env_key", ""),
        "codex_model": top_level.get("model", ""),
        "codex_effort": top_level.get("model_reasoning_effort", ""),
    }


def _secret_env_names(path=CODEX_SECRETS):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    return set(re.findall(r"(?m)^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", text))


def read_provider_catalog(config_path=CODEX_CONFIG, secrets_path=CODEX_SECRETS):
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    current = parse_top_level_strings(text).get("model_provider", "")
    secret_names = _secret_env_names(secrets_path)
    providers = []
    for match in re.finditer(r"(?ms)^\[model_providers\.([^]]+)]\s*\n(.*?)(?=^\[[^\n]+]\s*$|\Z)", text):
        name, body = match.group(1), match.group(2)
        values = {}
        for key, value in re.findall(r'^([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$', body, re.M):
            values[key] = value
        env_key = values.get("env_key", "")
        providers.append(
            {
                "name": name,
                "base_url": values.get("base_url", ""),
                "wire_api": values.get("wire_api", ""),
                "env_key": env_key,
                "key_set": bool(env_key and (env_key in secret_names or os.environ.get(env_key))),
                "active": name == current,
            }
        )
    return sorted(providers, key=lambda item: (not item["active"], item["name"]))


def _valid_model(value):
    return value if value in MODEL_IDS else "gpt-5.6-sol"


def _valid_effort(value):
    return value if value in EFFORTS else "xhigh"


def read_selection(path=SELECTION_FILE, codex_state=None):
    codex_state = codex_state or {}
    values = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return {
        "model": _valid_model(values.get("CLAUDEX_MODEL", codex_state.get("codex_model", ""))),
        "effort": _valid_effort(values.get("CLAUDEX_EFFORT", codex_state.get("codex_effort", ""))),
    }


def _validate_selection(model, effort):
    if model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {model}")
    if effort not in EFFORTS:
        raise ValueError(f"unsupported effort: {effort}")


def _atomic_write_text(path, content, prefix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temp_name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_selection_unlocked(model, effort, path):
    _atomic_write_text(
        path,
        f"CLAUDEX_MODEL={model}\nCLAUDEX_EFFORT={effort}\n",
        "selection.",
    )


def write_selection(model, effort, path=SELECTION_FILE, lock_path=LOCK_FILE):
    _validate_selection(model, effort)
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_selection_unlocked(model, effort, path)
    return {"model": model, "effort": effort}


def _write_gateway_route_unlocked(model, effort, path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    replacements = (
        (r'(?m)^(\s*- name: ")[^"]+(" # claudex-route-model)$', rf'\g<1>{model}\g<2>'),
        (r'(?m)^(\s*- name: ")[^"]+(" # claudex-effort-model)$', rf'\g<1>{model}\g<2>'),
        (r'(?m)^(\s*"reasoning\.effort": ")[^"]+(" # claudex-route-effort)$', rf'\g<1>{effort}\g<2>'),
    )
    for pattern, replacement in replacements:
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise ValueError(f"gateway route marker mismatch in {path}")
    _atomic_write_text(path, text, "config.route.")


def apply_selection(model, effort, selection_path=SELECTION_FILE, config_path=GATEWAY_CONFIG, lock_path=LOCK_FILE):
    _validate_selection(model, effort)
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_gateway_route_unlocked(model, effort, config_path)
        _write_selection_unlocked(model, effort, selection_path)
    return {"model": model, "effort": effort}


USAGE_STATUS = {"last_error": "", "last_sync": ""}


def init_usage_db(path=USAGE_DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                api_key TEXT NOT NULL,
                provider TEXT NOT NULL,
                auth_index TEXT NOT NULL,
                model TEXT NOT NULL,
                alias TEXT NOT NULL,
                reasoning_effort TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER NOT NULL,
                cache_creation_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                ttft_ms INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                status_code INTEGER NOT NULL,
                error TEXT NOT NULL
            )
            """
        )
    os.chmod(path, 0o600)


def store_usage_records(records, path=USAGE_DB):
    if not records:
        return 0
    init_usage_db(path)
    rows = []
    for record in records:
        tokens = record.get("tokens") or {}
        failure = record.get("fail") or {}
        rows.append(
            (
                record.get("request_id") or None,
                str(record.get("timestamp") or ""),
                str(record.get("api_key") or ""),
                str(record.get("provider") or ""),
                str(record.get("auth_index") or ""),
                str(record.get("model") or ""),
                str(record.get("alias") or ""),
                str(record.get("reasoning_effort") or ""),
                str(record.get("endpoint") or ""),
                int(tokens.get("input_tokens") or 0),
                int(tokens.get("output_tokens") or 0),
                int(tokens.get("reasoning_tokens") or 0),
                int(tokens.get("cache_read_tokens") or 0),
                int(tokens.get("cache_creation_tokens") or 0),
                int(tokens.get("total_tokens") or 0),
                int(record.get("latency_ms") or 0),
                int(record.get("ttft_ms") or 0),
                int(bool(record.get("failed"))),
                int(failure.get("status_code") or 0),
                "",
            )
        )
    with sqlite3.connect(path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO requests (
                request_id, timestamp, api_key, provider, auth_index, model, alias,
                reasoning_effort, endpoint, input_tokens, output_tokens,
                reasoning_tokens, cache_read_tokens, cache_creation_tokens,
                total_tokens, latency_ms, ttft_ms, failed, status_code, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted = connection.total_changes - before
        connection.execute(
            "DELETE FROM requests WHERE id NOT IN (SELECT id FROM requests ORDER BY id DESC LIMIT 5000)"
        )
    return inserted


def list_usage_records(limit=50, path=USAGE_DB):
    path = Path(path)
    if not path.exists():
        return []
    limit = max(1, min(int(limit), 200))
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def usage_count(path=USAGE_DB):
    path = Path(path)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0])


def _parse_usage_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("usage timestamp is not a string")
    normalized = value.replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+(?=(?:[+-]\d{2}:\d{2})?$)", r"\1", normalized)
    return datetime.fromisoformat(normalized)


def usage_summary(path=USAGE_DB, now=None):
    now = now or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    starts = {
        "day": today,
        "week": today - timedelta(days=today.weekday()),
        "month": today.replace(day=1),
    }
    periods = {
        name: {
            "requests": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
        }
        for name in starts
    }
    path = Path(path)
    if not path.exists():
        return periods

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT timestamp, input_tokens, output_tokens, reasoning_tokens,
                   cache_read_tokens, total_tokens
            FROM requests
            """
        ).fetchall()

    for row in rows:
        try:
            timestamp = _parse_usage_timestamp(row["timestamp"])
        except (AttributeError, ValueError):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=now.tzinfo)
        timestamp = timestamp.astimezone(now.tzinfo)
        input_tokens = int(row["input_tokens"] or 0)
        cache_read_tokens = int(row["cache_read_tokens"] or 0)
        uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)
        output_tokens = int(row["output_tokens"] or 0)
        total_tokens = int(row["total_tokens"] or input_tokens + output_tokens)
        for name, start in starts.items():
            if timestamp < start:
                continue
            periods[name]["requests"] += 1
            periods[name]["total_tokens"] += total_tokens
            periods[name]["input_tokens"] += uncached_input_tokens
            periods[name]["output_tokens"] += output_tokens
            periods[name]["reasoning_tokens"] += int(row["reasoning_tokens"] or 0)
            periods[name]["cache_read_tokens"] += cache_read_tokens
    return periods


def fetch_usage_queue(password=MANAGEMENT_PASSWORD):
    if not password:
        raise ValueError("management credential is unavailable")
    request = urllib.request.Request(
        "http://127.0.0.1:8317/v0/management/usage-queue?count=200",
        headers={"Authorization": f"Bearer {password}"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("usage queue returned an invalid payload")
    return payload


def usage_collector(stop_event):
    init_usage_db()
    while not stop_event.is_set():
        try:
            records = fetch_usage_queue()
            store_usage_records(records)
            USAGE_STATUS["last_error"] = ""
            if records:
                USAGE_STATUS["last_sync"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        except (OSError, ValueError, sqlite3.Error, urllib.error.URLError) as error:
            USAGE_STATUS["last_error"] = str(error)
        stop_event.wait(1)


def gateway_state():
    request = urllib.request.Request(
        "http://127.0.0.1:8317/v1/models?limit=1000",
        headers={"Authorization": "Bearer claudex-local"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            payload = json.load(response)
        models = sorted(item.get("id", "") for item in payload.get("data", []) if item.get("id"))
        return {"reachable": True, "models": models}
    except (OSError, ValueError, urllib.error.URLError):
        return {"reachable": False, "models": []}


def service_state():
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "cli-proxy-api.service"],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )
    return result.stdout.strip() == "active"


def _payload_text(payload, name, required=False, maximum=2048, strip=True):
    value = payload.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if strip:
        value = value.strip()
    if required and not value:
        raise ValueError(f"{name} is required")
    if len(value) > maximum or "\n" in value or "\r" in value:
        raise ValueError(f"{name} is invalid")
    return value


def _default_env_key(name):
    return f"{re.sub(r'[^A-Za-z0-9]', '_', name).upper()}_OPENAI_KEY"


def _run_provider_command(arguments, input_text=None):
    if not CODEX_PROVIDER_BACKEND.is_file():
        raise ValueError(f"provider backend is unavailable: {CODEX_PROVIDER_BACKEND}")
    result = subprocess.run(
        [sys.executable, str(CODEX_PROVIDER_BACKEND), *arguments],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "provider update failed"
        raise ValueError(message)
    return result.stdout


def _sync_provider_gateway():
    if not CODEX_SYNC.is_file():
        raise ValueError(f"gateway sync command is unavailable: {CODEX_SYNC}")
    result = subprocess.run(
        [str(CODEX_SYNC)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "gateway sync failed")
    subprocess.run(
        ["systemctl", "--user", "enable", "cli-proxy-api.service"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    subprocess.run(
        ["systemctl", "--user", "restart", "cli-proxy-api.service"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _provider_lock(path=PROVIDER_LOCK):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = path.open("a", encoding="utf-8")
    os.chmod(path, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def save_provider(payload):
    if not isinstance(payload, dict):
        raise ValueError("provider payload must be an object")
    name = _payload_text(payload, "name", required=True, maximum=64)
    base_url = _payload_text(payload, "base_url", required=True)
    env_key = _payload_text(payload, "env_key", maximum=128) or _default_env_key(name)
    api_key = _payload_text(payload, "api_key", maximum=8192, strip=False)

    with _provider_lock():
        providers = {item["name"]: item for item in read_provider_catalog()}
        existing = providers.get(name)
        if (existing is None or not existing["key_set"]) and not api_key:
            raise ValueError("api_key is required when the provider has no stored Key")
        if existing is not None and env_key and env_key != existing["env_key"] and not api_key:
            raise ValueError("api_key is required when env_key changes")
        command = ["update" if existing else "add", name, "--base-url", base_url]
        command.extend(["--env-key", env_key])
        command.extend(["--wire-api", "responses", "--skip-test"])
        _run_provider_command(command)
        if api_key:
            _run_provider_command(["set-key", name, "--stdin"], input_text=api_key)
    return name


def test_provider(payload):
    if not isinstance(payload, dict):
        raise ValueError("provider payload must be an object")
    name = _payload_text(payload, "name", required=True, maximum=64)
    base_url = _payload_text(payload, "base_url", required=True)
    env_key = _payload_text(payload, "env_key", maximum=128) or _default_env_key(name)
    api_key = _payload_text(payload, "api_key", maximum=8192, strip=False)
    model = _payload_text(payload, "model", required=True, maximum=64)
    if model not in MODEL_IDS:
        raise ValueError("model is not available")
    command = ["test", name, "--base-url", base_url, "--env-key", env_key, "--model", model]
    if api_key:
        command.append("--stdin")
    with _provider_lock():
        message = _run_provider_command(command, input_text=api_key or None).strip()
    status = "warning" if "API compatibility was not confirmed" in message else "success"
    return {"status": status, "message": message}


def switch_provider(payload):
    if not isinstance(payload, dict):
        raise ValueError("provider payload must be an object")
    name = _payload_text(payload, "name", required=True, maximum=64)
    with _provider_lock():
        providers = {item["name"]: item for item in read_provider_catalog()}
        provider = providers.get(name)
        if provider is None:
            raise ValueError(f"provider {name!r} not found")
        if not provider["key_set"]:
            raise ValueError(f"provider {name!r} has no API key")
        _run_provider_command(["switch", name])
        _sync_provider_gateway()
    return name


def delete_provider(payload):
    if not isinstance(payload, dict):
        raise ValueError("provider payload must be an object")
    name = _payload_text(payload, "name", required=True, maximum=64)
    confirm_name = _payload_text(payload, "confirm_name", required=True, maximum=64)
    if confirm_name != name:
        raise ValueError("provider deletion confirmation does not match")
    with _provider_lock():
        providers = {item["name"]: item for item in read_provider_catalog()}
        provider = providers.get(name)
        if provider is None:
            raise ValueError(f"provider {name!r} not found")
        if provider["active"]:
            raise ValueError("cannot delete the active provider; activate another provider first")
        _run_provider_command(["delete", name, "--yes"])
    return name


def build_state():
    codex = read_codex_state()
    return {
        "selection": read_selection(codex_state=codex),
        "client_route": {"model": CLIENT_MODEL, "effort": CLIENT_EFFORT},
        "provider": codex,
        "providers": read_provider_catalog(),
        "models": MODELS,
        "efforts": EFFORTS,
        "gateway": gateway_state(),
        "service_active": service_state(),
        "usage": {
            "count": usage_count(),
            "periods": usage_summary(),
            "last_sync": USAGE_STATUS["last_sync"],
            "error": USAGE_STATUS["last_error"],
        },
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Codex Routing Desk</title>
  <style>
    :root {
      --ink: #142126;
      --ink-soft: #4d5a5e;
      --paper: #f2ede1;
      --panel: #fbf8f0;
      --line: #c9c1b2;
      --signal: #ed6a3a;
      --teal: #197d78;
      --good: #287650;
      --bad: #b44835;
      --shadow: 0 18px 50px rgba(35, 43, 43, .12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(237, 106, 58, .16), transparent 28rem),
        radial-gradient(circle at 90% 85%, rgba(25, 125, 120, .13), transparent 32rem),
        linear-gradient(rgba(20, 33, 38, .035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20, 33, 38, .035) 1px, transparent 1px),
        var(--paper);
      background-size: auto, auto, 28px 28px, 28px 28px, auto;
    }
    button, input, select { font: inherit; }
    .shell { width: min(1480px, calc(100% - 40px)); margin: 0 auto; padding: 26px 0 44px; }
    .topbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-bottom: 18px; }
    .eyebrow, .mono {
      font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .eyebrow { margin: 0 0 7px; color: var(--signal); font-size: 12px; font-weight: 800; }
    h1 { margin: 0; font-size: clamp(31px, 3.7vw, 52px); line-height: .98; letter-spacing: -.045em; }
    .top-actions { display: flex; align-items: center; gap: 12px; padding-bottom: 1px; }
    .status-line { display: flex; align-items: center; gap: 9px; color: var(--ink-soft); font-size: 13px; white-space: nowrap; }
    .pulse { width: 10px; height: 10px; border-radius: 50%; background: var(--bad); box-shadow: 0 0 0 5px rgba(180,72,53,.12); }
    .pulse.good { background: var(--good); box-shadow: 0 0 0 5px rgba(40,118,80,.13); }
    .config-open {
      display: inline-flex;
      flex: 0 0 auto;
      align-items: center;
      gap: 8px;
      padding: 9px 13px;
      color: var(--ink);
      background: rgba(251,248,240,.72);
      border: 1px solid var(--line);
      border-radius: 999px;
      cursor: pointer;
      font: 800 11px "DejaVu Sans Mono", monospace;
      letter-spacing: .02em;
    }
    .config-open::before { color: var(--signal); content: "//"; }
    .config-open:hover { border-color: var(--signal); }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(290px, .8fr); gap: 14px; }
    .panel { background: rgba(251, 248, 240, .94); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow); overflow: hidden; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; padding: 15px 18px; border-bottom: 1px solid var(--line); }
    .panel-head h2 { margin: 0; font-size: 15px; letter-spacing: -.01em; }
    .shortcut { color: var(--ink-soft); font-size: 11px; }
    .model-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 14px; }
    .model-card {
      position: relative;
      min-height: 172px;
      padding: 16px;
      text-align: left;
      color: var(--ink);
      background: #f7f2e8;
      border: 1px solid var(--line);
      border-radius: 15px;
      cursor: pointer;
      transition: transform .18s ease, border-color .18s ease, background .18s ease;
      animation: rise .4s both;
    }
    .model-card:nth-child(2) { animation-delay: .06s; }
    .model-card:nth-child(3) { animation-delay: .12s; }
    .model-card:hover { transform: translateY(-3px); border-color: var(--ink); }
    button:focus-visible, input:focus-visible, select:focus-visible { outline: 3px solid rgba(25,125,120,.32); outline-offset: 2px; }
    .model-card.active { color: var(--panel); background: var(--ink); border-color: var(--ink); }
    .model-index { color: var(--signal); font: 800 11px "DejaVu Sans Mono", monospace; }
    .model-name { display: block; margin-top: 22px; font-size: 27px; font-weight: 800; letter-spacing: -.04em; }
    .model-role { display: block; margin-top: 3px; color: var(--teal); font: 700 11px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .active .model-role { color: #79d4cb; }
    .model-desc { display: block; margin-top: 14px; color: var(--ink-soft); font-size: 12px; line-height: 1.5; }
    .active .model-desc { color: #cbd4d3; }
    .effort-wrap { padding: 0 14px 16px; }
    .effort-label { display: flex; justify-content: space-between; margin: 2px 2px 8px; color: var(--ink-soft); font-size: 12px; }
    .effort { display: grid; grid-template-columns: repeat(5, 1fr); padding: 4px; border: 1px solid var(--line); border-radius: 13px; background: #ebe5d9; }
    .effort button { padding: 10px 5px; border: 0; border-radius: 9px; color: var(--ink-soft); background: transparent; cursor: pointer; font: 700 11px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .effort button.active { color: white; background: var(--teal); }
    .side { display: grid; align-content: start; gap: 14px; }
    .provider { padding: 18px; }
    .provider-name { margin: 3px 0 14px; font-size: 24px; font-weight: 800; letter-spacing: -.035em; }
    .facts { display: grid; gap: 8px; margin: 0; }
    .fact { display: grid; grid-template-columns: 72px 1fr; gap: 8px; padding-top: 9px; border-top: 1px solid var(--line); }
    .fact dt { color: var(--ink-soft); font: 700 10px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .fact dd { margin: 0; overflow-wrap: anywhere; font: 12px/1.45 "DejaVu Sans Mono", monospace; }
    .commit { padding: 14px 16px; background: var(--ink); color: var(--panel); border-color: var(--ink); }
    .commit-head { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .commit h3 { margin: 0; font-size: 14px; }
    .instant {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      width: 100%;
      margin-top: 10px;
      padding: 9px 11px;
      color: #dce5e3;
      background: #202f32;
      border: 1px solid #46575a;
      border-radius: 11px;
      cursor: pointer;
      text-align: left;
    }
    .instant:hover { border-color: #718285; }
    .instant-copy strong { font-size: 12px; }
    .instant-switch { position: relative; flex: 0 0 38px; height: 22px; border-radius: 20px; background: #536164; transition: background .18s ease; }
    .instant-switch::after { position: absolute; top: 3px; left: 3px; width: 16px; height: 16px; border-radius: 50%; background: #dce5e3; content: ""; transition: transform .18s ease; }
    .instant[aria-pressed="true"] .instant-switch { background: var(--teal); }
    .instant[aria-pressed="true"] .instant-switch::after { transform: translateX(16px); }
    .save { width: 100%; margin-top: 10px; padding: 11px 16px; border: 0; border-radius: 11px; color: #1c2527; background: var(--signal); cursor: pointer; font-weight: 900; transition: filter .15s ease, transform .15s ease; }
    .save:hover { filter: brightness(1.07); transform: translateY(-1px); }
    .save:disabled { color: #74807f; background: #435154; cursor: default; transform: none; }
    .save-state { color: #8f9d9b; font: 10px "DejaVu Sans Mono", monospace; white-space: nowrap; }
    .usage { margin-top: 14px; }
    .usage-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; margin: 0 2px 8px; }
    .usage-head h2 { margin: 0; font-size: 15px; }
    .usage-head p { margin: 0; color: var(--ink-soft); font-size: 11px; }
    .usage-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .usage-card { position: relative; padding: 16px 18px 15px; }
    .usage-card::before { position: absolute; inset: 0 0 auto; height: 4px; background: var(--teal); content: ""; }
    .usage-card:nth-child(2)::before { background: var(--signal); }
    .usage-card:nth-child(3)::before { background: var(--ink); }
    .usage-label { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; color: var(--ink-soft); font-size: 11px; }
    .usage-label strong { color: var(--ink); font-size: 13px; }
    .usage-total { display: block; margin-top: 12px; font: 800 clamp(28px, 3vw, 42px)/1 "DejaVu Sans Mono", monospace; letter-spacing: -.055em; }
    .usage-unit { margin-left: 7px; color: var(--ink-soft); font: 700 10px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .usage-breakdown { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 13px; padding-top: 11px; border-top: 1px solid var(--line); }
    .usage-breakdown span { color: var(--ink-soft); font-size: 10px; }
    .usage-breakdown strong { display: block; margin-top: 3px; color: var(--ink); font: 800 11px "DejaVu Sans Mono", monospace; }
    .last-request { display: grid; grid-template-columns: minmax(190px, .8fr) minmax(520px, 2.2fr); align-items: center; gap: 20px; margin-top: 10px; padding: 13px 18px; border-left: 4px solid var(--signal); }
    .last-request-title { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
    .last-request-label strong, .last-request-label span { display: block; }
    .last-request-label strong { font-size: 13px; }
    .last-request-label span { margin-top: 4px; color: var(--ink-soft); font: 10px "DejaVu Sans Mono", monospace; }
    .last-request-label .refresh-age { margin-top: 0; padding: 4px 8px; color: #9b3516; background: rgba(237,106,58,.14); border: 1px solid rgba(237,106,58,.42); border-radius: 999px; font-weight: 900; letter-spacing: .02em; }
    .last-request-breakdown { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .last-request-breakdown span { color: var(--ink-soft); font-size: 10px; }
    .last-request-breakdown strong { display: block; margin-top: 3px; color: var(--ink); font: 800 14px "DejaVu Sans Mono", monospace; }
    .ledger { margin-top: 14px; }
    .ledger-head { display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    .ledger-title { display: flex; align-items: baseline; gap: 10px; }
    .ledger-count { color: var(--signal); font: 800 11px "DejaVu Sans Mono", monospace; }
    .ledger-state { color: var(--ink-soft); font: 11px "DejaVu Sans Mono", monospace; }
    .table-scroll { max-height: 430px; overflow: auto; }
    table { width: 100%; min-width: 1300px; border-collapse: collapse; font-size: 12px; }
    thead th {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 12px 14px;
      color: var(--ink-soft);
      background: #eee8dc;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font: 800 10px "DejaVu Sans Mono", monospace;
      letter-spacing: .04em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    tbody td { height: 45px; padding: 9px 14px; border-bottom: 1px solid rgba(201,193,178,.65); white-space: nowrap; }
    tbody tr:nth-child(even) { background: rgba(232, 225, 212, .32); }
    tbody tr:hover { background: rgba(25, 125, 120, .07); }
    .metric { font-family: "DejaVu Sans Mono", monospace; font-variant-numeric: tabular-nums; }
    .model-cell { color: var(--teal); font-weight: 800; }
    .status { display: inline-flex; align-items: center; gap: 6px; font-weight: 800; }
    .status::before { width: 7px; height: 7px; border-radius: 50%; background: var(--good); content: ""; }
    .status.failed { color: var(--bad); }
    .status.failed::before { background: var(--bad); }
    .empty-row td { height: 110px; color: var(--ink-soft); text-align: center; }
    .foot { display: flex; justify-content: space-between; gap: 20px; margin-top: 15px; color: var(--ink-soft); font-size: 11px; }
    .config-layer { position: fixed; inset: 0; z-index: 20; }
    .config-layer[hidden] { display: none; }
    .config-scrim { position: absolute; inset: 0; width: 100%; border: 0; background: rgba(20,33,38,.42); backdrop-filter: blur(3px); cursor: default; }
    .config-drawer {
      position: absolute;
      top: 0;
      right: 0;
      width: min(470px, 100%);
      height: 100%;
      overflow: auto;
      padding: 28px;
      background: var(--panel);
      border-left: 1px solid var(--line);
      box-shadow: -24px 0 70px rgba(20,33,38,.22);
      animation: drawer-in .24s ease both;
    }
    .config-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; padding-bottom: 20px; border-bottom: 1px solid var(--line); }
    .config-head h2 { margin: 2px 0 0; font-size: 26px; letter-spacing: -.04em; }
    .config-close { width: 36px; height: 36px; color: var(--ink); background: transparent; border: 1px solid var(--line); border-radius: 50%; cursor: pointer; font-size: 20px; }
    .provider-picker { display: grid; grid-template-columns: 1fr auto; gap: 10px; margin-top: 22px; }
    .provider-picker select, .provider-form input {
      width: 100%;
      min-width: 0;
      padding: 11px 12px;
      color: var(--ink);
      background: #f6f1e7;
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .secondary-action { padding: 10px 13px; color: var(--teal); background: transparent; border: 1px solid var(--teal); border-radius: 10px; cursor: pointer; font-weight: 800; }
    .provider-form { display: grid; gap: 15px; margin-top: 22px; }
    .field { display: grid; gap: 7px; }
    .field label { color: var(--ink-soft); font: 800 10px "DejaVu Sans Mono", monospace; letter-spacing: .05em; text-transform: uppercase; }
    .field-note { color: var(--ink-soft); font-size: 10px; line-height: 1.45; }
    .key-state { display: inline-flex; width: fit-content; padding: 4px 8px; color: var(--bad); background: rgba(180,72,53,.08); border-radius: 999px; font: 800 10px "DejaVu Sans Mono", monospace; }
    .key-state.ready { color: var(--good); background: rgba(40,118,80,.1); }
    .provider-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 6px; }
    .provider-save, .provider-switch, .provider-test, .provider-delete { padding: 12px; border-radius: 11px; cursor: pointer; font-weight: 900; }
    .provider-test { color: var(--teal); background: transparent; border: 1px solid var(--teal); }
    .provider-save { color: #1c2527; background: var(--signal); border: 1px solid var(--signal); }
    .provider-switch { color: var(--panel); background: var(--ink); border: 1px solid var(--ink); }
    .provider-delete { color: var(--bad); background: transparent; border: 1px solid rgba(180,72,53,.65); }
    .provider-save:disabled, .provider-switch:disabled, .provider-test:disabled, .provider-delete:disabled { opacity: .45; cursor: not-allowed; }
    .provider-test-result {
      position: relative;
      display: grid;
      gap: 7px;
      min-width: 0;
      padding: 13px 14px 13px 38px;
      color: var(--teal);
      background: rgba(25,125,120,.07);
      border: 1px solid rgba(25,125,120,.32);
      border-radius: 11px;
    }
    .provider-test-result[hidden] { display: none; }
    .provider-test-result::before { position: absolute; top: 17px; left: 15px; width: 10px; height: 10px; border-radius: 50%; background: currentColor; content: ""; }
    .provider-test-result strong { font-size: 12px; }
    .provider-test-result span { color: var(--ink-soft); font: 10px/1.55 "DejaVu Sans Mono", monospace; overflow-wrap: anywhere; white-space: pre-wrap; }
    .provider-test-result.success { color: var(--good); background: rgba(40,118,80,.08); border-color: rgba(40,118,80,.32); }
    .provider-test-result.warning { color: #a84a20; background: rgba(237,106,58,.09); border-color: rgba(237,106,58,.4); }
    .provider-test-result.error { color: var(--bad); background: rgba(180,72,53,.08); border-color: rgba(180,72,53,.34); }
    .provider-test-result.running::before { animation: test-pulse 1s ease-in-out infinite alternate; }
    .config-footnote { margin: 18px 0 0; padding-top: 15px; color: var(--ink-soft); border-top: 1px solid var(--line); font-size: 10px; line-height: 1.55; }
    .toast { position: fixed; right: 22px; bottom: 22px; z-index: 30; max-width: 360px; padding: 13px 16px; border-radius: 11px; color: white; background: var(--good); box-shadow: var(--shadow); white-space: pre-line; transform: translateY(90px); opacity: 0; transition: .25s ease; }
    .toast.show { transform: translateY(0); opacity: 1; }
    .toast.error { background: var(--bad); }
    @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes drawer-in { from { opacity: 0; transform: translateX(28px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes test-pulse { from { opacity: .35; transform: scale(.78); } to { opacity: 1; transform: scale(1); } }
    @media (max-width: 900px) {
      .shell { width: calc(100% - 28px); }
      .model-card { min-height: 132px; padding: 12px; }
      .model-index { font-size: clamp(8px, 1vw, 10px); white-space: nowrap; }
      .model-name { margin-top: 18px; font-size: 24px; }
      .model-desc { display: none; }
    }
    @media (max-width: 700px) {
      .shell { width: min(100% - 22px, 680px); padding-top: 20px; }
      .topbar { align-items: flex-start; flex-direction: column; gap: 14px; margin-bottom: 16px; }
      .top-actions { width: 100%; align-items: flex-start; justify-content: space-between; }
      .workspace { grid-template-columns: 1fr; }
      .usage-head { align-items: flex-start; flex-direction: column; gap: 4px; }
      .usage-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .usage-card:last-child { grid-column: 1 / -1; }
      .usage-total { font-size: 34px; }
      .last-request { grid-template-columns: 1fr; gap: 13px; }
      .ledger-head { align-items: flex-start; flex-direction: column; }
      .table-scroll { max-height: 520px; }
      .foot { flex-direction: column; }
      .config-drawer { padding: 22px; }
    }
    @media (max-width: 520px) {
      .top-actions { gap: 10px; }
      .status-line { font-size: 11px; }
      .config-open { padding: 8px 11px; font-size: 10px; }
      .model-grid { gap: 7px; padding: 10px; }
      .model-card { min-height: 118px; padding: 10px; }
      .model-index { font-size: clamp(7px, 2vw, 9px); }
      .model-name { margin-top: 17px; font-size: 22px; }
      .model-role { font-size: 9px; }
      .effort-wrap { padding: 0 10px 12px; }
      .effort-label { font-size: 11px; }
      .effort button { padding: 9px 2px; font-size: 9px; }
      .provider { padding: 16px; }
      .provider-name { margin-bottom: 10px; }
      .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 10px; }
      .fact { display: block; padding-top: 8px; }
      .fact dd { margin-top: 5px; font-size: 10px; line-height: 1.4; }
      .fact:first-child dd { font-size: 9px; }
      .usage-grid { grid-template-columns: 1fr; }
      .usage-card:last-child { grid-column: auto; }
      .last-request-breakdown { grid-template-columns: repeat(2, 1fr); }
      .config-drawer { padding: 18px; }
    }
    @media (max-width: 360px) {
      .top-actions { flex-wrap: wrap; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div><p class="eyebrow">Local inference control</p><h1>Codex Routing Desk</h1></div>
      <div class="top-actions">
        <div class="status-line"><span id="pulse" class="pulse"></span><span id="health">Connecting to gateway...</span></div>
        <button id="provider-config-open" class="config-open" type="button">Provider config</button>
      </div>
    </header>
    <div class="workspace">
      <section class="panel" aria-labelledby="model-heading">
        <div class="panel-head"><h2 id="model-heading">GPT route</h2><span class="shortcut mono">keys 1—3</span></div>
        <div id="models" class="model-grid"></div>
        <div class="effort-wrap">
          <div class="effort-label"><span>Reasoning effort</span><span class="mono">Alt + 1—5</span></div>
          <div id="efforts" class="effort" role="group" aria-label="Reasoning effort"></div>
        </div>
      </section>
      <aside class="side">
        <section class="panel provider">
          <p class="eyebrow">Connection from Codex</p>
          <div id="provider-name" class="provider-name">—</div>
          <dl class="facts">
            <div class="fact"><dt>Base URL</dt><dd id="base-url">—</dd></div>
            <div class="fact"><dt>Wire</dt><dd id="wire-api">—</dd></div>
            <div class="fact"><dt>Env key</dt><dd id="env-key">—</dd></div>
            <div class="fact"><dt>Catalog</dt><dd id="catalog">—</dd></div>
          </dl>
        </section>
        <section class="panel commit">
          <div class="commit-head"><h3>Live route</h3><span id="save-state" class="save-state">Waiting…</span></div>
          <button id="instant" class="instant" type="button" aria-pressed="true">
            <span class="instant-copy"><strong>Instant switch</strong></span>
            <span class="instant-switch" aria-hidden="true"></span>
          </button>
          <button id="save" class="save" type="button" hidden disabled>Apply selection</button>
        </section>
      </aside>
    </div>
    <section class="usage" aria-labelledby="usage-heading">
      <div class="usage-head">
        <h2 id="usage-heading">Token usage</h2>
        <p>Calendar periods in local time · metadata only</p>
      </div>
      <div class="usage-grid">
        <article class="panel usage-card">
          <div class="usage-label"><strong>Today</strong><span>00:00 → now</span></div>
          <span id="usage-day-total" class="usage-total">0</span><span class="usage-unit">tokens</span>
          <div class="usage-breakdown">
            <span>Input<strong id="usage-day-input">0</strong></span>
            <span>Cache read<strong id="usage-day-cache-read">0</strong></span>
            <span>Output<strong id="usage-day-output">0</strong></span>
            <span>Requests<strong id="usage-day-requests">0</strong></span>
          </div>
        </article>
        <article class="panel usage-card">
          <div class="usage-label"><strong>This week</strong><span>Monday → now</span></div>
          <span id="usage-week-total" class="usage-total">0</span><span class="usage-unit">tokens</span>
          <div class="usage-breakdown">
            <span>Input<strong id="usage-week-input">0</strong></span>
            <span>Cache read<strong id="usage-week-cache-read">0</strong></span>
            <span>Output<strong id="usage-week-output">0</strong></span>
            <span>Requests<strong id="usage-week-requests">0</strong></span>
          </div>
        </article>
        <article class="panel usage-card">
          <div class="usage-label"><strong>This month</strong><span>Day 1 → now</span></div>
          <span id="usage-month-total" class="usage-total">0</span><span class="usage-unit">tokens</span>
          <div class="usage-breakdown">
            <span>Input<strong id="usage-month-input">0</strong></span>
            <span>Cache read<strong id="usage-month-cache-read">0</strong></span>
            <span>Output<strong id="usage-month-output">0</strong></span>
            <span>Requests<strong id="usage-month-requests">0</strong></span>
          </div>
        </article>
      </div>
      <article class="panel last-request">
        <div class="last-request-label">
          <div class="last-request-title"><strong>Last request</strong><span id="refresh-age" class="refresh-age" aria-live="polite">等待刷新</span></div>
          <span id="last-request-meta">No requests captured</span>
        </div>
        <div class="last-request-breakdown">
          <span>Input<strong id="last-request-input">0</strong></span>
          <span>Output<strong id="last-request-output">0</strong></span>
          <span>Cache read<strong id="last-request-cache-read">0</strong></span>
          <span>Cache hit<strong id="last-request-cache-hit">—</strong></span>
        </div>
      </article>
    </section>
    <section class="panel ledger" aria-labelledby="ledger-heading">
      <div class="panel-head ledger-head">
        <div class="ledger-title"><h2 id="ledger-heading">Request ledger</h2><span id="request-count" class="ledger-count">0 captured</span></div>
        <span id="ledger-state" class="ledger-state">Waiting for usage stream…</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr>
            <th>统计时间</th><th>API Key</th><th>凭据</th><th>实际模型</th><th>推理</th><th>接口</th>
            <th>输入</th><th>输出</th><th>推理 Token</th><th>缓存读取</th><th>缓存创建</th><th>缓存命中率</th><th>首 Token / 总耗时</th><th>状态</th>
          </tr></thead>
          <tbody id="requests"><tr class="empty-row"><td colspan="14">完成一次 Claude 请求后，这里会显示真实的上游路由和用量。</td></tr></tbody>
        </table>
      </div>
    </section>
    <footer class="foot"><span>Only request metadata is stored locally; prompts and responses are not recorded.</span><span class="mono">Ctrl+S save · auto refresh 3s</span></footer>
  </main>
  <div id="provider-config-layer" class="config-layer" hidden>
    <button id="provider-config-scrim" class="config-scrim" type="button" aria-label="Close provider configuration"></button>
    <aside class="config-drawer" role="dialog" aria-modal="true" aria-labelledby="provider-config-title">
      <header class="config-head">
        <div><p class="eyebrow">Shared Codex route</p><h2 id="provider-config-title">Provider config</h2></div>
        <button id="provider-config-close" class="config-close" type="button" aria-label="Close">x</button>
      </header>
      <div class="provider-picker">
        <select id="provider-picker" aria-label="Installed providers"></select>
        <button id="provider-new" class="secondary-action" type="button">New</button>
      </div>
      <form id="provider-form" class="provider-form">
        <div class="field">
          <label for="provider-edit-name">Provider name</label>
          <input id="provider-edit-name" name="name" maxlength="64" required autocomplete="off">
          <span class="field-note">Existing provider names cannot be renamed. Create a new provider instead.</span>
        </div>
        <div class="field">
          <label for="provider-edit-url">Base URL</label>
          <input id="provider-edit-url" name="base_url" type="url" maxlength="2048" placeholder="https://gateway.example/openai" required autocomplete="url">
        </div>
        <div class="field">
          <label for="provider-edit-env">Environment key</label>
          <input id="provider-edit-env" name="env_key" maxlength="128" placeholder="MY_PROVIDER_OPENAI_KEY" autocomplete="off">
          <span class="field-note">Leave blank to derive PROVIDER_NAME_OPENAI_KEY.</span>
        </div>
        <div class="field">
          <label for="provider-edit-wire">Wire API</label>
          <input id="provider-edit-wire" value="responses" readonly>
        </div>
        <div class="field">
          <label for="provider-edit-key">API Key</label>
          <input id="provider-edit-key" name="api_key" type="password" maxlength="8192" placeholder="Leave blank to keep the existing Key" autocomplete="new-password">
          <span id="provider-key-state" class="key-state">Key missing</span>
        </div>
        <div class="provider-actions">
          <button id="provider-test" class="provider-test" type="button">Test</button>
          <button id="provider-save" class="provider-save" type="submit">Save</button>
          <button id="provider-delete" class="provider-delete" type="button">Delete</button>
          <button id="provider-switch" class="provider-switch" type="button">Activate</button>
        </div>
        <div id="provider-test-result" class="provider-test-result" role="status" aria-live="polite" hidden>
          <strong id="provider-test-title"></strong>
          <span id="provider-test-detail"></span>
        </div>
      </form>
      <p class="config-footnote">The built-in provider backend handles all actions. Save stores the provider without changing the active route. The Key is stored locally with mode 0600 and is never returned to this page. Test sends a minimal live Responses request with the model currently selected on this page, which uses a small amount of Provider tokens. Activate switches Codex, regenerates the claudex gateway, and restarts CLIProxyAPI. Delete is available only for inactive providers.</p>
    </aside>
  </div>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script>
    const app = {
      state: null,
      draft: null,
      dirty: false,
      requests: [],
      rendered: {models: '', efforts: '', requests: ''},
      autoApply: localStorage.getItem('claudex-instant-switch') !== 'false',
      saving: false,
      saveTimer: null,
      providerSaving: false,
      providerAction: '',
      providerIsNew: false,
    };
    const $ = (id) => document.getElementById(id);
    const escapeText = (value) => value || '—';
    const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
    const formatTokens = (value) => {
      const number = Number(value || 0);
      if (number >= 1000000) return `${(number / 1000000).toFixed(1)}M`;
      if (number >= 1000) return `${(number / 1000).toFixed(1)}K`;
      return number.toLocaleString();
    };
    const formatTime = (value) => {
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString('zh-CN', {hour12: false});
    };
    function refreshRequestAge() {
      const timestamp = app.requests[0]?.timestamp;
      const requestedAt = timestamp ? new Date(timestamp).getTime() : NaN;
      if (!Number.isFinite(requestedAt)) {
        $('refresh-age').textContent = '暂无请求';
        return;
      }
      const seconds = Math.max(0, Math.floor((Date.now() - requestedAt) / 1000));
      $('refresh-age').textContent = seconds === 0 ? '刚刚请求' : `${seconds} 秒前请求`;
    }

    function renderUsage() {
      const periods = app.state.usage.periods || {};
      ['day', 'week', 'month'].forEach((name) => {
        const period = periods[name] || {};
        $(`usage-${name}-total`).textContent = formatTokens(period.total_tokens);
        $(`usage-${name}-input`).textContent = formatTokens(period.input_tokens);
        $(`usage-${name}-cache-read`).textContent = formatTokens(period.cache_read_tokens);
        $(`usage-${name}-output`).textContent = formatTokens(period.output_tokens);
        $(`usage-${name}-requests`).textContent = Number(period.requests || 0).toLocaleString();
      });
      const last = app.requests[0];
      const totalInput = Number(last?.input_tokens || 0);
      const cacheRead = Number(last?.cache_read_tokens || 0);
      const uncachedInput = Math.max(totalInput - cacheRead, 0);
      $('last-request-input').textContent = formatTokens(uncachedInput);
      $('last-request-output').textContent = formatTokens(last?.output_tokens);
      $('last-request-cache-read').textContent = formatTokens(cacheRead);
      $('last-request-cache-hit').textContent = totalInput > 0 ? `${(cacheRead / totalInput * 100).toFixed(1)}%` : '—';
      $('last-request-meta').textContent = last ? `${formatTime(last.timestamp)} · ${last.model || 'unknown model'}` : 'No requests captured';
      refreshRequestAge();
    }

    function renderRequests() {
      const rows = app.requests;
      $('request-count').textContent = `${app.state.usage.count} captured`;
      $('ledger-state').textContent = app.state.usage.error ? `Collector error: ${app.state.usage.error}` : 'Live metadata · local SQLite';
      const signature = JSON.stringify(rows);
      if (app.rendered.requests === signature) return;
      app.rendered.requests = signature;
      if (!rows.length) {
        $('requests').innerHTML = '<tr class="empty-row"><td colspan="14">完成一次 Claude 请求后，这里会显示真实的上游路由和用量。</td></tr>';
        return;
      }
      $('requests').innerHTML = rows.map((row) => {
        const totalInput = Number(row.input_tokens || 0);
        const cacheRead = Number(row.cache_read_tokens || 0);
        const freshInput = Math.max(totalInput - cacheRead, 0);
        const hitRate = totalInput > 0 ? `${(cacheRead / totalInput * 100).toFixed(1)}%` : '—';
        const credential = row.auth_index ? row.auth_index.slice(0, 10) : row.provider;
        const latency = Number(row.latency_ms || 0);
        const ttft = Number(row.ttft_ms || 0);
        const latencyText = `${ttft ? ttft.toLocaleString() : '—'} / ${latency ? latency.toLocaleString() : '—'}ms`;
        const generationTime = Math.max(latency - ttft, 0);
        return `<tr>
          <td class="metric">${escapeHtml(formatTime(row.timestamp))}</td>
          <td class="metric">${escapeHtml(row.api_key || 'claudex-local')}</td>
          <td class="metric" title="${escapeHtml(row.auth_index)}">${escapeHtml(credential || '—')}</td>
          <td class="model-cell">${escapeHtml(row.model || '—')}</td>
          <td class="metric">${escapeHtml(row.reasoning_effort || '—')}</td>
          <td class="metric">${escapeHtml(row.endpoint || '—')}</td>
          <td class="metric">${formatTokens(freshInput)}</td>
          <td class="metric">${formatTokens(row.output_tokens)}</td>
          <td class="metric">${formatTokens(row.reasoning_tokens)}</td>
          <td class="metric">${formatTokens(cacheRead)}</td>
          <td class="metric">${formatTokens(row.cache_creation_tokens)}</td>
          <td class="metric">${hitRate}</td>
          <td class="metric" title="首 Token ${ttft.toLocaleString()}ms · 后续生成 ${generationTime.toLocaleString()}ms">${latencyText}</td>
          <td><span class="status ${row.failed ? 'failed' : ''}">${row.failed ? `HTTP ${row.status_code || 'ERR'}` : 'OK'}</span></td>
        </tr>`;
      }).join('');
    }

    function render() {
      const state = app.state;
      if (!state) return;
      const modelCatalog = JSON.stringify(state.models);
      if (app.rendered.models !== modelCatalog) {
        app.rendered.models = modelCatalog;
        $('models').innerHTML = state.models.map((model, index) => `
          <button class="model-card" data-model="${model.id}" aria-pressed="false">
            <span class="model-index">0${index + 1} / ${model.id}</span>
            <span class="model-name">${model.name}</span><span class="model-role">${model.role}</span>
            <span class="model-desc">${model.description}</span>
          </button>`).join('');
        $('models').querySelectorAll('[data-model]').forEach((button) => button.addEventListener('click', () => chooseModel(button.dataset.model)));
      }
      $('models').querySelectorAll('[data-model]').forEach((button) => {
        const selected = app.draft.model === button.dataset.model;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-pressed', String(selected));
      });
      const effortCatalog = JSON.stringify(state.efforts);
      if (app.rendered.efforts !== effortCatalog) {
        app.rendered.efforts = effortCatalog;
        $('efforts').innerHTML = state.efforts.map((effort) => `<button data-effort="${effort}" aria-pressed="false">${effort}</button>`).join('');
        $('efforts').querySelectorAll('[data-effort]').forEach((button) => button.addEventListener('click', () => chooseEffort(button.dataset.effort)));
      }
      $('efforts').querySelectorAll('[data-effort]').forEach((button) => {
        const selected = app.draft.effort === button.dataset.effort;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-pressed', String(selected));
      });
      $('provider-name').textContent = escapeText(state.provider.provider);
      $('base-url').textContent = escapeText(state.provider.base_url);
      $('wire-api').textContent = escapeText(state.provider.wire_api);
      $('env-key').textContent = escapeText(state.provider.env_key);
      $('catalog').textContent = state.gateway.models.length ? state.gateway.models.join(' · ') : 'Unavailable';
      const healthy = state.service_active && state.gateway.reachable;
      const activeProvider = (state.providers || []).find((provider) => provider.active);
      $('pulse').classList.toggle('good', healthy);
      $('health').textContent = healthy
        ? 'Gateway online · 127.0.0.1:8317'
        : activeProvider && !activeProvider.key_set ? 'Provider setup required' : 'Gateway unavailable';
      $('instant').setAttribute('aria-pressed', String(app.autoApply));
      $('save').hidden = app.autoApply;
      $('save').disabled = app.saving || !app.dirty;
      $('save').textContent = app.saving ? 'Applying…' : app.dirty ? 'Apply selection' : 'Saved';
      $('save-state').textContent = app.saving ? 'Applying…' : `${app.draft.model} · ${app.draft.effort}`;
      renderUsage();
      renderRequests();
    }

    function scheduleSave() {
      clearTimeout(app.saveTimer);
      app.saveTimer = setTimeout(save, 120);
    }
    function selectionChanged() {
      app.dirty = true;
      render();
      if (app.autoApply) scheduleSave();
    }
    function chooseModel(model) {
      if (app.draft.model === model) return;
      app.draft.model = model;
      selectionChanged();
    }
    function chooseEffort(effort) {
      if (app.draft.effort === effort) return;
      app.draft.effort = effort;
      selectionChanged();
    }
    function toggleInstantSwitch() {
      app.autoApply = !app.autoApply;
      localStorage.setItem('claudex-instant-switch', String(app.autoApply));
      if (!app.autoApply) clearTimeout(app.saveTimer);
      render();
      if (app.autoApply && app.dirty) scheduleSave();
    }
    function notify(message, error = false) {
      const toast = $('toast'); toast.textContent = message; toast.className = `toast show${error ? ' error' : ''}`;
      clearTimeout(notify.timer); notify.timer = setTimeout(() => toast.classList.remove('show'), 3200);
    }
    function clearProviderTestResult() {
      const result = $('provider-test-result');
      result.hidden = true;
      result.className = 'provider-test-result';
      $('provider-test-title').textContent = '';
      $('provider-test-detail').textContent = '';
    }
    function setProviderTestResult(status, title, detail = '') {
      const result = $('provider-test-result');
      result.className = `provider-test-result ${status}`;
      $('provider-test-title').textContent = title;
      $('provider-test-detail').textContent = detail;
      result.hidden = false;
      result.scrollIntoView({block: 'nearest'});
    }

    function providerByName(name) {
      return (app.state?.providers || []).find((provider) => provider.name === name);
    }
    function populateProviderPicker(selectedName) {
      const providers = app.state?.providers || [];
      $('provider-picker').innerHTML = providers.length
        ? providers.map((provider) => `<option value="${escapeHtml(provider.name)}">${escapeHtml(provider.name)}${provider.active ? ' · active' : ''}</option>`).join('')
        : '<option value="">No providers configured</option>';
      if (selectedName && providerByName(selectedName)) $('provider-picker').value = selectedName;
    }
    function providerFormChanged(provider) {
      if (!provider) return true;
      return $('provider-edit-name').value.trim() !== provider.name
        || $('provider-edit-url').value.trim() !== provider.base_url
        || $('provider-edit-env').value.trim() !== provider.env_key
        || Boolean($('provider-edit-key').value);
    }
    function updateProviderActions(provider) {
      const busy = app.providerSaving;
      const changed = providerFormChanged(provider);
      $('provider-picker').disabled = busy;
      $('provider-new').disabled = busy;
      $('provider-test').disabled = busy;
      $('provider-save').disabled = busy || (!app.providerIsNew && !changed);
      $('provider-switch').disabled = busy || !provider || !provider.key_set || changed;
      $('provider-switch').title = changed ? 'Save changes before activation.' : '';
      $('provider-delete').disabled = busy || !provider || provider.active;
      $('provider-delete').title = provider?.active ? 'Activate another provider before deleting this one.' : '';
      $('provider-test').textContent = app.providerAction === 'test' ? 'Testing...' : 'Test';
      $('provider-save').textContent = app.providerAction === 'save' ? 'Saving...' : 'Save';
      $('provider-delete').textContent = app.providerAction === 'delete' ? 'Deleting...' : 'Delete';
      $('provider-switch').textContent = app.providerAction === 'activate' ? 'Activating...' : 'Activate';
    }
    function editProvider(provider) {
      clearProviderTestResult();
      app.providerIsNew = !provider;
      $('provider-edit-name').value = provider?.name || '';
      $('provider-edit-name').readOnly = Boolean(provider);
      $('provider-edit-url').value = provider?.base_url || '';
      $('provider-edit-env').value = provider?.env_key || '';
      $('provider-edit-key').value = '';
      $('provider-edit-key').placeholder = provider?.key_set ? 'Leave blank to keep the existing Key' : 'API Key required';
      $('provider-edit-key').required = !provider?.key_set;
      $('provider-key-state').textContent = provider?.key_set ? 'Key stored' : 'Key missing';
      $('provider-key-state').classList.toggle('ready', Boolean(provider?.key_set));
      updateProviderActions(provider);
    }
    function openProviderConfig() {
      const current = app.state?.provider?.provider || app.state?.providers?.[0]?.name;
      populateProviderPicker(current);
      editProvider(providerByName(current));
      $('provider-config-layer').hidden = false;
      document.body.style.overflow = 'hidden';
      $('provider-picker').focus();
    }
    function closeProviderConfig() {
      if (app.providerSaving) return;
      $('provider-config-layer').hidden = true;
      document.body.style.overflow = '';
      $('provider-config-open').focus();
    }
    function newProvider() {
      $('provider-picker').value = '';
      editProvider(null);
      $('provider-edit-name').focus();
    }
    function providerFormPayload() {
      return {
        name: $('provider-edit-name').value,
        base_url: $('provider-edit-url').value,
        env_key: $('provider-edit-env').value,
        api_key: $('provider-edit-key').value,
      };
    }
    async function providerStateRequest(path, payload, action, successMessage, selectedName) {
      if (app.providerSaving) return;
      app.providerSaving = true;
      app.providerAction = action;
      updateProviderActions(app.providerIsNew ? null : providerByName($('provider-picker').value));
      try {
        const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
        const state = await response.json();
        if (!response.ok) throw new Error(state.error || `HTTP ${response.status}`);
        app.state = state;
        if (!app.dirty) app.draft = {...state.selection};
        const nextName = typeof selectedName === 'function' ? selectedName(state) : selectedName;
        populateProviderPicker(nextName);
        editProvider(providerByName(nextName));
        render();
        notify(successMessage);
      } catch (error) {
        notify(`Provider ${action} failed: ${error.message}`, true);
      } finally {
        app.providerSaving = false;
        app.providerAction = '';
        if (!$('provider-config-layer').hidden) updateProviderActions(app.providerIsNew ? null : providerByName($('provider-picker').value));
      }
    }
    async function testProviderConnection() {
      if (app.providerSaving || !$('provider-form').reportValidity()) return;
      app.providerSaving = true;
      app.providerAction = 'test';
      setProviderTestResult('running', `Testing ${app.draft.model}...`, 'Sending a minimal live Responses request.');
      updateProviderActions(app.providerIsNew ? null : providerByName($('provider-picker').value));
      try {
        const response = await fetch('/api/providers/test', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...providerFormPayload(), model: app.draft.model})});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
        const warning = result.status === 'warning';
        setProviderTestResult(result.status, warning ? 'Reachable with warning' : 'Connection passed', result.message);
      } catch (error) {
        setProviderTestResult('error', 'Connection failed', error.message);
      } finally {
        app.providerSaving = false;
        app.providerAction = '';
        if (!$('provider-config-layer').hidden) updateProviderActions(app.providerIsNew ? null : providerByName($('provider-picker').value));
      }
    }
    function saveProvider(event) {
      event.preventDefault();
      if (!$('provider-form').reportValidity()) return;
      const payload = providerFormPayload();
      providerStateRequest('/api/providers/save', payload, 'save', 'Provider saved. Activate it to apply the route.', payload.name.trim());
    }
    function useSelectedProvider() {
      const provider = providerByName($('provider-picker').value);
      if (provider && !providerFormChanged(provider)) {
        providerStateRequest('/api/providers/switch', {name: provider.name}, 'activate', `Activated ${provider.name}.`, provider.name);
      }
    }
    function deleteSelectedProvider() {
      const provider = providerByName($('provider-picker').value);
      if (!provider || provider.active) return;
      if (!window.confirm(`Delete provider "${provider.name}" and its unshared stored Key?`)) return;
      providerStateRequest(
        '/api/providers/delete',
        {name: provider.name, confirm_name: provider.name},
        'delete',
        `Deleted ${provider.name}.`,
        (state) => state.provider.provider,
      );
    }

    async function refresh() {
      try {
        const [response, requestsResponse] = await Promise.all([
          fetch('/api/state', {cache: 'no-store'}),
          fetch('/api/requests?limit=80', {cache: 'no-store'}),
        ]);
        if (!response.ok || !requestsResponse.ok) throw new Error(`HTTP ${response.status}/${requestsResponse.status}`);
        const [state, requestPayload] = await Promise.all([response.json(), requestsResponse.json()]);
        app.state = state;
        app.requests = requestPayload.requests;
        if (!app.dirty) app.draft = {...state.selection};
        render();
      } catch (error) { notify(`Cannot read service state: ${error.message}`, true); }
    }

    async function save() {
      if (!app.dirty || app.saving) return;
      clearTimeout(app.saveTimer);
      const submitted = {...app.draft};
      let saveLatest = false;
      app.saving = true;
      render();
      try {
        const response = await fetch('/api/selection', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(submitted)});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        app.state = payload;
        const unchanged = app.draft.model === submitted.model && app.draft.effort === submitted.effort;
        if (unchanged) {
          app.dirty = false;
          app.draft = {...payload.selection};
        } else {
          saveLatest = app.autoApply;
        }
        notify('Live route updated. Active sessions switch on the next request.');
      } catch (error) { notify(`Save failed: ${error.message}`, true); }
      finally {
        app.saving = false;
        render();
        if (saveLatest) scheduleSave();
      }
    }

    $('instant').addEventListener('click', toggleInstantSwitch);
    $('save').addEventListener('click', save);
    $('provider-config-open').addEventListener('click', openProviderConfig);
    $('provider-config-close').addEventListener('click', closeProviderConfig);
    $('provider-config-scrim').addEventListener('click', closeProviderConfig);
    $('provider-new').addEventListener('click', newProvider);
    $('provider-picker').addEventListener('change', () => editProvider(providerByName($('provider-picker').value)));
    $('provider-form').addEventListener('input', () => {
      clearProviderTestResult();
      updateProviderActions(app.providerIsNew ? null : providerByName($('provider-picker').value));
    });
    $('provider-form').addEventListener('submit', saveProvider);
    $('provider-test').addEventListener('click', testProviderConnection);
    $('provider-delete').addEventListener('click', deleteSelectedProvider);
    $('provider-switch').addEventListener('click', useSelectedProvider);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !$('provider-config-layer').hidden) { closeProviderConfig(); return; }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's' && !$('provider-config-layer').hidden) {
        event.preventDefault(); $('provider-form').requestSubmit(); return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); return; }
      if (event.target.matches('input, textarea, select')) return;
      const index = Number(event.key) - 1;
      if (!event.altKey && index >= 0 && index < 3 && app.state) chooseModel(app.state.models[index].id);
      if (event.altKey && index >= 0 && index < 5 && app.state) { event.preventDefault(); chooseEffort(app.state.efforts[index]); }
    });
    refresh();
    setInterval(refreshRequestAge, 1000);
    setInterval(() => { if (!app.dirty) refresh(); }, 3000);
  </script>
</body>
</html>'''


class ManagerHandler(BaseHTTPRequestHandler):
    server_version = "CodexRoutingDesk/1.0"

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self):
        origin = self.headers.get("Origin")
        return not origin or origin in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}

    def _read_json(self):
        if not self._same_origin() or self.headers.get_content_type() != "application/json":
            raise PermissionError("request origin or content type is not allowed")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 16384:
            raise ValueError("invalid request size")
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/state":
            try:
                self._send_json(200, build_state())
            except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
                self._send_json(500, {"error": str(error)})
            return
        if self.path.startswith("/api/requests"):
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != "/api/requests":
                    self.send_error(404)
                    return
                values = urllib.parse.parse_qs(parsed.query)
                limit = int(values.get("limit", ["50"])[0])
                self._send_json(200, {"requests": list_usage_records(limit)})
            except (OSError, ValueError, sqlite3.Error) as error:
                self._send_json(400, {"error": str(error)})
            return
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path not in {
            "/api/selection",
            "/api/providers/save",
            "/api/providers/test",
            "/api/providers/switch",
            "/api/providers/delete",
        }:
            self.send_error(404)
            return
        try:
            payload = self._read_json()
            if self.path == "/api/selection":
                apply_selection(payload.get("model"), payload.get("effort"))
            elif self.path == "/api/providers/save":
                save_provider(payload)
            elif self.path == "/api/providers/test":
                self._send_json(200, test_provider(payload))
                return
            elif self.path == "/api/providers/switch":
                switch_provider(payload)
            else:
                delete_provider(payload)
            time.sleep(0.35)
            self._send_json(200, build_state())
        except PermissionError as error:
            self._send_json(403, {"error": str(error)})
        except (json.JSONDecodeError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
            self._send_json(400, {"error": str(error)})

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop_event = threading.Event()
    collector = threading.Thread(target=usage_collector, args=(stop_event,), name="usage-collector", daemon=True)
    collector.start()
    server = ThreadingHTTPServer((HOST, PORT), ManagerHandler)
    server.daemon_threads = True
    print(f"Codex Routing Desk listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        collector.join(timeout=2)
        server.server_close()


if __name__ == "__main__":
    main()

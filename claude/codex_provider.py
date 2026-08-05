#!/usr/bin/env python3
"""Manage Codex custom model providers, profiles, and API keys."""

from __future__ import annotations

import argparse
import curses
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
CONFIG = CODEX_HOME / "config.toml"
SECRETS = Path.home() / ".config" / "codex" / "secrets.env"

TOP_LEVEL_KEYS = (
    "model_provider",
    "model",
    "model_reasoning_effort",
    "model_reasoning_summary",
    "model_verbosity",
)


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def toml_quote(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        die("provider name must start with a letter and may only contain letters, numbers, hyphen, and underscore")
    reserved = {"openai", "ollama", "lmstudio"}
    if name in reserved:
        die(f"{name!r} is a reserved built-in provider id")
    return name


def validate_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        die("base_url must start with http:// or https://")
    if not parsed.hostname:
        die("base_url must include a host, for example http://127.0.0.1:3000/v1")
    return base_url.rstrip("/")


def default_env_key(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]", "_", name).upper()
    return f"{stem}_OPENAI_KEY"


def read_config() -> str:
    if not CONFIG.exists():
        CODEX_HOME.mkdir(parents=True, exist_ok=True)
        return ""
    return CONFIG.read_text()


def backup_config() -> None:
    if CONFIG.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(CONFIG, CONFIG.with_name(f"config.toml.backup.{stamp}"))


def table_pattern(table: str) -> re.Pattern[str]:
    escaped = re.escape(table)
    return re.compile(rf"(?ms)^\[{escaped}\]\n.*?(?=^\[[^\n]+\]\n|\Z)")


def remove_table(text: str, table: str) -> str:
    return table_pattern(table).sub("", text).rstrip() + "\n"


def upsert_top_level(text: str, updates: dict[str, str]) -> str:
    lines = text.splitlines()
    first_table = next((i for i, line in enumerate(lines) if line.startswith("[")), len(lines))
    prefix = lines[:first_table]
    suffix = lines[first_table:]

    seen: set[str] = set()
    new_prefix: list[str] = []
    for line in prefix:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in updates:
            new_prefix.append(f"{key} = {toml_quote(updates[key])}")
            seen.add(key)
        else:
            new_prefix.append(line)

    insert = [f"{key} = {toml_quote(value)}" for key, value in updates.items() if key not in seen]
    return "\n".join(insert + new_prefix + suffix).strip() + "\n"


def provider_block(name: str, base_url: str, env_key: str, wire_api: str) -> str:
    return "\n".join(
        [
            f"[model_providers.{name}]",
            f"name = {toml_quote(name)}",
            f"base_url = {toml_quote(base_url)}",
            f"wire_api = {toml_quote(wire_api)}",
            "requires_openai_auth = false",
            f"env_key = {toml_quote(env_key)}",
            "",
        ]
    )


def upsert_provider(name: str, base_url: str, env_key: str, wire_api: str) -> None:
    text = read_config()
    backup_config()
    text = remove_table(text, f"model_providers.{name}")
    CONFIG.write_text(text.rstrip() + "\n\n" + provider_block(name, base_url, env_key, wire_api))
    CONFIG.chmod(0o600)


def write_profile(name: str, model: str | None, effort: str | None, summary: str | None, verbosity: str | None) -> Path:
    profile = CODEX_HOME / f"{name}.config.toml"
    lines = [f"model_provider = {toml_quote(name)}"]
    if model:
        lines.append(f"model = {toml_quote(model)}")
    if effort:
        lines.append(f"model_reasoning_effort = {toml_quote(effort)}")
    if summary:
        lines.append(f"model_reasoning_summary = {toml_quote(summary)}")
    if verbosity:
        lines.append(f"model_verbosity = {toml_quote(verbosity)}")
    profile.write_text("\n".join(lines) + "\n")
    profile.chmod(0o600)
    return profile


def parse_profile(name: str) -> dict[str, str]:
    profile = CODEX_HOME / f"{name}.config.toml"
    if not profile.exists():
        return {}
    return dict(re.findall(r'^([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$', profile.read_text(), re.M))


def update_profile(
    name: str,
    model: str | None = None,
    effort: str | None = None,
    summary: str | None = None,
    verbosity: str | None = None,
) -> Path:
    current = parse_profile(name)
    return write_profile(
        name,
        model if model is not None else current.get("model"),
        effort if effort is not None else current.get("model_reasoning_effort"),
        summary if summary is not None else current.get("model_reasoning_summary"),
        verbosity if verbosity is not None else current.get("model_verbosity"),
    )


def profile_file(name: str) -> Path:
    return CODEX_HOME / f"{name}.config.toml"


def profile_seed(
    name: str,
    top: dict[str, str],
    existing: dict[str, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    if top.get("model_provider") == name:
        return (
            top.get("model"),
            top.get("model_reasoning_effort"),
            top.get("model_reasoning_summary"),
            top.get("model_verbosity"),
        )
    return (
        existing.get("model"),
        existing.get("model_reasoning_effort"),
        existing.get("model_reasoning_summary"),
        existing.get("model_verbosity"),
    )


def ensure_secrets_file() -> None:
    SECRETS.parent.mkdir(parents=True, exist_ok=True)
    SECRETS.parent.chmod(0o700)
    if not SECRETS.exists():
        SECRETS.write_text("# Codex provider API keys. Source this file from your shell rc.\n")
    SECRETS.chmod(0o600)


def upsert_secret(env_key: str, api_key: str) -> None:
    ensure_secrets_file()
    lines = SECRETS.read_text().splitlines()
    pattern = re.compile(rf"^\s*export\s+{re.escape(env_key)}=")
    replacement = f"export {env_key}={shell_single_quote(api_key)}"
    replaced = False
    out = []
    for line in lines:
        if pattern.match(line):
            out.append(replacement)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(replacement)
    SECRETS.write_text("\n".join(out).rstrip() + "\n")
    SECRETS.chmod(0o600)


def remove_secret(env_key: str) -> None:
    if not SECRETS.exists():
        return
    pattern = re.compile(rf"^\s*export\s+{re.escape(env_key)}=")
    lines = [line for line in SECRETS.read_text().splitlines() if not pattern.match(line)]
    SECRETS.write_text("\n".join(lines).rstrip() + "\n")
    SECRETS.chmod(0o600)


def parse_providers() -> dict[str, dict[str, str]]:
    text = read_config()
    providers: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"(?ms)^\[model_providers\.([^\]]+)\]\n(.*?)(?=^\[[^\n]+\]\n|\Z)", text):
        name, body = match.group(1), match.group(2)
        data: dict[str, str] = {}
        for key, value in re.findall(r'^([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$', body, re.M):
            data[key] = value
        providers[name] = data
    return providers


def parse_top_level() -> dict[str, str]:
    text = read_config()
    before_tables = text.split("\n[", 1)[0]
    return dict(re.findall(r'^([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$', before_tables, re.M))


def env_available(env_key: str) -> bool:
    if os.environ.get(env_key):
        return True
    if SECRETS.exists():
        return re.search(rf"^\s*export\s+{re.escape(env_key)}=", SECRETS.read_text(), re.M) is not None
    return False


def read_secret_value(env_key: str) -> str | None:
    if os.environ.get(env_key):
        return os.environ[env_key]
    if not SECRETS.exists():
        return None
    pattern = re.compile(rf"^\s*export\s+{re.escape(env_key)}=(.*)$")
    for line in SECRETS.read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            return None
        for part in parts:
            if part.startswith(f"{env_key}="):
                return part.split("=", 1)[1]
    return None


def test_provider_connection(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 5.0,
    model: str | None = None,
) -> tuple[bool, list[str]]:
    base_url = validate_base_url(base_url)
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    messages: list[str] = []

    try:
        with socket.create_connection((host, port), timeout=timeout):
            messages.append(f"tcp: ok ({host}:{port})")
    except OSError as exc:
        messages.append(f"tcp: failed ({host}:{port}) - {exc}")
        return False, messages

    if model:
        responses_url = f"{base_url}/responses"
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = json.dumps(
            {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Reply with exactly OK."}],
                    }
                ],
                "stream": True,
                "max_output_tokens": 16,
            }
        ).encode()
        request = Request(responses_url, data=payload, headers=headers, method="POST")
        messages.append(f"model: {model}")
        try:
            with urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                content_type = response.headers.get("Content-Type", "")
                answer_parts = []
                completed_text = ""
                if "text/event-stream" in content_type:
                    for raw_line in response:
                        line = raw_line.decode(errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        event_type = event.get("type")
                        if event_type == "response.output_text.delta":
                            answer_parts.append(str(event.get("delta", "")))
                        elif event_type == "response.output_text.done":
                            completed_text = str(event.get("text", ""))
                        elif event_type == "response.completed":
                            break
                else:
                    data = json.load(response)
                    if isinstance(data.get("output_text"), str):
                        completed_text = data["output_text"]
                    for item in data.get("output", []):
                        for part in item.get("content", []):
                            if part.get("type") == "output_text" and part.get("text"):
                                answer_parts.append(str(part["text"]))
            answer = "".join(answer_parts).strip() or completed_text.strip()
            messages.append(f"responses: ok POST {responses_url} -> {status}")
            if not (200 <= status < 300):
                return False, messages
            if not answer:
                messages.append("answer: missing output text")
                return False, messages
            answer = re.sub(r"\s+", " ", answer)[:240]
            messages.append(f"answer: {answer}")
            return True, messages
        except HTTPError as exc:
            messages.append(f"responses: POST {responses_url} -> {exc.code}")
            if exc.code in {401, 403}:
                messages.append("auth: rejected; check the API key")
            else:
                try:
                    error_data = json.loads(exc.read(4096))
                    detail = error_data.get("detail") or error_data.get("error", {}).get("message")
                except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                    detail = None
                if detail:
                    detail = re.sub(r"\s+", " ", str(detail))[:240]
                    messages.append(f"response: {detail}")
            return False, messages
        except URLError as exc:
            messages.append(f"responses: failed POST {responses_url} - {exc.reason}")
            return False, messages
        except OSError as exc:
            messages.append(f"responses: failed POST {responses_url} - {exc}")
            return False, messages
        except (json.JSONDecodeError, TypeError, AttributeError):
            messages.append("responses: returned an unreadable response")
            return False, messages

    models_url = f"{base_url}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(models_url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.getcode()
        messages.append(f"http: ok GET {models_url} -> {status}")
        return 200 <= status < 300, messages
    except HTTPError as exc:
        messages.append(f"http: GET {models_url} -> {exc.code}")
        if exc.code in {401, 403}:
            messages.append("auth: rejected; check the API key")
            return False, messages
        if exc.code == 404:
            messages.append("endpoint: /models not found; network is reachable but API compatibility was not confirmed")
            return True, messages
        if exc.code < 500:
            messages.append("endpoint: server responded, but API compatibility was not confirmed")
            return True, messages
        return False, messages
    except URLError as exc:
        messages.append(f"http: failed GET {models_url} - {exc.reason}")
        return False, messages
    except OSError as exc:
        messages.append(f"http: failed GET {models_url} - {exc}")
        return False, messages


def print_test_result(ok: bool, messages: list[str]) -> None:
    print("connection test:")
    for message in messages:
        print(f"  - {message}")
    print(f"result: {'ok' if ok else 'failed'}")


def cmd_add(args: argparse.Namespace) -> None:
    name = validate_name(args.name)
    args.base_url = validate_base_url(args.base_url)
    env_key = args.env_key or default_env_key(name)
    api_key = args.api_key
    if args.prompt_key:
        api_key = getpass.getpass(f"API key for {env_key}: ")

    if not args.skip_test:
        ok, messages = test_provider_connection(args.base_url, api_key or read_secret_value(env_key))
        print_test_result(ok, messages)
        if not ok:
            die("connection test failed; pass --skip-test to save anyway")

    upsert_provider(name, args.base_url, env_key, args.wire_api)
    ensure_secrets_file()

    if api_key:
        upsert_secret(env_key, api_key)

    profile = write_profile(name, args.model, args.effort, args.summary, args.verbosity)
    if args.activate:
        switch_provider(name, args.model)
    print(f"provider {name!r} saved")
    print(f"profile: {profile}")
    print(f"env key: {env_key} ({'set' if env_available(env_key) else 'not set'})")
    print(f"use: source {SECRETS} && codex --profile {name}")


def switch_provider(
    name: str,
    model: str | None = None,
    effort: str | None = None,
    summary: str | None = None,
    verbosity: str | None = None,
) -> None:
    providers = parse_providers()
    if name not in providers:
        die(f"provider {name!r} not found")
    updates = {"model_provider": name}
    profile = parse_profile(name)
    for key in TOP_LEVEL_KEYS:
        if key == "model_provider":
            continue
        if profile.get(key):
            updates[key] = profile[key]

    explicit_updates = {
        "model": model,
        "model_reasoning_effort": effort,
        "model_reasoning_summary": summary,
        "model_verbosity": verbosity,
    }
    for key, value in explicit_updates.items():
        if value is not None:
            updates[key] = value

    text = read_config()
    backup_config()
    CONFIG.write_text(upsert_top_level(text, updates))
    CONFIG.chmod(0o600)


def update_global_model_settings(
    name: str,
    model: str | None = None,
    effort: str | None = None,
    summary: str | None = None,
    verbosity: str | None = None,
) -> bool:
    if parse_top_level().get("model_provider") != name:
        return False

    updates: dict[str, str] = {}
    if model is not None:
        updates["model"] = model
    if effort is not None:
        updates["model_reasoning_effort"] = effort
    if summary is not None:
        updates["model_reasoning_summary"] = summary
    if verbosity is not None:
        updates["model_verbosity"] = verbosity
    if not updates:
        return False

    text = read_config()
    backup_config()
    CONFIG.write_text(upsert_top_level(text, updates))
    CONFIG.chmod(0o600)
    return True


def cmd_switch(args: argparse.Namespace) -> None:
    name = validate_name(args.name)
    if any(value is not None for value in (args.model, args.effort, args.summary, args.verbosity)):
        update_profile(name, args.model, args.effort, args.summary, args.verbosity)
    switch_provider(name, args.model, args.effort, args.summary, args.verbosity)
    top = parse_top_level()
    print(f"global default provider: {top.get('model_provider')}")
    print(f"global default model: {top.get('model', '(unchanged)')}")
    print("existing Codex sessions will not change; start a new session to use it")


def cmd_list(_: argparse.Namespace) -> None:
    top = parse_top_level()
    current = top.get("model_provider")
    providers = parse_providers()
    if not providers:
        print("no custom providers found")
        return
    for name in sorted(providers):
        data = providers[name]
        marker = "*" if name == current else " "
        env_key = data.get("env_key", "")
        status = "key:set" if env_key and env_available(env_key) else "key:missing"
        print(f"{marker} {name:16} {data.get('base_url', ''):40} env={env_key or '-'} {status}")


def cmd_show(args: argparse.Namespace) -> None:
    providers = parse_providers()
    name = validate_name(args.name)
    if name not in providers:
        die(f"provider {name!r} not found")
    data = providers[name]
    env_key = data.get("env_key", "")
    print(f"name: {name}")
    print(f"base_url: {data.get('base_url', '')}")
    print(f"wire_api: {data.get('wire_api', '')}")
    print(f"env_key: {env_key}")
    print(f"api_key: {'set' if env_key and env_available(env_key) else 'missing'}")
    print(f"profile: {CODEX_HOME / (name + '.config.toml')}")


def cmd_set_key(args: argparse.Namespace) -> None:
    providers = parse_providers()
    name = validate_name(args.name)
    if name not in providers:
        die(f"provider {name!r} not found")
    env_key = providers[name].get("env_key") or default_env_key(name)
    if args.stdin:
        api_key = sys.stdin.read().rstrip("\r\n")
    else:
        api_key = args.api_key or getpass.getpass(f"API key for {env_key}: ")
    if not api_key or "\n" in api_key or "\r" in api_key:
        die("API key must be a non-empty single line")
    upsert_secret(env_key, api_key)
    print(f"key saved for {name!r} in {SECRETS}")
    print(f"run: source {SECRETS}")


def cmd_update(args: argparse.Namespace) -> None:
    providers = parse_providers()
    name = validate_name(args.name)
    if name not in providers:
        die(f"provider {name!r} not found")

    current = providers[name]
    base_url = args.base_url if args.base_url is not None else current.get("base_url")
    env_key = args.env_key if args.env_key is not None else current.get("env_key", default_env_key(name))
    wire_api = args.wire_api if args.wire_api is not None else current.get("wire_api", "responses")

    if not base_url:
        die(f"provider {name!r} has no base_url; pass --base-url")
    base_url = validate_base_url(base_url)

    changed_provider = any(
        value is not None
        for value in (args.base_url, args.env_key, args.wire_api)
    )
    changed_profile = any(
        value is not None
        for value in (args.model, args.effort, args.summary, args.verbosity)
    )

    if changed_provider:
        upsert_provider(name, base_url, env_key, wire_api)
    if changed_profile:
        update_profile(name, args.model, args.effort, args.summary, args.verbosity)
        synced_global = update_global_model_settings(
            name,
            args.model,
            args.effort,
            args.summary,
            args.verbosity,
        )
    else:
        synced_global = False

    api_key = args.api_key
    if args.prompt_key:
        api_key = getpass.getpass(f"API key for {env_key}: ")

    if (changed_provider or api_key) and not args.skip_test:
        ok, messages = test_provider_connection(base_url, api_key or read_secret_value(env_key))
        print_test_result(ok, messages)
        if not ok:
            die("connection test failed; pass --skip-test to save anyway")

    if api_key:
        upsert_secret(env_key, api_key)

    if not changed_provider and not changed_profile and not api_key:
        print("nothing changed; pass one of --base-url/--model/--env-key/--wire-api/--api-key/--prompt-key")
        return

    print(f"provider {name!r} updated")
    print(f"base_url: {base_url}")
    print(f"env_key: {env_key} ({'set' if env_available(env_key) else 'not set'})")
    print(f"profile: {CODEX_HOME / (name + '.config.toml')}")
    if synced_global:
        print("global config: synced because this is the current provider")


def cmd_test(args: argparse.Namespace) -> None:
    providers = parse_providers()
    name = validate_name(args.name)
    if name not in providers and not args.base_url:
        die(f"provider {name!r} not found")
    data = providers.get(name, {})
    base_url = args.base_url or data.get("base_url", "")
    env_key = args.env_key or data.get("env_key", default_env_key(name))
    if args.stdin:
        api_key = sys.stdin.read().rstrip("\r\n")
    else:
        api_key = args.api_key or read_secret_value(env_key)
    ok, messages = test_provider_connection(base_url, api_key, timeout=args.timeout, model=args.model)
    print_test_result(ok, messages)
    if not ok:
        raise SystemExit(1)


def cmd_delete(args: argparse.Namespace) -> None:
    providers = parse_providers()
    name = validate_name(args.name)
    if name not in providers:
        die(f"provider {name!r} not found")
    if parse_top_level().get("model_provider") == name:
        die("cannot delete the active provider; activate another provider first")
    if not args.yes:
        die("pass --yes to confirm provider deletion")

    env_key = providers[name].get("env_key", default_env_key(name))
    backup_config()
    text = remove_table(read_config(), f"model_providers.{name}")
    CONFIG.write_text(text.rstrip() + "\n")
    CONFIG.chmod(0o600)
    profile_file(name).unlink(missing_ok=True)
    shared_key = any(
        provider_name != name and data.get("env_key", default_env_key(provider_name)) == env_key
        for provider_name, data in providers.items()
    )
    if not shared_key:
        remove_secret(env_key)
    print(f"provider {name!r} deleted")
    print(f"key: {'kept because it is shared' if shared_key else 'deleted'}")


def cmd_import_env(args: argparse.Namespace) -> None:
    providers = parse_providers()
    names = [validate_name(args.name)] if args.name else sorted(providers)
    imported = 0
    for name in names:
        if name not in providers:
            die(f"provider {name!r} not found")
        env_key = providers[name].get("env_key") or default_env_key(name)
        api_key = os.environ.get(env_key)
        if not api_key:
            print(f"{name}: skipped; {env_key} is not set in this shell")
            continue
        upsert_secret(env_key, api_key)
        imported += 1
        print(f"{name}: imported {env_key} into {SECRETS}")
    if imported:
        print(f"run in new shells or once now: source {SECRETS}")


def cmd_init_existing(args: argparse.Namespace) -> None:
    if not CONFIG.exists():
        die(f"{CONFIG} not found")

    providers = parse_providers()
    if not providers:
        die(f"no custom providers found in {CONFIG}")

    ensure_secrets_file()
    top = parse_top_level()
    current = top.get("model_provider")

    created: list[str] = []
    refreshed: list[str] = []
    kept: list[str] = []
    minimal: list[str] = []

    for name in providers:
        profile = profile_file(name)
        existed = profile.exists()
        if existed and not args.force:
            kept.append(name)
            continue

        existing = parse_profile(name) if existed else {}
        model, effort, summary, verbosity = profile_seed(name, top, existing)
        write_profile(name, model, effort, summary, verbosity)

        if existed:
            refreshed.append(name)
        else:
            created.append(name)

        if name != current and not any((model, effort, summary, verbosity)):
            minimal.append(name)

    print(f"config: {CONFIG}")
    print(f"providers discovered: {len(providers)}")
    print(f"current global provider: {current or '(not set)'}")

    for name, data in providers.items():
        env_key = data.get("env_key") or default_env_key(name)
        if name in created:
            profile_state = "profile:created"
        elif name in refreshed:
            profile_state = "profile:updated"
        else:
            profile_state = "profile:kept"
        key_state = "key:set" if env_available(env_key) else "key:missing"
        marker = "*" if name == current else " "
        print(f"{marker} {name:16} {profile_state:15} env={env_key} {key_state}")

    if created or refreshed:
        print(f"profiles path: {CODEX_HOME}")
    else:
        print("nothing to initialize; provider profiles already exist")

    if minimal:
        names = ", ".join(minimal)
        print("note: config.toml only stores one active model setup at the top level.")
        print(f"      non-current providers were initialized with minimal profiles: {names}")

    print(f"use: source {SECRETS} && codex --profile NAME")


def prompt_text(label: str, default: str | None = None, required: bool = False) -> str | None:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return None
        print("required")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def choose_provider(label: str = "provider") -> str | None:
    providers = sorted(parse_providers())
    if not providers:
        print("no custom providers found")
        return None
    if sys.stdin.isatty() and sys.stdout.isatty():
        choice = cursor_menu("Choose Provider", providers)
        return validate_name(choice) if choice else None
    for idx, name in enumerate(providers, 1):
        print(f"  {idx}) {name}")
    raw = prompt_text(label, required=True)
    if raw is None:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(providers):
        return providers[int(raw) - 1]
    return validate_name(raw)


def cursor_menu(title: str, labels: list[str]) -> str | None:
    def run(stdscr: curses.window) -> str | None:
        curses.curs_set(0)
        selected = 0
        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, title)
            stdscr.addstr(1, 0, "Use ↑/↓ or j/k, Enter to select, q/Esc to quit.")
            for idx, label in enumerate(labels):
                prefix = "➜ " if idx == selected else "  "
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                stdscr.addstr(idx + 3, 0, f"{prefix}{idx + 1}) {label}", attr)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(labels)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(labels)
            elif key in (curses.KEY_ENTER, 10, 13):
                return labels[selected]
            elif key in (27, ord("q")):
                return None
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(labels):
                    return labels[idx]

    try:
        return curses.wrapper(run)
    except curses.error:
        return None


def interactive_add() -> None:
    name = validate_name(prompt_text("provider name", required=True) or "")
    base_url = prompt_text("base_url", required=True)
    model = prompt_text("model (optional)")
    env_key = prompt_text("env_key", default=default_env_key(name))
    wire_api = prompt_text("wire_api", default="responses")
    api_key = None
    if prompt_yes_no("enter API key now", default=True):
        api_key = getpass.getpass(f"API key for {env_key}: ")
    activate = prompt_yes_no("make this the global default", default=False)
    args = argparse.Namespace(
        name=name,
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt_key=False,
        env_key=env_key,
        wire_api=wire_api,
        effort="medium",
        summary="concise",
        verbosity="medium",
        activate=activate,
        skip_test=not prompt_yes_no("test connection before saving", default=True),
    )
    cmd_add(args)


def interactive_update() -> None:
    name = choose_provider("provider number or name")
    if not name:
        return
    current = parse_providers()[name]
    profile = parse_profile(name)
    env_key_current = current.get("env_key", default_env_key(name))
    fields = [
        ("base_url", current.get("base_url", "")),
        ("env_key", env_key_current),
        ("model", profile.get("model", "")),
        ("model_reasoning_effort", profile.get("model_reasoning_effort", "medium")),
        ("done", ""),
    ]

    while True:
        labels = [f"{key} [{value}]" if value else key for key, value in fields]
        if sys.stdin.isatty() and sys.stdout.isatty():
            choice = cursor_menu(f"Update {name}", labels)
            if choice is None:
                return
            field = fields[labels.index(choice)][0]
        else:
            print("choose field to update")
            for idx, (key, value) in enumerate(fields, 1):
                suffix = f" [{value}]" if value else ""
                print(f"  {idx}) {key}{suffix}")
            raw = prompt_text("choose", default=str(len(fields)))
            if not raw or not raw.isdigit() or not (1 <= int(raw) <= len(fields)):
                print("invalid choice")
                continue
            field = fields[int(raw) - 1][0]
        if field == "done":
            return
        break

    old_value = dict(fields)[field]
    new_value = prompt_text(f"new {field}", default=old_value, required=True)
    if new_value == old_value:
        print("nothing changed")
        return

    args = argparse.Namespace(
        name=name,
        base_url=new_value if field == "base_url" else None,
        model=new_value if field == "model" else None,
        api_key=None,
        prompt_key=False,
        env_key=new_value if field == "env_key" else None,
        wire_api=None,
        effort=new_value if field == "model_reasoning_effort" else None,
        summary=None,
        verbosity=None,
        skip_test=False,
    )
    cmd_update(args)


def interactive_switch() -> None:
    name = choose_provider("provider number or name")
    if not name:
        return
    model = prompt_text("model override (optional)")
    args = argparse.Namespace(name=name, model=model, effort=None, summary=None, verbosity=None)
    cmd_switch(args)


def interactive_show() -> None:
    name = choose_provider("provider number or name")
    if name:
        cmd_show(argparse.Namespace(name=name))


def interactive_set_key() -> None:
    name = choose_provider("provider number or name")
    if name:
        cmd_set_key(argparse.Namespace(name=name, api_key=None))


def interactive_menu() -> None:
    actions = [
        ("list providers", lambda: cmd_list(argparse.Namespace())),
        ("add provider", interactive_add),
        ("update provider", interactive_update),
        ("quit", None),
    ]
    while True:
        labels = [label for label, _ in actions]
        if sys.stdin.isatty() and sys.stdout.isatty():
            choice_label = cursor_menu("Codex Provider Manager", labels)
            if choice_label is None:
                return
            idx = labels.index(choice_label)
        else:
            print("\nCodex Provider Manager")
            for idx, (label, _) in enumerate(actions, 1):
                print(f"  {idx}) {label}")
            choice = prompt_text("choose", default="1")
            if not choice:
                continue
            if not choice.isdigit() or not (1 <= int(choice) <= len(actions)):
                print("invalid choice")
                continue
            idx = int(choice) - 1
        label, action = actions[idx]
        if action is None:
            return
        print(f"\n== {label} ==")
        action()
        if sys.stdin.isatty() and sys.stdout.isatty():
            input("\nPress Enter to return to the menu...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage Codex model providers and API keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""examples:
  codex-provider
  codex-provider list
  codex-provider init-existing
  codex-provider show crs
  codex-provider add crs --base-url http://81.70.201.249:3000/openai --api-key sk-xxx
  codex-provider test crs
  codex-provider update zskj --base-url http://10.1.6.27/v1
  codex-provider update zskj --model gpt-5.4 --effort xhigh
  codex-provider update zskj --prompt-key
  codex-provider switch zskj --model gpt-5.4
  source {SECRETS} && codex --profile zskj
notes:
  Run without arguments to open an interactive menu.
  API keys are stored in {SECRETS}
  If update targets the current global provider, model settings also sync to {CONFIG}
  Use "codex-provider test NAME" for API checks; ping only accepts hosts/IPs, not http URLs.
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_existing = sub.add_parser(
        "init-existing",
        help="initialize provider profiles from the existing config.toml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""examples:
  codex-provider init-existing
  codex-provider init-existing --force
notes:
  Reads providers from {CONFIG}
  Creates missing {CODEX_HOME}/<provider>.config.toml profiles
  Only the current global provider can recover model settings from the top level of config.toml
""",
    )
    init_existing.add_argument("--force", action="store_true", help="rewrite existing provider profile files")
    init_existing.set_defaults(func=cmd_init_existing)

    add = sub.add_parser(
        "add",
        help="add or update a provider",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  codex-provider add myproxy --base-url http://example.com/v1 --api-key sk-xxx
  codex-provider add crs --base-url http://81.70.201.249:3000/openai --env-key CRS_OPENAI_KEY
  codex-provider add zskj --base-url http://10.1.6.27/v1 --activate
""",
    )
    add.add_argument("name")
    add.add_argument("--base-url", required=True)
    add.add_argument("--model", help="default model for this provider profile")
    add.add_argument("--api-key", help="API key; omit and use --prompt-key to avoid shell history")
    add.add_argument("--prompt-key", action="store_true", help="prompt for API key without echo")
    add.add_argument("--env-key")
    add.add_argument("--wire-api", default="responses")
    add.add_argument("--effort", default="medium")
    add.add_argument("--summary", default="concise")
    add.add_argument("--verbosity", default="medium")
    add.add_argument("--activate", action="store_true", help="also make this the global default")
    add.add_argument("--skip-test", action="store_true", help="save without testing TCP and /models")
    add.set_defaults(func=cmd_add)

    update = sub.add_parser(
        "update",
        help="modify one provider/profile field without replacing the rest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  codex-provider update zskj --base-url http://10.1.6.27/v1
  codex-provider update zskj --model gpt-5.4
  codex-provider update zskj --effort xhigh
  codex-provider update zskj --model gpt-5.4 --effort xhigh --verbosity medium
  codex-provider update zskj --prompt-key
""",
    )
    update.add_argument("name")
    update.add_argument("--base-url")
    update.add_argument("--model")
    update.add_argument("--api-key", help="API key; omit and use --prompt-key to avoid shell history")
    update.add_argument("--prompt-key", action="store_true", help="prompt for API key without echo")
    update.add_argument("--env-key")
    update.add_argument("--wire-api")
    update.add_argument("--effort")
    update.add_argument("--summary")
    update.add_argument("--verbosity")
    update.add_argument("--skip-test", action="store_true", help="save without testing TCP and /models")
    update.set_defaults(func=cmd_update)

    test = sub.add_parser(
        "test",
        help="test a provider with TCP, /models, and an optional live Responses request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  codex-provider test crs
  codex-provider test zskj
  codex-provider test crs --model gpt-5.6-sol
  codex-provider test crs --api-key sk-xxx
""",
    )
    test.add_argument("name")
    test.add_argument("--base-url", help="test this URL instead of the saved provider URL")
    test.add_argument("--env-key", help="read a stored Key using this environment variable name")
    test.add_argument("--model", help="send a minimal live Responses request using this model")
    test_key = test.add_mutually_exclusive_group()
    test_key.add_argument("--api-key")
    test_key.add_argument("--stdin", action="store_true", help="read a temporary API Key from standard input")
    test.add_argument("--timeout", type=float, default=30.0)
    test.set_defaults(func=cmd_test)

    delete = sub.add_parser("delete", help="delete an inactive provider and its unshared stored Key")
    delete.add_argument("name")
    delete.add_argument("--yes", action="store_true", help="confirm provider deletion")
    delete.set_defaults(func=cmd_delete)

    switch = sub.add_parser(
        "switch",
        help="make a provider the global default",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  codex-provider switch crs --model gpt-5.5
  codex-provider switch zskj --model gpt-5.4
  codex-provider switch zskj
""",
    )
    switch.add_argument("name")
    switch.add_argument("--model", help="also update the global default model")
    switch.add_argument("--effort")
    switch.add_argument("--summary")
    switch.add_argument("--verbosity")
    switch.set_defaults(func=cmd_switch)

    list_cmd = sub.add_parser("list", help="list providers without showing secrets")
    list_cmd.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="show one provider without showing its secret")
    show.add_argument("name")
    show.set_defaults(func=cmd_show)

    set_key = sub.add_parser(
        "set-key",
        help="set or replace a provider API key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  codex-provider set-key crs
  codex-provider set-key zskj
  codex-provider set-key crs --api-key sk-xxx
""",
    )
    set_key.add_argument("name")
    key_source = set_key.add_mutually_exclusive_group()
    key_source.add_argument("--api-key")
    key_source.add_argument("--stdin", action="store_true", help="read the API key from standard input")
    set_key.set_defaults(func=cmd_set_key)

    import_env = sub.add_parser(
        "import-env",
        help="copy current shell env keys into the private secrets file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  codex-provider import-env
  codex-provider import-env crs
  CRS_OPENAI_KEY=sk-xxx codex-provider import-env crs
""",
    )
    import_env.add_argument("name", nargs="?")
    import_env.set_defaults(func=cmd_import_env)

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        interactive_menu()
        return
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

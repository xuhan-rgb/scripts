#!/usr/bin/env python3
import fcntl
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
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
STATE_DIR = Path.home() / ".cli-proxy-api"
SELECTION_FILE = STATE_DIR / "selection.conf"
LOCK_FILE = STATE_DIR / "selection.lock"
CODEX_DIR = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
CODEX_CONFIG = CODEX_DIR / "config.toml"


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


def write_selection(model, effort, path=SELECTION_FILE, lock_path=LOCK_FILE):
    if model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {model}")
    if effort not in EFFORTS:
        raise ValueError(f"unsupported effort: {effort}")

    path = Path(path)
    lock_path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        descriptor, temp_name = tempfile.mkstemp(prefix="selection.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temp_file:
                temp_file.write(f"CLAUDEX_MODEL={model}\nCLAUDEX_EFFORT={effort}\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return {"model": model, "effort": effort}


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


def build_state():
    codex = read_codex_state()
    return {
        "selection": read_selection(codex_state=codex),
        "provider": codex,
        "models": MODELS,
        "efforts": EFFORTS,
        "gateway": gateway_state(),
        "service_active": service_state(),
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
    button { font: inherit; }
    .shell { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 34px 0 48px; }
    .topbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 22px; }
    .eyebrow, .mono {
      font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .eyebrow { margin: 0 0 7px; color: var(--signal); font-size: 12px; font-weight: 800; }
    h1 { margin: 0; font-size: clamp(31px, 4vw, 55px); line-height: .98; letter-spacing: -.045em; }
    .status-line { display: flex; align-items: center; gap: 9px; padding-bottom: 5px; color: var(--ink-soft); font-size: 13px; }
    .pulse { width: 10px; height: 10px; border-radius: 50%; background: var(--bad); box-shadow: 0 0 0 5px rgba(180,72,53,.12); }
    .pulse.good { background: var(--good); box-shadow: 0 0 0 5px rgba(40,118,80,.13); }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(290px, .8fr); gap: 18px; }
    .panel { background: rgba(251, 248, 240, .94); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow); overflow: hidden; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; padding: 18px 20px; border-bottom: 1px solid var(--line); }
    .panel-head h2 { margin: 0; font-size: 15px; letter-spacing: -.01em; }
    .shortcut { color: var(--ink-soft); font-size: 11px; }
    .model-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 16px; }
    .model-card {
      position: relative;
      min-height: 190px;
      padding: 18px;
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
    .model-card:focus-visible, .effort button:focus-visible, .save:focus-visible { outline: 3px solid rgba(25,125,120,.32); outline-offset: 2px; }
    .model-card.active { color: var(--panel); background: var(--ink); border-color: var(--ink); }
    .model-index { color: var(--signal); font: 800 11px "DejaVu Sans Mono", monospace; }
    .model-name { display: block; margin-top: 27px; font-size: 28px; font-weight: 800; letter-spacing: -.04em; }
    .model-role { display: block; margin-top: 3px; color: var(--teal); font: 700 11px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .active .model-role { color: #79d4cb; }
    .model-desc { display: block; margin-top: 18px; color: var(--ink-soft); font-size: 12px; line-height: 1.5; }
    .active .model-desc { color: #cbd4d3; }
    .effort-wrap { padding: 0 16px 18px; }
    .effort-label { display: flex; justify-content: space-between; margin: 3px 2px 10px; color: var(--ink-soft); font-size: 12px; }
    .effort { display: grid; grid-template-columns: repeat(5, 1fr); padding: 4px; border: 1px solid var(--line); border-radius: 13px; background: #ebe5d9; }
    .effort button { padding: 11px 5px; border: 0; border-radius: 9px; color: var(--ink-soft); background: transparent; cursor: pointer; font: 700 11px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .effort button.active { color: white; background: var(--teal); }
    .side { display: grid; align-content: start; gap: 18px; }
    .provider { padding: 20px; }
    .provider-name { margin: 4px 0 20px; font-size: 25px; font-weight: 800; letter-spacing: -.035em; }
    .facts { display: grid; gap: 13px; margin: 0; }
    .fact { display: grid; grid-template-columns: 78px 1fr; gap: 10px; padding-top: 12px; border-top: 1px solid var(--line); }
    .fact dt { color: var(--ink-soft); font: 700 10px "DejaVu Sans Mono", monospace; text-transform: uppercase; }
    .fact dd { margin: 0; overflow-wrap: anywhere; font: 12px/1.45 "DejaVu Sans Mono", monospace; }
    .commit { padding: 20px; background: var(--ink); color: var(--panel); border-color: var(--ink); }
    .commit h3 { margin: 0; font-size: 16px; }
    .commit p { margin: 8px 0 18px; color: #bfcac9; font-size: 12px; line-height: 1.55; }
    .save { width: 100%; padding: 13px 16px; border: 0; border-radius: 11px; color: #1c2527; background: var(--signal); cursor: pointer; font-weight: 900; transition: filter .15s ease, transform .15s ease; }
    .save:hover { filter: brightness(1.07); transform: translateY(-1px); }
    .save:disabled { color: #74807f; background: #435154; cursor: default; transform: none; }
    .save-state { margin-top: 11px; min-height: 18px; color: #8f9d9b; font: 11px "DejaVu Sans Mono", monospace; }
    .foot { display: flex; justify-content: space-between; gap: 20px; margin-top: 17px; color: var(--ink-soft); font-size: 11px; }
    .toast { position: fixed; right: 22px; bottom: 22px; max-width: 360px; padding: 13px 16px; border-radius: 11px; color: white; background: var(--good); box-shadow: var(--shadow); transform: translateY(90px); opacity: 0; transition: .25s ease; }
    .toast.show { transform: translateY(0); opacity: 1; }
    .toast.error { background: var(--bad); }
    @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 820px) {
      .shell { width: min(100% - 22px, 680px); padding-top: 22px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .workspace { grid-template-columns: 1fr; }
      .model-grid { grid-template-columns: 1fr; }
      .model-card { min-height: 135px; }
      .model-name { margin-top: 15px; }
      .foot { flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div><p class="eyebrow">Local inference control</p><h1>Codex Routing Desk</h1></div>
      <div class="status-line"><span id="pulse" class="pulse"></span><span id="health">Connecting to gateway...</span></div>
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
          <h3>Apply as Claude default</h3>
          <p>Saved independently from Codex. New Claude sessions use this route; existing sessions keep their in-session selection.</p>
          <button id="save" class="save" disabled>Saved</button>
          <div id="save-state" class="save-state">Waiting for state…</div>
        </section>
      </aside>
    </div>
    <footer class="foot"><span>Provider credentials stay in the environment selected by Codex.</span><span class="mono">Ctrl+S save · auto refresh 10s</span></footer>
  </main>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script>
    const app = { state: null, draft: null, dirty: false };
    const $ = (id) => document.getElementById(id);
    const escapeText = (value) => value || '—';

    function render() {
      const state = app.state;
      if (!state) return;
      $('models').innerHTML = state.models.map((model, index) => `
        <button class="model-card ${app.draft.model === model.id ? 'active' : ''}" data-model="${model.id}" aria-pressed="${app.draft.model === model.id}">
          <span class="model-index">0${index + 1} / ${model.id}</span>
          <span class="model-name">${model.name}</span><span class="model-role">${model.role}</span>
          <span class="model-desc">${model.description}</span>
        </button>`).join('');
      $('models').querySelectorAll('[data-model]').forEach((button) => button.addEventListener('click', () => chooseModel(button.dataset.model)));
      $('efforts').innerHTML = state.efforts.map((effort) => `<button class="${app.draft.effort === effort ? 'active' : ''}" data-effort="${effort}" aria-pressed="${app.draft.effort === effort}">${effort}</button>`).join('');
      $('efforts').querySelectorAll('[data-effort]').forEach((button) => button.addEventListener('click', () => chooseEffort(button.dataset.effort)));
      $('provider-name').textContent = escapeText(state.provider.provider);
      $('base-url').textContent = escapeText(state.provider.base_url);
      $('wire-api').textContent = escapeText(state.provider.wire_api);
      $('env-key').textContent = escapeText(state.provider.env_key);
      $('catalog').textContent = state.gateway.models.length ? state.gateway.models.join(' · ') : 'Unavailable';
      const healthy = state.service_active && state.gateway.reachable;
      $('pulse').classList.toggle('good', healthy);
      $('health').textContent = healthy ? 'Gateway online · 127.0.0.1:8317' : 'Gateway unavailable';
      $('save').disabled = !app.dirty;
      $('save').textContent = app.dirty ? 'Apply selection' : 'Saved';
      $('save-state').textContent = app.dirty ? `${app.draft.model} · ${app.draft.effort}` : `Active: ${app.draft.model} · ${app.draft.effort}`;
    }

    function chooseModel(model) { app.draft.model = model; app.dirty = true; render(); }
    function chooseEffort(effort) { app.draft.effort = effort; app.dirty = true; render(); }
    function notify(message, error = false) {
      const toast = $('toast'); toast.textContent = message; toast.className = `toast show${error ? ' error' : ''}`;
      clearTimeout(notify.timer); notify.timer = setTimeout(() => toast.classList.remove('show'), 3200);
    }

    async function refresh() {
      try {
        const response = await fetch('/api/state', {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const state = await response.json();
        app.state = state;
        if (!app.dirty) app.draft = {...state.selection};
        render();
      } catch (error) { notify(`Cannot read service state: ${error.message}`, true); }
    }

    async function save() {
      if (!app.dirty) return;
      $('save').disabled = true; $('save').textContent = 'Applying…';
      try {
        const response = await fetch('/api/selection', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(app.draft)});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        app.dirty = false; app.state = payload; app.draft = {...payload.selection}; render(); notify('Selection saved for new Claude sessions.');
      } catch (error) { render(); notify(`Save failed: ${error.message}`, true); }
    }

    $('save').addEventListener('click', save);
    document.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); return; }
      if (event.target.matches('input, textarea, select')) return;
      const index = Number(event.key) - 1;
      if (!event.altKey && index >= 0 && index < 3 && app.state) chooseModel(app.state.models[index].id);
      if (event.altKey && index >= 0 && index < 5 && app.state) { event.preventDefault(); chooseEffort(app.state.efforts[index]); }
    });
    refresh(); setInterval(() => { if (!app.dirty) refresh(); }, 10000);
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
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self._send_json(500, {"error": str(error)})
            return
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/selection":
            self.send_error(404)
            return
        if not self._same_origin() or self.headers.get_content_type() != "application/json":
            self._send_json(403, {"error": "request origin or content type is not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            write_selection(payload.get("model"), payload.get("effort"))
            self._send_json(200, build_state())
        except (json.JSONDecodeError, OSError, ValueError, subprocess.SubprocessError) as error:
            self._send_json(400, {"error": str(error)})

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    server = ThreadingHTTPServer((HOST, PORT), ManagerHandler)
    print(f"Codex Routing Desk listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

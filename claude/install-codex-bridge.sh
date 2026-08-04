#!/usr/bin/env bash
set -euo pipefail

readonly CLIPROXY_VERSION="${CLIPROXY_VERSION:-7.2.116}"
readonly SERVICE_NAME="cli-proxy-api.service"
readonly MANAGER_SERVICE_NAME="claudex-manager.service"
readonly STATE_DIR="${HOME}/.cli-proxy-api"
readonly BIN_DIR="${HOME}/.local/bin"
readonly LIB_DIR="${HOME}/.local/lib/claudex"
readonly UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
readonly UNIT_FILE="${UNIT_DIR}/${SERVICE_NAME}"
readonly MANAGER_UNIT_FILE="${UNIT_DIR}/${MANAGER_SERVICE_NAME}"
readonly SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

if [[ ${EUID} -eq 0 ]]; then
  fail "run this installer as the desktop user, not root"
fi

[[ ${CLIPROXY_VERSION} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "invalid CLIPROXY_VERSION"

for command_name in awk cmp cp curl date grep install mktemp mv printenv python3 readlink sed sha256sum sort systemctl tar; do
  require_command "${command_name}"
done

case "$(uname -m)" in
  x86_64) asset_arch="amd64" ;;
  aarch64 | arm64) asset_arch="arm64" ;;
  *) fail "unsupported architecture: $(uname -m)" ;;
esac

[[ $(uname -s) == "Linux" ]] || fail "only Linux is supported"
systemctl --user show-environment >/dev/null 2>&1 || fail "systemd user manager is unavailable"

[[ -f ${SCRIPT_DIR}/codex_bridge_manager.py ]] || fail "missing manager source: ${SCRIPT_DIR}/codex_bridge_manager.py"

mkdir -p "${BIN_DIR}" "${LIB_DIR}" "${STATE_DIR}" "${UNIT_DIR}"
chmod 700 "${STATE_DIR}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "${tmp_dir}"
}
trap cleanup EXIT

binary_file="${BIN_DIR}/cli-proxy-api"
if [[ ! -x ${binary_file} ]] || ! "${binary_file}" -h 2>&1 | grep -q "Version: ${CLIPROXY_VERSION}"; then
  asset="CLIProxyAPI_${CLIPROXY_VERSION}_linux_${asset_arch}.tar.gz"
  release_url="https://github.com/router-for-me/CLIProxyAPI/releases/download/v${CLIPROXY_VERSION}"

  printf 'Downloading CLIProxyAPI %s...\n' "${CLIPROXY_VERSION}"
  curl --fail --location --retry 3 --silent --show-error \
    --output "${tmp_dir}/${asset}" "${release_url}/${asset}"
  curl --fail --location --retry 3 --silent --show-error \
    --output "${tmp_dir}/checksums.txt" "${release_url}/checksums.txt"

  expected_checksum="$(awk -v name="${asset}" '$2 == name {print $1}' "${tmp_dir}/checksums.txt")"
  [[ -n ${expected_checksum} ]] || fail "release checksum is missing for ${asset}"
  actual_checksum="$(sha256sum "${tmp_dir}/${asset}" | awk '{print $1}')"
  [[ ${actual_checksum} == "${expected_checksum}" ]] || fail "checksum verification failed for ${asset}"

  tar -xzf "${tmp_dir}/${asset}" -C "${tmp_dir}" cli-proxy-api
  install -m 0755 "${tmp_dir}/cli-proxy-api" "${binary_file}"
fi

cat >"${tmp_dir}/${SERVICE_NAME}" <<'EOF'
[Unit]
Description=Claude Code bridge for the active Codex provider
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/.cli-proxy-api
ExecStart=%h/.local/bin/cli-proxy-api -config %h/.cli-proxy-api/config.yaml -local-model
Restart=on-failure
RestartSec=3
UMask=0077

[Install]
WantedBy=default.target
EOF
install -m 0644 "${tmp_dir}/${SERVICE_NAME}" "${UNIT_FILE}"

cat >"${tmp_dir}/claude-codex-sync" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

readonly STATE_DIR="${HOME}/.cli-proxy-api"
readonly CONFIG_FILE="${STATE_DIR}/config.yaml"
readonly SELECTION_FILE="${STATE_DIR}/selection.conf"
readonly CODEX_DIR="${CODEX_HOME:-${HOME}/.codex}"
readonly CODEX_CONFIG="${CODEX_DIR}/config.toml"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

toml_top_value() {
  awk -v wanted="$1" '
    /^[[:space:]]*\[/ { exit }
    $0 ~ "^[[:space:]]*" wanted "[[:space:]]*=" {
      line = $0
      sub(/^[^"]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' "${CODEX_CONFIG}"
}

toml_provider_value() {
  awk -v target="[model_providers.$1]" -v wanted="$2" '
    {
      section = $0
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", section)
    }
    section == target { active = 1; next }
    active && section ~ /^\[/ { exit }
    active && $0 ~ "^[[:space:]]*" wanted "[[:space:]]*=" {
      line = $0
      sub(/^[^"]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' "${CODEX_CONFIG}"
}

yaml_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "${value}"
}

[[ -r ${CODEX_CONFIG} ]] || fail "Codex config is not readable: ${CODEX_CONFIG}"

provider="$(toml_top_value model_provider)"
codex_model="$(toml_top_value model)"
codex_effort="$(toml_top_value model_reasoning_effort)"
[[ -n ${provider} ]] || fail "model_provider is missing from ${CODEX_CONFIG}"
[[ ${provider} =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsupported model_provider name: ${provider}"

mkdir -p "${STATE_DIR}"
chmod 700 "${STATE_DIR}"
if [[ ! -f ${SELECTION_FILE} ]]; then
  case "${codex_model}" in
    gpt-5.6-sol | gpt-5.6-terra | gpt-5.6-luna) initial_model="${codex_model}" ;;
    *) initial_model="gpt-5.6-sol" ;;
  esac
  case "${codex_effort}" in
    low | medium | high | xhigh | max) initial_effort="${codex_effort}" ;;
    *) initial_effort="xhigh" ;;
  esac
  tmp_selection="$(mktemp "${STATE_DIR}/selection.conf.tmp.XXXXXX")"
  printf 'CLAUDEX_MODEL=%s\nCLAUDEX_EFFORT=%s\n' "${initial_model}" "${initial_effort}" >"${tmp_selection}"
  chmod 600 "${tmp_selection}"
  mv "${tmp_selection}" "${SELECTION_FILE}"
fi

selection_value() {
  awk -F= -v wanted="$1" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "${SELECTION_FILE}"
}
model="$(selection_value CLAUDEX_MODEL)"
effort="$(selection_value CLAUDEX_EFFORT)"
case "${model}" in
  gpt-5.6-sol | gpt-5.6-terra | gpt-5.6-luna) ;;
  *) fail "Claude model must be gpt-5.6-sol, gpt-5.6-terra, or gpt-5.6-luna; got ${model}" ;;
esac
case "${effort}" in
  low | medium | high | xhigh | max) ;;
  *) fail "unsupported Claude reasoning effort: ${effort}" ;;
esac

base_url="$(toml_provider_value "${provider}" base_url)"
wire_api="$(toml_provider_value "${provider}" wire_api)"
env_key="$(toml_provider_value "${provider}" env_key)"
[[ ${base_url} == http://* || ${base_url} == https://* ]] || fail "provider ${provider} has no valid base_url"
[[ ${wire_api} == "responses" ]] || fail "provider ${provider} must use wire_api = \"responses\""
[[ ${env_key} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail "provider ${provider} has no valid env_key"
api_key="$(printenv "${env_key}" || true)"
[[ -n ${api_key} ]] || fail "environment variable ${env_key} required by provider ${provider} is not set"
[[ ${base_url} != *$'\n'* && ${api_key} != *$'\n'* ]] || fail "provider values must not contain newlines"

tmp_config="$(mktemp "${STATE_DIR}/config.yaml.tmp.XXXXXX")"
cleanup() {
  if [[ -n ${tmp_config:-} && -e ${tmp_config} ]]; then
    rm -f -- "${tmp_config}"
  fi
}
trap cleanup EXIT

cat >"${tmp_config}" <<YAML
# Generated from ${CODEX_CONFIG}; active provider: ${provider}
host: "127.0.0.1"
port: 8317

tls:
  enable: false

auth-dir: "~/.cli-proxy-api"
api-keys:
  - "claudex-local"

debug: false
logging-to-file: false
usage-statistics-enabled: false
disable-image-generation: true

# Ignore stale OAuth files; requests use the API-key provider selected by Codex.
oauth-excluded-models:
  codex:
    - "*"

codex-api-key:
  - api-key: "$(yaml_escape "${api_key}")"
    base-url: "$(yaml_escape "${base_url}")"
    models:
      - name: "gpt-5.6-sol"
        alias: "gpt-5.6-sol"
        display-name: "GPT 5.6 Sol"
      - name: "gpt-5.6-terra"
        alias: "gpt-5.6-terra"
        display-name: "GPT 5.6 Terra"
      - name: "gpt-5.6-luna"
        alias: "gpt-5.6-luna"
        display-name: "GPT 5.6 Luna"
YAML
chmod 600 "${tmp_config}"

if [[ ! -f ${CONFIG_FILE} ]] || ! cmp -s "${tmp_config}" "${CONFIG_FILE}"; then
  if [[ ${CLAUDEX_SYNC_BACKUP:-0} == 1 && -f ${CONFIG_FILE} ]]; then
    backup_file="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp -p "${CONFIG_FILE}" "${backup_file}"
    printf 'Backed up the previous config to %s\n' "${backup_file}" >&2
  fi
  mv "${tmp_config}" "${CONFIG_FILE}"
  tmp_config=""
  systemctl --user try-restart cli-proxy-api.service >/dev/null 2>&1 || true
fi

printf '%s\n%s\n%s\n' "${model}" "${effort}" "${provider}"
EOF
install -m 0755 "${tmp_dir}/claude-codex-sync" "${BIN_DIR}/claude-codex-sync"
install -m 0755 "${SCRIPT_DIR}/codex_bridge_manager.py" "${LIB_DIR}/codex_bridge_manager.py"

cat >"${tmp_dir}/claudex" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

extra_args=()
if [[ $(basename "$0") == "claudex-yolo" ]]; then
  extra_args+=(--dangerously-skip-permissions)
fi

usage() {
  printf '%s\n' \
    'Usage: claudex [--pick] [--gpt-model MODEL] [--gpt-effort EFFORT] [-- CLAUDE_ARGS...]' \
    'Models: gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna' \
    'Effort: low, medium, high, xhigh, max'
}

selected_model=""
selected_effort=""
pick=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pick)
      pick=true
      shift
      ;;
    --gpt-model)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      selected_model="$2"
      shift 2
      ;;
    --gpt-effort)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      selected_effort="$2"
      shift 2
      ;;
    --gpt-help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *) break ;;
  esac
done

sync_output="$("${HOME}/.local/bin/claude-codex-sync")"
mapfile -t active_route <<<"${sync_output}"
[[ ${#active_route[@]} -ge 3 ]] || { printf 'Failed to read the active route.\n' >&2; exit 1; }

if [[ ${pick} == true ]]; then
  models=(gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna)
  efforts=(low medium high xhigh max)
  printf 'Select GPT model (current: %s):\n' "${active_route[0]}"
  select choice in "${models[@]}"; do
    [[ -n ${choice} ]] && { selected_model="${choice}"; break; }
  done
  printf 'Select reasoning effort (current: %s):\n' "${active_route[1]}"
  select choice in "${efforts[@]}"; do
    [[ -n ${choice} ]] && { selected_effort="${choice}"; break; }
  done
fi

model="${selected_model:-${CLAUDEX_MODEL:-${active_route[0]}}}"
effort="${selected_effort:-${CLAUDEX_EFFORT:-${active_route[1]}}}"
case "${model}" in
  gpt-5.6-sol | gpt-5.6-terra | gpt-5.6-luna) ;;
  *)
    printf 'Unsupported model: %s\n' "${model}" >&2
    exit 2
    ;;
esac
case "${effort}" in
  low | medium | high | xhigh | max) ;;
  *)
    printf 'Unsupported effort: %s\n' "${effort}" >&2
    exit 2
    ;;
esac

exec env \
  -u NO_COLOR \
  -u CLAUDE_CODE_AUTO_COMPACT_WINDOW \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8317 \
  ANTHROPIC_AUTH_TOKEN=claudex-local \
  ANTHROPIC_DEFAULT_OPUS_MODEL=gpt-5.6-sol \
  ANTHROPIC_DEFAULT_SONNET_MODEL=gpt-5.6-terra \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=gpt-5.6-luna \
  CLAUDE_CODE_SUBAGENT_MODEL=gpt-5.6-terra \
  CLAUDE_CODE_ALWAYS_ENABLE_EFFORT=1 \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=0 \
  claude --model "${model}" --effort "${effort}" "${extra_args[@]}" "$@"
EOF
install -m 0755 "${tmp_dir}/claudex" "${BIN_DIR}/claudex"
ln -sfn claudex "${BIN_DIR}/claudex-yolo"

cat >"${tmp_dir}/${MANAGER_SERVICE_NAME}" <<'EOF'
[Unit]
Description=Codex Routing Desk
After=cli-proxy-api.service
Wants=cli-proxy-api.service

[Service]
Type=simple
WorkingDirectory=%h/.local/lib/claudex
ExecStart=/usr/bin/env python3 %h/.local/lib/claudex/codex_bridge_manager.py
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
UMask=0077

[Install]
WantedBy=default.target
EOF
install -m 0644 "${tmp_dir}/${MANAGER_SERVICE_NAME}" "${MANAGER_UNIT_FILE}"

cat >"${tmp_dir}/claudex-ui" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

readonly URL="http://127.0.0.1:8320"
systemctl --user start claudex-manager.service
if [[ ${1:-} == "--no-open" ]]; then
  printf '%s\n' "${URL}"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${URL}" >/dev/null 2>&1
else
  printf '%s\n' "${URL}"
fi
EOF
install -m 0755 "${tmp_dir}/claudex-ui" "${BIN_DIR}/claudex-ui"

systemctl --user daemon-reload
sync_output="$(CLAUDEX_SYNC_BACKUP=1 "${BIN_DIR}/claude-codex-sync")"
mapfile -t synced_route <<<"${sync_output}"
systemctl --user enable "${SERVICE_NAME}" >/dev/null
systemctl --user restart "${SERVICE_NAME}"
systemctl --user enable "${MANAGER_SERVICE_NAME}" >/dev/null
systemctl --user restart "${MANAGER_SERVICE_NAME}"

models_json=""
for _ in {1..20}; do
  if models_json="$(curl --fail --silent --show-error \
    -H 'Authorization: Bearer claudex-local' \
    'http://127.0.0.1:8317/v1/models?limit=1000' 2>/dev/null)"; then
    break
  fi
  sleep 1
done
[[ -n ${models_json} ]] || fail "CLIProxyAPI did not become ready"

manager_ready=false
for _ in {1..20}; do
  if curl --fail --silent 'http://127.0.0.1:8320/healthz' >/dev/null 2>&1; then
    manager_ready=true
    break
  fi
  sleep 1
done
[[ ${manager_ready} == true ]] || fail "Codex Routing Desk did not become ready"

actual_models="$(printf '%s' "${models_json}" \
  | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' \
  | sed -E 's/.*"([^"]+)"$/\1/' \
  | sort)"
expected_models="$(printf '%s\n' gpt-5.6-luna gpt-5.6-sol gpt-5.6-terra | sort)"
[[ ${actual_models} == "${expected_models}" ]] || {
  printf 'Expected models:\n%s\n' "${expected_models}" >&2
  printf 'Visible models:\n%s\n' "${actual_models}" >&2
  fail "unexpected model catalog"
}

printf 'Installed Claude-to-Codex bridge with models:\n%s\n' "${actual_models}"
printf 'Active route: provider=%s model=%s effort=%s\n' \
  "${synced_route[2]}" "${synced_route[0]}" "${synced_route[1]}"
if ! command -v claude >/dev/null 2>&1; then
  printf 'warning: Claude Code is not installed; install it before running claudex.\n' >&2
fi
printf 'Launch with: claudex, claudex --pick, or claudex-yolo\n'
printf 'Open the visual model switcher with: claudex-ui\n'

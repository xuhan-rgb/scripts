#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
readonly BIN_DIR="${HOME}/.local/bin"
readonly CODEX_DIR="${CODEX_HOME:-${HOME}/.codex}"
readonly CODEX_CONFIG="${CODEX_DIR}/config.toml"
readonly CODEX_PROVIDER_SOURCE="${SCRIPT_DIR}/codex_provider.py"
readonly CODEX_PROVIDER="${BIN_DIR}/codex-provider"
readonly SECRETS_DIR="${HOME}/.config/codex"
readonly SECRETS_FILE="${SECRETS_DIR}/secrets.env"
readonly BASHRC="${HOME}/.bashrc"
readonly DEFAULT_PROVIDER="${CLAUDEX_DEFAULT_PROVIDER:-crs_local}"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

for command_name in awk cat chmod cmp cp find grep install mkdir mktemp mv printenv python3 readlink rm sort stat; do
  require_command "${command_name}"
done

case "${DEFAULT_PROVIDER}" in
  crs | crs_local | sorryios | zskj) ;;
  *) fail "CLAUDEX_DEFAULT_PROVIDER must be crs, crs_local, sorryios, or zskj" ;;
esac

[[ -f ${CODEX_PROVIDER_SOURCE} ]] || fail "missing provider manager: ${CODEX_PROVIDER_SOURCE}"

update_bashrc() {
  local mode=600
  local bashrc_tmp
  if [[ -e ${BASHRC} ]]; then
    mode="$(stat -c '%a' "${BASHRC}")"
  fi
  bashrc_tmp="$(mktemp "${HOME}/.bashrc.tmp.XXXXXX")"
  if [[ -f ${BASHRC} ]]; then
    awk '
      $0 == "# >>> scripts AI yolo aliases >>>" { managed = 1; next }
      $0 == "# <<< scripts AI yolo aliases <<<" { managed = 0; next }
      $0 == "[ -f \"$HOME/.config/codex/secrets.env\" ] && source \"$HOME/.config/codex/secrets.env\"" { next }
      $0 ~ "^[[:space:]]*alias[[:space:]]+(codex-yolo|claude-yolo|claudex-yolo)[[:space:]]*=" { next }
      !managed { print }
    ' "${BASHRC}" >"${bashrc_tmp}"
  fi
  cat >>"${bashrc_tmp}" <<'EOF'

# >>> scripts AI yolo aliases >>>
[ -f "$HOME/.config/codex/secrets.env" ] && source "$HOME/.config/codex/secrets.env"
alias codex-yolo='codex --dangerously-bypass-approvals-and-sandbox'
alias claude-yolo="claude --dangerously-skip-permissions --disable-slash-commands --strict-mcp-config --mcp-config '{\"mcpServers\":{}}'"
alias claudex-yolo='claudex --dangerously-skip-permissions'
# <<< scripts AI yolo aliases <<<
EOF
  chmod "${mode}" "${bashrc_tmp}"
  mv "${bashrc_tmp}" "${BASHRC}"
}

printf '[1/3] Updating AI yolo aliases in %s\n' "${BASHRC}"
update_bashrc

disable_claude_plugins() {
  [[ -f ${HOME}/.claude/settings.json ]] || return 0
  command -v claude >/dev/null 2>&1 || return 0
  [[ -f ${HOME}/.claude/settings.json.before-disabled-extensions ]] \
    || cp -p "${HOME}/.claude/settings.json" "${HOME}/.claude/settings.json.before-disabled-extensions"
  chmod 600 "${HOME}/.claude/settings.json.before-disabled-extensions"
  while IFS= read -r plugin; do
    [[ -n ${plugin} ]] || continue
    claude plugin disable "${plugin}" >/dev/null
  done < <(
    claude plugin list --json 2>/dev/null | python3 -c '
import json
import sys

for plugin in json.load(sys.stdin):
    if plugin.get("enabled") and plugin.get("id"):
        print(plugin["id"])
'
  )
}

disable_claude_plugins

mkdir -p "${BIN_DIR}" "${CODEX_DIR}" "${SECRETS_DIR}"
chmod 700 "${CODEX_DIR}" "${SECRETS_DIR}"
install -m 0755 "${CODEX_PROVIDER_SOURCE}" "${CODEX_PROVIDER}"
printf '[2/3] Configuring Codex providers and credentials with codex-provider\n'

toml_top_value() {
  awk -v wanted="$1" '
    /^[[:space:]]*\[/ { exit }
    $0 ~ "^[[:space:]]*" wanted "[[:space:]]*=" {
      line = $0
      sub(/^[^\"]*\"/, "", line)
      sub(/\".*$/, "", line)
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
      sub(/^[^\"]*\"/, "", line)
      sub(/\".*$/, "", line)
      print line
      exit
    }
  ' "${CODEX_CONFIG}"
}

if [[ ! -f ${CODEX_CONFIG} ]]; then
  config_tmp="$(mktemp "${CODEX_DIR}/config.toml.tmp.XXXXXX")"
  cat >"${config_tmp}" <<EOF
model_provider = "${DEFAULT_PROVIDER}"
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"
model_reasoning_summary = "concise"
model_verbosity = "medium"
approval_policy = "on-request"
sandbox_mode = "danger-full-access"
personality = "pragmatic"
hide_agent_reasoning = true
web_search = "live"
disable_response_storage = true
preferred_auth_method = "apikey"
service_tier = "default"

[features]
apply_patch_freeform = true
plan_tool = true
rmcp_client = true
view_image_tool = true
parallel = true
fast_mode = true

[sandbox_workspace_write]
network_access = true

[mcp_servers.openaiDeveloperDocs]
url = "https://developers.openai.com/mcp"
EOF
  chmod 600 "${config_tmp}"
  mv "${config_tmp}" "${CODEX_CONFIG}"
fi
chmod 600 "${CODEX_CONFIG}"

ensure_provider() {
  local name="$1"
  local base_url="$2"
  local env_key="$3"
  local model="$4"
  local effort="$5"
  if ! "${CODEX_PROVIDER}" show "${name}" >/dev/null 2>&1; then
    "${CODEX_PROVIDER}" add "${name}" \
      --base-url "${base_url}" \
      --env-key "${env_key}" \
      --wire-api responses \
      --model "${model}" \
      --effort "${effort}" \
      --summary concise \
      --verbosity medium \
      --skip-test >/dev/null
  fi
}

ensure_provider crs "http://81.70.201.249:3000/openai" CRS_OPENAI_KEY gpt-5.5 medium
ensure_provider crs_local "http://127.0.0.1:3000/openai" CRS_OPENAI_KEY gpt-5.6-sol xhigh
ensure_provider sorryios "https://sorryios.ai/codex" SORRYIOS_OPENAI_KEY gpt-5.5 xhigh
ensure_provider zskj "http://10.1.6.27/v1" ZSKJ_OPENAI_KEY gpt-5.4 xhigh

ensure_profile() {
  local name="$1"
  local model="$2"
  local effort="$3"
  local profile="${CODEX_DIR}/${name}.config.toml"
  local profile_tmp
  [[ -f ${profile} ]] && return
  profile_tmp="$(mktemp "${CODEX_DIR}/${name}.config.toml.tmp.XXXXXX")"
  cat >"${profile_tmp}" <<EOF
model_provider = "${name}"
model = "${model}"
model_reasoning_effort = "${effort}"
model_reasoning_summary = "concise"
model_verbosity = "medium"
EOF
  chmod 600 "${profile_tmp}"
  mv "${profile_tmp}" "${profile}"
}

ensure_profile crs gpt-5.5 medium
ensure_profile crs_local gpt-5.6-sol xhigh
ensure_profile sorryios gpt-5.5 xhigh
ensure_profile zskj gpt-5.4 xhigh

disable_extension_tables() {
  local config_file="$1"
  local config_tmp
  config_tmp="$(mktemp "${config_file}.tmp.XXXXXX")"
  awk '
    /^\[/ {
      if (extension && !seen_enabled) print "enabled = false"
      extension = ($0 ~ /^\[mcp_servers\.[^]]+\]$/ || $0 ~ /^\[plugins\.[^]]+\]$/)
      seen_enabled = 0
      print
      next
    }
    extension && /^[[:space:]]*enabled[[:space:]]*=/ {
      if (!seen_enabled) print "enabled = false"
      seen_enabled = 1
      next
    }
    { print }
    END {
      if (extension && !seen_enabled) print "enabled = false"
    }
  ' "${config_file}" >"${config_tmp}"
  if cmp -s "${config_tmp}" "${config_file}"; then
    rm -f -- "${config_tmp}"
  else
    chmod 600 "${config_tmp}"
    mv "${config_tmp}" "${config_file}"
  fi
}

disable_codex_skills() {
  local config_tmp
  local skill_path
  local escaped_path
  config_tmp="$(mktemp "${CODEX_CONFIG}.skills.XXXXXX")"
  awk '
    $0 == "# >>> scripts disabled Codex skills >>>" { managed = 1; next }
    $0 == "# <<< scripts disabled Codex skills <<<" { managed = 0; next }
    !managed { print }
  ' "${CODEX_CONFIG}" >"${config_tmp}"
  printf '\n# >>> scripts disabled Codex skills >>>\n' >>"${config_tmp}"
  while IFS= read -r -d '' skill_path; do
    escaped_path="${skill_path//\\/\\\\}"
    escaped_path="${escaped_path//\"/\\\"}"
    printf '[[skills.config]]\npath = "%s"\nenabled = false\n\n' "${escaped_path}" >>"${config_tmp}"
  done < <(
    find "${CODEX_DIR}/skills" "${HOME}/.agents/skills" "${HOME}/.claude/skills" \
      -type f -name SKILL.md -print0 2>/dev/null | sort -zu
  )
  printf '# <<< scripts disabled Codex skills <<<\n' >>"${config_tmp}"
  if cmp -s "${config_tmp}" "${CODEX_CONFIG}"; then
    rm -f -- "${config_tmp}"
  else
    chmod 600 "${config_tmp}"
    mv "${config_tmp}" "${CODEX_CONFIG}"
  fi
}

for config_file in "${CODEX_CONFIG}" "${CODEX_DIR}"/*.config.toml; do
  [[ -f ${config_file} ]] || continue
  [[ -f ${config_file}.before-disabled-extensions ]] \
    || cp -p "${config_file}" "${config_file}.before-disabled-extensions"
  chmod 600 "${config_file}.before-disabled-extensions"
  disable_extension_tables "${config_file}"
done
disable_codex_skills

active_provider="$(toml_top_value model_provider)"
if [[ -z ${active_provider} ]] || ! grep -Fqx "[model_providers.${active_provider}]" "${CODEX_CONFIG}"; then
  "${CODEX_PROVIDER}" switch "${DEFAULT_PROVIDER}" \
    --model gpt-5.6-sol \
    --effort xhigh \
    --summary concise \
    --verbosity medium >/dev/null
  active_provider="${DEFAULT_PROVIDER}"
fi

if [[ -n ${CLAUDEX_SECRETS_FILE:-} ]]; then
  [[ -r ${CLAUDEX_SECRETS_FILE} ]] || fail "CLAUDEX_SECRETS_FILE is not readable: ${CLAUDEX_SECRETS_FILE}"
  if [[ $(readlink -f "${CLAUDEX_SECRETS_FILE}") != $(readlink -f "${SECRETS_FILE}") ]]; then
    install -m 0600 "${CLAUDEX_SECRETS_FILE}" "${SECRETS_FILE}"
  fi
fi

if [[ ! -f ${SECRETS_FILE} ]]; then
  secrets_tmp="$(mktemp "${SECRETS_DIR}/secrets.env.tmp.XXXXXX")"
  printf '# Codex provider API keys. Managed by codex-provider.\n' >"${secrets_tmp}"
  install -m 0600 "${secrets_tmp}" "${SECRETS_FILE}"
  rm -f -- "${secrets_tmp}"
fi
chmod 600 "${SECRETS_FILE}"

"${CODEX_PROVIDER}" import-env >/dev/null
set -a
# shellcheck disable=SC1090
source "${SECRETS_FILE}"
set +a

active_env_key="$(toml_provider_value "${active_provider}" env_key)"
[[ ${active_env_key} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
  || fail "active provider ${active_provider} has no valid env_key"

if [[ -z $(printenv "${active_env_key}" || true) ]]; then
  if [[ -t 0 && -t 1 && ${CLAUDEX_NONINTERACTIVE:-0} != 1 ]]; then
    "${CODEX_PROVIDER}" set-key "${active_provider}"
    set -a
    # shellcheck disable=SC1090
    source "${SECRETS_FILE}"
    set +a
  else
    fail "${active_env_key} is missing; export it or pass CLAUDEX_SECRETS_FILE=/path/to/secrets.env"
  fi
fi

printf 'Codex configured: provider=%s config=%s\n' "${active_provider}" "${CODEX_CONFIG}"
printf 'Provider manager: %s\n' "${CODEX_PROVIDER}"
printf 'Shell aliases installed in: %s\n' "${BASHRC}"

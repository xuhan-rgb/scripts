#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
readonly BIN_DIR="${HOME}/.local/bin"
readonly CODEX_DIR="${CODEX_HOME:-${HOME}/.codex}"
readonly CODEX_CONFIG="${CODEX_DIR}/config.toml"
readonly CODEX_PROVIDER_SOURCE="${SCRIPT_DIR}/codex_provider.py"
readonly CODEX_AUTH_SOURCE="${SCRIPT_DIR}/switch-codex-auth.sh"
readonly CODEX_USAGE_SOURCE="${SCRIPT_DIR}/codex-usage"
readonly CODEX_USAGE_WIDGET_SOURCE="${SCRIPT_DIR}/../codex-usage-widget"
readonly ACCOUNT_MANAGER_QT_SOURCE="${SCRIPT_DIR}/codex_account_manager_qt.py"
readonly ACCOUNT_MANAGER_BACKEND_SOURCE="${SCRIPT_DIR}/codex_account_manager_backend.py"
readonly ACCOUNT_MANAGER_ICON_SOURCE="${SCRIPT_DIR}/codex-account-manager.svg"
readonly ACCOUNT_MANAGER_LIB_DIR="${HOME}/.local/lib/codex-account-manager"
readonly ACCOUNT_MANAGER_ICON_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
readonly APPLICATIONS_DIR="${HOME}/.local/share/applications"
readonly AUTOSTART_DIR="${HOME}/.config/autostart"
readonly SECRETS_DIR="${HOME}/.config/codex"
readonly SECRETS_FILE="${SECRETS_DIR}/secrets.env"
readonly BASHRC="${HOME}/.bashrc"
readonly CLAUDE_DIR="${HOME}/.claude"
readonly CLAUDE_SETTINGS_YOLO="${CLAUDE_DIR}/settings.yolo.json"
readonly CLAUDE_SETTINGS_CLAUDEX_YOLO="${CLAUDE_DIR}/settings.claudex-yolo.json"
readonly CODEX_CONFIG_YOLO="${CODEX_DIR}/yolo.config.toml"
readonly DEFAULT_PROVIDER="${CLAUDEX_DEFAULT_PROVIDER:-crs_local}"
readonly DEFAULT_ENABLED_SKILLS="agent-reach brainstorming grill-me grill-with-docs handoff tdd"
readonly INTERNAL_SKILLS="domain-modeling grilling"
readonly YOLO_MINIMAL_SKILLS="agent-reach brainstorming domain-modeling grilling tdd"
readonly AGENT_REACH_VERSION="1.5.0"
readonly AGENT_REACH_SOURCE="https://github.com/Panniantong/Agent-Reach/archive/refs/tags/v${AGENT_REACH_VERSION}.zip"
ENABLED_SKILLS="${DEFAULT_ENABLED_SKILLS}"

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
[[ -f ${CODEX_AUTH_SOURCE} ]] || fail "missing Codex auth switcher: ${CODEX_AUTH_SOURCE}"
[[ -f ${CODEX_USAGE_SOURCE} ]] || fail "missing Codex usage command: ${CODEX_USAGE_SOURCE}"
[[ -f ${CODEX_USAGE_WIDGET_SOURCE} ]] \
  || fail "missing Codex usage widget: ${CODEX_USAGE_WIDGET_SOURCE}"
[[ -f ${ACCOUNT_MANAGER_QT_SOURCE} ]] \
  || fail "missing Qt account manager: ${ACCOUNT_MANAGER_QT_SOURCE}"
[[ -f ${ACCOUNT_MANAGER_BACKEND_SOURCE} ]] \
  || fail "missing Qt account manager backend: ${ACCOUNT_MANAGER_BACKEND_SOURCE}"
[[ -f ${ACCOUNT_MANAGER_ICON_SOURCE} ]] \
  || fail "missing Qt account manager icon: ${ACCOUNT_MANAGER_ICON_SOURCE}"

create_claude_yolo_settings() {
  mkdir -p "${CLAUDE_DIR}"
  python3 - "${CLAUDE_DIR}/skills" "${CLAUDE_SETTINGS_YOLO}" \
    "${CLAUDE_SETTINGS_CLAUDEX_YOLO}" "${YOLO_MINIMAL_SKILLS}" <<'PY'
import json
import re
import sys
from pathlib import Path

skills_root = Path(sys.argv[1])
yolo_path = Path(sys.argv[2])
claudex_yolo_path = Path(sys.argv[3])
minimal_skills = set(sys.argv[4].split())
skill_names = set(minimal_skills)

if skills_root.is_dir():
    for skill_file in skills_root.rglob("SKILL.md"):
        name = skill_file.parent.name
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue
        frontmatter = re.search(r"\A---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter:
            match = re.search(r"^name:\s*([^#\n]+)", frontmatter.group(1), re.MULTILINE)
            if match:
                name = match.group(1).strip().strip("'\"")
        skill_names.add(name)

overrides = {
    name: "on" if name in minimal_skills else "off"
    for name in sorted(skill_names)
}
yolo_path.write_text(
    json.dumps({"skillOverrides": overrides}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

claudex_overrides = dict(overrides)
claudex_overrides["claude-api"] = "off"
claudex_yolo_path.write_text(
    json.dumps(
        {
            "availableModels": ["claudex-router[1m]"],
            "skillOverrides": claudex_overrides,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  chmod 600 "${CLAUDE_SETTINGS_YOLO}" "${CLAUDE_SETTINGS_CLAUDEX_YOLO}"
  printf 'Created minimal skill settings for claude-yolo: %s\n' "${CLAUDE_SETTINGS_YOLO}" >&2
  printf 'Created minimal skill settings for claudex-yolo: %s\n' \
    "${CLAUDE_SETTINGS_CLAUDEX_YOLO}" >&2
}

create_codex_yolo_config() {
  mkdir -p "${CODEX_DIR}"

  local config_tmp
  local enabled
  local enabled_skill_names=" "
  local escaped_path
  local skill_name
  local skill_path
  config_tmp="$(mktemp "${CODEX_CONFIG_YOLO}.tmp.XXXXXX")"

  awk '
    /^\[\[skills\.config\]\]/ { in_skill = 1; next }
    in_skill && /^$/ { in_skill = 0; next }
    in_skill { next }
    $0 == "# >>> scripts disabled Codex skills >>>" { managed = 1; next }
    $0 == "# <<< scripts disabled Codex skills <<<" { managed = 0; next }
    !managed && !in_skill { print }
  ' "${CODEX_CONFIG}" >"${config_tmp}"

  printf '\n# >>> scripts disabled Codex skills >>>\n' >>"${config_tmp}"

  while IFS= read -r -d '' skill_path; do
    escaped_path="${skill_path//\\/\\\\}"
    escaped_path="${escaped_path//\"/\\\"}"
    skill_name="$(basename "$(dirname "${skill_path}")")"
    enabled=false
    if [[ " ${YOLO_MINIMAL_SKILLS} " == *" ${skill_name} "* ]] \
      && [[ ${enabled_skill_names} != *" ${skill_name} "* ]]; then
      enabled=true
      enabled_skill_names+="${skill_name} "
    fi
    printf '[[skills.config]]\npath = "%s"\nenabled = %s\n\n' \
      "${escaped_path}" "${enabled}" >>"${config_tmp}"
  done < <(
    find "${CODEX_DIR}/skills" "${HOME}/.agents/skills" "${HOME}/.claude/skills" \
      -type f -name SKILL.md -print0 2>/dev/null | sort -zu
  )

  printf '# <<< scripts disabled Codex skills <<<\n' >>"${config_tmp}"

  chmod 600 "${config_tmp}"
  mv "${config_tmp}" "${CODEX_CONFIG_YOLO}"
  printf 'Created minimal skill config for codex-yolo: %s\n' "${CODEX_CONFIG_YOLO}" >&2
}

update_bashrc() {
  local mode=600
  local bashrc_tmp
  if [[ -e ${BASHRC} ]]; then
    mode="$(stat -c '%a' "${BASHRC}")"
  fi
  bashrc_tmp="$(mktemp "${HOME}/.bashrc.tmp.XXXXXX")"
  cat >"${bashrc_tmp}" <<'EOF'
# >>> scripts AI alias cleanup >>>
unalias codex 2>/dev/null || true
# <<< scripts AI alias cleanup <<<

EOF
  if [[ -f ${BASHRC} ]]; then
    awk '
      $0 == "# >>> scripts AI alias cleanup >>>" { cleanup = 1; next }
      $0 == "# <<< scripts AI alias cleanup <<<" { cleanup = 0; next }
      $0 == "# >>> scripts AI yolo aliases >>>" { managed = 1; next }
      $0 == "# <<< scripts AI yolo aliases <<<" { managed = 0; next }
      $0 == "[ -f \"$HOME/.config/codex/secrets.env\" ] && source \"$HOME/.config/codex/secrets.env\"" { next }
      $0 ~ "^[[:space:]]*alias[[:space:]]+(codex|codex-yolo|claude-yolo|claudex-yolo|codex-account-manager)[[:space:]]*=" { next }
      !cleanup && !managed { print }
    ' "${BASHRC}" >>"${bashrc_tmp}"
  fi
  cat >>"${bashrc_tmp}" <<'EOF'

# >>> scripts AI yolo aliases >>>
[ -f "$HOME/.config/codex/secrets.env" ] && source "$HOME/.config/codex/secrets.env"
codex() {
  codex-auth run -- "$@"
}
alias codex-yolo='codex-auth run -- --dangerously-bypass-approvals-and-sandbox -p yolo'
alias claude-yolo='claude --dangerously-skip-permissions --settings ~/.claude/settings.yolo.json'
alias claudex-yolo='CLAUDEX_YOLO=1 claudex'
# <<< scripts AI yolo aliases <<<
EOF
  chmod "${mode}" "${bashrc_tmp}"
  mv "${bashrc_tmp}" "${BASHRC}"
}

plugin_install_path() {
  claude plugin list --json 2>/dev/null | python3 -c '
import json
import sys

wanted = sys.argv[1]
for plugin in json.load(sys.stdin):
    if plugin.get("id") == wanted and plugin.get("installPath"):
        print(plugin["installPath"])
        break
' "$1"
}

ensure_claude_plugin_source() {
  local plugin="$1"
  local plugin_path
  printf 'Checking Claude plugin source: %s\n' "${plugin}" >&2
  plugin_path="$(plugin_install_path "${plugin}" || true)"
  if [[ -z ${plugin_path} ]]; then
    printf 'Installing Claude plugin source: %s\n' "${plugin}" >&2
    claude plugin install "${plugin}" --scope user >/dev/null
    plugin_path="$(plugin_install_path "${plugin}" || true)"
  fi
  [[ -d ${plugin_path} ]] || fail "Claude plugin source is unavailable: ${plugin}"
  printf 'Claude plugin source ready: %s\n' "${plugin}" >&2
  printf '%s\n' "${plugin_path}"
}

copy_skill_if_missing() {
  local source="$1"
  local name="$2"
  local skill_root
  local target
  [[ -f ${source}/SKILL.md ]] || fail "Skill source is incomplete: ${source}"
  for skill_root in "${HOME}/.agents/skills" "${HOME}/.claude/skills"; do
    target="${skill_root}/${name}"
    [[ -f ${target}/SKILL.md ]] && continue
    mkdir -p "${skill_root}"
    cp -R "${source}" "${target}"
  done
}

install_agent_reach() {
  local venv_dir="${HOME}/.local/share/agent-reach-venv"
  local executable
  executable="$(command -v agent-reach || true)"
  if [[ -z ${executable} ]] \
    || [[ $("${executable}" --version 2>/dev/null || true) != "Agent Reach v${AGENT_REACH_VERSION}" ]]; then
    if command -v uv >/dev/null 2>&1; then
      UV_TOOL_BIN_DIR="${BIN_DIR}" uv tool install --force "${AGENT_REACH_SOURCE}" >/dev/null
    else
      python3 -c 'import ensurepip, venv' >/dev/null 2>&1 \
        || fail "Agent Reach requires uv or python3-venv; install python3-venv or rerun with CLAUDEX_AGENT_REACH=0"
      python3 -m venv "${venv_dir}"
      "${venv_dir}/bin/python" -m pip install --upgrade "${AGENT_REACH_SOURCE}" >/dev/null
      install -m 0755 "${venv_dir}/bin/agent-reach" "${BIN_DIR}/agent-reach"
    fi
  fi
  if [[ -x ${BIN_DIR}/agent-reach ]]; then
    executable="${BIN_DIR}/agent-reach"
  else
    executable="$(command -v agent-reach || true)"
  fi
  [[ -n ${executable} ]] \
    && [[ $("${executable}" --version 2>/dev/null || true) == "Agent Reach v${AGENT_REACH_VERSION}" ]] \
    || fail "Agent Reach v${AGENT_REACH_VERSION} installation failed"

  if [[ ! -f ${HOME}/.agents/skills/agent-reach/SKILL.md \
    && ! -f ${HOME}/.claude/skills/agent-reach/SKILL.md ]]; then
    mkdir -p "${HOME}/.agents/skills" "${HOME}/.claude/skills"
    "${executable}" skill --install >/dev/null
  elif [[ ! -f ${HOME}/.agents/skills/agent-reach/SKILL.md ]]; then
    copy_skill_if_missing "${HOME}/.claude/skills/agent-reach" agent-reach
  elif [[ ! -f ${HOME}/.claude/skills/agent-reach/SKILL.md ]]; then
    copy_skill_if_missing "${HOME}/.agents/skills/agent-reach" agent-reach
  fi
  [[ -f ${HOME}/.agents/skills/agent-reach/SKILL.md \
    && -f ${HOME}/.claude/skills/agent-reach/SKILL.md ]] \
    || fail "Agent Reach skill installation failed"
}

configure_agent_reach_choice() {
  local choice="${CLAUDEX_AGENT_REACH:-}"
  local executable

  case "${choice}" in
    0)
      ENABLED_SKILLS="${ENABLED_SKILLS#agent-reach }"
      printf 'Agent Reach disabled by CLAUDEX_AGENT_REACH=0\n'
      return
      ;;
    1 | "") ;;
    *) fail "CLAUDEX_AGENT_REACH must be 0 or 1" ;;
  esac

  executable="$(command -v agent-reach || true)"
  if [[ -n ${executable} ]] \
    && [[ $("${executable}" --version 2>/dev/null || true) == "Agent Reach v${AGENT_REACH_VERSION}" ]]; then
    return
  fi
  command -v uv >/dev/null 2>&1 && return
  [[ ${CLAUDEX_SKIP_SKILL_INSTALL:-0} == 1 ]] && return

  if [[ -z ${choice} ]]; then
    if [[ -t 0 && -t 1 && ${CLAUDEX_NONINTERACTIVE:-0} != 1 ]]; then
      printf 'uv is not installed. Enable Agent Reach using a Python virtual environment? [y/N] '
      IFS= read -r choice || choice=""
      case "${choice,,}" in
        y | yes) choice=1 ;;
        *) choice=0 ;;
      esac
    else
      choice=0
      printf 'uv is not installed; skipping optional Agent Reach in non-interactive mode.\n'
    fi
  fi

  if [[ ${choice} == 0 ]]; then
    ENABLED_SKILLS="${ENABLED_SKILLS#agent-reach }"
    printf 'Agent Reach will not be enabled; the other selected skills are unchanged.\n'
    return
  fi

  python3 -c 'import ensurepip, venv' >/dev/null 2>&1 \
    || fail "Agent Reach requires python3-venv when uv is unavailable; install it or rerun and choose no"
}

install_selected_skills() {
  local matt_path
  local superpowers_path
  [[ ${CLAUDEX_SKIP_SKILL_INSTALL:-0} == 1 ]] && return 0
  command -v claude >/dev/null 2>&1 || fail "missing required command: claude"

  mkdir -p "${BIN_DIR}"
  if [[ " ${ENABLED_SKILLS} " == *" agent-reach "* ]]; then
    install_agent_reach
  fi
  matt_path="$(ensure_claude_plugin_source mattpocock-skills@claude-plugins-official)"
  superpowers_path="$(ensure_claude_plugin_source superpowers@claude-plugins-official)"

  copy_skill_if_missing "${matt_path}/skills/productivity/grill-me" grill-me
  copy_skill_if_missing "${matt_path}/skills/engineering/tdd" tdd
  copy_skill_if_missing "${matt_path}/skills/productivity/handoff" handoff
  copy_skill_if_missing "${matt_path}/skills/engineering/grill-with-docs" grill-with-docs
  copy_skill_if_missing "${matt_path}/skills/productivity/grilling" grilling
  copy_skill_if_missing "${matt_path}/skills/engineering/domain-modeling" domain-modeling
  copy_skill_if_missing "${superpowers_path}/skills/brainstorming" brainstorming
}

disable_claude_plugins() {
  command -v claude >/dev/null 2>&1 || return 0
  if [[ -f ${HOME}/.claude/settings.json ]]; then
    [[ -f ${HOME}/.claude/settings.json.before-disabled-extensions ]] \
      || cp -p "${HOME}/.claude/settings.json" "${HOME}/.claude/settings.json.before-disabled-extensions"
    chmod 600 "${HOME}/.claude/settings.json.before-disabled-extensions"
  fi
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

configure_claude_skill_overrides() {
  # 默认的 claude 命令不设置 skillOverrides，允许使用所有 skills
  # 只有 claude-yolo 通过 settings.yolo.json 限制 skills
  printf 'Skipping Claude skillOverrides - default command uses all available skills\n' >&2

  # 确保目录存在
  mkdir -p "${HOME}/.claude"
  chmod 700 "${HOME}/.claude"

  # 如果 settings.json 不存在，创建一个空的
  local settings_file="${HOME}/.claude/settings.json"
  if [[ ! -f ${settings_file} ]]; then
    echo '{}' > "${settings_file}"
    chmod 600 "${settings_file}"
  fi
}

provider_manager() {
  python3 "${CODEX_PROVIDER_SOURCE}" "$@"
}

printf '[1/3] Initializing Codex providers\n'
mkdir -p "${BIN_DIR}" "${CODEX_DIR}" "${SECRETS_DIR}" "${ACCOUNT_MANAGER_LIB_DIR}" \
  "${ACCOUNT_MANAGER_ICON_DIR}" "${APPLICATIONS_DIR}" "${AUTOSTART_DIR}"
chmod 700 "${CODEX_DIR}" "${SECRETS_DIR}"
rm -f -- "${BIN_DIR}/codex-provider"
install -m 0755 "${CODEX_AUTH_SOURCE}" "${BIN_DIR}/codex-auth"
rm -f -- "${BIN_DIR}/codex-usage"
install -m 0755 "${CODEX_USAGE_SOURCE}" "${BIN_DIR}/codex-usage"
rm -f -- "${BIN_DIR}/codex-usage-widget"
install -m 0755 "${CODEX_USAGE_WIDGET_SOURCE}" "${BIN_DIR}/codex-usage-widget"
install -m 0755 "${ACCOUNT_MANAGER_QT_SOURCE}" \
  "${ACCOUNT_MANAGER_LIB_DIR}/codex_account_manager_qt.py"
install -m 0644 "${ACCOUNT_MANAGER_BACKEND_SOURCE}" \
  "${ACCOUNT_MANAGER_LIB_DIR}/codex_account_manager_backend.py"
install -m 0644 "${CODEX_PROVIDER_SOURCE}" \
  "${ACCOUNT_MANAGER_LIB_DIR}/codex_provider.py"
install -m 0644 "${ACCOUNT_MANAGER_ICON_SOURCE}" \
  "${ACCOUNT_MANAGER_LIB_DIR}/codex-account-manager.svg"
install -m 0644 "${ACCOUNT_MANAGER_ICON_SOURCE}" \
  "${ACCOUNT_MANAGER_ICON_DIR}/codex-account-manager.svg"

cat >"${BIN_DIR}/codex-account-manager" <<EOF
#!/usr/bin/env bash
if [[ -x "${BIN_DIR}/codex-usage-widget" ]]; then
  "${BIN_DIR}/codex-usage-widget" stop >/dev/null 2>&1 || true
fi
exec /usr/bin/env python3 "${ACCOUNT_MANAGER_LIB_DIR}/codex_account_manager_qt.py" "\$@"
EOF
chmod 755 "${BIN_DIR}/codex-account-manager"

cat >"${APPLICATIONS_DIR}/codex-account-manager.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Codex Account Manager
Comment=Switch Codex ChatGPT accounts and API providers
Exec=${BIN_DIR}/codex-account-manager
TryExec=${BIN_DIR}/codex-account-manager
Icon=${ACCOUNT_MANAGER_ICON_DIR}/codex-account-manager.svg
Terminal=false
Categories=Utility;
StartupNotify=true
StartupWMClass=Codex Account Manager
EOF
chmod 644 "${APPLICATIONS_DIR}/codex-account-manager.desktop"

cat >"${AUTOSTART_DIR}/codex-account-manager.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Codex Account Manager
Comment=Keep Codex account selection and quota available in the system tray
Exec=${BIN_DIR}/codex-account-manager --background
TryExec=${BIN_DIR}/codex-account-manager
Icon=${ACCOUNT_MANAGER_ICON_DIR}/codex-account-manager.svg
Terminal=false
X-GNOME-Autostart-enabled=true
NoDisplay=true
StartupWMClass=Codex Account Manager
EOF
chmod 644 "${AUTOSTART_DIR}/codex-account-manager.desktop"

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

upsert_top_level_number() {
  local key="$1"
  local value="$2"
  local config_tmp
  config_tmp="$(mktemp "${CODEX_CONFIG}.tmp.XXXXXX")"
  awk -v wanted="${key}" -v replacement="${key} = ${value}" '
    BEGIN { in_prefix = 1 }
    in_prefix && /^[[:space:]]*\[/ {
      if (!seen) print replacement
      seen = 1
      in_prefix = 0
    }
    in_prefix && $0 ~ "^[[:space:]]*" wanted "[[:space:]]*=" {
      if (!seen) print replacement
      seen = 1
      next
    }
    { print }
    END {
      if (!seen) print replacement
    }
  ' "${CODEX_CONFIG}" >"${config_tmp}"
  if cmp -s "${config_tmp}" "${CODEX_CONFIG}"; then
    rm -f -- "${config_tmp}"
  else
    chmod 600 "${config_tmp}"
    mv "${config_tmp}" "${CODEX_CONFIG}"
  fi
}

if [[ ! -f ${CODEX_CONFIG} ]]; then
  config_tmp="$(mktemp "${CODEX_DIR}/config.toml.tmp.XXXXXX")"
  cat >"${config_tmp}" <<EOF
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
upsert_top_level_number model_context_window 372000
upsert_top_level_number model_auto_compact_token_limit 244800

ensure_provider() {
  local name="$1"
  local base_url="$2"
  local env_key="$3"
  local model="$4"
  local effort="$5"
  if ! provider_manager show "${name}" >/dev/null 2>&1; then
    provider_manager add "${name}" \
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

disable_extension_tables() {
  local config_file="$1"
  local config_tmp
  config_tmp="$(mktemp "${config_file}.tmp.XXXXXX")"
  awk '
    /^\[/ {
      if (extension && !seen_enabled) print "enabled = false"
      nested_mcp = ($0 ~ /^\[mcp_servers\..+\.env\]$/)
      extension = (!nested_mcp && ($0 ~ /^\[mcp_servers\.[^]]+\]$/ || $0 ~ /^\[plugins\.[^]]+\]$/))
      seen_enabled = 0
      print
      next
    }
    nested_mcp && /^[[:space:]]*enabled[[:space:]]*=/ { next }
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

configure_codex_skills() {
  local config_tmp
  local enabled
  local enabled_skill_names=" "
  local escaped_path
  local skill_name
  local skill_path
  config_tmp="$(mktemp "${CODEX_CONFIG}.skills.XXXXXX")"
  awk '
    /^\[\[skills\.config\]\]/ { in_skill = 1; next }
    in_skill && /^$/ { in_skill = 0; next }
    in_skill { next }
    $0 == "# >>> scripts disabled Codex skills >>>" { managed = 1; next }
    $0 == "# <<< scripts disabled Codex skills <<<" { managed = 0; next }
    !managed { print }
  ' "${CODEX_CONFIG}" >"${config_tmp}"
  printf '\n# >>> scripts disabled Codex skills >>>\n' >>"${config_tmp}"
  while IFS= read -r -d '' skill_path; do
    escaped_path="${skill_path//\\/\\\\}"
    escaped_path="${escaped_path//\"/\\\"}"
    skill_name="$(basename "$(dirname "${skill_path}")")"
    enabled=false
    if [[ " ${YOLO_MINIMAL_SKILLS} " == *" ${skill_name} "* ]] \
      && [[ ${enabled_skill_names} != *" ${skill_name} "* ]]; then
      enabled=true
      enabled_skill_names+="${skill_name} "
    fi
    printf '[[skills.config]]\npath = "%s"\nenabled = %s\n\n' \
      "${escaped_path}" "${enabled}" >>"${config_tmp}"
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

active_provider="$(toml_top_value model_provider)"
if [[ ${active_provider} != openai ]] \
  && { [[ -z ${active_provider} ]] || ! grep -Fqx "[model_providers.${active_provider}]" "${CODEX_CONFIG}"; }; then
  provider_manager switch "${DEFAULT_PROVIDER}" >/dev/null
  active_provider="${DEFAULT_PROVIDER}"
fi
provider_manager init-existing >/dev/null

if [[ -n ${CLAUDEX_SECRETS_FILE:-} ]]; then
  [[ -r ${CLAUDEX_SECRETS_FILE} ]] || fail "CLAUDEX_SECRETS_FILE is not readable: ${CLAUDEX_SECRETS_FILE}"
  if [[ $(readlink -f "${CLAUDEX_SECRETS_FILE}") != $(readlink -f "${SECRETS_FILE}") ]]; then
    install -m 0600 "${CLAUDEX_SECRETS_FILE}" "${SECRETS_FILE}"
  fi
fi

if [[ ! -f ${SECRETS_FILE} ]]; then
  secrets_tmp="$(mktemp "${SECRETS_DIR}/secrets.env.tmp.XXXXXX")"
  printf '# Codex provider API keys. Managed by the local routing desk.\n' >"${secrets_tmp}"
  install -m 0600 "${secrets_tmp}" "${SECRETS_FILE}"
  rm -f -- "${secrets_tmp}"
fi
chmod 600 "${SECRETS_FILE}"

provider_manager import-env >/dev/null
set -a
# shellcheck disable=SC1090
source "${SECRETS_FILE}"
set +a

if [[ ${active_provider} == openai ]]; then
  printf 'Codex account mode is active; use codex-auth api to restore the saved API provider.\n'
else
  active_env_key="$(toml_provider_value "${active_provider}" env_key)"
  if [[ ! ${active_env_key} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    printf 'Provider %s has no valid env_key yet; configure it at http://127.0.0.1:8320 after installation.\n' \
      "${active_provider}"
  elif [[ -z $(printenv "${active_env_key}" || true) ]]; then
    printf 'Provider %s has no Key yet; configure it at http://127.0.0.1:8320 after installation.\n' \
      "${active_provider}"
  fi
fi

printf 'Codex configured: provider=%s config=%s\n' "${active_provider}" "${CODEX_CONFIG}"

printf '[2/3] Updating aliases, skills, and extension policy\n'
configure_agent_reach_choice
update_bashrc
install_selected_skills
create_claude_yolo_settings
disable_claude_plugins
configure_claude_skill_overrides
for config_file in "${CODEX_CONFIG}" "${CODEX_DIR}"/*.config.toml; do
  [[ -f ${config_file} ]] || continue
  [[ -f ${config_file}.before-disabled-extensions ]] \
    || cp -p "${config_file}" "${config_file}.before-disabled-extensions"
  chmod 600 "${config_file}.before-disabled-extensions"
  disable_extension_tables "${config_file}"
done
configure_codex_skills
create_codex_yolo_config

printf 'Shell aliases installed in: %s\n' "${BASHRC}"
printf 'Codex auth switch installed: %s (use: codex-auth status)\n' "${BIN_DIR}/codex-auth"
printf 'Codex usage command installed: %s (use: codex-usage)\n' "${BIN_DIR}/codex-usage"
printf 'Codex usage widget installed: %s (use: codex-usage-widget)\n' \
  "${BIN_DIR}/codex-usage-widget"
printf 'Native Codex account manager installed: %s (use: codex-account-manager)\n' \
  "${BIN_DIR}/codex-account-manager"

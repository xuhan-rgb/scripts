#!/usr/bin/env bash
set -euo pipefail

readonly CODEX_DIR="${CODEX_HOME:-${HOME}/.codex}"
readonly CODEX_CONFIG="${CODEX_DIR}/config.toml"
readonly CODEX_YOLO_CONFIG="${CODEX_DIR}/yolo.config.toml"
readonly STATE_DIR="${HOME}/.config/codex"
readonly API_PROVIDER_FILE="${STATE_DIR}/api-provider"
readonly ACTIVE_ACCOUNT_FILE="${STATE_DIR}/active-account"
readonly ACCOUNTS_DIR="${CODEX_ACCOUNTS_DIR:-${HOME}/.local/share/codex/accounts}"
readonly DELETED_ACCOUNTS_DIR="${CODEX_DELETED_ACCOUNTS_DIR:-${HOME}/.local/share/codex/deleted-accounts}"
readonly CODEX_BIN="${CODEX_AUTH_CODEX_BIN:-$(type -P codex || true)}"
readonly -a ACCOUNT_CONFIG_OVERRIDES=(
  -c 'model_provider="openai"'
  -c 'preferred_auth_method="chatgpt"'
  -c 'forced_login_method="chatgpt"'
  -c 'cli_auth_credentials_store="file"'
)
AUTO_PENDING_DIR=""

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: codex-auth account [--device-auth]
       codex-auth api [PROVIDER]
       codex-auth add NAME [--device-auth]
       codex-auth add-auto [--device-auth]
       codex-auth use NAME
       codex-auth remove NAME --yes
       codex-auth list
       codex-auth run [--account NAME] [--] [CODEX_ARGS...]
       codex-auth status

Modes:
  account  Use the built-in OpenAI provider and sign in with ChatGPT.
  api      Select a custom API provider, or restore the last one when omitted.
  add      Create or finish login for an isolated named ChatGPT account.
  add-auto Create an account and use the authenticated login email as its name.
  use      Select the named account used by future Codex processes.
  remove   Move a named account to a recoverable archive; shared conversations remain.
  list     List ChatGPT accounts without displaying credentials.
  run      Start Codex pinned to the selected account; conversations stay shared.
  status   Show the active mode and provider without displaying credentials.
EOF
}

require_codex() {
  [[ -n ${CODEX_BIN} && -x ${CODEX_BIN} ]] || fail "codex executable not found"
}

validate_account_name() {
  local account="$1"
  [[ ${account} =~ ^[a-z0-9][a-z0-9._+@-]{0,127}$ ]] \
    || fail "invalid account name: use a lowercase email address or a safe local alias"
}

account_home() {
  validate_account_name "$1"
  printf '%s/%s\n' "${ACCOUNTS_DIR}" "$1"
}

write_state_file() {
  local path="$1"
  local value="$2"
  local state_tmp
  mkdir -p "$(dirname "${path}")"
  chmod 700 "$(dirname "${path}")"
  state_tmp="$(mktemp "${path}.tmp.XXXXXX")"
  printf '%s\n' "${value}" >"${state_tmp}"
  chmod 600 "${state_tmp}"
  mv "${state_tmp}" "${path}"
}

active_account() {
  local account=""
  [[ -f ${ACTIVE_ACCOUNT_FILE} ]] || return 1
  IFS= read -r account <"${ACTIVE_ACCOUNT_FILE}" || true
  validate_account_name "${account}"
  [[ -d $(account_home "${account}") ]] || fail "active account does not exist: ${account}"
  printf '%s\n' "${account}"
}

save_active_account() {
  write_state_file "${ACTIVE_ACCOUNT_FILE}" "$1"
}

clear_active_account() {
  rm -f -- "${ACTIVE_ACCOUNT_FILE}"
}

ensure_shared_link() {
  local target="$1"
  local link="$2"
  if [[ -L ${link} ]]; then
    [[ $(readlink -f "${link}") == "$(readlink -f "${target}")" ]] \
      || fail "account path points to an unexpected target: ${link}"
  elif [[ -e ${link} ]]; then
    fail "account path must be a shared symlink: ${link}"
  else
    ln -s "${target}" "${link}"
  fi
}

ensure_shared_directory() {
  local account_dir="$1"
  local name="$2"
  mkdir -p "${CODEX_DIR}/${name}"
  ensure_shared_link "${CODEX_DIR}/${name}" "${account_dir}/${name}"
}

ensure_shared_file() {
  local account_dir="$1"
  local name="$2"
  local create_if_missing="${3:-0}"
  if [[ ! -e ${CODEX_DIR}/${name} && ${create_if_missing} == 1 ]]; then
    touch "${CODEX_DIR}/${name}"
    chmod 600 "${CODEX_DIR}/${name}"
  fi
  [[ -e ${CODEX_DIR}/${name} || -L ${CODEX_DIR}/${name} ]] || return 0
  ensure_shared_link "${CODEX_DIR}/${name}" "${account_dir}/${name}"
}

prepare_account_home() {
  local account="$1"
  local account_dir
  local config_profile
  account_dir="$(account_home "${account}")"
  [[ -d ${account_dir} ]] || fail "named account does not exist: ${account}"
  chmod 700 "${account_dir}"

  ensure_shared_directory "${account_dir}" sessions
  ensure_shared_directory "${account_dir}" archived_sessions
  ensure_shared_directory "${account_dir}" thread-writer-locks
  ensure_shared_directory "${account_dir}" attachments
  for name in skills rules plugins; do
    [[ -d ${CODEX_DIR}/${name} ]] && ensure_shared_link "${CODEX_DIR}/${name}" "${account_dir}/${name}"
  done

  ensure_shared_file "${account_dir}" history.jsonl 1
  ensure_shared_file "${account_dir}" session_index.jsonl 1
  ensure_shared_file "${account_dir}" config.toml
  ensure_shared_file "${account_dir}" yolo.config.toml
  ensure_shared_file "${account_dir}" AGENTS.md
  for config_profile in "${CODEX_DIR}"/*.config.toml; do
    [[ -f ${config_profile} ]] || continue
    ensure_shared_link "${config_profile}" "${account_dir}/$(basename "${config_profile}")"
  done
  printf '%s\n' "${account_dir}"
}

account_email_from_auth() {
  python3 - "$1" <<'PY'
import base64
import json
import re
import sys
from pathlib import Path

try:
    auth = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    token = auth["tokens"]["id_token"]
    segment = token.split(".", 2)[1]
    payload = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
except (KeyError, IndexError, OSError, UnicodeDecodeError, ValueError):
    raise SystemExit(1)

pending = [payload]
while pending:
    value = pending.pop()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() == "email" and isinstance(child, str):
                email = child.strip().lower()
                if len(email) <= 128 and re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
                    print(email)
                    raise SystemExit(0)
            if isinstance(child, (dict, list)):
                pending.append(child)
    elif isinstance(value, list):
        pending.extend(child for child in value if isinstance(child, (dict, list)))
raise SystemExit(1)
PY
}

legacy_chatgpt_login_available() {
  python3 - "${CODEX_DIR}/auth.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    auth = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

if not (
    isinstance(auth, dict)
    and auth.get("auth_mode") == "chatgpt"
    and isinstance(auth.get("tokens"), dict)
):
    raise SystemExit(1)
PY
}

archive_pending_account() {
  local archive_dir
  local pending_name
  local stamp
  [[ -n ${AUTO_PENDING_DIR} && -d ${AUTO_PENDING_DIR} ]] || return 0
  pending_name="$(basename "${AUTO_PENDING_DIR}")"
  mkdir -p "${DELETED_ACCOUNTS_DIR}" || return 0
  chmod 700 "${DELETED_ACCOUNTS_DIR}" || true
  printf -v stamp '%(%Y%m%d_%H%M%S)T' -1
  archive_dir="${DELETED_ACCOUNTS_DIR}/${pending_name}.${stamp}"
  if mv -- "${AUTO_PENDING_DIR}" "${archive_dir}"; then
    chmod 700 "${archive_dir}" || true
    printf 'Incomplete account login archived: %s\n' "${archive_dir}" >&2
  fi
  AUTO_PENDING_DIR=""
}

run_account_codex() {
  local account="$1"
  shift
  local account_dir
  require_codex
  account_dir="$(prepare_account_home "${account}")"
  CODEX_HOME="${account_dir}" CODEX_SQLITE_HOME="${CODEX_DIR}" \
    "${CODEX_BIN}" "${ACCOUNT_CONFIG_OVERRIDES[@]}" "$@"
}

add_named_account() {
  local account="${1:-}"
  local login_option="${2:-}"
  local account_dir
  [[ -n ${account} ]] || fail "add requires an account name"
  [[ $# -le 2 ]] || fail "too many arguments for add"
  validate_account_name "${account}"
  case "${login_option}" in
    "" | --device-auth) ;;
    *) fail "add accepts only --device-auth after the account name" ;;
  esac

  mkdir -p "${ACCOUNTS_DIR}"
  chmod 700 "${ACCOUNTS_DIR}"
  account_dir="$(account_home "${account}")"
  mkdir -p "${account_dir}"
  chmod 700 "${account_dir}"
  prepare_account_home "${account}" >/dev/null

  if ! run_account_codex "${account}" login status >/dev/null 2>&1; then
    if [[ ${login_option} == --device-auth ]]; then
      run_account_codex "${account}" login --device-auth
    else
      run_account_codex "${account}" login
    fi
  fi
  run_account_codex "${account}" login status >/dev/null 2>&1 \
    || fail "ChatGPT login did not complete for account: ${account}"

  if [[ ! -f ${ACTIVE_ACCOUNT_FILE} ]]; then
    save_active_account "${account}"
  fi
  printf 'Named Codex account ready: %s\n' "${account}"
}

add_automatic_account() {
  local login_option="${1:-}"
  local pending_account
  local pending_home
  local email
  local target_home
  local stamp

  [[ $# -le 1 ]] || fail "add-auto accepts only --device-auth"
  case "${login_option}" in
    "" | --device-auth) ;;
    *) fail "add-auto accepts only --device-auth" ;;
  esac

  mkdir -p "${ACCOUNTS_DIR}"
  chmod 700 "${ACCOUNTS_DIR}"
  printf -v stamp '%(%Y%m%d_%H%M%S)T' -1
  pending_account="pending-login-${stamp}-$$-${RANDOM}"
  pending_home="$(account_home "${pending_account}")"
  mkdir "${pending_home}"
  chmod 700 "${pending_home}"
  AUTO_PENDING_DIR="${pending_home}"
  trap archive_pending_account EXIT
  prepare_account_home "${pending_account}" >/dev/null

  if ! run_account_codex "${pending_account}" login status >/dev/null 2>&1; then
    if [[ ${login_option} == --device-auth ]]; then
      run_account_codex "${pending_account}" login --device-auth \
        || fail "ChatGPT device login did not complete"
    else
      run_account_codex "${pending_account}" login \
        || fail "ChatGPT login did not complete"
    fi
  fi
  run_account_codex "${pending_account}" login status >/dev/null 2>&1 \
    || fail "ChatGPT login did not complete"

  email="$(account_email_from_auth "${pending_home}/auth.json" || true)"
  [[ -n ${email} && ${email} == *@* ]] \
    || fail "the authenticated account did not provide an email address"
  validate_account_name "${email}"
  target_home="$(account_home "${email}")"
  [[ ! -e ${target_home} && ! -L ${target_home} ]] \
    || fail "an account named ${email} already exists; existing credentials were not changed"

  mv -- "${pending_home}" "${target_home}"
  chmod 700 "${target_home}"
  AUTO_PENDING_DIR=""
  trap - EXIT
  if [[ ! -f ${ACTIVE_ACCOUNT_FILE} ]]; then
    save_active_account "${email}"
  fi
  printf 'Named Codex account ready: %s\n' "${email}"
}

use_named_account() {
  local account="${1:-}"
  [[ -n ${account} ]] || fail "use requires an account name"
  [[ $# -eq 1 ]] || fail "too many arguments for use"
  prepare_account_home "${account}" >/dev/null
  run_account_codex "${account}" login status >/dev/null 2>&1 \
    || fail "named account is not logged in: ${account}"
  save_active_account "${account}"
  printf 'Future Codex processes will use account: %s\n' "${account}"
}

remove_named_account() {
  local account="${1:-}"
  local confirmation="${2:-}"
  local account_dir
  local archive_dir
  local selected=""
  local stamp

  [[ -n ${account} ]] || fail "remove requires an account name"
  validate_account_name "${account}"
  [[ ${confirmation} == --yes ]] || fail "pass --yes to confirm account removal"
  [[ $# -eq 2 ]] || fail "remove accepts only NAME --yes"

  account_dir="$(account_home "${account}")"
  [[ -d ${account_dir} && ! -L ${account_dir} ]] \
    || fail "named account does not exist or is not a managed directory: ${account}"
  if [[ -f ${ACTIVE_ACCOUNT_FILE} ]]; then
    selected="$(active_account)"
  fi

  mkdir -p "${DELETED_ACCOUNTS_DIR}"
  chmod 700 "${DELETED_ACCOUNTS_DIR}"
  printf -v stamp '%(%Y%m%d_%H%M%S)T' -1
  archive_dir="${DELETED_ACCOUNTS_DIR}/${account}.${stamp}.$$"
  [[ ! -e ${archive_dir} && ! -L ${archive_dir} ]] \
    || fail "account archive target already exists: ${archive_dir}"
  mv -- "${account_dir}" "${archive_dir}"
  chmod 700 "${archive_dir}"
  if [[ ${selected} == "${account}" ]]; then
    clear_active_account
  fi

  printf 'Named Codex account archived: %s\n' "${account}"
  printf 'Recovery path: %s\n' "${archive_dir}"
  printf 'Shared Codex conversations were not deleted.\n'
}

list_accounts() {
  local account_dir
  local account
  local legacy_email
  local legacy_label="unnamed"
  local provider
  local selected=""
  local found=0
  if [[ -f ${ACTIVE_ACCOUNT_FILE} ]]; then
    selected="$(active_account)"
  fi
  if legacy_chatgpt_login_available; then
    legacy_email="$(account_email_from_auth "${CODEX_DIR}/auth.json" || true)"
    if [[ -n ${legacy_email} ]]; then
      legacy_label="${legacy_email} (unnamed)"
    fi
    provider="$(toml_top_value "${CODEX_CONFIG}" model_provider)"
    if [[ -z ${selected} && ${provider} == openai ]]; then
      printf '* %s\n' "${legacy_label}"
    else
      printf '  %s\n' "${legacy_label}"
    fi
    found=1
  fi
  for account_dir in "${ACCOUNTS_DIR}"/*; do
    [[ -d ${account_dir} ]] || continue
    account="$(basename "${account_dir}")"
    [[ ${account} == pending-login-* ]] && continue
    validate_account_name "${account}"
    if [[ ${account} == "${selected}" ]]; then
      printf '* %s\n' "${account}"
    else
      printf '  %s\n' "${account}"
    fi
    found=1
  done
  [[ ${found} == 1 ]] || printf 'No Codex accounts.\n'
}

run_codex() {
  local account=""
  require_codex
  if [[ ${1:-} == --account ]]; then
    [[ $# -ge 2 ]] || fail "--account requires a name"
    account="$2"
    validate_account_name "${account}"
    shift 2
  elif [[ -f ${ACTIVE_ACCOUNT_FILE} ]]; then
    account="$(active_account)"
  fi
  [[ ${1:-} == -- ]] && shift

  if [[ -z ${account} ]]; then
    exec env CODEX_HOME="${CODEX_DIR}" "${CODEX_BIN}" "$@"
  fi
  local account_dir
  account_dir="$(prepare_account_home "${account}")"
  exec env CODEX_HOME="${account_dir}" CODEX_SQLITE_HOME="${CODEX_DIR}" "${CODEX_BIN}" \
    "${ACCOUNT_CONFIG_OVERRIDES[@]}" "$@"
}

toml_top_value() {
  local file="$1"
  local key="$2"
  awk -v wanted="${key}" '
    /^[[:space:]]*\[/ { exit }
    $0 ~ "^[[:space:]]*" wanted "[[:space:]]*=" {
      line = $0
      sub(/^[^"]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' "${file}"
}

has_provider() {
  local file="$1"
  local provider="$2"
  awk -v target="[model_providers.${provider}]" '
    {
      line = $0
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (line == target) found = 1
    }
    END { exit !found }
  ' "${file}"
}

upsert_top_level_string() {
  local file="$1"
  local key="$2"
  local value="$3"
  local config_tmp
  config_tmp="$(mktemp "${file}.tmp.XXXXXX")"
  awk -v wanted="${key}" -v replacement="${key} = \"${value}\"" '
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
  ' "${file}" >"${config_tmp}"
  chmod 600 "${config_tmp}"
  mv "${config_tmp}" "${file}"
}

remove_top_level_key() {
  local file="$1"
  local key="$2"
  local config_tmp
  config_tmp="$(mktemp "${file}.tmp.XXXXXX")"
  awk -v wanted="${key}" '
    BEGIN { in_prefix = 1 }
    in_prefix && /^[[:space:]]*\[/ { in_prefix = 0 }
    in_prefix && $0 ~ "^[[:space:]]*" wanted "[[:space:]]*=" { next }
    { print }
  ' "${file}" >"${config_tmp}"
  chmod 600 "${config_tmp}"
  mv "${config_tmp}" "${file}"
}

save_api_provider() {
  local provider="$1"
  write_state_file "${API_PROVIDER_FILE}" "${provider}"
}

saved_api_provider() {
  local provider=""
  if [[ -f ${API_PROVIDER_FILE} ]]; then
    IFS= read -r provider <"${API_PROVIDER_FILE}" || true
  fi
  [[ ${provider} =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  printf '%s\n' "${provider}"
}

set_account_mode() {
  local login_option="${1:-}"
  local current_provider
  local login_status

  case "${login_option}" in
    "" | --device-auth) ;;
    *) fail "account accepts only --device-auth" ;;
  esac

  current_provider="$(toml_top_value "${CODEX_CONFIG}" model_provider)"
  if [[ ${current_provider} != openai ]]; then
    [[ ${current_provider} =~ ^[A-Za-z0-9._-]+$ ]] \
      || fail "current model_provider is missing or invalid"
    has_provider "${CODEX_CONFIG}" "${current_provider}" \
      || fail "current API provider is not defined: ${current_provider}"
    save_api_provider "${current_provider}"
  fi

  upsert_top_level_string "${CODEX_CONFIG}" model_provider openai
  upsert_top_level_string "${CODEX_CONFIG}" preferred_auth_method chatgpt
  upsert_top_level_string "${CODEX_CONFIG}" forced_login_method chatgpt
  if [[ -f ${CODEX_YOLO_CONFIG} ]]; then
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" model_provider openai
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" preferred_auth_method chatgpt
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" forced_login_method chatgpt
  fi

  clear_active_account

  login_status="$(codex login status 2>&1 || true)"
  if [[ ${login_status} != *ChatGPT* ]]; then
    if [[ ${login_option} == --device-auth ]]; then
      codex login --device-auth
    else
      codex login
    fi
  fi

  printf 'Codex mode: account (provider=openai)\n'
}

set_api_mode() {
  local requested_provider="${1:-}"
  local provider
  local current_provider
  current_provider="$(toml_top_value "${CODEX_CONFIG}" model_provider)"

  if [[ -n ${requested_provider} ]]; then
    [[ ${requested_provider} =~ ^[A-Za-z0-9._-]+$ ]] \
      || fail "invalid API provider: ${requested_provider}"
    provider="${requested_provider}"
  elif [[ ${current_provider} != openai ]] && has_provider "${CODEX_CONFIG}" "${current_provider}"; then
    provider="${current_provider}"
  else
    provider="$(saved_api_provider || true)"
  fi

  [[ -n ${provider} ]] || fail "no saved API provider; configure one before switching to account mode"
  has_provider "${CODEX_CONFIG}" "${provider}" \
    || fail "API provider is not defined: ${provider}"
  save_api_provider "${provider}"

  upsert_top_level_string "${CODEX_CONFIG}" model_provider "${provider}"
  upsert_top_level_string "${CODEX_CONFIG}" preferred_auth_method apikey
  remove_top_level_key "${CODEX_CONFIG}" forced_login_method
  if [[ -f ${CODEX_YOLO_CONFIG} ]]; then
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" model_provider "${provider}"
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" preferred_auth_method apikey
    remove_top_level_key "${CODEX_YOLO_CONFIG}" forced_login_method
  fi

  clear_active_account

  printf 'Codex mode: api (provider=%s)\n' "${provider}"
}

show_status() {
  local account
  local account_dir
  local provider
  if [[ -f ${ACTIVE_ACCOUNT_FILE} ]]; then
    account="$(active_account)"
    account_dir="$(account_home "${account}")"
    printf 'mode: named-account\naccount: %s\nprovider: openai\nCODEX_HOME: %s\n' \
      "${account}" "${account_dir}"
    run_account_codex "${account}" login status || true
    return
  fi
  provider="$(toml_top_value "${CODEX_CONFIG}" model_provider)"
  if [[ ${provider} == openai ]]; then
    printf 'mode: account\nprovider: openai\n'
    codex login status || true
  else
    printf 'mode: api\nprovider: %s\n' "${provider:-unknown}"
  fi
}

[[ -r ${CODEX_CONFIG} ]] || fail "Codex config is not readable: ${CODEX_CONFIG}"

case "${1:-}" in
  account)
    [[ $# -le 2 ]] || fail "too many arguments for account mode"
    set_account_mode "${2:-}"
    ;;
  api)
    [[ $# -le 2 ]] || fail "api accepts at most one provider name"
    set_api_mode "${2:-}"
    ;;
  add)
    shift
    add_named_account "$@"
    ;;
  add-auto)
    shift
    add_automatic_account "$@"
    ;;
  use)
    shift
    use_named_account "$@"
    ;;
  remove)
    shift
    remove_named_account "$@"
    ;;
  list)
    [[ $# -eq 1 ]] || fail "list does not accept additional arguments"
    list_accounts
    ;;
  run)
    shift
    run_codex "$@"
    ;;
  status)
    [[ $# -eq 1 ]] || fail "status does not accept additional arguments"
    show_status
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

readonly CODEX_DIR="${CODEX_HOME:-${HOME}/.codex}"
readonly CODEX_CONFIG="${CODEX_DIR}/config.toml"
readonly CODEX_YOLO_CONFIG="${CODEX_DIR}/yolo.config.toml"
readonly STATE_DIR="${HOME}/.config/codex"
readonly API_PROVIDER_FILE="${STATE_DIR}/api-provider"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: codex-auth account [--device-auth]
       codex-auth api
       codex-auth status

Modes:
  account  Use the built-in OpenAI provider and sign in with ChatGPT.
  api      Restore the last custom API provider.
  status   Show the active mode and provider without displaying credentials.
EOF
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
  local state_tmp
  mkdir -p "${STATE_DIR}"
  chmod 700 "${STATE_DIR}"
  state_tmp="$(mktemp "${API_PROVIDER_FILE}.tmp.XXXXXX")"
  printf '%s\n' "${provider}" >"${state_tmp}"
  chmod 600 "${state_tmp}"
  mv "${state_tmp}" "${API_PROVIDER_FILE}"
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
  local provider
  local current_provider
  current_provider="$(toml_top_value "${CODEX_CONFIG}" model_provider)"

  if [[ ${current_provider} != openai ]] && has_provider "${CODEX_CONFIG}" "${current_provider}"; then
    provider="${current_provider}"
    save_api_provider "${provider}"
  else
    provider="$(saved_api_provider || true)"
  fi

  [[ -n ${provider} ]] || fail "no saved API provider; configure one before switching to account mode"
  has_provider "${CODEX_CONFIG}" "${provider}" \
    || fail "saved API provider is no longer defined: ${provider}"

  upsert_top_level_string "${CODEX_CONFIG}" model_provider "${provider}"
  upsert_top_level_string "${CODEX_CONFIG}" preferred_auth_method apikey
  remove_top_level_key "${CODEX_CONFIG}" forced_login_method
  if [[ -f ${CODEX_YOLO_CONFIG} ]]; then
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" model_provider "${provider}"
    upsert_top_level_string "${CODEX_YOLO_CONFIG}" preferred_auth_method apikey
    remove_top_level_key "${CODEX_YOLO_CONFIG}" forced_login_method
  fi

  printf 'Codex mode: api (provider=%s)\n' "${provider}"
}

show_status() {
  local provider
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
    [[ $# -eq 1 ]] || fail "api mode does not accept additional arguments"
    set_api_mode
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

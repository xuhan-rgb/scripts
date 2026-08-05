#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export CLAUDEX_SETUP_PHASE=provider
exec bash "${SCRIPT_DIR}/setup-codex.sh"

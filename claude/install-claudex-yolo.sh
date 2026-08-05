#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export CLAUDEX_SETUP_PHASE=extensions
exec bash "${SCRIPT_DIR}/install-codex-bridge.sh"

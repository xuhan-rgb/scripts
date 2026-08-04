#!/usr/bin/env bash
set -Eeuo pipefail

readonly MEDIA_SCHEMA="org.gnome.settings-daemon.plugins.media-keys"
readonly SHORTCUT_SCHEMA="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
readonly SHORTCUT_BASE="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if (( EUID == 0 )); then
    die "Run this script as the logged-in desktop user, not with sudo."
fi

command -v apt-get >/dev/null || die "This installer supports Ubuntu/Debian only (apt-get not found)."
[[ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]] || die "No desktop session found. Run this from a terminal inside GNOME."

packages=(
    flameshot \
    gnome-settings-daemon-common \
    libglib2.0-bin \
    wl-clipboard \
    xclip \
    xdg-desktop-portal \
    xdg-desktop-portal-gnome
)

missing_packages=()
for package in "${packages[@]}"; do
    status="$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)"
    [[ "$status" == "install ok installed" ]] || missing_packages+=("$package")
done

if ((${#missing_packages[@]} == 0)); then
    printf '%s\n' '[1/3] Required packages are already installed; skipping APT.'
else
    printf '[1/3] Installing missing packages: %s\n' "${missing_packages[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
fi

command -v gsettings >/dev/null || die "gsettings was not installed successfully."
gsettings list-schemas | grep -Fx "$MEDIA_SCHEMA" >/dev/null \
    || die "GNOME keyboard settings were not found. Make sure the desktop is GNOME."

helper_path="$HOME/.local/bin/flameshot-save-path"
mkdir -p "${helper_path%/*}"

printf '%s\n' '[2/3] Installing the Alt+S screenshot helper...'
install -m 0755 /dev/stdin "$helper_path" <<'HELPER_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail

runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
exec 9>"$runtime_dir/flameshot-save-path-${UID}.lock"
flock -n 9 || exit 0

output_path="$(mktemp "/tmp/screenshot_$(date +%Y%m%d_%H%M%S)_XXXXXX.png")" || exit 1
cleanup() {
    [[ -s "$output_path" ]] || rm -f -- "$output_path"
}
trap cleanup EXIT

# --raw writes the selected area as PNG; cleanup removes an aborted capture.
if ! flameshot gui --raw >"$output_path" || [[ ! -s "$output_path" ]]; then
    exit 0
fi

# Clipboard providers stay alive after this script exits, so they must not
# inherit the capture lock or all later hotkey presses will be ignored.
flock -u 9
exec 9>&-

copied=false
if [[ -n "${WAYLAND_DISPLAY:-}" ]] && command -v wl-copy >/dev/null; then
    if printf '%s' "$output_path" | wl-copy --type text/plain; then
        copied=true
    fi
fi

if [[ "$copied" == false ]] && command -v xclip >/dev/null; then
    if printf '%s' "$output_path" | xclip -selection clipboard; then
        copied=true
    fi
fi

printf '%s\n' "$output_path"
if [[ "$copied" == false ]]; then
    printf 'Screenshot saved, but its path could not be copied.\n' >&2
    exit 1
fi
HELPER_SCRIPT

declare -a shortcut_paths=()
while IFS= read -r path; do
    shortcut_paths+=("$path")
done < <(
    gsettings get "$MEDIA_SCHEMA" custom-keybindings \
        | grep -oE "'/[^']+/'" \
        | tr -d "'" \
        || true
)

read_shortcut_value() {
    local path="$1"
    local key="$2"
    local value

    value="$(gsettings get "$SHORTCUT_SCHEMA:$path" "$key")"
    value="${value#\'}"
    value="${value%\'}"
    printf '%s' "$value"
}

path_is_registered() {
    local wanted="$1"
    local path

    for path in "${shortcut_paths[@]}"; do
        [[ "$path" == "$wanted" ]] && return 0
    done
    return 1
}

find_shortcut_path() {
    local wanted_name="$1"
    local wanted_binding="$2"
    local path

    # Reuse the binding owner so one key never launches two commands.
    for path in "${shortcut_paths[@]}"; do
        if [[ "$(read_shortcut_value "$path" binding)" == "$wanted_binding" ]]; then
            printf '%s' "$path"
            return 0
        fi
    done

    # Reuse this installer's entry if its binding was changed manually.
    for path in "${shortcut_paths[@]}"; do
        if [[ "$(read_shortcut_value "$path" name)" == "$wanted_name" ]]; then
            printf '%s' "$path"
            return 0
        fi
    done

    return 1
}

new_shortcut_path() {
    local index=0
    local candidate

    while :; do
        candidate="${SHORTCUT_BASE}custom${index}/"
        if ! path_is_registered "$candidate"; then
            printf '%s' "$candidate"
            return 0
        fi
        ((index += 1))
    done
}

CONFIGURED_PATH=""
configure_shortcut() {
    local name="$1"
    local command="$2"
    local binding="$3"
    local path

    path="$(find_shortcut_path "$name" "$binding" || true)"
    if [[ -z "$path" ]]; then
        path="$(new_shortcut_path)"
        shortcut_paths+=("$path")
    fi

    gsettings set "$SHORTCUT_SCHEMA:$path" name "$name"
    gsettings set "$SHORTCUT_SCHEMA:$path" command "$command"
    gsettings set "$SHORTCUT_SCHEMA:$path" binding "$binding"
    CONFIGURED_PATH="$path"
}

write_shortcut_list() {
    local serialized="["
    local separator=""
    local path

    for path in "${shortcut_paths[@]}"; do
        serialized+="${separator}'${path}'"
        separator=", "
    done
    serialized+="]"
    gsettings set "$MEDIA_SCHEMA" custom-keybindings "$serialized"
}

verify_shortcut() {
    local path="$1"
    local expected_command="$2"
    local expected_binding="$3"

    [[ "$(read_shortcut_value "$path" command)" == "$expected_command" ]] || return 1
    [[ "$(read_shortcut_value "$path" binding)" == "$expected_binding" ]] || return 1
    path_is_registered "$path"
}

printf '%s\n' '[3/3] Configuring GNOME shortcuts...'
configure_shortcut "Flameshot GUI" "flameshot gui" "<Alt>a"
gui_shortcut_path="$CONFIGURED_PATH"

save_command="$helper_path"
configure_shortcut "Flameshot Save Path" "$save_command" "<Alt>s"
save_shortcut_path="$CONFIGURED_PATH"

write_shortcut_list

verify_shortcut "$gui_shortcut_path" "flameshot gui" "<Alt>a" \
    || die "Alt+A shortcut verification failed."
verify_shortcut "$save_shortcut_path" "$save_command" "<Alt>s" \
    || die "Alt+S shortcut verification failed."

printf '\nInstallation complete:\n'
printf '  Alt+A  -> flameshot gui\n'
printf '  Alt+S  -> save a selected area under /tmp and copy its full path\n'
printf '\nIf the shortcuts do not work immediately, log out and back in.\n'

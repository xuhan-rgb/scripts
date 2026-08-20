#!/usr/bin/env bash
# Build and install ScholarVault without changing the existing mdview command.
set -euo pipefail

die() {
    printf 'ScholarVault installer: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == "Linux" ]] || die "only Linux is supported by this installer"
command -v apt-get >/dev/null || die "Ubuntu/Debian APT is required"

repository_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_directory="$repository_directory/scholarvault"
build_directory="$repository_directory/build/scholarvault-release"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
install_directory="$HOME/.local/lib/scholarvault"
command_path="$HOME/.local/bin/scholarvault"
desktop_path="$data_home/applications/scholarvault.desktop"
icon_root="$data_home/icons/hicolor"

[[ -f "$source_directory/CMakeLists.txt" ]] || die "ScholarVault sources are missing"

mdview_path="$(command -v mdview 2>/dev/null || true)"
mdview_target=""
if [[ -n "$mdview_path" ]]; then
    mdview_target="$(readlink -f "$mdview_path")"
fi

required_packages=(
    build-essential
    cmake
    desktop-file-utils
    git
    libx11-dev
    libsqlite3-dev
    ninja-build
    qt6-base-dev
    qt6-pdf-dev
    python3-websocket
    tmux
    x11-utils
    xterm
)
missing_packages=()
for package in "${required_packages[@]}"; do
    status="$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)"
    [[ "$status" == "installed" ]] || missing_packages+=("$package")
done

if ((${#missing_packages[@]} == 0)); then
    printf '%s\n' '[1/4] Build and runtime packages are already installed; skipping APT.'
else
    printf '[1/4] Installing missing packages: %s\n' "${missing_packages[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
fi

printf '%s\n' '[2/4] Configuring and building the C++20 / Qt 6 application...'
cmake -S "$source_directory" -B "$build_directory" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSCHOLARVAULT_BUILD_GUI=ON \
    -DSCHOLARVAULT_BUILD_TESTS=ON
cmake --build "$build_directory"
ctest --test-dir "$build_directory" --output-on-failure
[[ -x "$build_directory/scholarvault" ]] || die "Qt 6 executable was not produced"

printf '%s\n' '[3/4] Installing the independent scholarvault command...'
mkdir -p "$install_directory" "${command_path%/*}" "${desktop_path%/*}"
cmake --install "$build_directory" --prefix "$install_directory"
ln -sfn "$install_directory/bin/scholarvault" "$command_path"
install -m 0644 "$repository_directory/desktop/scholarvault.desktop" "$desktop_path"
for size in 32 64 128 256 512; do
    icon_directory="$icon_root/${size}x${size}/apps"
    mkdir -p "$icon_directory"
    install -m 0644 \
        "$repository_directory/desktop/icons/hicolor/${size}x${size}/apps/scholarvault.png" \
        "$icon_directory/scholarvault.png"
done
desktop-file-edit --set-key=Exec --set-value="$command_path" "$desktop_path"
desktop-file-edit --set-key=TryExec --set-value="$command_path" "$desktop_path"
desktop-file-validate "$desktop_path"
update-desktop-database "${desktop_path%/*}"
if command -v gtk-update-icon-cache >/dev/null; then
    gtk-update-icon-cache -f -t "$icon_root" >/dev/null 2>&1 || true
fi

printf '%s\n' '[4/4] Verifying command isolation...'
[[ "$(readlink -f "$command_path")" == "$install_directory/bin/scholarvault" ]] \
    || die "scholarvault command points to an unexpected file"
if [[ -n "$mdview_path" ]]; then
    [[ "$(command -v mdview)" == "$mdview_path" ]] || die "mdview command path changed"
    [[ "$(readlink -f "$mdview_path")" == "$mdview_target" ]] || die "mdview command target changed"
fi

printf '\nScholarVault installation complete:\n'
printf '  New command:   %s\n' "$command_path"
printf '  Desktop entry: %s\n' "$desktop_path"
if [[ -n "$mdview_path" ]]; then
    printf '  Preserved:     mdview -> %s\n' "$mdview_target"
else
    printf '%s\n' '  Preserved:     mdview was not installed and was not created or changed'
fi

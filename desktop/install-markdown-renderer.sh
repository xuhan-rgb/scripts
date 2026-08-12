#!/usr/bin/env bash
# Install the complete Markdown/LaTeX mdview desktop workflow on Ubuntu/Debian.
#
# This single installer covers:
#   1. PyQt/Markdown/Matplotlib preview dependencies and X11 Chrome embedding.
#   2. Pandoc -> XeLaTeX PDF dependencies, Chinese fonts, and TikZ.
#   3. Mermaid's pinned Node package and a local Chrome/Chromium check.
#   4. The ~/.local/bin/mdview command, desktop entry, and Markdown MIME defaults.
#
# Run as the logged-in desktop user:
#   bash ~/scripts/desktop/install-markdown-renderer.sh
# sudo asks for a password only when an APT package is actually missing.

set -Eeuo pipefail
umask 022

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if ((EUID == 0)); then
    die "Run this installer as the logged-in desktop user, not with sudo."
fi

command -v apt-get >/dev/null \
    || die "This installer supports Ubuntu/Debian only (apt-get not found)."
command -v dpkg-query >/dev/null \
    || die "dpkg-query was not found."

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(dirname -- "$script_directory")"

required_files=(
    "$repository_directory/markdown_editor.py"
    "$repository_directory/desktop/markdown-renderer.desktop"
    "$repository_directory/markdown_pdf/template.tex"
    "$repository_directory/markdown_pdf/callout-boxes.lua"
    "$repository_directory/markdown_renderer_node/package.json"
    "$repository_directory/markdown_renderer_node/package-lock.json"
)
for required_file in "${required_files[@]}"; do
    [[ -f "$required_file" ]] || die "Required project file is missing: $required_file"
done
[[ -d /tmp && -w /tmp ]] \
    || die "/tmp must exist and be writable for copied preview images."

find_browser() {
    local candidate
    for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
        if command -v "$candidate" >/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

packages=(
    desktop-file-utils
    fonts-noto-cjk
    graphviz
    libx11-6
    libglib2.0-bin
    pandoc
    poppler-utils
    python3-markdown
    python3-matplotlib
    python3-pyqt5
    texlive-lang-chinese
    texlive-latex-extra
    texlive-pictures
    texlive-xetex
    x11-utils
    xdg-utils
)

if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
    packages+=(nodejs npm)
fi

browser="$(find_browser || true)"
if [[ -z "$browser" ]]; then
    for browser_package in chromium chromium-browser; do
        if apt-cache show "$browser_package" >/dev/null 2>&1; then
            packages+=("$browser_package")
            break
        fi
    done
fi

missing_packages=()
for package in "${packages[@]}"; do
    status="$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)"
    [[ "$status" == "install ok installed" ]] || missing_packages+=("$package")
done

if ((${#missing_packages[@]} == 0)); then
    printf '%s\n' '[1/5] System packages are already installed; skipping APT.'
else
    printf '[1/5] Installing missing packages: %s\n' "${missing_packages[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
fi

printf '%s\n' '[2/5] Verifying preview, PDF, and X11 engines...'
python3 - <<'PY' \
    || die "Python preview dependencies could not be imported."
import markdown
import matplotlib
import PyQt5
PY

for command in pandoc xelatex kpsewhich dot pdftocairo pdftotext pdfinfo xprop xwininfo; do
    command -v "$command" >/dev/null \
        || die "$command was not installed successfully."
done
for package in ctexart.cls standalone.cls tcolorbox.sty fvextra.sty tikz.sty; do
    kpsewhich "$package" >/dev/null \
        || die "The required TeX package $package is missing."
done

browser="$(find_browser || true)"
[[ -n "$browser" ]] \
    || die "Install Google Chrome or Chromium, then rerun this installer."

printf '%s\n' '[3/5] Installing the pinned Mermaid package when needed...'
mermaid_directory="$repository_directory/markdown_renderer_node"
if [[ -f "$mermaid_directory/node_modules/mermaid/dist/mermaid.min.js" ]]; then
    printf '%s\n' '      Mermaid is already installed; skipping npm.'
elif [[ -f "$mermaid_directory/package-lock.json" ]]; then
    npm ci --no-audit --no-fund --prefix "$mermaid_directory"
else
    die "Mermaid package-lock.json is missing; refusing an unpinned npm install."
fi
[[ -f "$mermaid_directory/node_modules/mermaid/dist/mermaid.min.js" ]] \
    || die "The pinned Mermaid browser bundle was not installed successfully."

printf '%s\n' '[4/5] Installing the mdview command and desktop entry...'
command_path="$HOME/.local/bin/mdview"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
desktop_path="$data_home/applications/markdown-renderer.desktop"
mkdir -p "${command_path%/*}" "${desktop_path%/*}"
chmod 0755 "$repository_directory/markdown_editor.py"
ln -sfn "$repository_directory/markdown_editor.py" "$command_path"
install -m 0644 "$repository_directory/desktop/markdown-renderer.desktop" "$desktop_path"
desktop-file-edit \
    --set-key=Exec \
    --set-value="$command_path %f" \
    "$desktop_path"
desktop-file-edit \
    --set-key=TryExec \
    --set-value="$command_path" \
    "$desktop_path"
desktop-file-validate "$desktop_path"
update-desktop-database "${desktop_path%/*}"

printf '%s\n' '[5/5] Registering mdview for Markdown and LaTeX documents...'
mime_types=(
    text/markdown
    text/x-markdown
    text/x-tex
    text/x-latex
    application/x-tex
)
for mime_type in "${mime_types[@]}"; do
    xdg-mime default markdown-renderer.desktop "$mime_type"
done

"$command_path" --help >/dev/null
[[ "$(readlink -f "$command_path")" == "$repository_directory/markdown_editor.py" ]] \
    || die "The mdview command link points to an unexpected file."
for mime_type in "${mime_types[@]}"; do
    [[ "$(xdg-mime query default "$mime_type")" == "markdown-renderer.desktop" ]] \
        || die "Failed to register markdown-renderer.desktop for $mime_type."
done

printf '\nInstallation complete:\n'
printf '  Command:       %s [document.md|document.tex]\n' "$command_path"
printf '  PDF pipeline:  Pandoc -> XeLaTeX (ctexart)\n'
printf '  PDF preview:   Real pages + coordinate-accurate Poppler selection\n'
printf '  Image copies:  Clipboard + /tmp/mdview-selection-*.png\n'
printf '  ChatGPT:       Embedded Chrome + automatic downloaded TeX rendering\n'
printf '  TikZ preview:  XeLaTeX -> pdftocairo (PDF export stays vector)\n'
printf '  Mermaid:       %s + %s\n' "$(command -v npm)" "$browser"
printf '  Desktop entry: %s\n' "$desktop_path"
printf '  MIME default:  %s\n' "$(xdg-mime query default text/markdown)"
printf '  TeX default:   %s\n' "$(xdg-mime query default text/x-tex)"

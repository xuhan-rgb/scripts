#!/usr/bin/env bash
set -euo pipefail

profile_root="${HOME}/.zotero/zotero"
manage_process=1

usage() {
    cat <<'EOF'
用法：fix-zotero-ime-candidate-position.sh [选项]

修复 Zotero 的 focusmanager.testmode=true，解决 Linux 中文候选窗固定在左下角的问题。

选项：
  --profile-root PATH       使用指定的 Zotero profile 根目录
  --no-process-control      不关闭或重启 Zotero（仅用于已关闭 Zotero 的高级场景和测试）
  -h, --help                显示帮助
EOF
}

while (($#)); do
    case "$1" in
        --profile-root)
            if (($# < 2)); then
                printf '%s\n' '错误：--profile-root 需要一个路径。' >&2
                exit 2
            fi
            profile_root=$2
            shift 2
            ;;
        --no-process-control)
            manage_process=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '错误：未知选项：%s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$profile_root" ]]; then
    printf '错误：找不到 Zotero profile 目录：%s\n' "$profile_root" >&2
    exit 1
fi

mapfile -d '' config_files < <(
    find "$profile_root" -mindepth 2 -maxdepth 2 -type f \
        \( -name prefs.js -o -name user.js \) -print0
)

if ((${#config_files[@]} == 0)); then
    printf '错误：%s 中没有找到 prefs.js 或 user.js。\n' "$profile_root" >&2
    exit 1
fi

true_pattern='^[[:space:]]*user_pref\([[:space:]]*"focusmanager\.testmode"[[:space:]]*,[[:space:]]*true[[:space:]]*\);[[:space:]]*$'
files_to_fix=()
for config_file in "${config_files[@]}"; do
    if grep -Eq "$true_pattern" "$config_file"; then
        files_to_fix+=("$config_file")
    fi
done

if ((${#files_to_fix[@]} == 0)); then
    printf '%s\n' '检查完成：未发现 focusmanager.testmode=true，无需修改。'
    exit 0
fi

zotero_was_running=0
if ((manage_process)); then
    mapfile -t zotero_pids < <(
        pgrep -f '(^|/)zotero-bin -app( |$)' 2>/dev/null || true
    )
    if ((${#zotero_pids[@]} > 0)); then
        zotero_was_running=1
        printf '正在正常关闭 Zotero（PID：%s）...\n' "${zotero_pids[*]}"
        kill -TERM "${zotero_pids[@]}"
        for _ in $(seq 1 50); do
            still_running=0
            for zotero_pid in "${zotero_pids[@]}"; do
                if kill -0 "$zotero_pid" 2>/dev/null; then
                    still_running=1
                    break
                fi
            done
            ((still_running == 0)) && break
            sleep 0.2
        done
        for zotero_pid in "${zotero_pids[@]}"; do
            if kill -0 "$zotero_pid" 2>/dev/null; then
                printf '错误：Zotero 进程 %s 未正常退出，未修改配置。\n' "$zotero_pid" >&2
                exit 1
            fi
        done
    fi
fi

backup_suffix="bak.focusmanager-testmode-$(date +%Y%m%d-%H%M%S)-$$"
fixed_count=0
for config_file in "${files_to_fix[@]}"; do
    backup_file="${config_file}.${backup_suffix}"
    cp -a -- "$config_file" "$backup_file"
    sed -E -i \
        's/^([[:space:]]*user_pref\([[:space:]]*"focusmanager\.testmode"[[:space:]]*,[[:space:]]*)true([[:space:]]*\);[[:space:]]*)$/\1false\2/' \
        "$config_file"
    if grep -Eq "$true_pattern" "$config_file"; then
        cp -a -- "$backup_file" "$config_file"
        printf '错误：修复 %s 失败，已从备份恢复。\n' "$config_file" >&2
        exit 1
    fi
    printf '已修复：%s\n' "$config_file"
    printf '备份：%s\n' "$backup_file"
    ((fixed_count += 1))
done

printf '完成：已修复 %d 个配置文件。\n' "$fixed_count"

if ((zotero_was_running)); then
    restarted=0
    if command -v gtk-launch >/dev/null 2>&1; then
        for desktop_id in zotero-current zotero; do
            if [[ -f "${HOME}/.local/share/applications/${desktop_id}.desktop" || \
                  -f "/usr/share/applications/${desktop_id}.desktop" ]]; then
                setsid -f gtk-launch "$desktop_id"
                printf '已通过桌面入口重新启动 Zotero：%s\n' "$desktop_id"
                restarted=1
                break
            fi
        done
    fi
    if ((restarted == 0)); then
        printf '%s\n' '修复已完成，但没有找到 Zotero 桌面入口；请手动重新启动 Zotero。'
    fi
fi

# Zotero Linux 中文候选窗固定在左下角

## 结论

如果 Zotero 可以切换中英文、可以唤起中文候选窗，但候选窗始终出现在窗口或屏幕左下角，而终端等其他程序正常，先检查 Zotero 的 Gecko 偏好：

```js
user_pref("focusmanager.testmode", true);
```

`focusmanager.testmode` 是 Gecko 自动化测试专用选项。设为 `true` 会绕过正常的操作系统焦点管理，使 Gecko 无法记录当前编辑窗口，也就无法把插入光标矩形交给输入法。候选窗只能使用左下角一类回退位置。

本次问题的最小修复是将该偏好恢复为 `false`。修复不需要重装 IBus/Fcitx5，也不需要修改 Zotero XPI。

## 已验证环境与症状

- Ubuntu 22.04、GNOME X11。
- Zotero 使用 Gecko/Firefox ESR 140 系列运行时。
- IBus 和 Fcitx5 都能输入中文，但候选窗位置都异常。
- Zotero 的 PDF 批注、普通文本框和右侧 AI 输入框均受影响。
- 终端中的中文输入正常。
- 切换输入法本身可以工作，问题仅是候选窗不跟随光标。

这些现象表明问题更可能位于 Zotero/Gecko 的焦点或光标坐标上，而不是具体拼音引擎。

## 快速检查

Zotero 配置目录中的 profile 名称不是固定值，先搜索所有配置文件：

```bash
rg -n --hidden \
  --glob 'prefs.js' \
  --glob 'user.js' \
  'focusmanager\.testmode' \
  ~/.zotero/zotero
```

异常配置示例：

```text
~/.zotero/zotero/<profile>/prefs.js:
user_pref("focusmanager.testmode", true);
```

## 一键修复

在 Ubuntu/Debian 桌面环境运行：

```bash
bash ~/scripts/desktop/fix-zotero-ime-candidate-position.sh
```

脚本会：

1. 扫描 `~/.zotero/zotero/*/prefs.js` 和 `user.js`。
2. 仅在发现 `focusmanager.testmode=true` 时正常关闭 Zotero。
3. 为每个实际修改的文件创建带时间戳的同目录备份。
4. 只把该偏好的 `true` 改为 `false`，不改其他设置。
5. 验证异常值已经消失。
6. 如果 Zotero 原本正在运行，通过现有桌面入口重新启动。

脚本可以重复运行。配置已经正常时只会输出“无需修改”，不会关闭 Zotero，也不会重复生成备份。

查看帮助：

```bash
bash ~/scripts/desktop/fix-zotero-ime-candidate-position.sh --help
```

需要检查其他 profile 根目录时可以显式指定：

```bash
bash ~/scripts/desktop/fix-zotero-ime-candidate-position.sh \
  --profile-root /path/to/zotero/profiles
```

`--no-process-control` 不会关闭或重启 Zotero，仅用于 Zotero 已经关闭的高级场景和自动化测试。不要用该参数修改正在运行的 Zotero，因为退出时 Zotero 可能覆盖 `prefs.js`。

## 最小修复

### 方法一：通过 Zotero 配置编辑器

优先在 Zotero 的高级设置/配置编辑器中搜索：

```text
focusmanager.testmode
```

将其恢复为 `false`，然后完整退出并重新启动 Zotero。

### 方法二：关闭 Zotero 后修改 prefs.js

不要在 Zotero 运行时直接修改 `prefs.js`，退出时程序可能覆盖文件。先正常关闭 Zotero 并确认进程已结束：

```bash
pgrep -a -f '^.*/zotero-bin -app' || true
```

然后把对应 profile 的设置改为：

```js
user_pref("focusmanager.testmode", false);
```

重新启动后，该行可能从 `prefs.js` 中自动消失。这是正常现象：`false` 是默认值，Gecko 不一定继续保存默认配置。再次检查时，“不存在”与“显式为 false”都表示未开启测试模式。

## 确定性诊断：Gecko IME 日志

如果配置中没有明显异常，可以临时启用 Gecko 的输入法日志。日志可能包含输入行为信息，不要在诊断期间输入密码或隐私内容。

以下命令需要按实际 Zotero 安装路径调整：

```bash
env \
  GTK_IM_MODULE=fcitx \
  XMODIFIERS=@im=fcitx \
  GDK_BACKEND=x11 \
  MOZ_ENABLE_WAYLAND=0 \
  MOZ_LOG='IMEHandler:4,sync' \
  MOZ_LOG_FILE=/tmp/zotero-ime.log \
  /path/to/zotero-bin -app /path/to/app/application.ini
```

Zotero 启动后，只在目标输入框输入一段无敏感信息的测试拼音，再退出程序。Mozilla 日志文件通常带有额外后缀：

```bash
ls -1 /tmp/zotero-ime.log*
rg -n \
  'SetCursorPosition|OnFocusWindow|mLastFocusedWindow|FAILED' \
  /tmp/zotero-ime.log*.moz_log
```

本次故障的决定性日志是：

```text
SetCursorPosition(), FAILED, due to no focused window
the caller isn't focused window, mLastFocusedWindow=0x0
```

同时可以看到 Gecko 收到了编辑器内部焦点变化，却没有有效的顶层焦点窗口：

```text
OnFocusChangeInGecko(aFocus=true)
mLastFocusedWindow=0x0
```

这意味着失败发生在 Gecko 调用输入法之前。继续更换拼音引擎或候选窗主题通常不会改变结果。

## 正常机制

Gecko 正常处理候选窗位置的流程是：

1. 记录当前获得焦点的 `nsWindow`。
2. 查询当前 selection/caret 的矩形。
3. 把设备像素换算为 GTK 坐标。
4. 调用 `gtk_im_context_set_cursor_location()` 把位置交给 IBus/Fcitx5。

如果 `mLastFocusedWindow` 为空，`SetCursorPosition()` 会提前返回，第四步不会执行。因此“候选窗在左下角”是光标位置没有被提交后的表现，不是 Fcitx5 主动选择的布局。

Mozilla 上游记录过相同日志和相同根因：

- [Mozilla Bug 1701047](https://bugzilla.mozilla.org/show_bug.cgi?id=1701047)
- [Gecko `IMContextWrapper::SetCursorPosition()`](https://searchfox.org/firefox-main/source/widget/gtk/IMContextWrapper.cpp)

## 验证修复

修复后至少验证以下项目：

1. 在 Zotero 普通文本框中输入拼音，候选窗位于插入光标附近。
2. 在 PDF 批注或笔记编辑器中重复测试。
3. 在插件提供的输入框中重复测试；候选窗不应固定在左下角。
4. 测试文本不要提交，验证完成后清空。
5. 如果启用了临时日志，随后用普通启动器重启，并确认进程没有 `MOZ_LOG` 环境变量。

Fcitx5 可以辅助确认 Zotero 输入上下文已经获得焦点：

```bash
dbus-send --session --print-reply \
  --dest=org.fcitx.Fcitx5 \
  /controller \
  org.fcitx.Fcitx.Controller1.DebugInfo \
  | rg 'program:Zotero.*focus:1'
```

修复后的预期输出包含：

```text
program:Zotero frontend:dbus ... focus:1
```

## UI 自动化注意事项

Zotero 可能同时暴露主窗口和置顶通知窗口。使用 `xdotool search --class zotero | tail -n 1` 可能误选 `WM_CLASS=Alert` 的“进度”窗口，造成焦点测试假失败。

应选择 `WM_CLASS` 中包含 `Navigator` 的主窗口：

```bash
main_window=$(
  xdotool search --onlyvisible --class zotero |
    while read -r window_id; do
      if xprop -id "$window_id" WM_CLASS 2>/dev/null | rg -q 'Navigator'; then
        printf '%s\n' "$window_id"
      fi
    done |
    head -n 1
)
```

如果一个 `Alert` 窗口持续抢焦点，应先正常关闭该通知，再进行输入验证。

## 本次排查中无效或非根因的方向

### 重装 IBus 或更换 Fcitx5

两套输入法都在 Zotero 中复现，而终端正常。输入法引擎变化没有修复 Gecko 的焦点状态，因此不是有效修复。

### 强制 XIM

将 `GTK_IM_MODULE` 改成 `xim` 后，测试中拼音预编辑不可见，候选窗也没有恢复。该方案已回滚，Zotero 应继续使用原生 GTK Fcitx5 模块：

```bash
GTK_IM_MODULE=fcitx
XMODIFIERS=@im=fcitx
```

### 调整 DPI 或文本缩放

缩放错误通常表现为候选窗相对光标有比例偏移；本次是固定落在左下角，而且 Gecko 日志明确显示没有 focused window。没有证据支持修改全局缩放。

### 修改 XPI

故障同时出现在 Zotero 自带编辑区域和插件输入框，并且 Gecko 的顶层焦点日志已经失败。XPI 不是根因，不应为了候选窗定位去修改插件。

## 排障顺序建议

以后遇到类似问题，按以下顺序处理可以减少无效重装：

1. 确认其他应用是否正常，区分系统输入法故障与单应用故障。
2. 确认 Zotero 实际加载的 GTK 输入法模块。
3. 搜索 `focusmanager.testmode`。
4. 用 `MOZ_LOG=IMEHandler:4,sync` 捕获一次最小复现。
5. 只有在日志显示焦点和光标矩形正常时，再排查 Fcitx/IBus、DPI、X11/Wayland 或桌面环境。

检查当前 Zotero 加载的模块：

```bash
zotero_pid=$(pgrep -o -f '^.*/zotero-bin -app')
rg 'im-(fcitx5|ibus|xim)|libFcitx' /proc/"$zotero_pid"/maps
```

这套顺序把“输入法是否运行”和“应用是否提供正确光标坐标”分开验证，能避免把 Gecko 焦点问题误判为输入法安装问题。

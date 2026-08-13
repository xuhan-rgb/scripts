# ScholarVault PDF-to-ChatGPT and Workspace Persistence Design

Date: 2026-08-13

## 1. Goals

This change has four user-visible goals:

1. A paper selected in the library tree always opens its matching PDF.
2. A paper can be sent as an original PDF attachment to the embedded ChatGPT session from the paper's context menu.
3. Closing and reopening ScholarVault restores the previous reading workspace.
4. A Codex session continues running after ScholarVault closes and can be reattached later.

The implementation remains a native C++20 and Qt 6 desktop application. `mdview` and its command mapping are outside this change and must remain unchanged.

## 2. User Interface

### 2.1 Main toolbar

The main toolbar contains only:

- `选择 Vault`
- `同步 Zotero`

The following toolbar actions are removed:

- `新建话题`
- `刷新目录`

### 2.2 Library context menu

The library tree uses a node-sensitive custom context menu instead of a shared static action list.

- Right-clicking a topic shows no context menu.
- Right-clicking a paper shows exactly one action: `发送 PDF 给 ChatGPT`.
- `新建论文`, `删除话题`, `添加相关代码`, and `同步当前论文` are not shown in the library context menu.

Opening the paper context menu first makes the clicked paper the current selection. The upload action receives the project represented by that exact model index; it must not infer the target from a stale reader or previous selection.

The existing PDF reader actions for downloading available arXiv LaTeX and related GitHub code are not changed by this context-menu cleanup.

### 2.3 Upload feedback

Selecting `发送 PDF 给 ChatGPT` performs this sequence:

1. Open the selected paper and verify that `<project>/paper/paper.pdf` exists and is a regular PDF file.
2. Switch the assistant panel to the ChatGPT tab.
3. Start or reconnect the ScholarVault-managed ChatGPT Chrome window if necessary.
4. Show a background task named `发送 PDF 给 ChatGPT：<paper title>`.
5. Disable the paper upload action while that upload is active.
6. Attach the PDF to the current ChatGPT conversation.
7. On success, focus the ChatGPT composer and report `PDF 已附加，可以开始提问`.

ScholarVault never presses ChatGPT's send button. The user verifies the attachment and writes the question.

Only one ChatGPT PDF upload can run at a time. Starting an upload for another paper while one is active reports that an upload is already in progress.

## 3. Paper Selection Correctness

The reader follows `QItemSelectionModel::currentChanged`, not only the mouse-specific `QTreeView::clicked` signal. This covers mouse selection, keyboard navigation, model restoration, and programmatic selection.

`MainWindow::activateIndex` resolves the project directly from the current model index, opens that project in `ProjectReader`, and updates the assistant's project path. Selecting a topic does not clear the currently open paper.

A regression test creates a temporary Vault with two PDF projects, changes the tree's current index twice, and verifies that the reader follows both project IDs. This test reproduces the reported stale-PDF behavior before the fix.

## 4. ChatGPT Upload Architecture

### 4.1 Managed Chrome session

ScholarVault continues to use its private persistent Chrome profile:

```text
~/.local/share/ScholarVault/chrome-profile
```

New ScholarVault-managed Chrome processes are launched with:

- `--remote-debugging-address=127.0.0.1`
- `--remote-debugging-port=0`
- the existing private `--user-data-dir`
- X11 and app-window arguments already used by ScholarVault

Port `0` makes Chrome choose a random local port and write it to `DevToolsActivePort` inside the private profile. The debugging endpoint is never bound to a non-loopback interface.

An already-running managed Chrome process that lacks `DevToolsActivePort` cannot acquire remote debugging dynamically. On the first PDF upload after upgrading, ScholarVault displays a confirmation dialog explaining that its managed ChatGPT window must restart. If accepted, ScholarVault terminates only the Chrome browser process whose command line contains the exact ScholarVault profile path, then relaunches the same profile with local debugging enabled. Cookies and login state remain in that profile. If declined, the upload is cancelled without affecting Chrome.

### 4.2 Upload helper

The Chrome DevTools upload adapter is an installed helper at:

```text
<install-prefix>/libexec/scholarvault-chatgpt-upload
```

It uses Python's standard HTTP library plus the Ubuntu/Debian `python3-websocket` package. The installer declares `python3-websocket` as a runtime dependency. No API key, ChatGPT credential, or browser cookie is passed to the helper.

The C++ `ChatGptUploadTask` invokes the helper through `QProcess` with the private profile path and canonical PDF path. The helper emits newline-delimited JSON progress events on standard output. C++ validates and displays those events in the existing background task panel.

The helper performs these Chrome DevTools Protocol operations:

1. Read `DevToolsActivePort` and enumerate Chrome targets.
2. Select the visible target whose URL belongs to `https://chatgpt.com/`.
3. Locate the existing `input[type=file]` element, including pierced child documents where supported.
4. If the input is not present, activate ChatGPT's attachment control by stable accessibility attributes, then retry the input lookup.
5. Use `DOM.setFileInputFiles` with the canonical current-project PDF path.
6. Wait until the input or attachment UI reports the expected filename.
7. Focus the message composer without submitting it.

There is no coordinate-based mouse fallback. If ChatGPT changes its document structure and the file input cannot be found, the task fails with a clear message instead of clicking an uncertain screen location.

### 4.3 Path and security checks

Before the helper starts, C++ verifies all of the following:

- A current project exists.
- The context-menu project ID still matches a project in the active Vault model.
- The canonical PDF path is inside the canonical project directory.
- The file exists, is regular, uses a `.pdf` suffix, and begins with a PDF signature.

The helper accepts only an absolute path supplied by the C++ process. It does not scan the user's filesystem.

## 5. Workspace State Persistence

### 5.1 Storage boundary

Workspace state is local device state and is stored through `QSettings("ScholarVault", "ScholarVault")`. It is not written into a paper directory, `project.json`, the Vault, or a Nutstore-synchronized path.

The state contains:

- Last Vault path.
- Main-window geometry, maximized state, and dock state.
- The three main splitter sizes.
- Expanded topic IDs.
- Selected project ID.
- Reader tab index.
- PDF zoom mode and custom zoom factor.
- Visible PDF page and offset within that page.
- Assistant tab index.
- Whether ChatGPT was open.
- Last active ChatGPT conversation URL.
- Whether a Codex terminal was attached and its project ID.

The state does not contain ChatGPT messages, cookies, credentials, PDF content, Codex terminal output, or API keys.

### 5.2 Save timing

Relevant UI changes schedule a single-shot state snapshot with a 400 ms debounce. A normal close forces an immediate final snapshot. At worst, an abnormal process termination loses the final 400 ms of UI-only movement; the active Codex process remains protected by `tmux`.

The state writer uses `QSettings::sync()` after a forced close snapshot. Ordinary debounced snapshots rely on the normal QSettings write path.

### 5.3 Restoration order

Restoration occurs in this order:

1. Restore the last valid Vault path and build the library model.
2. Lazily fetch and expand topics identified by their stable IDs.
3. Resolve and select the saved project ID.
4. Open that project and load its PDF.
5. Restore the reader tab.
6. After PDF layout is ready, restore zoom mode/factor, visible page, and page-relative offset.
7. Restore main-window geometry, dock state, and splitter sizes, clamped to the current screen.
8. Restore the assistant tab.
9. Reopen the saved ChatGPT conversation when ChatGPT was previously open.
10. Reattach the saved project's Codex session when Codex was previously attached.

Missing Vaults, topics, projects, PDFs, screens, or invalid stored values are skipped independently. A stale item never prevents the rest of the window from opening. When the saved project is unavailable, the library remains loaded with no paper selected.

### 5.4 PDF view state

`PdfDocumentView` exposes a small value state containing zoom mode, custom zoom factor, visible page, and page-relative vertical offset. Page-relative state is used instead of a raw scrollbar value so restoration remains stable when the window width, DPI, or fit-to-width scale changes.

Restoration waits for the document to reach a usable state and for page layout to complete. The restored page and offset are clamped to the current document.

## 6. ChatGPT Conversation Restoration

While an embedded ChatGPT window is active, the C++ application periodically reads Chrome's local `/json/list` endpoint using `QNetworkAccessManager`. It stores the matching ChatGPT target URL only when that URL changes. This does not require WebSocket traffic and avoids spawning a helper process for polling.

On restart, ScholarVault launches the ChatGPT app window with the saved `https://chatgpt.com/...` URL when it is valid. Any non-HTTPS URL or host other than `chatgpt.com` is rejected and replaced with `https://chatgpt.com/`.

During the one-time migration from an old managed Chrome process without debugging enabled, the exact active URL cannot be read automatically. The conversation remains in ChatGPT history, but the first restarted window may open the ChatGPT home page. All later managed sessions can restore the active conversation URL.

## 7. Persistent Codex Sessions

The installer adds `tmux` as a runtime dependency. Each paper uses a deterministic session name derived only from its stable project ID:

```text
scholarvault-<sanitized-project-id>
```

Starting Codex performs one of two operations:

- If the project's session exists, attach to it.
- Otherwise, create the session in the project directory and start `codex` inside it.

The embedded `xterm` is a display client for the `tmux` session. Closing ScholarVault terminates or detaches the embedded xterm but does not kill the tmux session or Codex process. Reopening ScholarVault and selecting the same paper reattaches to that session.

Switching papers detaches the current xterm and attaches or creates the selected paper's session. It does not terminate another paper's background session.

The Codex panel provides an explicit `结束后台 Codex 会话` action. It requires confirmation and runs `tmux kill-session` only for the current project's deterministic session. This is the only normal ScholarVault action that terminates a persistent Codex session.

If `tmux` is unavailable, ScholarVault reports the missing dependency and does not fall back to a non-persistent Codex launch.

## 8. Error Handling

User-facing failures are specific and leave existing sessions intact:

- Missing or invalid project PDF: no Chrome action is attempted.
- ChatGPT not logged in: the embedded page remains open and asks the user to log in before retrying.
- Managed Chrome restart declined: upload is cancelled.
- DevTools port or target unavailable: task fails and ChatGPT remains usable manually.
- ChatGPT file input not found: task reports that the web upload integration needs updating.
- Attachment rejected by ChatGPT: the rejection text or a generic upload failure is reported; no prompt is sent.
- `tmux` unavailable: Codex session is not started.
- Saved state references missing data: only that state element is skipped.

No automatic retry uploads a PDF twice. The user explicitly triggers each retry.

## 9. Verification

Automated verification includes:

1. Main-window selection test: two project selections open the matching projects.
2. Context-menu test: a topic has no actions and a paper has only `发送 PDF 给 ChatGPT`.
3. Toolbar test: `新建话题` and `刷新目录` are absent; `选择 Vault` and `同步 Zotero` remain.
4. Upload target validation tests for missing, non-PDF, outside-project, and valid PDF paths.
5. Upload task tests using a fake helper that emits success, malformed progress, and failure output.
6. Python helper tests with mocked DevTools responses for target selection, file-input discovery, attachment success, and DOM-change failure.
7. Workspace-state round-trip test using temporary QSettings storage.
8. Missing-state recovery tests for removed Vaults, projects, and screens.
9. PDF view-state tests at fit-width and custom zoom, including page-relative restoration.
10. Codex session command tests using a fake `tmux` executable for create, attach, switch, and explicit terminate flows.
11. Existing Debug and Release CTest suites.
12. Installer syntax and dependency checks, installed-binary smoke test, and verification that the `mdview` command target and script hash are unchanged.

A live ChatGPT upload test is not part of the default suite because it requires a logged-in personal session and depends on the current ChatGPT web document. It may be run only when explicitly requested, using the current selected PDF and without automatically submitting a prompt.

## 10. Acceptance Criteria

The feature is complete when all of the following hold:

- Changing the library's current paper always changes the reader to the matching project.
- The main toolbar and context menus match Section 2 exactly.
- Right-clicking a paper uploads that paper's original PDF to the embedded ChatGPT conversation and never sends a prompt.
- Upload progress and actionable failures are visible in ScholarVault.
- Restarting ScholarVault restores the last valid workspace and ChatGPT conversation.
- Closing ScholarVault does not terminate a running Codex task; reopening reattaches it.
- A user can explicitly terminate the current paper's persistent Codex session.
- State and ChatGPT URLs remain local and are not written into the Vault.
- All scoped automated tests pass in Debug and Release builds.
- `mdview` remains unchanged.

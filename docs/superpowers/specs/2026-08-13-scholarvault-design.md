# ScholarVault Desktop Rewrite Design

Date: 2026-08-13

## Goal

Replace the single-file `mdview` growth path with a responsive desktop research workspace. A user creates a real filesystem Topic, then creates a Research Project inside it from either an arXiv link or a local PDF. Each project may own related GitHub repositories, notes, annotations, source files, and AI sessions.

## Confirmed decisions

- Product name: ScholarVault.
- Framework: C++20 with Qt 6 Widgets.
- One Research Project belongs to exactly one Topic.
- A Topic and Research Project are real directories, not virtual collections.
- A project starts from one arXiv link or one PDF file.
- A local PDF project may be imported from a read-only Zotero library selection.
- A project may have multiple explicitly supplied public GitHub repository links.
- The Codex integration is a fully interactive PTY terminal rooted at the active project.
- ChatGPT preserves the existing remote-debugging Chrome integration instead of creating a second login store.
- Network, Git, archive extraction, LaTeX compilation, and indexing never run on the GUI thread.
- The existing `mdview` remains usable while ScholarVault is built alongside it.

## Product layout

The main window has four persistent regions:

1. Left sidebar: Vault selector and the `Topic -> Research Project` tree.
2. Center reader: PDF, TeX/source, Markdown notes, and project overview tabs.
3. Right assistant: ChatGPT and Codex Terminal tabs.
4. Bottom task strip: active and failed downloads, clones, extraction, and compilation jobs.

The initial reader opens the PDF fit-to-width. TeX is compiled only when its preview is first opened or when the user reloads after a source change. Switching tree nodes never recompiles an unchanged document.

## Filesystem model

The Vault root is user-selectable. Its default is the hidden XDG data path `~/.local/share/ScholarVault/Vault`. The former default `~/Documents/ScholarVault` is moved there on the next launch only when it is the recorded default and the new destination does not exist. Selecting a folder already synchronized by Nutstore makes the durable project data sync through the Nutstore desktop client. Generated preview pages, thumbnails, logs, and transient downloads stay outside the Vault in the platform cache directory.

```text
<vault>/
  vault.json
  topics/
    <topic-slug>/
      topic.json
      <nested-topic-slug>/
        topic.json
      <project-slug>/
        project.json
        paper/paper.pdf
        source/original
        source/extracted/...
        code/<owner>-<repository>/...
        notes/notes.md
        annotations/annotations.json
```

`project.json` is the durable source of truth for origin, arXiv metadata, related repositories, file paths, and job status. Paths stored in metadata are relative to the project directory so a synchronized Vault can move between machines.

Topic and project directory names are sanitized for Linux filesystems. IDs stored in `topic.json` and `project.json` remain stable when a display name changes.

## Project creation flows

### From arXiv

1. Parse and normalize modern or legacy arXiv IDs from an `abs`, `pdf`, or bare-ID input.
2. Fetch public metadata and determine the current version.
3. Create the project atomically in a temporary sibling directory.
4. Download the PDF and original `e-print` payload.
5. Detect gzip/tar, bare TeX, or PDF-only payload; reject archive paths that escape `source/extracted`.
6. Write project metadata and rename the temporary directory into place.
7. Show partial failures as retryable jobs. A PDF-only or unavailable-source paper remains a valid project.

### From PDF

1. Copy the selected PDF into an atomic temporary project directory; never move or modify the original.
2. Use PDF metadata and first-page text to suggest a title.
3. If an arXiv ID is detected, record it as a suggestion. Network enrichment is a separate user action.
4. Write metadata and rename the project into place.

### From Zotero

1. Add `From Zotero` as a third project source and default to the detected Zotero data directory.
2. While Zotero is running, read current bibliographic metadata through its loopback API; if it is unavailable, read `zotero.sqlite` in read-only mode.
3. Show a searchable, single-selection list containing title, authors, year, and PDF filename.
4. Resolve only local `file://` or Zotero `storage:` PDF attachments. Never write to the Zotero database or attachment directory.
5. Copy the selected PDF through the same atomic project creation path as a local PDF.
6. Record the Zotero parent item key, attachment key, data directory, and original attachment path in `project.json` for provenance.

The searchable dialog imports one selected paper. The global `Sync Zotero` operation scans every local Zotero PDF, mirrors the Zotero collection hierarchy as nested real Topic directories, and creates missing projects. Unfiled attachments go into `未分类`. A PDF is copied, never symlinked, so the ScholarVault project remains usable if Zotero moves or removes the source attachment. The current one-project/one-topic invariant chooses the lexicographically first collection if a future Zotero item belongs to several collections; no duplicate project is created. Annotation synchronization and writes back to Zotero remain outside this delivery.

Imported papers retain a live one-way identity link. `Sync Zotero` matches the stored item and attachment keys, refreshes Zotero title/author/year metadata, and atomically replaces a changed local PDF copy. It never writes to Zotero. If the original attachment key disappeared and the item has multiple PDF attachments, synchronization reports ambiguity instead of choosing one.

The main window raises and requests activation after its first event-loop turn. It does not stay permanently above other applications.

Topic deletion is recoverable: after confirmation, the complete real directory is moved to `<vault>/.trash/topics/`. ScholarVault does not recursively erase a topic from the UI.

### Related GitHub code

1. Validate a complete public `https://github.com/<owner>/<repo>` URL and reject embedded credentials.
2. Query or fetch the default branch, resolve its exact commit SHA, then clone into `code/` in a background process.
3. Record normalized URL, default branch, pinned commit, local relative path, and clone status.
4. Multiple repositories are allowed. Duplicate normalized URLs in one project are rejected.
5. Failed or interrupted clones leave no completed repository entry and can be retried.

## Modules

```text
scholarvault/
  CMakeLists.txt
  include/scholarvault/  public C++ domain and storage interfaces
  src/
    domain.cpp           immutable IDs and project/topic metadata
    storage.cpp          Vault paths, atomic writes, directory operations
    arxiv.cpp            ID parsing, metadata/source/PDF acquisition
    archives.cpp         safe source archive detection and extraction
    github.cpp           URL parsing and clone planning
    zotero.cpp           read-only SQLite catalog and attachment path resolution
    tasks.cpp            background job abstraction and progress events
    main.cpp             Qt application bootstrap
    ui/
      main_window.cpp    shell and dock/tab composition
      library_model.cpp  lazy Topic/Project model
      project_dialogs.cpp creation and GitHub forms
      zotero_import_dialog.cpp local API loading, search, and single selection
      reader.cpp         PDF/source/note tabs with lazy construction
      task_panel.cpp     job progress and retry controls
      terminal.cpp       PTY-backed Codex terminal
      chatgpt.cpp        isolated adapter around existing Chrome embedding
```

Domain and storage modules use the C++ standard library and do not depend on Qt Widgets. UI modules call services through small interfaces, allowing filesystem and parsing behavior to be tested without a display server.

## Responsiveness rules

- Use `QPdfDocument` with an asynchronous visible-page renderer rather than rasterizing every page to HTML. Render at device scale with at least 1.5× supersampling, cache only the visible page neighborhood, coalesce resize events, and support mouse text selection plus clipboard copy.
- Create heavyweight reader, ChatGPT, and terminal widgets lazily.
- Use a bounded `QThreadPool` for network/archive/index work and `QProcess` for Git and LaTeX.
- Stream progress events at a limited rate; never rebuild the full tree for each byte or subprocess line.
- Cache compiled TeX by source-tree fingerprint and compiler options.
- Watch active files for changes but debounce reloads; opening the same project reuses the current reader state.
- Persist tree expansion and selected project independently from document content.

## PTY terminal

Qt does not provide a terminal emulator. The MVP uses a PTY subprocess plus a terminal widget adapter. The process starts as `codex` with the active project as its working directory. Project switches do not silently move a live terminal; the user starts a new session for the new project. Closing the app first requests graceful termination, then stops the child after a timeout.

## ChatGPT

The existing remote Chrome target/window integration is moved behind a `ChatGptPanel` adapter. It is created only when the ChatGPT tab is opened. The X11 client is reparented directly and its parent is verified periodically; resize events are coalesced instead of starting a helper process for every splitter movement. If no debuggable Chrome session exists, the panel shows setup instructions and leaves the rest of ScholarVault functional.

Opening either ChatGPT or Codex is also the demand signal for arXiv source. Imports store the PDF and arXiv ID only. If source is not cached yet, the assistant launch downloads `https://arxiv.org/e-print/<id>` in the background, validates archive paths, and installs the archive plus extracted tree into the current project. A missing or failed source download falls back to PDF and does not block the assistant.

## Nutstore synchronization

The first implementation uses filesystem synchronization: the user selects a Nutstore-synchronized directory as the Vault root. ScholarVault writes portable relative paths and atomic JSON files. It detects common Nutstore conflict filenames and surfaces them; it does not silently merge project metadata.

Direct WebDAV synchronization is deferred until the local filesystem model and conflict behavior are stable. Credentials must never be stored in project files.

## Error handling

- Project creation is atomic: incomplete temporary directories are recoverable and not shown as completed projects.
- Network and subprocess errors include the failed stage and a short diagnostic.
- Source unavailability is a valid state, not a project-creation failure.
- Unsafe archive members are skipped and reported.
- Existing destination repositories are never overwritten.
- Malformed metadata is shown as a damaged project and is not rewritten automatically.

## Verification

Automated tests cover:

- arXiv URL/ID normalization, including legacy IDs and versions.
- GitHub URL normalization, credential rejection, and duplicate detection.
- Topic/project name sanitization and containment.
- Atomic creation from a local PDF.
- arXiv source handling for tar/gzip, bare TeX, PDF-only, and traversal attempts.
- Zotero attachment path resolution and read-only SQLite catalog fallback.
- Zotero-origin metadata round-trips without modifying the source attachment.
- JSON round-trips and relative-path portability.
- Project tree discovery and one-topic ownership.
- Background task result/progress delivery without Qt GUI imports in the core.

Scoped runtime checks cover building against Qt 6 and creating the main window with the offscreen Qt platform. No live UI screenshot check is part of this implementation unless explicitly requested.

## Initial delivery boundary

The first runnable delivery includes Vault selection, real Topics, arXiv/PDF/Zotero project creation, project discovery, related GitHub cloning, native PDF reading, source/file browsing, a lazy ChatGPT adapter, a PTY Codex panel, and background task progress. Rich PDF annotation editing and direct WebDAV synchronization remain follow-up phases; their durable directories and metadata fields are reserved now without speculative UI.

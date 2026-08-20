# ScholarVault Document Workspace

ScholarVault organizes papers, source documents, code repositories, annotations, and AI-assisted research as a local, filesystem-backed workspace.

## Language

**ScholarVault**:
The desktop research workspace that manages papers, TeX sources, code repositories, annotations, and AI-assisted reading.
_Avoid_: mdview, Markdown Renderer

**Workspace (Vault)**:
A filesystem root whose real child folders, source documents, code, and related assets form one ScholarVault research workspace.
_Avoid_: Library, virtual collection

**Workspace Folder**:
A real filesystem directory inside a Workspace; moving or renaming it changes the corresponding path on disk.
_Avoid_: Category, collection, virtual folder

**Topic**:
A real filesystem directory inside a Workspace that groups Research Projects about the same research question or subject. Each Research Project belongs to exactly one Topic.
_Avoid_: Virtual collection, favorite, temporary filter

**Research Project**:
A real filesystem directory created from either one arXiv link or one PDF file. It owns the paper PDF, any available source files, metadata, notes, annotations, and links to related code repositories.
_Avoid_: Paper cache, temporary download

**Related Code Repository**:
A public GitHub repository explicitly attached to a Research Project by URL. ScholarVault stores its normalized URL and pinned commit, and may clone it into the project's `code/` directory.
_Avoid_: Automatically guessed repository, unrelated GitHub link

**Codex Terminal**:
An embedded, fully interactive PTY session running the Codex CLI with the active research project as its working directory.
_Avoid_: Codex task panel, non-interactive Codex output

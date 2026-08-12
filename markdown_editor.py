#!/usr/bin/env python3
"""Native Markdown and complete XeLaTeX document previewer."""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import html
import io
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
import weakref
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import markdown
import websocket
from PyQt5.QtCore import (
    QPointF,
    QRectF,
    QSettings,
    QTimer,
    QUrl,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QTextDocument,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStyle,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)


APP_NAME = "Markdown Renderer"
CHATGPT_URL = QUrl("https://chatgpt.com/")
AUTO_OPEN_DOCUMENT_SUFFIXES = {".tex", ".latex", ".ltx", ".md", ".markdown"}
MARKDOWN_EXTENSIONS = ("extra", "sane_lists")
DEFAULT_BORDER_COLOR = "#2563eb"
DEFAULT_BACKGROUND_COLOR = "#ffffff"
DEFAULT_LINE_HEIGHT = 1.45
PDF_SUPPORT_DIRECTORY = Path(__file__).resolve().parent / "markdown_pdf"
PDF_TEMPLATE_PATH = PDF_SUPPORT_DIRECTORY / "template.tex"
PDF_FILTER_PATH = PDF_SUPPORT_DIRECTORY / "callout-boxes.lua"
PREVIEW_CSS = """
body { background-color: __BACKGROUND__; color: __TEXT__; margin: 20px 56px 48px; }
.markdown-body { max-width: 900px; margin: 0 auto; padding: 4px 0 0; color: __TEXT__; font-family: "Noto Serif CJK SC", "Noto Serif CJK JP", "DejaVu Serif", serif; font-size: 16px; line-height: __LINE_HEIGHT__; }
p { margin: 0.45em 0; }
h1, h2, h3, h4, h5, h6 { color: __TEXT__; font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans", sans-serif; font-weight: 700; line-height: 1.25; margin: 1.2em 0 0.5em; }
.report-content > p { text-indent: 2em; }
h1 { font-size: 2em; }
h2 { font-size: 1.5em; }
h3 { font-size: 1.25em; }
h4, h5, h6 { font-size: 1.05em; }
ol, ul { margin: 0.45em 0 0.7em; padding-left: 1.8em; }
li { margin: 0.2em 0; padding-left: 0.15em; }
li p { margin: 0.25em 0; }
hr { border: 0; border-top: 1px solid __BORDER__; margin: 2em 0; }
blockquote { background: __SURFACE__; color: __TEXT__; border-left: 4px solid __ACCENT__; margin: 1.2em 0; padding: 0.7em 1em; }
blockquote p { margin: 0.25em 0; }
code { background: __SURFACE__; color: __TEXT__; font-family: "DejaVu Sans Mono", monospace; padding: 0.12em 0.32em; }
pre.code-block { background-color: #111827; color: #e5e7eb; font-family: "DejaVu Sans Mono", monospace; font-size: 0.92em; line-height: 1.5; margin: 1.2em 0; padding: 1em 1.15em; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 1.25em 0; width: 100%; }
th, td { border: 1px solid __BORDER__; padding: 0.6em 0.8em; text-align: left; vertical-align: top; }
th { background: __SURFACE__; font-weight: 600; }
img { max-width: 100%; }
a { color: __ACCENT__; text-decoration: none; }
.report-cover { background: __BACKGROUND__; border: 1px solid __BORDER__; min-height: 650px; padding: 76px 54px 44px; text-align: center; }
.report-title { font-size: 2.25em; margin: 1.2em 0 0.8em; }
.report-subtitle { color: __MUTED__; font-family: "Noto Serif CJK SC", "Noto Serif CJK JP", serif; font-size: 1.16em; line-height: 1.8; margin: 0 auto; }
.cover-question { background: __SURFACE__; border: 1px solid __ACCENT__; margin: 4em auto 0; padding: 0.9em 1.4em; }
.cover-question h2 { color: __ACCENT__; font-size: 1em; margin: 0 0 0.45em; }
.cover-question p { font-family: "Noto Serif CJK SC", "Noto Serif CJK JP", serif; font-size: 1.12em; font-weight: 600; margin: 0; }
.report-preface { background: __SURFACE__; border-left: 4px solid __ACCENT__; margin: 2em 0; padding: 0.8em 1.2em; }
.math-display { margin: 0.55em 0; text-align: center; }
.math-display img { max-width: 96%; }
.math-inline { vertical-align: middle; }
.math-fallback { background: #fff7ed; border: 1px solid #fdba74; color: #9a3412; padding: 0.7em; }
.mermaid-diagram { margin: 1.4em 0; text-align: center; }
.mermaid-diagram img { max-width: 96%; }
.tikz-diagram { margin: 1.4em 0; text-align: center; }
.tikz-diagram img { max-width: 96%; }
"""

PANDOC_RAW_TEX_RE = re.compile(r"`([\s\S]*?)`\{=tex\}")
BACKSLASH_DISPLAY_RE = re.compile(r"\\\[\s*([\s\S]*?)\s*\\\]")
BACKSLASH_DISPLAY_LINE_RE = re.compile(r"(?m)^[ \t]*\\\[(.*)\\\][ \t]*$")
BACKSLASH_INLINE_RE = re.compile(r"\\\(([^\n]+?)\\\)")
DOLLAR_DISPLAY_RE = re.compile(r"\$\$\s*([\s\S]*?)\s*\$\$")
DOLLAR_INLINE_RE = re.compile(
    r"(?<!\\)(?<!\$)\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\$)"
)
FENCE_START_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
CITATION_ARTIFACT_RE = re.compile("\ue200cite\ue202[^\ue201]*\ue201")
LIST_MATH_RE = re.compile(r"(?m)^(\s*[-*+]\s+)\(([^)\n]+)\)(：)")
HTML_CODE_BLOCK_RE = re.compile(
    r'<pre><code(?: class="language-([^" ]+)")?>([\s\S]*?)</code></pre>'
)
PDF_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([^\r\n]*)$")
TIKZ_PICTURE_RE = re.compile(
    r"\\begin\s*\{tikzpicture\}[\s\S]*\\end\s*\{tikzpicture\}"
)
TIKZ_FORBIDDEN_COMMAND_RE = re.compile(
    r"\\(?:documentclass|usepackage|input|include|openin|openout|read|write|"
    r"catcode|csname)\b|\\(?:begin|end)\s*\{document\}"
)
TIKZ_LIBRARIES = (
    "arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric,"
    "shapes.multipart,matrix,chains,decorations.pathreplacing"
)


def split_front_matter(source: str) -> tuple[dict[str, str], str]:
    """Return simple YAML string metadata and the remaining Markdown body."""
    lines = source.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, source
    metadata: dict[str, str] = {}
    current_key: str | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return metadata, "".join(lines[index + 1 :]).lstrip("\r\n")
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$", lines[index].rstrip())
        if match:
            current_key = match.group(1)
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            metadata[current_key] = value
        elif current_key and lines[index][:1].isspace():
            continuation = lines[index].strip()
            if continuation:
                metadata[current_key] = (
                    f"{metadata[current_key]} {continuation}".strip()
                )
    return {}, source


def strip_front_matter(source: str) -> str:
    """Remove a leading YAML metadata block without treating it as Markdown."""
    return split_front_matter(source)[1]


def _comparable_title(value: str) -> str:
    return re.sub(r"[\s\"'“”‘’`]+", "", value).casefold()


def split_report_preview(
    source: str,
) -> tuple[dict[str, str], str, str, str]:
    """Split report metadata, cover question, preface, and main Markdown."""
    metadata, body = split_front_matter(source)
    title = metadata.get("title", "").strip()
    if not title:
        return metadata, "", "", body

    lines = body.splitlines(keepends=True)
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start == len(lines):
        return metadata, "", "", ""
    title_match = re.match(r"^#\s+(.+?)\s*$", lines[start].rstrip("\r\n"))
    if not title_match or _comparable_title(title_match.group(1)) != _comparable_title(title):
        return metadata, "", "", body

    separator = None
    for index in range(start + 1, min(len(lines), start + 80)):
        if re.match(r"^\s*-{3,}\s*$", lines[index]):
            separator = index
            break
    if separator is None:
        return metadata, "", "", "".join(lines[start + 1 :]).lstrip("\r\n")

    preamble_lines = lines[start + 1 : separator]
    main = "".join(lines[separator + 1 :]).lstrip("\r\n")
    cursor = 0
    while cursor < len(preamble_lines) and not preamble_lines[cursor].strip():
        cursor += 1
    question = ""
    if cursor < len(preamble_lines) and re.match(
        r"^##\s+核心问题\s*$", preamble_lines[cursor].strip()
    ):
        cursor += 1
        while cursor < len(preamble_lines) and not preamble_lines[cursor].strip():
            cursor += 1
        question_lines: list[str] = []
        while cursor < len(preamble_lines) and preamble_lines[cursor].strip():
            question_lines.append(preamble_lines[cursor].strip())
            cursor += 1
        question = " ".join(question_lines)
    preface = "".join(preamble_lines[cursor:]).strip()
    return metadata, question, preface, main


def prepare_report_source_for_pdf(source: str) -> str:
    """Move a recognized report preamble into PDF metadata and normalize headings."""
    metadata, question, _preface, main = split_report_preview(source)
    if metadata.get("title") and question and main:
        additions = {
            "question": question,
            "report-type": "研究笔记 / 技术报告",
            "date": date.today().isoformat(),
            "header-left": metadata["title"],
        }
        lines = source.splitlines(keepends=True)
        closing = next(
            (
                index
                for index in range(1, len(lines))
                if lines[index].strip() in {"---", "..."}
            ),
            None,
        )
        if closing is not None:
            inserted = [
                f"{key}: {json.dumps(value, ensure_ascii=False)}\n"
                for key, value in additions.items()
                if key not in metadata
            ]
            source = (
                "".join(lines[:closing])
                + "".join(inserted)
                + lines[closing]
                + "\n"
                + main.lstrip("\r\n")
            )

    return re.sub(
        r"(?m)^(#{1,6}[ \t]+)\d+(?:\.\d+)*(?:[.)])?[ \t]+",
        r"\1",
        source,
    )


def transform_outside_fences(source: str, transform) -> str:
    """Apply a text transform without changing fenced code blocks."""
    output: list[str] = []
    normal: list[str] = []
    fence_char = ""
    fence_length = 0

    def flush_normal() -> None:
        if normal:
            output.append(transform("".join(normal)))
            normal.clear()

    for line in source.splitlines(keepends=True):
        if fence_char:
            output.append(line)
            stripped = line.lstrip(" ")
            if (
                len(line) - len(stripped) <= 3
                and stripped.startswith(fence_char * fence_length)
                and not stripped[fence_length:].strip(fence_char).strip()
            ):
                fence_char = ""
                fence_length = 0
            continue

        match = FENCE_START_RE.match(line)
        if match:
            flush_normal()
            marker = match.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            output.append(line)
        else:
            normal.append(line)

    flush_normal()
    return "".join(output)


def metadata_flag(metadata: dict[str, str], *keys: str) -> bool:
    """Return whether one of the simple YAML metadata flags is enabled."""
    value = next((metadata[key] for key in keys if key in metadata), "")
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def prepare_preview_heading_source(
    source: str, metadata: dict[str, str]
) -> tuple[str, bool]:
    """Normalize explicit heading numbers when Pandoc-style numbering is enabled."""
    numbered = metadata_flag(metadata, "numbersections", "number-sections")
    if not numbered:
        return source, False
    return (
        transform_outside_fences(
            source,
            lambda chunk: re.sub(
                r"(?m)^(#{1,6}[ \t]+)\d+(?:\.\d+)*(?:[.)])?[ \t]+",
                r"\1",
                chunk,
            ),
        ),
        True,
    )


def number_html_headings(body: str) -> str:
    """Prefix rendered headings with Pandoc-like hierarchical section numbers."""
    counters = [0] * 6

    def replace_heading(match: re.Match[str]) -> str:
        level = int(match.group(1))
        counters[level - 1] += 1
        for index in range(level, len(counters)):
            counters[index] = 0
        prefix = ".".join(str(value) for value in counters[:level] if value)
        return (
            f"<h{level}{match.group(2)}>"
            + prefix
            + " "
            + match.group(3)
            + f"</h{level}>"
        )

    return re.sub(
        r"<h([1-6])([^>]*)>([\s\S]*?)</h\1>",
        replace_heading,
        body,
    )


def normalize_math_tex(tex: str) -> str:
    """Convert Pandoc raw-TeX fragments into ordinary LaTeX."""
    normalized = PANDOC_RAW_TEX_RE.sub(lambda match: match.group(1), tex)
    normalized = (
        normalized.replace(r"\_", "_")
        .replace(r"\^", "^")
        .replace(r"\*", "*")
        .replace(r"\[", "[")
        .replace(r"\]", "]")
        .strip()
    )
    return re.sub(
        r"(?<![A-Za-z{\\])(MLP|MSE|Flatten|stopgrad|vec|Future|Cost)(?![A-Za-z}])",
        lambda match: rf"\operatorname{{{match.group(1)}}}",
        normalized,
    )


def normalize_math_markup(source: str) -> str:
    """Normalize Pandoc-style display math to portable ``$$`` blocks."""

    def normalize_chunk(chunk: str) -> str:
        chunk = PANDOC_RAW_TEX_RE.sub(lambda match: match.group(1), chunk)

        def replace_display(match: re.Match[str]) -> str:
            return f"\n\n$$\n{normalize_math_tex(match.group(1))}\n$$\n\n"

        chunk = BACKSLASH_DISPLAY_LINE_RE.sub(replace_display, chunk)
        return BACKSLASH_DISPLAY_RE.sub(replace_display, chunk)

    return transform_outside_fences(source, normalize_chunk)


@lru_cache(maxsize=1)
def _math_font_config() -> dict[str, object]:
    try:
        from matplotlib import font_manager

        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            return {
                "mathtext.fontset": "custom",
                "mathtext.rm": family,
                "mathtext.bf": family,
                "mathtext.it": "DejaVu Serif:italic",
                "mathtext.cal": "STIXGeneral",
                "mathtext.sf": "DejaVu Sans",
                "mathtext.tt": "DejaVu Sans Mono",
                "mathtext.fallback": "stix",
                "font.cursive": ["DejaVu Sans"],
                "figure.facecolor": "none",
                "savefig.facecolor": "none",
                "savefig.transparent": True,
            }
    except (ImportError, RuntimeError, ValueError):
        pass
    return {}


def prepare_mathtext_expression(tex: str) -> str:
    """Adapt common LaTeX text fragments to Matplotlib's math subset."""
    expression = normalize_math_tex(tex)
    expression = re.sub(
        r"\\xrightarrow\s*\{([^{}]*)\}",
        lambda match: r"\overset{\mathrm{" + match.group(1) + r"}}{\longrightarrow}",
        expression,
    )
    expression = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", expression)
    expression = re.sub(r"\\mathbb\s+([A-Za-z])", r"\\mathbb{\1}", expression)
    expression = re.sub(
        r"\\(?:rm|mathrm)\s+([A-Za-z]+)",
        r"\\mathrm{\1}",
        expression,
    )
    expression = re.sub(
        r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])",
        r"\\frac{\1}{\2}",
        expression,
    )
    expression = expression.replace(r"\boxed{", "{")
    expression = expression.replace(r"\left\|", r"\left\Vert")
    expression = expression.replace(r"\right\|", r"\right\Vert")
    expression = re.sub(r"\\le(?![A-Za-z])", r"\\leq", expression)

    text_fragments: list[str] = []

    def stash_text(match: re.Match[str]) -> str:
        text_fragments.append(match.group(1))
        return f"\x02TEXT{len(text_fragments) - 1}\x02"

    expression = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", stash_text, expression)
    expression = re.sub(
        r"[\u3400-\u9fff]+",
        lambda match: r"\mathrm{" + match.group(0) + "}",
        expression,
    )
    for index, text in enumerate(text_fragments):
        expression = expression.replace(
            f"\x02TEXT{index}\x02",
            r"\mathrm{" + text + "}",
        )
    return re.sub(r"\s+", " ", expression).strip()


@lru_cache(maxsize=512)
def render_math_data_url(tex: str, color: str = "#111827") -> str | None:
    """Render one LaTeX expression to a PNG data URL for QTextBrowser."""
    expression = prepare_mathtext_expression(tex)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from matplotlib import rc_context
            from matplotlib.mathtext import math_to_image

            image = io.BytesIO()
            with rc_context(_math_font_config()):
                math_to_image(
                    f"${expression}$",
                    image,
                    format="png",
                    dpi=170,
                    color=color,
                )
    except (ImportError, RuntimeError, ValueError):
        return None

    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_math_blocks(source: str, text_color: str) -> str:
    def replace_display_math(match: re.Match[str]) -> str:
        tex = normalize_math_tex(match.group(1))
        data_url = render_math_data_url(tex, text_color)
        if data_url is None:
            return (
                '\n<div class="math-fallback"><code>'
                + html.escape(tex)
                + "</code></div>\n"
            )
        return (
            '\n<div class="math-display"><img src="'
            + data_url
            + '" alt="数学公式"></div>\n'
        )

    def replace_inline_math(match: re.Match[str]) -> str:
        data_url = render_math_data_url(match.group(1), text_color)
        if data_url is None:
            return "<code>" + html.escape(match.group(1)) + "</code>"
        return '<img class="math-inline" src="' + data_url + '" alt="数学公式">'

    def replace_math_in_chunk(chunk: str) -> str:
        chunk = DOLLAR_DISPLAY_RE.sub(replace_display_math, chunk)
        chunk = BACKSLASH_INLINE_RE.sub(replace_inline_math, chunk)
        return DOLLAR_INLINE_RE.sub(replace_inline_math, chunk)

    return transform_outside_fences(
        source,
        replace_math_in_chunk,
    )


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", r"\n")


MATH_SUBSCRIPT_CHARACTERS = str.maketrans(
    {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
        "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ",
        "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
    }
)


def format_compact_math_label(value: str) -> str:
    """Convert short Markdown/ASCII math labels to readable Unicode text."""
    def convert_expression(expression: str) -> str:
        expression = normalize_math_tex(expression)
        expression = expression.replace(r"\epsilon", "ε").replace(r"\Delta", "Δ")
        expression = re.sub(r"\\(?:operatorname|mathrm|text)\{([^{}]+)\}", r"\1", expression)
        expression = re.sub(
            r"\b([A-Za-z])_hat\(([^)]+)\)",
            lambda match: match.group(1) + "\N{COMBINING CIRCUMFLEX ACCENT}"
            + ("(" + match.group(2) + ")").translate(MATH_SUBSCRIPT_CHARACTERS),
            expression,
        )
        expression = re.sub(
            r"\b([A-Za-z])_\(([^)]+)\)",
            lambda match: match.group(1)
            + ("(" + match.group(2) + ")").translate(MATH_SUBSCRIPT_CHARACTERS),
            expression,
        )
        expression = re.sub(
            r"\b([A-Za-zε])_\{([^{}]+)\}",
            lambda match: match.group(1)
            + match.group(2).translate(MATH_SUBSCRIPT_CHARACTERS),
            expression,
        )
        expression = re.sub(
            r"\b([A-Za-zε])_([A-Za-z0-9+\-]+)",
            lambda match: match.group(1)
            + match.group(2).translate(MATH_SUBSCRIPT_CHARACTERS),
            expression,
        )
        return expression.replace("{", "").replace("}", "")

    value = re.sub(r"\$([^$\n]+)\$", lambda match: convert_expression(match.group(1)), value)
    value = re.sub(r"\\\(([^\n]+?)\\\)", lambda match: convert_expression(match.group(1)), value)
    return convert_expression(value)


def mermaid_script_path() -> Path | None:
    bundled = (
        Path(__file__).resolve().with_name("markdown_renderer_node")
        / "node_modules/mermaid/dist/mermaid.min.js"
    )
    if bundled.is_file():
        return bundled
    return None


def browser_executable() -> str | None:
    """Return a Chrome-compatible executable for Mermaid rendering."""
    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        executable = shutil.which(candidate)
        if executable:
            return executable
    return None


def render_mermaid_with_browser(source: str) -> str | None:
    mermaid_script = mermaid_script_path()
    chrome = browser_executable()
    if mermaid_script is None or chrome is None:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="mdview-mermaid-") as directory:
            root = Path(directory)
            input_path = root / "diagram.html"
            input_path.write_text(
                "<!doctype html><meta charset=\"utf-8\">"
                "<style>body{margin:0;background:transparent}</style>"
                '<div class="mermaid">'
                + html.escape(source)
                + "</div>"
                + f'<script src="{mermaid_script.as_uri()}"></script>'
                + "<script>mermaid.initialize({startOnLoad:true,"
                + "securityLevel:'strict',theme:'default',"
                + "flowchart:{htmlLabels:false}});</script>",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--allow-file-access-from-files",
                    "--virtual-time-budget=3000",
                    "--dump-dom",
                    input_path.as_uri(),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            if result.returncode != 0:
                return None
            rendered_html = result.stdout.decode("utf-8", errors="replace")
            start = rendered_html.find('<svg id="mermaid-')
            end = rendered_html.rfind("</svg>")
            if start < 0 or end < start:
                return None
            svg = rendered_html[start : end + len("</svg>")]
            view_box_match = re.search(r'<svg\b[^>]*\bviewBox="([^"]+)"', svg)
            if not view_box_match:
                return None
            try:
                _, _, raw_width, raw_height = map(
                    float, view_box_match.group(1).split()
                )
            except (TypeError, ValueError):
                return None
            if raw_width <= 0 or raw_height <= 0:
                return None
            scale = min(1.0, 4096 / raw_width, 8192 / raw_height)
            width = max(32, math.ceil(raw_width * scale))
            height = max(32, math.ceil(raw_height * scale))
            padding = 4
            raster_path = root / "raster.html"
            output_path = root / "diagram.png"
            raster_path.write_text(
                "<!doctype html><meta charset=\"utf-8\"><style>"
                f"html,body{{margin:0;padding:0;width:{width + 2 * padding}px;"
                f"height:{height + 2 * padding}px;overflow:hidden;"
                "background:transparent}}"
                f"svg{{display:block;margin:{padding}px;width:{width}px!important;"
                f"height:{height}px!important;max-width:none!important}}"
                "</style>"
                + svg,
                encoding="utf-8",
            )
            screenshot = subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--allow-file-access-from-files",
                    "--default-background-color=00000000",
                    "--hide-scrollbars",
                    f"--window-size={width + 2 * padding},{height + 2 * padding}",
                    f"--screenshot={output_path}",
                    raster_path.as_uri(),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            if screenshot.returncode != 0 or not output_path.is_file():
                return None
            image = output_path.read_bytes()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not image.startswith(b"\x89PNG"):
        return None
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


@lru_cache(maxsize=64)
def render_mermaid_data_url(source: str) -> str | None:
    """Render standard Mermaid, with a Graphviz flowchart fallback."""
    rendered = render_mermaid_with_browser(source)
    if rendered:
        return rendered
    if not shutil.which("dot"):
        return None
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines or not re.match(r"^(?:flowchart|graph)\s+", lines[0], re.I):
        return None
    direction_match = re.match(r"^(?:flowchart|graph)\s+(LR|RL|TB|TD|BT)", lines[0], re.I)
    rankdir = {"TD": "TB"}.get(
        direction_match.group(1).upper() if direction_match else "TB",
        direction_match.group(1).upper() if direction_match else "TB",
    )
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    node_re = re.compile(
        r'^([A-Za-z_][\w]*)\s*(?:\[\s*"?(.*?)"?\s*\]|\(\s*"?(.*?)"?\s*\)|\{\s*"?(.*?)"?\s*\})\s*$'
    )
    edge_re = re.compile(
        r"([A-Za-z_][\w]*)\s*(?:-->|---|-.->|==>)\s*(?:\|([^|]*)\|\s*)?([A-Za-z_][\w]*)"
    )
    for line in lines[1:]:
        node_match = node_re.match(line)
        if node_match:
            label = next(value for value in node_match.groups()[1:] if value is not None)
            nodes[node_match.group(1)] = re.sub(r"<br\s*/?>", "\n", label, flags=re.I).strip('"')
        for edge_match in edge_re.finditer(line):
            edges.append(
                (edge_match.group(1), edge_match.group(3), edge_match.group(2) or "")
            )
    if not edges:
        return None
    dot_lines = [
        "digraph MarkdownFlowchart {",
        f'graph [rankdir={rankdir}, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.55"];',
        'node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#2563eb", fontname="Noto Sans CJK SC", fontsize=11];',
        'edge [color="#475569", fontname="Noto Sans CJK SC", fontsize=10];',
    ]
    for node_id, label in nodes.items():
        dot_lines.append(f'"{_dot_escape(node_id)}" [label="{_dot_escape(label)}"];')
    for source_id, target_id, label in edges:
        label_attr = f' [label="{_dot_escape(label)}"]' if label else ""
        dot_lines.append(
            f'"{_dot_escape(source_id)}" -> "{_dot_escape(target_id)}"{label_attr};'
        )
    dot_lines.append("}")
    try:
        result = subprocess.run(
            ["dot", "-Tpng"],
            input="\n".join(dot_lines).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.startswith(b"\x89PNG"):
        return None
    return "data:image/png;base64," + base64.b64encode(result.stdout).decode("ascii")


def visual_tikz_source(language: str, source: str) -> str | None:
    """Return a supported TikZ picture from tikz or visual tex fences."""
    if language not in {"tikz", "tex", "latex"}:
        return None
    if not TIKZ_PICTURE_RE.search(source):
        return None
    if TIKZ_FORBIDDEN_COMMAND_RE.search(source):
        return None
    return source.strip()


@lru_cache(maxsize=32)
def render_tikz_data_url(source: str) -> str | None:
    """Compile a self-contained tikzpicture to a cropped PNG for Qt preview."""
    source = visual_tikz_source("tikz", source) or ""
    xelatex = shutil.which("xelatex")
    pdftocairo = shutil.which("pdftocairo")
    if not source or not xelatex or not pdftocairo:
        return None
    document_source = rf"""\documentclass[border=6pt]{{standalone}}
\usepackage[UTF8,fontset=none]{{ctex}}
\usepackage{{amsmath,amssymb}}
\usepackage{{xcolor}}
\usepackage{{tikz}}
\usetikzlibrary{{{TIKZ_LIBRARIES}}}
\setmainfont{{DejaVu Serif}}
\setsansfont{{Noto Sans CJK SC}}
\setCJKmainfont{{Noto Serif CJK SC}}
\setCJKsansfont{{Noto Sans CJK SC}}
\pagestyle{{empty}}
\begin{{document}}
{source}
\end{{document}}
"""
    try:
        with tempfile.TemporaryDirectory(prefix="mdview-tikz-") as directory:
            root = Path(directory)
            tex_path = root / "diagram.tex"
            pdf_path = root / "diagram.pdf"
            output_prefix = root / "diagram-preview"
            output_path = root / "diagram-preview.png"
            tex_path.write_text(document_source, encoding="utf-8")
            compiled = subprocess.run(
                [
                    xelatex,
                    "-no-shell-escape",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={root}",
                    str(tex_path),
                ],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            if compiled.returncode != 0 or not pdf_path.is_file():
                return None
            rasterized = subprocess.run(
                [
                    pdftocairo,
                    "-png",
                    "-singlefile",
                    "-r",
                    "160",
                    "-transp",
                    str(pdf_path),
                    str(output_prefix),
                ],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=False,
            )
            if rasterized.returncode != 0 or not output_path.is_file():
                return None
            image = output_path.read_bytes()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not image.startswith(b"\x89PNG"):
        return None
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def ascii_flow_to_mermaid(source: str) -> str | None:
    """Convert recognizable ASCII flow diagrams into Mermaid flowcharts."""
    if not re.search(r"(?:\||\bv\b|↓|-->|----|├|└)", source):
        return None

    def diagram(lines: list[str]) -> str:
        diagram_source = "flowchart TD\n" + "\n".join(lines)
        return re.sub(
            r'(\[\")([^\"]*)(\"\])',
            lambda match: match.group(1)
            + format_compact_math_label(match.group(2))
            + match.group(3),
            diagram_source,
        )

    if "Action-aware latent" in source and "Latent World Model" in source:
        return diagram(
            [
                'I0["当前帧 I_t"] --> E0["Encoder Eθ"] --> V0["V_t"]',
                'V0 --> P["Planner Pφ(V_t,G_t,S_t)"] --> W["W_t"]',
                'V0 --> A["Action-aware latent<br/>MLP([V_t, vec(W_t)])"]',
                'W --> A --> F["Latent World Model Fψ"] --> VP["Predicted latent V_hat(t+1)"]',
                'I1["真实未来帧 I_(t+1)"] --> E1["Encoder Eθ"] --> SG["stopgrad(V_(t+1))"]',
                'VP --> L["MSE loss"]',
                'SG --> L',
            ]
        )
    if "Candidate trajectories" in source and "Select best trajectory" in source:
        return diagram(
            [
                'S["Current state"] --> C["Candidate trajectories"]',
                'C --> W1["W1"] --> M1["World Model"] --> F1["Future"] --> K1["Cost"]',
                'C --> W2["W2"] --> M2["World Model"] --> F2["Future"] --> K2["Cost"]',
                'C --> W3["W3"] --> M3["World Model"] --> F3["Future"] --> K3["Cost"]',
                'K1 --> B["Select best trajectory"]',
                'K2 --> B',
                'K3 --> B',
            ]
        )
    if "Sparse Encoder" in source and "Future latent" in source:
        return diagram(
            [
                'L["LiDAR"] --> E["Sparse Encoder"] --> S["Sparse latent"]',
                'S --> A["Spatial query / voxel alignment"] --> V["V_t"]',
                'V --> P["Planner"] --> W["W_t"]',
                'V --> M["World Model"] --> F["Future latent"]',
            ]
        )

    connector_re = re.compile(r"^[\s|vV↓/\\+\-.<>=├└─]+$")
    labels = []
    connectors = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if connector_re.fullmatch(stripped):
            connectors += 1
            continue
        if stripped.startswith("(") and stripped.endswith(")"):
            continue
        labels.append(stripped)
    if connectors < 2 or len(labels) < 2:
        return None
    node_lines = []
    for index, label in enumerate(labels):
        safe_label = html.escape(format_compact_math_label(label), quote=True).replace(
            "\n", "<br/>"
        )
        node_lines.append(f'N{index}["{safe_label}"]')
        if index:
            node_lines.append(f"N{index - 1} --> N{index}")
    return diagram(node_lines)


def render_code_blocks(body: str) -> str:
    def replace_code(match: re.Match[str]) -> str:
        language = (match.group(1) or "").lower()
        escaped_code = match.group(2)
        code = html.unescape(escaped_code)
        tikz_source = visual_tikz_source(language, code)
        if tikz_source:
            data_url = render_tikz_data_url(tikz_source)
            if data_url:
                return (
                    '<div class="tikz-diagram"><img src="'
                    + data_url
                    + '" alt="TikZ 流程图"></div>'
                )
        mermaid_source = code if language == "mermaid" else None
        if language in {"text", "txt"}:
            mermaid_source = ascii_flow_to_mermaid(code)
        if mermaid_source:
            data_url = render_mermaid_data_url(mermaid_source)
            if data_url:
                return (
                    '<div class="mermaid-diagram"><img src="'
                    + data_url
                    + '" alt="流程图"></div>'
                )
        language_class = f" language-{html.escape(language)}" if language else ""
        return f'<pre class="code-block{language_class}">{escaped_code}</pre>'

    return HTML_CODE_BLOCK_RE.sub(replace_code, body)


def text_color_for_background(background_color: str) -> str:
    color = QColor(background_color)
    if not color.isValid():
        color = QColor(DEFAULT_BACKGROUND_COLOR)
    return "#f8fafc" if color.lightness() < 128 else "#1f2937"


def normalized_line_height(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LINE_HEIGHT
    return min(2.2, max(1.1, parsed))


def extract_toc_entries(source: str, max_depth: int = 3) -> list[tuple[int, str, str]]:
    """Extract heading level, visible title, and preview anchor for the left TOC."""
    metadata, _question, _preface, body_source = split_report_preview(source)
    body_source, numbered = prepare_preview_heading_source(body_source, metadata)
    renderer = markdown.Markdown(
        extensions=[*MARKDOWN_EXTENSIONS, "toc"],
        output_format="html5",
    )
    renderer.convert(normalize_math_markup(body_source))
    entries: list[tuple[int, str, str]] = []
    counters = [0] * 6

    def append_tokens(tokens: list[dict[str, object]]) -> None:
        for token in tokens:
            level = int(token["level"])
            counters[level - 1] += 1
            for index in range(level, len(counters)):
                counters[index] = 0
            if level <= max_depth:
                title = re.sub(r"<[^>]+>", "", str(token["name"]))
                if numbered:
                    prefix = ".".join(
                        str(value) for value in counters[:level] if value
                    )
                    title = f"{prefix} {title}"
                entries.append(
                    (
                        level,
                        format_compact_math_label(html.unescape(title)),
                        str(token["id"]),
                    )
                )
            append_tokens(token.get("children", []))

    append_tokens(getattr(renderer, "toc_tokens", []))
    return entries


def render_markdown(
    source: str,
    background_color: str = DEFAULT_BACKGROUND_COLOR,
    line_height: float = DEFAULT_LINE_HEIGHT,
) -> str:
    """Render Markdown source to an HTML document suitable for QTextBrowser."""
    background = QColor(background_color)
    if not background.isValid():
        background = QColor(DEFAULT_BACKGROUND_COLOR)
    background_name = background.name()
    text_color = text_color_for_background(background_name)
    dark_background = background.lightness() < 128
    surface_color = "#1f2937" if dark_background else "#f7f7f8"
    border_color = "#374151" if dark_background else "#e5e7eb"
    muted_color = "#cbd5e1" if dark_background else "#5f6368"
    accent_color = "#60a5fa" if dark_background else "#2563eb"
    metadata, question, preface_source, body_source = split_report_preview(source)
    body_source, numbered_sections = prepare_preview_heading_source(
        body_source, metadata
    )

    def prepare_fragment(fragment: str) -> str:
        prepared_fragment = normalize_math_markup(fragment)
        prepared_fragment = transform_outside_fences(
            prepared_fragment,
            lambda chunk: CITATION_ARTIFACT_RE.sub("", chunk),
        )
        prepared_fragment = transform_outside_fences(
            prepared_fragment,
            lambda chunk: LIST_MATH_RE.sub(
                lambda match: (
                    match.group(1)
                    + r"\("
                    + normalize_math_tex(match.group(2))
                    + r"\)"
                    + match.group(3)
                ),
                chunk,
            ),
        )
        return render_math_blocks(prepared_fragment, text_color)

    prepared = prepare_fragment(body_source)
    extensions = [*MARKDOWN_EXTENSIONS, "toc"]
    renderer = markdown.Markdown(extensions=extensions, output_format="html5")
    body = renderer.convert(prepared)
    if numbered_sections:
        body = number_html_headings(body)
    body = render_code_blocks(body)
    parts: list[str] = []
    title = metadata.get("title", "").strip()
    if title:
        cover_parts = [
            '<section class="report-cover">',
            f'<h1 class="report-title">{html.escape(title)}</h1>',
        ]
        subtitle = metadata.get("subtitle", "").strip()
        if subtitle:
            cover_parts.append(
                f'<p class="report-subtitle">{html.escape(subtitle)}</p>'
            )
        if question:
            cover_parts.extend(
                [
                    '<div class="cover-question"><h2>核心问题</h2>',
                    f"<p>{html.escape(question)}</p></div>",
                ]
            )
        cover_parts.append("</section>")
        parts.append("".join(cover_parts))

    if preface_source:
        preface = markdown.markdown(
            prepare_fragment(preface_source),
            extensions=list(MARKDOWN_EXTENSIONS),
            output_format="html5",
        )
        parts.append(f'<section class="report-preface">{preface}</section>')
    parts.append(f'<article class="report-content">{body}</article>')

    css = (
        PREVIEW_CSS.replace("__BACKGROUND__", background_name)
        .replace("__TEXT__", text_color)
        .replace("__SURFACE__", surface_color)
        .replace("__BORDER__", border_color)
        .replace("__MUTED__", muted_color)
        .replace("__ACCENT__", accent_color)
        .replace("__LINE_HEIGHT__", f"{normalized_line_height(line_height):.2f}")
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f'<style>{css}</style></head><body><main class="markdown-body">'
        + "".join(parts)
        + "</main></body></html>"
    )


def prepare_markdown_for_pdf(source: str, assets_directory: Path) -> str:
    """Normalize Markdown and replace supported diagrams with local PNG files."""
    assets_directory.mkdir(parents=True, exist_ok=True)
    source = prepare_report_source_for_pdf(source)
    prepared = normalize_math_markup(source)
    prepared = transform_outside_fences(
        prepared,
        lambda chunk: CITATION_ARTIFACT_RE.sub("", chunk),
    )
    prepared = transform_outside_fences(
        prepared,
        lambda chunk: LIST_MATH_RE.sub(
            lambda match: (
                match.group(1)
                + "$"
                + normalize_math_tex(match.group(2))
                + "$"
                + match.group(3)
            ),
            chunk,
        ),
    )
    lines = prepared.splitlines(keepends=True)
    output: list[str] = []
    diagram_number = 0
    index = 0

    while index < len(lines):
        opening = PDF_FENCE_OPEN_RE.match(lines[index].rstrip("\r\n"))
        if not opening:
            output.append(lines[index])
            index += 1
            continue

        marker = opening.group(1)
        info = opening.group(2).strip()
        language = info.split(None, 1)[0].strip("{}.").lower() if info else ""
        closing = re.compile(
            rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*$"
        )
        end = index + 1
        while end < len(lines) and not closing.match(lines[end].rstrip("\r\n")):
            end += 1
        if end == len(lines):
            output.extend(lines[index:])
            break

        code = "".join(lines[index + 1 : end]).rstrip("\r\n")
        tikz_source = visual_tikz_source(language, code)
        if tikz_source:
            output.append(
                "\n\n\\begin{center}\n"
                + tikz_source
                + "\n\\end{center}\n\n"
            )
            index = end + 1
            continue
        mermaid_source = code if language == "mermaid" else None
        if language in {"text", "txt"}:
            mermaid_source = ascii_flow_to_mermaid(code)

        data_url = (
            render_mermaid_data_url(mermaid_source) if mermaid_source else None
        )
        if not data_url:
            output.extend(lines[index : end + 1])
            index = end + 1
            continue

        diagram_number += 1
        asset_name = f"diagram-{diagram_number:03d}.png"
        encoded = data_url.split(",", 1)[1]
        asset_path = (assets_directory / asset_name).resolve()
        asset_path.write_bytes(base64.b64decode(encoded))
        output.append(f"\n\n![]({asset_path.as_posix()})\n\n")
        index = end + 1

    return "".join(output)


def latex_pdf_engine() -> str | None:
    """Return XeLaTeX only when the required PDF toolchain is available."""
    if shutil.which("pandoc") and shutil.which("xelatex"):
        return "xelatex"
    return None


def pandoc_pdf_command(
    output_path: Path,
    base_directory: Path,
    engine: str,
    assets_directory: Path | None = None,
    include_toc: bool = False,
) -> list[str]:
    resource_paths = [str(base_directory)]
    if assets_directory is not None:
        resource_paths.append(str(assets_directory))
    command = [
        "pandoc",
        "--from=markdown+tex_math_dollars",
        "--standalone",
        f"--pdf-engine={engine}",
        f"--template={PDF_TEMPLATE_PATH}",
        f"--lua-filter={PDF_FILTER_PATH}",
        "--toc-depth=3",
        "--no-highlight",
        f"--resource-path={os.pathsep.join(resource_paths)}",
        f"--output={output_path}",
    ]
    command.insert(7, "--toc" if include_toc else "--metadata=toc=false")
    return command


def open_local_file(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.expanduser().resolve())))


def export_pdf(
    source: str,
    output_path: Path,
    *,
    base_directory: Path | None = None,
    include_toc: bool = False,
) -> str:
    """Export Markdown through Pandoc, the project template, and XeLaTeX."""
    output_path = output_path.expanduser().resolve()
    base_directory = (base_directory or Path.cwd()).expanduser().resolve()
    engine = latex_pdf_engine()
    if engine is None:
        missing = []
        if not shutil.which("pandoc"):
            missing.append("Pandoc")
        if not shutil.which("xelatex"):
            missing.append("XeLaTeX")
        raise RuntimeError(
            "PDF 导出需要 Pandoc → XeLaTeX 工具链；当前缺少："
            + "、".join(missing)
            + "。\n可在 Ubuntu 执行：sudo apt install pandoc texlive-xetex "
            "texlive-lang-chinese texlive-latex-extra texlive-pictures fonts-noto-cjk"
        )
    if not PDF_TEMPLATE_PATH.is_file() or not PDF_FILTER_PATH.is_file():
        raise RuntimeError("PDF 导出模板不完整，请重新安装 mdview。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mdview-pdf-") as temporary:
        assets_directory = Path(temporary) / "assets"
        prepared = prepare_markdown_for_pdf(source, assets_directory)
        command = pandoc_pdf_command(
            output_path,
            base_directory,
            engine,
            assets_directory,
            include_toc,
        )
        result = subprocess.run(
            command,
            input=prepared,
            text=True,
            cwd=base_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "未知错误").strip()
            if len(details) > 4000:
                details = details[-4000:]
            raise RuntimeError(f"Pandoc/XeLaTeX 导出失败：\n{details}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("PDF 文件未能生成")
    return "Pandoc + XeLaTeX"


LATEX_TOC_RE = re.compile(r"\\tableofcontents\b")


def _split_latex_comment(line: str) -> tuple[str, str]:
    """Split one LaTeX line at its first unescaped percent sign."""
    for index, character in enumerate(line):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return line[:index], line[index:]
    return line, ""


def latex_has_toc(source: str) -> bool:
    """Return whether a complete LaTeX source enables a table of contents."""
    return any(
        LATEX_TOC_RE.search(_split_latex_comment(line)[0])
        for line in source.splitlines()
    )


def configure_latex_toc(source: str, include_toc: bool) -> str:
    """Override the table of contents in an in-memory LaTeX compilation copy."""
    if include_toc:
        if latex_has_toc(source):
            return source
        begin_document = re.compile(r"\\begin\s*\{document\}")
        if not begin_document.search(source):
            return source
        return begin_document.sub(
            lambda match: match.group(0) + "\n\\tableofcontents\n\\clearpage",
            source,
            count=1,
        )

    output: list[str] = []
    for line in source.splitlines(keepends=True):
        code, comment = _split_latex_comment(line)
        output.append(LATEX_TOC_RE.sub("", code) + comment)
    return "".join(output)


def _read_braced_group(value: str, offset: int) -> tuple[str, int] | None:
    while offset < len(value) and value[offset].isspace():
        offset += 1
    if offset >= len(value) or value[offset] != "{":
        return None
    depth = 0
    start = offset + 1
    for index in range(offset, len(value)):
        if value[index] == "{" and (index == 0 or value[index - 1] != "\\"):
            depth += 1
        elif value[index] == "}" and (index == 0 or value[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return value[start:index], index + 1
    return None


def extract_latex_toc_entries(
    toc_source: str,
    destination_targets: dict[str, int | tuple[int, float]] | None = None,
) -> list[tuple[int, str, str, float]]:
    """Read titles, physical pages, and page-relative positions from a .toc file."""
    destination_targets = destination_targets or {}
    level_map = {"section": 1, "subsection": 2, "subsubsection": 3}
    entries: list[tuple[int, str, str, float]] = []
    for line in toc_source.splitlines():
        marker = line.find(r"\contentsline")
        if marker < 0:
            continue
        offset = marker + len(r"\contentsline")
        groups: list[str] = []
        for _ in range(4):
            parsed = _read_braced_group(line, offset)
            if parsed is None:
                break
            group, offset = parsed
            groups.append(group)
        if len(groups) < 3 or groups[0].strip() not in level_map:
            continue
        kind, raw_title, logical_page = groups[:3]
        destination = groups[3].strip() if len(groups) >= 4 else ""
        number = ""
        title = raw_title.strip()
        if title.startswith(r"\numberline"):
            parsed_number = _read_braced_group(title, len(r"\numberline"))
            if parsed_number is not None:
                number, title_offset = parsed_number
                title = title[title_offset:].strip()
        title = title.replace(r"\protect", "").replace("~", " ")
        title = format_compact_math_label(title)
        title = re.sub(r"\\(?:textbf|textit|emph|mbox)\{([^{}]*)\}", r"\1", title)
        title = re.sub(r"\\[A-Za-z@]+\*?", "", title)
        title = re.sub(r"[{}]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        if number:
            title = f"{number.strip()} {title}".strip()
        try:
            fallback_page = int(logical_page.strip())
        except ValueError:
            fallback_page = 1
        target = (
            destination_targets.get(destination)
            if destination and destination != "Doc-Start"
            else None
        )
        if target is None:
            target = destination_targets.get(
                f"page.{logical_page.strip()}", (fallback_page, 0.0)
            )
        if isinstance(target, tuple):
            physical_page, vertical_ratio = target
        else:
            physical_page, vertical_ratio = target, 0.0
        entries.append(
            (
                level_map[kind.strip()],
                title,
                f"pdf-page-{physical_page}",
                max(0.0, min(1.0, float(vertical_ratio))),
            )
        )
    return entries


def pdf_destination_targets(pdf_path: Path) -> dict[str, tuple[int, float]]:
    """Map named PDF destinations to physical pages and top-relative positions."""
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return {}
    try:
        page_info = subprocess.run(
            [pdfinfo, str(pdf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        destinations_info = subprocess.run(
            [pdfinfo, "-dests", str(pdf_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if page_info.returncode != 0 or destinations_info.returncode != 0:
        return {}
    page_output = page_info.stdout.decode("utf-8", errors="replace")
    size_match = re.search(
        r"^Page size:\s*[0-9.]+\s+x\s+([0-9.]+)\s+pts",
        page_output,
        flags=re.MULTILINE,
    )
    page_height = float(size_match.group(1)) if size_match else 0.0
    destinations: dict[str, tuple[int, float]] = {}
    output = destinations_info.stdout.decode("utf-8", errors="replace")
    for line in output.splitlines():
        match = re.match(
            r'^\s*(\d+)\s+\[\s*(.*?)\s*\]\s+"([^"]+)"\s*$', line
        )
        if not match:
            continue
        physical_page = int(match.group(1))
        destination = match.group(3)
        vertical_ratio = 0.0
        coordinates = match.group(2).split()
        if len(coordinates) >= 3 and coordinates[0] == "XYZ" and page_height > 0:
            try:
                y_from_bottom = float(coordinates[2])
            except ValueError:
                pass
            else:
                vertical_ratio = (page_height - y_from_bottom) / page_height
        destinations[destination] = (
            physical_page,
            max(0.0, min(1.0, vertical_ratio)),
        )
    return destinations


def pdf_destination_pages(pdf_path: Path) -> dict[str, int]:
    """Compatibility view of named destinations containing page numbers only."""
    return {
        destination: target[0]
        for destination, target in pdf_destination_targets(pdf_path).items()
    }


def compile_latex_document(
    source: str,
    output_path: Path,
    *,
    base_directory: Path | None = None,
    toc_output_path: Path | None = None,
) -> str:
    """Compile a complete LaTeX document twice with XeLaTeX."""
    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError(
            "完整 TeX 预览需要 XeLaTeX。\n"
            "Ubuntu 可执行：sudo apt install texlive-xetex texlive-lang-chinese "
            "texlive-latex-extra texlive-pictures"
        )
    if not re.search(r"\\documentclass(?:\[[^]]*\])?\s*\{", source):
        raise RuntimeError("该 .tex 文件缺少 \\documentclass，无法作为完整文档编译。")
    if not re.search(r"\\begin\s*\{document\}", source):
        raise RuntimeError("该 .tex 文件缺少 \\begin{document}。")

    output_path = output_path.expanduser().resolve()
    base_directory = (base_directory or Path.cwd()).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    texinputs = f"{base_directory}//{os.pathsep}"
    if environment.get("TEXINPUTS"):
        texinputs += environment["TEXINPUTS"]
    environment["TEXINPUTS"] = texinputs

    with tempfile.TemporaryDirectory(prefix="mdview-latex-document-") as temporary:
        build_directory = Path(temporary)
        source_path = build_directory / "source.tex"
        compiled_path = build_directory / "mdview-document.pdf"
        source_path.write_text(source, encoding="utf-8")
        command = [
            xelatex,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-jobname=mdview-document",
            f"-output-directory={build_directory}",
            str(source_path),
        ]
        for _pass in range(2):
            result = subprocess.run(
                command,
                cwd=base_directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                details = (result.stdout or result.stderr or b"").decode(
                    "utf-8", errors="replace"
                ).strip()
                if len(details) > 5000:
                    details = details[-5000:]
                raise RuntimeError(f"XeLaTeX 编译失败：\n{details}")
        if not compiled_path.is_file() or compiled_path.stat().st_size == 0:
            raise RuntimeError("XeLaTeX 未生成 PDF。")
        shutil.copy2(compiled_path, output_path)
        compiled_toc = build_directory / "mdview-document.toc"
        if toc_output_path is not None and compiled_toc.is_file():
            toc_output_path = toc_output_path.expanduser().resolve()
            toc_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(compiled_toc, toc_output_path)
    return "XeLaTeX"


def extract_pdf_text_layout(
    pdf_path: Path,
) -> list[tuple[float, float, list[tuple[float, float, float, float, str, int]]]]:
    """Extract selectable PDF words and their exact page-space boxes."""
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise RuntimeError(
            "PDF 文字选择需要 pdftotext。\n"
            "Ubuntu 可执行：sudo apt install poppler-utils"
        )
    result = subprocess.run(
        [pdftotext, "-bbox-layout", str(pdf_path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or b"").decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(f"PDF 文字层提取失败：\n{details}")
    try:
        root = ElementTree.fromstring(result.stdout)
    except ElementTree.ParseError as error:
        raise RuntimeError(f"PDF 文字层解析失败：{error}") from error

    pages: list[
        tuple[
            float,
            float,
            list[tuple[float, float, float, float, str, int]],
        ]
    ] = []
    for page in root.findall(".//{*}page"):
        try:
            page_width = float(page.attrib["width"])
            page_height = float(page.attrib["height"])
        except (KeyError, ValueError):
            continue
        words: list[tuple[float, float, float, float, str, int]] = []
        for line_number, line in enumerate(page.findall(".//{*}line")):
            for word in line.findall("./{*}word"):
                text = "".join(word.itertext()).strip()
                if not text:
                    continue
                try:
                    x_min = float(word.attrib["xMin"])
                    x_max = float(word.attrib["xMax"])
                    y_min = float(word.attrib["yMin"])
                    y_max = float(word.attrib["yMax"])
                except (KeyError, ValueError):
                    continue
                words.append((x_min, y_min, x_max, y_max, text, line_number))
        words.sort(key=lambda value: (value[5], value[0]))
        pages.append((page_width, page_height, words))
    return pages


def render_pdf_page_image(image_path: Path) -> str:
    """Render one PDF page without asking Qt to reflow an invisible text copy."""
    image = QImage(str(image_path))
    if image.isNull():
        raise RuntimeError(f"无法读取 PDF 页面图像：{image_path}")
    image_width = image.width()
    image_height = image.height()
    image_url = html.escape(image_path.as_uri(), quote=True)
    return (
        f'<table class="pdf-page-image" width="{image_width}" '
        f'height="{image_height}" cellspacing="0" cellpadding="0" '
        f'background="{image_url}"><tbody><tr height="{image_height}">'
        "<td></td></tr></tbody></table>"
    )


def render_pdf_pages_html(
    pdf_path: Path,
    pages_directory: Path,
    background_color: str = DEFAULT_BACKGROUND_COLOR,
) -> str:
    """Rasterize real PDF pages and return a continuous in-app preview."""
    pdftocairo = shutil.which("pdftocairo")
    if not pdftocairo:
        raise RuntimeError(
            "软件内 PDF 页面预览需要 pdftocairo。\n"
            "Ubuntu 可执行：sudo apt install poppler-utils"
        )
    pdf_path = pdf_path.expanduser().resolve()
    pages_directory = pages_directory.expanduser().resolve()
    pages_directory.mkdir(parents=True, exist_ok=True)
    for old_page in pages_directory.glob("page-*.png"):
        old_page.unlink()
    output_prefix = pages_directory / "page"
    result = subprocess.run(
        [
            pdftocairo,
            "-png",
            "-r",
            "120",
            str(pdf_path),
            str(output_prefix),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or b"").decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(f"PDF 页面预览生成失败：\n{details}")

    def page_number(path: Path) -> int:
        match = re.search(r"-(\d+)\.png$", path.name)
        return int(match.group(1)) if match else 0

    pages = sorted(pages_directory.glob("page-*.png"), key=page_number)
    if not pages:
        raise RuntimeError("PDF 页面预览没有生成任何页面。")
    background = QColor(background_color)
    if not background.isValid():
        background = QColor(DEFAULT_BACKGROUND_COLOR)
    page_html = "".join(
        f'<a name="pdf-page-{index}">&#8203;</a>'
        f'<section class="pdf-page" id="pdf-page-{index}">'
        + render_pdf_page_image(path)
        + "</section>"
        for index, path in enumerate(pages, 1)
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        f"body{{margin:24px;background:{background.name()};}}"
        ".pdf-document{max-width:992px;margin:0 auto;}"
        ".pdf-page{margin:0 auto 22px;background:#fff;box-shadow:0 3px 14px rgba(15,23,42,.18);}"
        ".pdf-page img{display:block;width:100%;height:auto;}"
        ".pdf-page-image{border:0;table-layout:fixed;}"
        ".pdf-page-image td{border:0;background:transparent;}"
        "</style></head><body><main class=\"pdf-document\">"
        + page_html
        + "</main></body></html>"
    )


def save_preview_image(
    image: QImage,
    output_directory: Path = Path("/tmp"),
) -> Path:
    """Save a copied preview image under a unique, readable PNG name."""
    if image.isNull():
        raise RuntimeError("无法保存空图片。")
    output_directory = output_directory.expanduser().resolve()
    if not output_directory.is_dir():
        raise RuntimeError(f"图片保存目录不存在：{output_directory}")
    file_descriptor, filename = tempfile.mkstemp(
        prefix=f"mdview-selection-{datetime.now():%Y%m%d-%H%M%S}-",
        suffix=".png",
        dir=output_directory,
    )
    os.close(file_descriptor)
    output_path = Path(filename)
    if not image.save(str(output_path), "PNG"):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"图片保存失败：{output_path}")
    return output_path


def build_chatgpt_edit_prompt(
    source_path: Path | None,
    selected_text: str,
    rendered_location: str,
) -> str:
    source_path = source_path.expanduser().resolve() if source_path else None
    suffix = source_path.suffix.casefold() if source_path else ".md"
    is_latex = suffix in {".tex", ".latex", ".ltx"}
    document_kind = "完整 LaTeX" if is_latex else "Markdown"
    output_suffix = suffix if suffix in AUTO_OPEN_DOCUMENT_SUFFIXES else ".md"
    source_label = str(source_path) if source_path else "当前打开但尚未保存的文档"
    return f"""请修改你在本次对话中提供、且我已下载的{document_kind}源文件。

本地文件：{source_label}
渲染位置：{rendered_location}

请在源文件中定位下面的选中原文。PDF 中的公式、空格或换行可能与源代码不完全一致；遇到这种情况，请结合物理页码、相邻文字和语义定位，不要修改其他相似段落。

<<<MDVIEW_SELECTED_TEXT
{selected_text.strip()}
MDVIEW_SELECTED_TEXT

执行约束：
1. 只修改上述原文对应的位置，不要顺带重写其他章节。
2. 保留原有文档结构、公式、引用、标签、目录和排版风格，除非修改要求明确涉及它们。
3. 修改后自行检查上下文是否连贯、语法是否有效。
4. 返回可下载的完整 `{output_suffix}` 文件，不要只返回修改片段或 diff。

修改要求（由我补充）：
"""


@dataclass(frozen=True)
class RemoteChromeSession:
    executable: str
    user_data_dir: str | None
    profile_directory: str | None
    debug_port: int
    pid: int


@dataclass(frozen=True)
class ChromeDebugTarget:
    target_id: str
    url: str
    websocket_url: str


def chrome_download_directory(
    session: RemoteChromeSession,
    fallback_home: Path | None = None,
) -> Path:
    """Return the download directory configured for the remote Chrome profile."""
    fallback = (fallback_home or Path.home()).expanduser() / "Downloads"
    if not session.user_data_dir:
        return fallback.resolve()
    preferences_path = (
        Path(session.user_data_dir).expanduser()
        / (session.profile_directory or "Default")
        / "Preferences"
    )
    try:
        preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
        configured = preferences.get("download", {}).get("default_directory")
    except (OSError, json.JSONDecodeError, AttributeError):
        configured = None
    if not isinstance(configured, str) or not configured.strip():
        return fallback.resolve()
    return Path(os.path.expandvars(configured)).expanduser().resolve()


def chrome_debug_json(debug_port: int, endpoint: str):
    url = f"http://127.0.0.1:{debug_port}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise OSError(f"无法读取 Chrome 调试端点 {endpoint}：{error}") from error


def chrome_debug_targets(debug_port: int) -> list[ChromeDebugTarget]:
    targets = []
    for item in chrome_debug_json(debug_port, "/json/list"):
        target_id = item.get("id")
        websocket_url = item.get("webSocketDebuggerUrl")
        if item.get("type") != "page" or not target_id or not websocket_url:
            continue
        targets.append(
            ChromeDebugTarget(
                target_id=str(target_id),
                url=str(item.get("url", "")),
                websocket_url=str(websocket_url),
            )
        )
    return targets


def chrome_browser_websocket_url(debug_port: int) -> str:
    websocket_url = chrome_debug_json(debug_port, "/json/version").get(
        "webSocketDebuggerUrl"
    )
    if not websocket_url:
        raise OSError("Chrome 调试端点没有提供浏览器 WebSocket 地址。")
    return str(websocket_url)


def new_chatgpt_target(
    known_target_ids: set[str],
    targets: list[ChromeDebugTarget],
) -> ChromeDebugTarget | None:
    for target in targets:
        hostname = (urllib.parse.urlparse(target.url).hostname or "").casefold()
        if target.target_id in known_target_ids:
            continue
        if hostname == "chatgpt.com" or hostname.endswith(".chatgpt.com"):
            return target
    return None


class ChromeTargetDownloadCapture:
    """Capture completed downloads initiated by one Chrome page target."""

    def __init__(
        self,
        target: ChromeDebugTarget,
        browser_websocket_url: str,
        download_directory: Path,
        *,
        socket_factory=None,
    ) -> None:
        self.target_id = target.target_id
        self.download_directory = Path(download_directory).expanduser().resolve()
        self.socket_factory = socket_factory or websocket.create_connection
        self.request_id = 0
        target_socket = self.socket_factory(
            target.websocket_url,
            timeout=2,
            suppress_origin=True,
        )
        try:
            frame_tree = self.request(target_socket, "Page.getFrameTree").get(
                "frameTree", {}
            )
            self.frame_ids = self.collect_frame_ids(frame_tree)
        finally:
            target_socket.close()
        self.browser_socket = self.socket_factory(
            browser_websocket_url,
            timeout=2,
            suppress_origin=True,
        )
        self.request(
            self.browser_socket,
            "Browser.setDownloadBehavior",
            {"behavior": "default", "eventsEnabled": True},
        )
        self.browser_socket.settimeout(0)
        self.downloads: dict[str, str] = {}
        self.pending_completed: set[Path] = set()

    def request(self, connection, method: str, params=None):
        self.request_id += 1
        request_id = self.request_id
        message = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        connection.send(json.dumps(message))
        while True:
            response = json.loads(connection.recv())
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise OSError(response["error"].get("message", "Chrome 调试命令失败。"))
            return response.get("result", {})

    @classmethod
    def collect_frame_ids(cls, frame_tree) -> set[str]:
        frame_ids = set()
        frame_id = frame_tree.get("frame", {}).get("id")
        if frame_id:
            frame_ids.add(str(frame_id))
        for child in frame_tree.get("childFrames", []):
            frame_ids.update(cls.collect_frame_ids(child))
        return frame_ids

    def poll_completed_downloads(self) -> list[Path]:
        completed = self.take_existing_completed_paths()
        while True:
            try:
                raw_message = self.browser_socket.recv()
            except (
                BlockingIOError,
                TimeoutError,
                websocket.WebSocketTimeoutException,
            ):
                break
            except websocket.WebSocketConnectionClosedException:
                break
            message = json.loads(raw_message)
            method = message.get("method")
            params = message.get("params", {})
            if method == "Browser.downloadWillBegin":
                filename = str(params.get("suggestedFilename", ""))
                if (
                    params.get("frameId") in self.frame_ids
                    and Path(filename).suffix.casefold()
                    in AUTO_OPEN_DOCUMENT_SUFFIXES
                ):
                    self.downloads[str(params.get("guid", ""))] = filename
                continue
            if method != "Browser.downloadProgress":
                continue
            guid = str(params.get("guid", ""))
            filename = self.downloads.get(guid)
            state = params.get("state")
            if filename is None:
                continue
            if state == "canceled":
                self.downloads.pop(guid, None)
                continue
            if state != "completed":
                continue
            self.downloads.pop(guid, None)
            file_path = params.get("filePath")
            completed_path = (
                Path(str(file_path))
                if file_path
                else self.download_directory / filename
            )
            self.pending_completed.add(completed_path.expanduser().resolve())
        completed.extend(self.take_existing_completed_paths())
        return completed

    def take_existing_completed_paths(self) -> list[Path]:
        completed = []
        for path in list(self.pending_completed):
            if not path.is_file():
                continue
            self.pending_completed.remove(path)
            completed.append(path)
        return completed

    def close(self, *, close_target: bool = False) -> bool:
        target_closed = False
        if close_target:
            try:
                self.browser_socket.settimeout(2)
                result = self.request(
                    self.browser_socket,
                    "Target.closeTarget",
                    {"targetId": self.target_id},
                )
                target_closed = bool(result.get("success"))
            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
                websocket.WebSocketException,
            ) as error:
                logging.warning("无法关闭内嵌 ChatGPT 页面 target：%s", error)
        self.browser_socket.close()
        return target_closed


@dataclass(frozen=True)
class X11WindowInfo:
    window_id: int
    wm_class: str
    pid: int | None
    width: int
    height: int
    is_viewable: bool


def find_remote_chrome(
    debug_port: int = 9223,
    proc_root: Path = Path("/proc"),
) -> RemoteChromeSession | None:
    """Find the top-level Chrome process serving the requested debug port."""
    port_argument = f"--remote-debugging-port={debug_port}"
    for commandline_path in sorted(proc_root.glob("[0-9]*/cmdline")):
        try:
            commandline_values = [
                value.decode("utf-8", errors="replace")
                for value in commandline_path.read_bytes().split(b"\0")
                if value
            ]
        except (OSError, PermissionError):
            continue
        if len(commandline_values) == 1 and " --" in commandline_values[0]:
            arguments = shlex.split(commandline_values[0])
        else:
            arguments = commandline_values
        if not arguments or port_argument not in arguments:
            continue
        if any(argument.startswith("--type=") for argument in arguments):
            continue
        executable_name = Path(arguments[0]).name.casefold()
        if "chrome" not in executable_name and "chromium" not in executable_name:
            continue
        user_data_dir = next(
            (
                argument.partition("=")[2]
                for argument in arguments
                if argument.startswith("--user-data-dir=")
            ),
            None,
        )
        profile_directory = next(
            (
                argument.partition("=")[2]
                for argument in arguments
                if argument.startswith("--profile-directory=")
            ),
            None,
        )
        return RemoteChromeSession(
            executable=arguments[0],
            user_data_dir=user_data_dir,
            profile_directory=profile_directory,
            debug_port=debug_port,
            pid=int(commandline_path.parent.name),
        )
    return None


def remote_chrome_app_command(session: RemoteChromeSession) -> list[str]:
    """Build the command that asks the existing Chrome process for an app window."""
    command = [session.executable]
    if session.user_data_dir:
        command.append(f"--user-data-dir={session.user_data_dir}")
    if session.profile_directory:
        command.append(f"--profile-directory={session.profile_directory}")
    command.extend(
        [
            f"--remote-debugging-port={session.debug_port}",
            "--new-window",
            f"--app={CHATGPT_URL.toString()}",
        ]
    )
    return command


def parse_x11_client_ids(output: str) -> set[int]:
    """Parse _NET_CLIENT_LIST output into X11 window IDs."""
    return {int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", output)}


def x11_client_window_ids() -> set[int]:
    completed = subprocess.run(
        ["xprop", "-root", "_NET_CLIENT_LIST"],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    if completed.returncode != 0:
        raise OSError(completed.stderr.strip() or "无法读取 X11 窗口列表。")
    return parse_x11_client_ids(completed.stdout)


def read_x11_window_info(window_id: int) -> X11WindowInfo | None:
    properties = subprocess.run(
        [
            "xprop",
            "-id",
            hex(window_id),
            "WM_CLASS",
            "_NET_WM_PID",
            "_NET_WM_WINDOW_TYPE",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    geometry = subprocess.run(
        ["xwininfo", "-id", hex(window_id)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    if properties.returncode != 0 or geometry.returncode != 0:
        return None
    class_match = re.search(r"^WM_CLASS.*?=\s*(.+)$", properties.stdout, re.M)
    pid_match = re.search(r"^_NET_WM_PID.*?=\s*(\d+)$", properties.stdout, re.M)
    width_match = re.search(r"^\s*Width:\s*(\d+)$", geometry.stdout, re.M)
    height_match = re.search(r"^\s*Height:\s*(\d+)$", geometry.stdout, re.M)
    if width_match is None or height_match is None:
        return None
    return X11WindowInfo(
        window_id=window_id,
        wm_class=class_match.group(1) if class_match else "",
        pid=int(pid_match.group(1)) if pid_match else None,
        width=int(width_match.group(1)),
        height=int(height_match.group(1)),
        is_viewable="Map State: IsViewable" in geometry.stdout,
    )


def x11_window_matches_session(
    window: X11WindowInfo,
    session: RemoteChromeSession,
) -> bool:
    window_class = window.wm_class.casefold()
    return (
        "chatgpt.com" in window_class
        and ("chrome" in window_class or "chromium" in window_class)
        and window.pid == session.pid
        and window.is_viewable
        and window.width >= 320
        and window.height >= 320
    )


class X11WindowController:
    """Reparent a Chrome window directly through libX11."""

    class SetWindowAttributes(ctypes.Structure):
        _fields_ = [
            ("background_pixmap", ctypes.c_ulong),
            ("background_pixel", ctypes.c_ulong),
            ("border_pixmap", ctypes.c_ulong),
            ("border_pixel", ctypes.c_ulong),
            ("bit_gravity", ctypes.c_int),
            ("win_gravity", ctypes.c_int),
            ("backing_store", ctypes.c_int),
            ("backing_planes", ctypes.c_ulong),
            ("backing_pixel", ctypes.c_ulong),
            ("save_under", ctypes.c_int),
            ("event_mask", ctypes.c_long),
            ("do_not_propagate_mask", ctypes.c_long),
            ("override_redirect", ctypes.c_int),
            ("colormap", ctypes.c_ulong),
            ("cursor", ctypes.c_ulong),
        ]

    def __init__(self, *, library=None, display=None) -> None:
        if library is None:
            library_path = ctypes.util.find_library("X11")
            if library_path is None:
                raise OSError("没有找到 libX11。")
            library = ctypes.CDLL(library_path)
        self.library = library
        if isinstance(library, ctypes.CDLL):
            self._configure_signatures()
        if display is None:
            display = self.library.XOpenDisplay(None)
        if not display:
            raise OSError("无法连接当前 X11 DISPLAY。")
        self.display = display
        self.closed = False

    def _configure_signatures(self) -> None:
        display_pointer = ctypes.c_void_p
        window = ctypes.c_ulong
        self.library.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.library.XOpenDisplay.restype = display_pointer
        self.library.XUnmapWindow.argtypes = [display_pointer, window]
        self.library.XReparentWindow.argtypes = [
            display_pointer,
            window,
            window,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.XMoveResizeWindow.argtypes = [
            display_pointer,
            window,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.library.XMapWindow.argtypes = [display_pointer, window]
        self.library.XDestroyWindow.argtypes = [display_pointer, window]
        self.library.XFlush.argtypes = [display_pointer]
        self.library.XSync.argtypes = [display_pointer, ctypes.c_int]
        self.library.XCloseDisplay.argtypes = [display_pointer]
        self.library.XChangeWindowAttributes.argtypes = [
            display_pointer,
            window,
            ctypes.c_ulong,
            ctypes.POINTER(self.SetWindowAttributes),
        ]
        self.library.XQueryTree.argtypes = [
            display_pointer,
            window,
            ctypes.POINTER(window),
            ctypes.POINTER(window),
            ctypes.POINTER(ctypes.POINTER(window)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.library.XFree.argtypes = [ctypes.c_void_p]
        for name in (
            "XUnmapWindow",
            "XReparentWindow",
            "XMoveResizeWindow",
            "XMapWindow",
            "XDestroyWindow",
            "XFlush",
            "XSync",
            "XCloseDisplay",
            "XChangeWindowAttributes",
            "XQueryTree",
            "XFree",
        ):
            getattr(self.library, name).restype = ctypes.c_int

    def parent_window_id(self, window_id: int) -> int | None:
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        child_count = ctypes.c_uint()
        result = self.library.XQueryTree(
            self.display,
            window_id,
            ctypes.byref(root),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(child_count),
        )
        if children:
            self.library.XFree(children)
        return int(parent.value) if result else None

    def reparent(
        self,
        child_id: int,
        parent_id: int,
        width: int,
        height: int,
    ) -> None:
        attributes = self.SetWindowAttributes()
        attributes.override_redirect = 1
        self.library.XUnmapWindow(self.display, child_id)
        self.library.XSync(self.display, False)
        self.library.XChangeWindowAttributes(
            self.display,
            child_id,
            1 << 9,
            ctypes.byref(attributes),
        )
        self.library.XReparentWindow(self.display, child_id, parent_id, 0, 0)
        self.library.XMoveResizeWindow(
            self.display,
            child_id,
            0,
            0,
            max(width, 1),
            max(height, 1),
        )
        self.library.XMapWindow(self.display, child_id)
        self.library.XSync(self.display, False)
        actual_parent = self.parent_window_id(child_id)
        if actual_parent != parent_id:
            actual = hex(actual_parent) if actual_parent is not None else "unknown"
            raise OSError(
                f"Chrome 窗口父节点校验失败：期望 {hex(parent_id)}，实际 {actual}。"
            )

    def resize(self, window_id: int, width: int, height: int) -> None:
        self.library.XMoveResizeWindow(
            self.display,
            window_id,
            0,
            0,
            max(width, 1),
            max(height, 1),
        )
        self.library.XFlush(self.display)

    def destroy(self, window_id: int) -> None:
        self.library.XDestroyWindow(self.display, window_id)
        self.library.XSync(self.display, False)

    def close(self) -> None:
        if self.closed:
            return
        self.library.XCloseDisplay(self.display)
        self.closed = True


class EmbeddedChromeWidget(QWidget):
    """Host a real Chrome app window inside Qt through an X11 foreign window."""

    attach_failed = pyqtSignal(str)
    document_downloaded = pyqtSignal(str)
    ATTACH_STABILITY_POLLS = 8
    WINDOW_SCAN_BATCH_SIZE = 4

    def __init__(
        self,
        session: RemoteChromeSession,
        parent=None,
        *,
        x11_connection=None,
        download_directory: Path | None = None,
        known_debug_targets: list[ChromeDebugTarget] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.session = session
        self.x11_connection = x11_connection or X11WindowController()
        self.window_id: int | None = None
        self.poll_attempts = 0
        self.candidate_window_id: int | None = None
        self.candidate_seen_count = 0
        self.known_window_ids = x11_client_window_ids()
        self.known_window_candidates = sorted(self.known_window_ids)
        self.known_window_scan_index = 0
        self.download_directory = (
            Path(download_directory).expanduser().resolve()
            if download_directory is not None
            else chrome_download_directory(session)
        )
        self.download_capture: ChromeTargetDownloadCapture | None = None
        self.target_poll_attempts = 0
        try:
            debug_targets = (
                chrome_debug_targets(session.debug_port)
                if known_debug_targets is None
                else known_debug_targets
            )
        except OSError as error:
            logging.warning("ChatGPT 下载捕获不可用：%s", error)
            self.known_debug_target_ids: set[str] | None = None
        else:
            self.known_debug_target_ids = {
                target.target_id for target in debug_targets
            }

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("正在连接远程 Chrome…", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.layout.addWidget(self.status_label)

        try:
            self.chrome_process = subprocess.Popen(
                remote_chrome_app_command(session),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            raise OSError(f"无法启动 Chrome：{error}") from error

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(150)
        self.poll_timer.timeout.connect(self.attach_new_chrome_window)
        self.poll_timer.start()
        self.guard_timer = QTimer(self)
        self.guard_timer.setInterval(1000)
        self.guard_timer.timeout.connect(self.verify_embedded_window)
        self.target_timer = QTimer(self)
        self.target_timer.setInterval(100)
        self.target_timer.timeout.connect(self.attach_download_capture)
        self.download_timer = QTimer(self)
        self.download_timer.setInterval(250)
        self.download_timer.timeout.connect(self.poll_download_capture)
        if self.known_debug_target_ids is not None:
            self.target_timer.start()

    def attach_download_capture(self) -> None:
        self.target_poll_attempts += 1
        try:
            targets = chrome_debug_targets(self.session.debug_port)
            target = new_chatgpt_target(self.known_debug_target_ids or set(), targets)
            if target is None:
                if self.target_poll_attempts >= 100:
                    self.target_timer.stop()
                return
            browser_websocket_url = chrome_browser_websocket_url(
                self.session.debug_port
            )
            self.download_capture = ChromeTargetDownloadCapture(
                target,
                browser_websocket_url,
                self.download_directory,
            )
        except (OSError, websocket.WebSocketException) as error:
            self.target_timer.stop()
            logging.warning("无法连接内嵌 ChatGPT 下载事件：%s", error)
            return
        self.target_timer.stop()
        self.download_timer.start()

    def poll_download_capture(self) -> None:
        if self.download_capture is None:
            return
        try:
            completed_paths = self.download_capture.poll_completed_downloads()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.download_timer.stop()
            logging.warning("读取内嵌 ChatGPT 下载事件失败：%s", error)
            return
        for path in completed_paths:
            self.document_downloaded.emit(str(path))

    def attach_new_chrome_window(self) -> None:
        self.poll_attempts += 1
        try:
            current_window_ids = x11_client_window_ids()
        except OSError as error:
            self.fail_attachment(str(error))
            return
        new_window_ids = current_window_ids - self.known_window_ids
        if self.candidate_window_id is not None:
            candidates = [self.candidate_window_id]
        elif new_window_ids:
            candidates = list(sorted(new_window_ids))
        else:
            start = self.known_window_scan_index
            stop = min(
                start + self.WINDOW_SCAN_BATCH_SIZE,
                len(self.known_window_candidates),
            )
            candidates = self.known_window_candidates[start:stop]
            self.known_window_scan_index = stop
        matched_window_id = None
        for window_id in candidates:
            window = read_x11_window_info(window_id)
            if window is not None and x11_window_matches_session(window, self.session):
                matched_window_id = window_id
                break
        if matched_window_id is not None:
            if matched_window_id == self.candidate_window_id:
                self.candidate_seen_count += 1
            else:
                self.candidate_window_id = matched_window_id
                self.candidate_seen_count = 1
            if self.candidate_seen_count >= self.ATTACH_STABILITY_POLLS:
                self.attach_window(matched_window_id)
                return
        else:
            self.candidate_window_id = None
            self.candidate_seen_count = 0
        if self.poll_attempts >= 100:
            self.fail_attachment(
                "Chrome 已启动，但 15 秒内没有找到新的 ChatGPT 窗口。"
            )

    def attach_window(self, window_id: int) -> None:
        try:
            self.x11_connection.reparent(
                window_id,
                int(self.winId()),
                self.width(),
                self.height(),
            )
        except OSError as error:
            self.fail_attachment(f"无法嵌入 Chrome 窗口：{error}")
            return
        self.poll_timer.stop()
        self.window_id = window_id
        self.status_label.hide()
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.guard_timer.start()

    def verify_embedded_window(self) -> None:
        if self.window_id is None:
            return
        parent_id = int(self.winId())
        if self.x11_connection.parent_window_id(self.window_id) == parent_id:
            return
        try:
            self.x11_connection.reparent(
                self.window_id,
                parent_id,
                self.width(),
                self.height(),
            )
        except OSError as error:
            self.fail_attachment(f"Chrome 窗口脱离面板且无法重新接管：{error}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.window_id is None:
            return
        try:
            self.x11_connection.resize(
                self.window_id,
                event.size().width(),
                event.size().height(),
            )
        except OSError as error:
            self.fail_attachment(f"无法调整 Chrome 窗口大小：{error}")

    def fail_attachment(self, message: str) -> None:
        self.poll_timer.stop()
        self.guard_timer.stop()
        self.status_label.show()
        self.status_label.setText(message)
        self.attach_failed.emit(message)

    def shutdown(self) -> None:
        self.poll_timer.stop()
        self.guard_timer.stop()
        self.target_timer.stop()
        self.download_timer.stop()
        target_closed = False
        if self.download_capture is not None:
            target_closed = self.download_capture.close(close_target=True)
            self.download_capture = None
        if target_closed:
            self.window_id = None
        if self.window_id is not None:
            self.x11_connection.destroy(self.window_id)
            self.window_id = None
        self.x11_connection.close()


def create_chatgpt_browser(parent=None) -> EmbeddedChromeWidget:
    """Embed the real remotely-debuggable Chrome window in the current X11 app."""
    if os.environ.get("XDG_SESSION_TYPE", "").casefold() == "wayland":
        raise OSError("Chrome 窗口嵌入只支持 X11，当前桌面正在使用 Wayland。")
    if not os.environ.get("DISPLAY"):
        raise OSError("没有检测到 X11 DISPLAY，无法嵌入 Chrome 窗口。")
    missing_tools = [tool for tool in ("xprop", "xwininfo") if shutil.which(tool) is None]
    if missing_tools:
        raise OSError(
            f"缺少 X11 工具：{', '.join(missing_tools)}。请运行 mdview 安装脚本。"
        )
    try:
        debug_port = int(os.environ.get("MDVIEW_CHROME_DEBUG_PORT", "9223"))
    except ValueError as error:
        raise OSError("MDVIEW_CHROME_DEBUG_PORT 必须是整数端口。") from error
    session = find_remote_chrome(debug_port)
    if session is None:
        raise OSError(
            f"没有找到使用 --remote-debugging-port={debug_port} 的 Chrome。\n\n"
            "请先启动远程调试 Chrome，再打开 ChatGPT 面板。"
        )
    return EmbeddedChromeWidget(session, parent)


class MarkdownPreview(QTextBrowser):
    """Rendered Markdown preview with explicit rich selection copy actions."""

    image_saved = pyqtSignal(str)
    image_save_failed = pyqtSignal(str)
    chatgpt_edit_requested = pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pdf_pages: list[
            tuple[
                float,
                float,
                list[tuple[float, float, float, float, str, int]],
            ]
        ] = []
        self._pdf_images: list[Path] = []
        self._pdf_words: list[tuple[int, int]] = []
        self._pdf_selection_start: int | None = None
        self._pdf_selection_end: int | None = None
        self._pdf_selecting = False
        self.image_output_directory = Path("/tmp")

    def setHtml(self, text: str) -> None:
        self._clear_pdf_mode()
        super().setHtml(text)

    def set_pdf_document(
        self,
        page_html: str,
        page_images: list[Path],
        text_pages: list[
            tuple[
                float,
                float,
                list[tuple[float, float, float, float, str, int]],
            ]
        ],
    ) -> None:
        """Display PDF pages and select text using the PDF's own word boxes."""
        self._pdf_pages = text_pages
        self._pdf_images = page_images
        self._pdf_words = [
            (page_index, word_index)
            for page_index, page in enumerate(text_pages)
            for word_index in range(len(page[2]))
        ]
        self._pdf_selection_start = None
        self._pdf_selection_end = None
        self._pdf_selecting = False
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        super().setHtml(page_html)
        self.viewport().setCursor(Qt.IBeamCursor)

    def _clear_pdf_mode(self) -> None:
        self._pdf_pages = []
        self._pdf_images = []
        self._pdf_words = []
        self._pdf_selection_start = None
        self._pdf_selection_end = None
        self._pdf_selecting = False
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.viewport().unsetCursor()

    def _pdf_page_frames(self) -> list[QRectF]:
        if not self._pdf_pages:
            return []
        layout = self.document().documentLayout()
        return [
            layout.frameBoundingRect(frame)
            for frame in self.document().rootFrame().childFrames()
        ][: len(self._pdf_pages)]

    def _pdf_word_document_rect(self, ordinal: int) -> QRectF:
        if ordinal < 0 or ordinal >= len(self._pdf_words):
            return QRectF()
        page_index, word_index = self._pdf_words[ordinal]
        frames = self._pdf_page_frames()
        if page_index >= len(frames):
            return QRectF()
        page_width, page_height, words = self._pdf_pages[page_index]
        x_min, y_min, x_max, y_max, _text, _line = words[word_index]
        frame = frames[page_index]
        return QRectF(
            frame.x() + x_min * frame.width() / page_width,
            frame.y() + y_min * frame.height() / page_height,
            (x_max - x_min) * frame.width() / page_width,
            (y_max - y_min) * frame.height() / page_height,
        )

    def _pdf_word_at(self, viewport_position) -> int | None:
        document_point = QPointF(
            viewport_position.x() + self.horizontalScrollBar().value(),
            viewport_position.y() + self.verticalScrollBar().value(),
        )
        frames = self._pdf_page_frames()
        page_index = next(
            (
                index
                for index, frame in enumerate(frames)
                if frame.adjusted(0, -8, 0, 8).contains(document_point)
            ),
            None,
        )
        if page_index is None:
            return None
        page_width, page_height, words = self._pdf_pages[page_index]
        frame = frames[page_index]
        page_point = QPointF(
            (document_point.x() - frame.x()) * page_width / frame.width(),
            (document_point.y() - frame.y()) * page_height / frame.height(),
        )
        candidates = [
            (word_index, word)
            for word_index, word in enumerate(words)
            if word[1] - 3 <= page_point.y() <= word[3] + 3
        ]
        if not candidates:
            candidates = list(enumerate(words))

        def distance(candidate) -> float:
            _index, (x_min, y_min, x_max, y_max, _text, _line) = candidate
            dx = max(x_min - page_point.x(), 0, page_point.x() - x_max)
            dy = max(y_min - page_point.y(), 0, page_point.y() - y_max)
            return dx * dx + dy * dy

        word_index, _word = min(candidates, key=distance)
        try:
            return self._pdf_words.index((page_index, word_index))
        except ValueError:
            return None

    def _pdf_selected_ordinals(self) -> range:
        if self._pdf_selection_start is None or self._pdf_selection_end is None:
            return range(0)
        start, end = sorted((self._pdf_selection_start, self._pdf_selection_end))
        return range(start, end + 1)

    def _pdf_selection_text(self) -> str:
        output: list[str] = []
        previous_page_line: tuple[int, int] | None = None
        previous_x_max: float | None = None
        for ordinal in self._pdf_selected_ordinals():
            page_index, word_index = self._pdf_words[ordinal]
            x_min, _y_min, x_max, _y_max, text, line_number = self._pdf_pages[
                page_index
            ][2][word_index]
            page_line = (page_index, line_number)
            if previous_page_line is not None and page_line != previous_page_line:
                output.append("\n")
            elif previous_x_max is not None and x_min - previous_x_max > 0.8:
                output.append(" ")
            output.append(text)
            previous_page_line = page_line
            previous_x_max = x_max
        return "".join(output)

    def _pdf_selection_rects(self) -> list[QRectF]:
        horizontal = self.horizontalScrollBar().value()
        vertical = self.verticalScrollBar().value()
        return [
            self._pdf_word_document_rect(ordinal).translated(-horizontal, -vertical)
            for ordinal in self._pdf_selected_ordinals()
        ]

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._pdf_pages:
            return
        painter = QPainter(self.viewport())
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(14, 165, 233, 150))
        for selection_rect in self._pdf_selection_rects():
            painter.drawRect(selection_rect.adjusted(-1, -1, 1, 1))
        painter.end()

    def mousePressEvent(self, event) -> None:
        if not self._pdf_pages or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        ordinal = self._pdf_word_at(event.pos())
        if ordinal is None:
            self._pdf_selection_start = None
            self._pdf_selection_end = None
            self.viewport().update()
            return
        self._pdf_selection_start = ordinal
        self._pdf_selection_end = ordinal
        self._pdf_selecting = True
        self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        if not self._pdf_pages or not self._pdf_selecting:
            super().mouseMoveEvent(event)
            return
        ordinal = self._pdf_word_at(event.pos())
        if ordinal is not None:
            self._pdf_selection_end = ordinal
            self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:
        if self._pdf_pages and event.button() == Qt.LeftButton:
            self._pdf_selecting = False
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._pdf_pages and event.matches(QKeySequence.Copy):
            self.copy_selection_as_text()
            return
        super().keyPressEvent(event)

    def copy_selection_as_text(self) -> None:
        if self._pdf_pages:
            selected = self._pdf_selection_text()
            if selected:
                QApplication.clipboard().setText(selected)
            return
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        selected = cursor.selectedText().replace("\u2029", "\n").replace("\u2028", "\n")
        QApplication.clipboard().setText(selected)

    def selected_text_and_location(self) -> tuple[str, str]:
        if self._pdf_pages:
            ordinals = list(self._pdf_selected_ordinals())
            if not ordinals:
                return "", ""
            pages = sorted({self._pdf_words[ordinal][0] + 1 for ordinal in ordinals})
            if len(pages) == 1:
                location = f"PDF 物理页码：第 {pages[0]} 页"
            else:
                location = f"PDF 物理页码：第 {pages[0]}–{pages[-1]} 页"
            return self._pdf_selection_text(), location
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return "", ""
        selected = cursor.selectedText().replace("\u2029", "\n").replace(
            "\u2028", "\n"
        )
        return selected, "实时渲染预览（请按选中原文精确搜索）"

    def request_chatgpt_edit(self) -> None:
        selected_text, location = self.selected_text_and_location()
        if selected_text:
            self.chatgpt_edit_requested.emit(selected_text, location)

    def copy_selection_as_image(self) -> None:
        if self._pdf_pages:
            self._copy_pdf_selection_as_image()
            return
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        document = QTextDocument()
        document.setDocumentMargin(12)
        document.setDefaultFont(self.document().defaultFont())
        document.setHtml(cursor.selection().toHtml())
        document.setTextWidth(min(900, max(320, self.viewport().width() - 32)))
        size = document.documentLayout().documentSize()
        image = QImage(
            max(1, math.ceil(size.width())),
            max(1, math.ceil(size.height())),
            QImage.Format_ARGB32_Premultiplied,
        )
        image.fill(self.palette().base().color())
        painter = QPainter(image)
        document.drawContents(painter)
        painter.end()
        self._publish_copied_image(image)

    def _publish_copied_image(self, image: QImage) -> None:
        QApplication.clipboard().setImage(image)
        try:
            output_path = save_preview_image(image, self.image_output_directory)
        except (OSError, RuntimeError) as error:
            self.image_save_failed.emit(str(error))
            return
        self.image_saved.emit(str(output_path))

    def _copy_pdf_selection_as_image(self) -> None:
        selected = list(self._pdf_selected_ordinals())
        if not selected:
            return
        page_groups: dict[int, list[int]] = {}
        for ordinal in selected:
            page_index, word_index = self._pdf_words[ordinal]
            page_groups.setdefault(page_index, []).append(word_index)
        crops: list[QImage] = []
        for page_index, word_indices in page_groups.items():
            if page_index >= len(self._pdf_images):
                continue
            image = QImage(str(self._pdf_images[page_index]))
            if image.isNull():
                continue
            page_width, page_height, words = self._pdf_pages[page_index]
            boxes = [words[index] for index in word_indices]
            left = max(0, math.floor(min(box[0] for box in boxes) * image.width() / page_width) - 6)
            top = max(0, math.floor(min(box[1] for box in boxes) * image.height() / page_height) - 6)
            right = min(image.width(), math.ceil(max(box[2] for box in boxes) * image.width() / page_width) + 6)
            bottom = min(image.height(), math.ceil(max(box[3] for box in boxes) * image.height() / page_height) + 6)
            crops.append(image.copy(left, top, right - left, bottom - top))
        if not crops:
            return
        width = max(crop.width() for crop in crops)
        height = sum(crop.height() for crop in crops) + 8 * (len(crops) - 1)
        output = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        output.fill(Qt.white)
        painter = QPainter(output)
        y = 0
        for crop in crops:
            painter.drawImage(0, y, crop)
            y += crop.height() + 8
        painter.end()
        self._publish_copied_image(output)

    def create_preview_context_menu(self):
        if self._pdf_pages:
            menu = QMenu(self)
            has_selection = bool(list(self._pdf_selected_ordinals()))
            copy_text = menu.addAction("复制为文字")
            copy_text.setEnabled(has_selection)
            copy_text.triggered.connect(self.copy_selection_as_text)
            copy_image = menu.addAction("复制为图片")
            copy_image.setEnabled(has_selection)
            copy_image.triggered.connect(self.copy_selection_as_image)
            chatgpt_edit = menu.addAction("复制为 ChatGPT 对话…")
            chatgpt_edit.setEnabled(has_selection)
            chatgpt_edit.triggered.connect(self.request_chatgpt_edit)
            return menu
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        cursor_has_selection = self.textCursor().hasSelection()
        copy_text = menu.addAction("复制为文字")
        copy_text.setEnabled(cursor_has_selection)
        copy_text.triggered.connect(self.copy_selection_as_text)
        copy_image = menu.addAction("复制为图片")
        copy_image.setEnabled(cursor_has_selection)
        copy_image.triggered.connect(self.copy_selection_as_image)
        chatgpt_edit = menu.addAction("复制为 ChatGPT 对话…")
        chatgpt_edit.setEnabled(cursor_has_selection)
        chatgpt_edit.triggered.connect(self.request_chatgpt_edit)
        return menu

    def contextMenuEvent(self, event) -> None:
        menu = self.create_preview_context_menu()
        menu.exec_(event.globalPos())
        menu.deleteLater()


class PdfExportDialog(QDialog):
    """Collect one-time PDF export options without changing app settings."""

    def __init__(
        self,
        suggested_path: Path,
        *,
        include_toc: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出 PDF")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        path_label = QLabel("输出文件")
        layout.addWidget(path_label)
        path_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(suggested_path.expanduser()))
        self.output_edit.setPlaceholderText("选择 PDF 保存位置和文件名")
        path_row.addWidget(self.output_edit, 1)
        browse_button = QPushButton("浏览…")
        browse_button.clicked.connect(self.browse_output_path)
        path_row.addWidget(browse_button)
        layout.addLayout(path_row)

        self.toc_checkbox = QCheckBox("生成正文目录（TOC）")
        self.toc_checkbox.setChecked(include_toc)
        self.toc_checkbox.setToolTip("仅影响本次导出，不修改 Markdown/LaTeX 源文件")
        layout.addWidget(self.toc_checkbox)
        self.open_checkbox = QCheckBox("导出后打开 PDF")
        self.open_checkbox.setChecked(True)
        layout.addWidget(self.open_checkbox)

        hint = QLabel("这些选项只用于本次导出，不会写入全局设置。")
        hint.setObjectName("exportHint")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("导出")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_output_path(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "选择 PDF 保存位置",
            self.output_edit.text().strip(),
            "PDF 文档 (*.pdf)",
        )
        if selected:
            path = Path(selected)
            if path.suffix.casefold() != ".pdf":
                path = path.with_suffix(".pdf")
            self.output_edit.setText(str(path))

    def output_path(self) -> Path:
        path = Path(self.output_edit.text().strip()).expanduser()
        if path.suffix.casefold() != ".pdf":
            path = path.with_suffix(".pdf")
        return path

    def include_toc(self) -> bool:
        return self.toc_checkbox.isChecked()

    def open_after_export(self) -> bool:
        return self.open_checkbox.isChecked()


class MarkdownWindow(QMainWindow):
    def __init__(
        self,
        initial_path: Path | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or QSettings("Codex Tools", APP_NAME)
        self.current_path: Path | None = None
        self.document_mode = "markdown"
        self.preview_cache_path = Path(
            tempfile.mkdtemp(prefix="mdview-document-preview-")
        )
        self._preview_cache_finalizer = weakref.finalize(
            self,
            shutil.rmtree,
            self.preview_cache_path,
            ignore_errors=True,
        )
        self.latest_tex_pdf_path: Path | None = None
        self.border_color = DEFAULT_BORDER_COLOR
        self.background_color = DEFAULT_BACKGROUND_COLOR
        self.setProperty("tocVisible", False)
        self.line_height = normalized_line_height(
            self.settings.value("lineHeight", DEFAULT_LINE_HEIGHT)
        )
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 800)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("markdownSource")
        self.editor.setPlaceholderText("在这里输入 Markdown/LaTeX，或点击“打开文件”。")
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setFont(QFont("monospace", 11))

        self.preview = MarkdownPreview()
        self.preview.setObjectName("markdownPreview")
        self.preview.setOpenExternalLinks(True)
        self.preview.image_saved.connect(
            lambda path: self.statusBar().showMessage(
                f"图片已复制并保存到：{path}", 10_000
            )
        )
        self.preview.image_save_failed.connect(
            lambda error: self.statusBar().showMessage(
                f"图片已复制，但保存到 /tmp 失败：{error}", 10_000
            )
        )
        self.preview.chatgpt_edit_requested.connect(
            self.copy_chatgpt_edit_request
        )

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("sourcePreviewSplitter")
        splitter.addWidget(self.editor)
        splitter.addWidget(self.preview)
        splitter.setSizes([0, 1280])
        self.setCentralWidget(splitter)
        self.editor.hide()

        self.toc_tree = QTreeWidget()
        self.toc_tree.setObjectName("documentToc")
        self.toc_tree.setHeaderHidden(True)
        self.toc_tree.setIndentation(16)
        self.toc_tree.itemClicked.connect(self.navigate_to_toc_item)
        self.toc_dock = QDockWidget("目录", self)
        self.toc_dock.setObjectName("documentTocDock")
        self.toc_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.toc_dock.setMinimumWidth(220)
        self.toc_dock.setWidget(self.toc_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.toc_dock)

        self.chatgpt_view = None
        self.chatgpt_dock = QDockWidget("ChatGPT", self)
        self.chatgpt_dock.setObjectName("chatgptDock")
        self.chatgpt_dock.setProperty("tocVisible", False)
        self.chatgpt_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.chatgpt_dock.setMinimumWidth(380)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.chatgpt_dock)
        self.splitDockWidget(self.chatgpt_dock, self.toc_dock, Qt.Horizontal)
        self.chatgpt_dock.hide()

        toolbar = QToolBar("文档", self)
        toolbar.setObjectName("documentToolBar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolBar = toolbar
        self.addToolBar(toolbar)
        self.open_action = QAction(
            self.style().standardIcon(QStyle.SP_DialogOpenButton),
            "打开文件",
            self,
        )
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setToolTip("打开 Markdown 或完整 LaTeX 文档")
        self.open_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(self.open_action)
        self.toc_action = QAction("显示目录", self)
        self.toc_action.setCheckable(True)
        self.toc_action.setChecked(False)
        self.toc_action.triggered.connect(self.set_toc_visible)
        toolbar.addAction(self.toc_action)
        self.toc_dock.visibilityChanged.connect(self.sync_toc_action)
        self.toc_dock.hide()
        self.chatgpt_action = QAction("ChatGPT", self)
        self.chatgpt_action.setCheckable(True)
        self.chatgpt_action.setToolTip("在目录左侧嵌入或隐藏远程 Chrome ChatGPT")
        self.chatgpt_action.triggered.connect(self.set_chatgpt_visible)
        toolbar.addAction(self.chatgpt_action)
        self.chatgpt_dock.visibilityChanged.connect(
            self.handle_chatgpt_visibility_changed
        )
        self.source_action = QAction("显示原文", self)
        self.source_action.setCheckable(True)
        self.source_action.triggered.connect(self.set_source_visible)
        toolbar.addAction(self.source_action)
        toolbar.addSeparator()
        self.border_action = QAction("边框颜色", self)
        self.border_action.triggered.connect(self.choose_border_color)
        self.background_action = QAction("背景颜色", self)
        self.background_action.triggered.connect(self.choose_background_color)
        self.line_height_action = QAction(f"行间距：{self.line_height:.2f}", self)
        self.line_height_action.triggered.connect(self.choose_line_height)
        self.preview_pdf_action = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            "预览 PDF",
            self,
        )
        self.preview_pdf_action.setToolTip("生成临时 PDF 并用系统阅读器打开")
        self.preview_pdf_action.triggered.connect(self.preview_pdf)
        toolbar.addAction(self.preview_pdf_action)
        self.export_action = QAction(
            self.style().standardIcon(QStyle.SP_DialogSaveButton),
            "导出 PDF",
            self,
        )
        self.export_action.triggered.connect(self.export_pdf_dialog)
        toolbar.addAction(self.export_action)
        toolbar.addSeparator()

        self.settings_menu = QMenu(self)
        self.settings_menu.addAction(self.background_action)
        self.settings_menu.addAction(self.border_action)
        self.settings_menu.addAction(self.line_height_action)
        self.settings_button = QToolButton(self)
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setText("设置")
        self.settings_button.setIcon(
            self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        )
        self.settings_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.settings_button.setPopupMode(QToolButton.InstantPopup)
        self.settings_button.setMenu(self.settings_menu)
        self.settings_button.setToolTip("外观和 Markdown 行间距")
        toolbar.addWidget(self.settings_button)

        self.path_button = QPushButton("")
        self.path_button.setObjectName("documentPathButton")
        self.path_button.setFlat(True)
        self.path_button.setCursor(Qt.PointingHandCursor)
        self.path_button.clicked.connect(self.copy_current_file_path)
        self.statusBar().addPermanentWidget(self.path_button, 1)

        self.set_border_color(str(self.settings.value("borderColor", DEFAULT_BORDER_COLOR)))
        self.set_background_color(
            str(self.settings.value("backgroundColor", DEFAULT_BACKGROUND_COLOR))
        )

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(120)
        self.render_timer.timeout.connect(self.refresh_preview)
        self.editor.textChanged.connect(self.render_timer.start)

        if initial_path is not None:
            self.load_file(initial_path)
        else:
            self.refresh_preview()

    def set_toc_visible(self, visible: bool) -> None:
        self.toc_dock.setVisible(visible)
        self.sync_toc_action(visible)

    def sync_toc_action(self, visible: bool) -> None:
        self.toc_action.setChecked(visible)
        self.toc_action.setText("隐藏目录" if visible else "显示目录")
        self.setProperty("tocVisible", visible)
        self.chatgpt_dock.setProperty("tocVisible", visible)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        self.chatgpt_dock.style().unpolish(self.chatgpt_dock)
        self.chatgpt_dock.style().polish(self.chatgpt_dock)
        self.chatgpt_dock.update()

    def set_chatgpt_visible(self, visible: bool) -> None:
        if not visible:
            self.release_chatgpt_view()
        if visible and self.chatgpt_view is None:
            try:
                self.chatgpt_view = create_chatgpt_browser(self.chatgpt_dock)
            except OSError as error:
                self.chatgpt_view = None
                self.chatgpt_action.blockSignals(True)
                self.chatgpt_action.setChecked(False)
                self.chatgpt_action.blockSignals(False)
                QMessageBox.warning(self, "无法打开 ChatGPT", str(error))
                return
            document_downloaded = getattr(
                self.chatgpt_view,
                "document_downloaded",
                None,
            )
            if document_downloaded is not None:
                document_downloaded.connect(self.open_downloaded_document)
            self.chatgpt_dock.setWidget(self.chatgpt_view)
        self.chatgpt_dock.setVisible(visible)
        self.sync_chatgpt_action(visible)
        if visible:
            self.resizeDocks(
                [self.chatgpt_dock, self.toc_dock],
                [460, 250],
                Qt.Horizontal,
            )

    def sync_chatgpt_action(self, visible: bool) -> None:
        self.chatgpt_action.setChecked(visible)
        self.chatgpt_action.setText("隐藏 ChatGPT" if visible else "ChatGPT")

    def handle_chatgpt_visibility_changed(self, visible: bool) -> None:
        self.sync_chatgpt_action(visible)
        if not visible:
            self.release_chatgpt_view()

    def release_chatgpt_view(self) -> None:
        if self.chatgpt_view is None:
            return
        view = self.chatgpt_view
        self.chatgpt_view = None
        shutdown = getattr(view, "shutdown", None)
        if shutdown is not None:
            shutdown()
        self.chatgpt_dock.setWidget(None)
        view.deleteLater()

    def open_downloaded_document(self, filename: str) -> None:
        document_path = Path(filename).expanduser().resolve()
        if (
            document_path.suffix.casefold() not in AUTO_OPEN_DOCUMENT_SUFFIXES
            or not document_path.is_file()
        ):
            return
        if not self.load_file(document_path):
            return
        self.statusBar().showMessage(
            f"已在当前窗口打开 ChatGPT 下载：{document_path}",
            5000,
        )

    def copy_chatgpt_edit_request(
        self,
        selected_text: str,
        rendered_location: str,
    ) -> None:
        prompt = build_chatgpt_edit_prompt(
            self.current_path,
            selected_text,
            rendered_location,
        )
        QApplication.clipboard().setText(prompt)
        self.statusBar().showMessage(
            "已复制 ChatGPT 定位对话；粘贴后在末尾填写修改要求。",
            8000,
        )

    def closeEvent(self, event) -> None:
        self.release_chatgpt_view()
        super().closeEvent(event)

    def navigate_to_toc_item(self, item: QTreeWidgetItem, _column: int) -> None:
        anchor = item.data(0, Qt.UserRole)
        if not anchor:
            return
        anchor = str(anchor)
        self.preview.scrollToAnchor(anchor)
        vertical_ratio = item.data(0, Qt.UserRole + 1)
        page_match = re.fullmatch(r"pdf-page-(\d+)", anchor)
        if vertical_ratio is None or page_match is None:
            return
        scrollbar = self.preview.verticalScrollBar()
        page_top = scrollbar.value()
        page_number = int(page_match.group(1))
        self.preview.scrollToAnchor(f"pdf-page-{page_number + 1}")
        next_page_top = scrollbar.value()
        if next_page_top <= page_top and page_number > 1:
            self.preview.scrollToAnchor(f"pdf-page-{page_number - 1}")
            previous_page_top = scrollbar.value()
            page_span = page_top - previous_page_top
        else:
            page_span = next_page_top - page_top
        if page_span <= 0:
            self.preview.scrollToAnchor(anchor)
            return
        target = round(page_top + float(vertical_ratio) * page_span)
        scrollbar.setValue(target)

    def refresh_toc(self, source: str) -> None:
        self.toc_tree.clear()
        if self.document_mode == "latex":
            return
        parents: list[tuple[int, QTreeWidgetItem]] = []
        for level, title, anchor in extract_toc_entries(source):
            while parents and parents[-1][0] >= level:
                parents.pop()
            item = QTreeWidgetItem([title])
            item.setData(0, Qt.UserRole, anchor)
            if parents:
                parents[-1][1].addChild(item)
            else:
                self.toc_tree.addTopLevelItem(item)
            parents.append((level, item))
        self.toc_tree.expandToDepth(1)

    def refresh_latex_toc(self, toc_path: Path, pdf_path: Path) -> None:
        self.toc_tree.clear()
        if not toc_path.is_file():
            return
        try:
            toc_source = toc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        parents: list[tuple[int, QTreeWidgetItem]] = []
        for level, title, anchor, vertical_ratio in extract_latex_toc_entries(
            toc_source, pdf_destination_targets(pdf_path)
        ):
            while parents and parents[-1][0] >= level:
                parents.pop()
            item = QTreeWidgetItem([title])
            item.setData(0, Qt.UserRole, anchor)
            item.setData(0, Qt.UserRole + 1, vertical_ratio)
            if parents:
                parents[-1][1].addChild(item)
            else:
                self.toc_tree.addTopLevelItem(item)
            parents.append((level, item))
        self.toc_tree.expandToDepth(1)

    def set_source_visible(self, visible: bool) -> None:
        self.editor.setVisible(visible)
        self.source_action.setChecked(visible)
        self.source_action.setText("隐藏原文" if visible else "显示原文")
        self.centralWidget().setSizes([640, 640] if visible else [0, 1280])

    def set_border_color(self, color: str) -> None:
        selected = QColor(color)
        if not selected.isValid():
            selected = QColor(DEFAULT_BORDER_COLOR)
        self.border_color = selected.name()
        self.settings.setValue("borderColor", self.border_color)
        self.apply_app_theme()

    def choose_border_color(self) -> None:
        current = QColor(str(self.settings.value("borderColor", DEFAULT_BORDER_COLOR)))
        selected = QColorDialog.getColor(current, self, "选择窗口内边框颜色")
        if selected.isValid():
            self.set_border_color(selected.name())

    def set_background_color(self, color: str) -> None:
        selected = QColor(color)
        if not selected.isValid():
            selected = QColor(DEFAULT_BACKGROUND_COLOR)
        self.background_color = selected.name()
        self.settings.setValue("backgroundColor", self.background_color)
        self.apply_app_theme()
        if hasattr(self, "preview"):
            self.refresh_preview()

    def choose_background_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self.background_color),
            self,
            "选择软件背景颜色",
        )
        if selected.isValid():
            self.set_background_color(selected.name())

    def set_line_height(self, value: float) -> None:
        self.line_height = normalized_line_height(value)
        self.settings.setValue("lineHeight", self.line_height)
        self.line_height_action.setText(f"行间距：{self.line_height:.2f}")
        if self.document_mode == "markdown":
            self.refresh_preview()

    def choose_line_height(self) -> None:
        selected, accepted = QInputDialog.getDouble(
            self,
            "调整正文行间距",
            "行高倍数（1.10–2.20）：",
            self.line_height,
            1.1,
            2.2,
            2,
        )
        if accepted:
            self.set_line_height(selected)

    def apply_app_theme(self) -> None:
        text_color = text_color_for_background(self.background_color)
        muted_color = "#cbd5e1" if text_color == "#f8fafc" else "#475569"
        input_background = "#0f172a" if text_color == "#f8fafc" else "#ffffff"
        hover_color = "#1e293b" if text_color == "#f8fafc" else "#eef2ff"
        pressed_color = "#334155" if text_color == "#f8fafc" else "#dbeafe"
        self.setStyleSheet(
            "QMainWindow, QToolBar, QStatusBar, QSplitter, QTextBrowser, QPlainTextEdit,"
            " QDockWidget, QTreeWidget {"
            f" background-color: {self.background_color}; color: {text_color};"
            " }"
            "QMainWindow {"
            f" border: 4px solid {self.border_color};"
            " }"
            "QMainWindow::separator { background-color: transparent; width: 4px; }"
            'QMainWindow[tocVisible="true"]::separator {'
            f" background-color: {self.border_color}; width: 4px;"
            " }"
            "QToolBar {"
            f" border: 0; border-bottom: 2px solid {self.border_color};"
            " spacing: 5px; padding: 6px 8px;"
            " }"
            "QToolButton {"
            f" background-color: {self.background_color}; color: {text_color};"
            " border: 0; border-radius: 6px; padding: 6px 10px;"
            " }"
            "QToolButton:hover {"
            f" background-color: {hover_color};"
            " }"
            "QToolButton:pressed, QToolButton:checked {"
            f" background-color: {pressed_color}; color: {text_color};"
            " }"
            "QToolBar::separator {"
            f" background-color: {self.border_color}; width: 1px; margin: 5px 4px;"
            " }"
            "QMenu {"
            f" background-color: {self.background_color}; color: {text_color};"
            f" border: 1px solid {self.border_color}; padding: 5px;"
            " }"
            "QMenu::item { padding: 7px 28px 7px 10px; border-radius: 4px; }"
            "QMenu::item:selected {"
            f" background-color: {hover_color};"
            " }"
            "QStatusBar {"
            f" border-top: 2px solid {self.border_color};"
            " }"
            "QSplitter#sourcePreviewSplitter {"
            f" border: 3px solid {self.border_color};"
            " }"
            "QDockWidget#chatgptDock { border-right: 0; }"
            "QDockWidget#chatgptDock::title {"
            f" background-color: {self.background_color}; color: {text_color};"
            " padding: 6px; text-align: left; font-weight: 600;"
            " }"
            "QDockWidget#documentTocDock {"
            f" border-right: 2px solid {self.border_color};"
            " }"
            "QDockWidget#documentTocDock::title {"
            f" background-color: {self.background_color}; color: {text_color};"
            " padding: 6px; text-align: left; font-weight: 600;"
            " }"
            "QTreeWidget#documentToc {"
            f" border: 0; border-top: 1px solid {self.border_color};"
            " padding: 5px; outline: 0;"
            " }"
            "QTreeWidget#documentToc::item { padding: 5px 3px; }"
            "QTreeWidget#documentToc::item:selected {"
            f" background-color: {self.border_color}; color: #ffffff;"
            " }"
            "QPlainTextEdit#markdownSource {"
            f" background-color: {input_background}; color: {text_color};"
            " }"
            "QPushButton#documentPathButton {"
            f" color: {muted_color}; text-align: left; border: 0; padding: 1px 4px;"
            " }"
        )

    def copy_current_file_path(self) -> None:
        if self.current_path is None:
            return
        file_path = str(self.current_path)
        QApplication.clipboard().setText(file_path)
        self.statusBar().showMessage(f"文件路径已复制：{file_path}", 3000)

    def default_include_toc(self) -> bool:
        if self.document_mode == "latex":
            return latex_has_toc(self.editor.toPlainText())
        return False

    def build_current_pdf(
        self, output_path: Path, *, include_toc: bool | None = None
    ) -> str:
        base_directory = self.current_path.parent if self.current_path else Path.cwd()
        if include_toc is None:
            include_toc = self.default_include_toc()
        if self.document_mode == "latex":
            return compile_latex_document(
                configure_latex_toc(
                    self.editor.toPlainText(), include_toc
                ),
                output_path,
                base_directory=base_directory,
            )
        return export_pdf(
            self.editor.toPlainText(),
            output_path,
            base_directory=base_directory,
            include_toc=include_toc,
        )

    def export_pdf_dialog(self) -> None:
        suggested = (
            self.current_path.with_suffix(".pdf")
            if self.current_path
            else Path.home() / "document.pdf"
        )
        dialog = PdfExportDialog(
            suggested,
            include_toc=self.default_include_toc(),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        output_path = dialog.output_path().resolve()
        if output_path.exists():
            answer = QMessageBox.question(
                self,
                "覆盖已有 PDF？",
                f"文件已经存在：\n{output_path}\n\n是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        try:
            backend = self.build_current_pdf(
                output_path,
                include_toc=dialog.include_toc(),
            )
        except (OSError, RuntimeError) as error:
            QMessageBox.critical(self, "PDF 导出失败", str(error))
            return
        self.statusBar().showMessage(f"已通过 {backend} 导出：{output_path}", 10_000)
        if dialog.open_after_export() and not open_local_file(output_path):
            QMessageBox.warning(
                self,
                "无法打开 PDF",
                f"PDF 已生成，但系统阅读器未能打开：\n{output_path}",
            )

    def preview_pdf(self) -> None:
        cache_root = Path(
            os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        ) / "mdview"
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            stem = self.current_path.stem if self.current_path else "document"
            output_path = cache_root / f"{stem}-preview.pdf"
            backend = self.build_current_pdf(output_path)
        except (OSError, RuntimeError) as error:
            QMessageBox.critical(self, "PDF 预览失败", str(error))
            return
        self.statusBar().showMessage(f"已通过 {backend} 生成 PDF 预览", 10_000)
        if not open_local_file(output_path):
            QMessageBox.warning(
                self,
                "无法打开 PDF",
                f"临时 PDF 已生成，但系统阅读器未能打开：\n{output_path}",
            )

    def open_file_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "打开文档",
            str(self.current_path.parent if self.current_path else Path.home()),
            "支持的文档 (*.md *.markdown *.tex *.latex *.ltx);;"
            "Markdown 文档 (*.md *.markdown);;"
            "LaTeX 文档 (*.tex *.latex *.ltx);;文本文件 (*.txt);;所有文件 (*)",
        )
        if selected:
            self.load_file(Path(selected))

    def load_file(self, path: Path) -> bool:
        try:
            source = path.expanduser().read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            QMessageBox.critical(self, "无法打开文件", f"{path}\n\n{error}")
            return False
        self.current_path = path.expanduser().resolve()
        self.document_mode = (
            "latex"
            if self.current_path.suffix.casefold() in {".tex", ".latex", ".ltx"}
            else "markdown"
        )
        self.render_timer.stop()
        self.render_timer.setInterval(900 if self.document_mode == "latex" else 120)
        self.editor.blockSignals(True)
        self.editor.setPlainText(source)
        self.editor.blockSignals(False)
        latex_mode = self.document_mode == "latex"
        self.line_height_action.setEnabled(not latex_mode)
        self.toc_action.setEnabled(True)
        if self.toc_action.isChecked():
            self.toc_dock.show()
        self.setWindowTitle(f"{self.current_path.name} — {APP_NAME}")
        self.path_button.setText(str(self.current_path))
        self.path_button.setToolTip("点击复制当前文件的完整路径")
        self.refresh_preview()
        return True

    def refresh_preview(self) -> None:
        source = self.editor.toPlainText()
        if self.document_mode == "latex":
            self.refresh_latex_preview(source)
            return
        if self.current_path is not None:
            self.preview.document().setBaseUrl(
                QUrl.fromLocalFile(f"{self.current_path.parent}/")
            )
        self.preview.setHtml(
            render_markdown(source, self.background_color, self.line_height)
        )
        self.refresh_toc(source)

    def refresh_latex_preview(self, source: str) -> None:
        cache_root = self.preview_cache_path
        pdf_path = cache_root / "latex-preview.pdf"
        toc_path = cache_root / "latex-preview.toc"
        pages_directory = cache_root / "latex-pages"
        if toc_path.exists():
            toc_path.unlink()
        try:
            backend = compile_latex_document(
                source,
                pdf_path,
                base_directory=self.current_path.parent if self.current_path else Path.cwd(),
                toc_output_path=toc_path,
            )
            preview_html = render_pdf_pages_html(
                pdf_path,
                pages_directory,
                self.background_color,
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            self.latest_tex_pdf_path = None
            self.toc_tree.clear()
            self.preview.setHtml(
                "<!doctype html><meta charset=\"utf-8\"><style>"
                "body{font-family:'Noto Sans CJK SC',sans-serif;margin:28px;}"
                ".error{white-space:pre-wrap;color:#b91c1c;background:#fff7ed;"
                "border:1px solid #fdba74;border-radius:8px;padding:16px;}"
                "</style><div class=\"error\"><b>LaTeX 编译失败</b>\n\n"
                + html.escape(str(error))
                + "</div>"
            )
            self.statusBar().showMessage("LaTeX 编译失败；错误已显示在预览区", 10_000)
            return
        self.latest_tex_pdf_path = pdf_path
        self.preview.document().setBaseUrl(QUrl.fromLocalFile(f"{pages_directory}/"))
        page_images = sorted(
            pages_directory.glob("page-*.png"),
            key=lambda path: int(re.search(r"-(\d+)\.png$", path.name).group(1)),
        )
        self.preview.set_pdf_document(
            preview_html,
            page_images,
            extract_pdf_text_layout(pdf_path),
        )
        self.refresh_latex_toc(toc_path, pdf_path)
        self.statusBar().showMessage(f"已通过 {backend} 更新完整 TeX 预览", 5000)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="打开原生 Markdown/LaTeX 原文与渲染预览窗口。"
    )
    parser.add_argument(
        "file", nargs="?", type=Path, help="启动时打开的 Markdown 或 LaTeX 文件"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    signal_heartbeat = QTimer()
    signal_heartbeat.setInterval(200)
    signal_heartbeat.timeout.connect(lambda: None)
    signal_heartbeat.start()
    window = MarkdownWindow(args.file)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

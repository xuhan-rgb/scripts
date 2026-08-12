import os
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRectF, QSettings, QSizeF, Qt
from PyQt5.QtGui import QImage, QPainter, QTextDocument
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QSplitter,
    QTreeWidgetItem,
)

from markdown_editor import (
    ChromeDebugTarget,
    ChromeTargetDownloadCapture,
    MarkdownPreview,
    MarkdownWindow,
    PdfExportDialog,
    RemoteChromeSession,
    X11WindowInfo,
    X11WindowController,
    ascii_flow_to_mermaid,
    build_chatgpt_edit_prompt,
    chatgpt_performance_script,
    compile_latex_document,
    configure_latex_toc,
    chrome_download_directory,
    create_chatgpt_browser,
    extract_pdf_text_layout,
    extract_toc_entries,
    extract_latex_toc_entries,
    export_pdf,
    normalize_math_markup,
    new_chatgpt_target,
    open_local_file,
    pandoc_pdf_command,
    find_remote_chrome,
    parse_x11_client_ids,
    pdf_destination_targets,
    prepare_markdown_for_pdf,
    remote_chrome_app_command,
    x11_window_matches_session,
    render_mermaid_data_url,
    render_math_data_url,
    render_markdown,
    render_pdf_pages_html,
    render_tikz_data_url,
    save_preview_image,
    strip_front_matter,
)


SAMPLE = r"""---
title: 自动驾驶 Latent World Model 深度解析：LAW、VLA 与稀疏 LiDAR World
  Model
---

# LAW 世界模型核心定位

> 使用当前场景 latent 预测未来场景 latent。

``` text
V_t -> World Model -> V_(t+1)
```

\[ A_t^i=\operatorname{MLP}([v_t^i,\operatorname{vec}(W_t)]) \]
"""


class MarkdownRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_front_matter_is_not_rendered_as_a_heading(self):
        body = strip_front_matter(SAMPLE)
        self.assertTrue(body.startswith("# LAW"))
        self.assertNotIn("title:", body)

        html = render_markdown(SAMPLE)
        self.assertIn(">LAW 世界模型核心定位</h1>", html)
        self.assertNotIn("<h2>Model</h2>", html)

    def test_renders_sample_quote_and_fenced_code(self):
        html = render_markdown(SAMPLE)
        self.assertIn("<blockquote>", html)
        self.assertIn('<pre class="code-block language-text">', html)
        self.assertNotIn("<pre><code", html)
        self.assertIn("V_t -&gt; World Model -&gt; V_(t+1)", html)

    def test_preview_uses_a_centered_reading_layout(self):
        html = render_markdown(
            "## 核心问题\n\n正文说明。\n\n1. 第一个问题\n2. 第二个问题"
        )

        self.assertIn('<main class="markdown-body">', html)
        self.assertIn('font-family: "Noto Serif CJK SC"', html)
        self.assertIn("max-width: 900px", html)
        self.assertIn("margin: 20px 56px 48px", html)
        self.assertIn("padding: 4px 0 0", html)
        self.assertIn("line-height: 1.45", html)
        self.assertIn("li { margin: 0.2em 0;", html)
        self.assertNotIn("h2 { border-bottom:", html)

    def test_preview_line_height_is_configurable(self):
        html = render_markdown("正文", line_height=1.42)

        self.assertIn("line-height: 1.42", html)

    def test_preview_keeps_word_like_side_margins(self):
        html = render_markdown("# 标题\n\n正文")

        self.assertIn("margin: 20px 56px 48px", html)
        self.assertIn("padding: 4px 0 0", html)

    def test_preview_matches_pdf_section_numbering(self):
        html = render_markdown(
            "---\nnumbersections: true\n---\n\n"
            "# 阅读说明\n\n# 1. Executive Summary\n\n## 1.1 方法\n"
        )

        self.assertIn(">1 阅读说明</h1>", html)
        self.assertIn(">2 Executive Summary</h1>", html)
        self.assertIn(">2.1 方法</h2>", html)
        self.assertNotIn("2 1. Executive Summary", html)

    def test_report_metadata_builds_a_cover_without_duplicate_inline_toc(self):
        source = '''---
subtitle: 基于 LAW（ICLR 2025）及相关 World-Model Planning
  工作的系统整理
title: 自动驾驶世界模型：从辅助训练到在线规划
---

# 自动驾驶世界模型：从“辅助训练”到“在线规划”

## 核心问题

预测出来的未来，在系统中到底拿来做什么？

本报告回答三个问题。

1. 训练
2. 在线规划

---

# 1. Executive Summary

正文。

## 1.1 三类路线
'''
        html = render_markdown(source)

        self.assertIn('class="report-cover"', html)
        self.assertIn('class="report-title"', html)
        self.assertIn("基于 LAW（ICLR 2025）及相关 World-Model Planning 工作的系统整理", html)
        self.assertIn('class="cover-question"', html)
        self.assertIn("预测出来的未来，在系统中到底拿来做什么？", html)
        self.assertNotIn('class="report-toc"', html)
        self.assertNotIn("<h2>目录</h2>", html)
        self.assertIn(">1. Executive Summary</h1>", html)
        self.assertIn(">1.1 三类路线</h2>", html)
        self.assertNotIn('<h1>自动驾驶世界模型：从“辅助训练”到“在线规划”</h1>', html)

    def test_renders_display_latex_as_an_image(self):
        html = render_markdown(SAMPLE)
        self.assertIn('class="math-display"', html)
        self.assertIn('src="data:image/png;base64,', html)
        self.assertNotIn(r"\operatorname{MLP}", html)

    def test_normalizes_pandoc_raw_tex_inside_display_math(self):
        source = r"\[ (V_t, W_t) `\rightarrow `{=tex}`\hat{V}`{=tex}\_{t+`\Delta`{=tex}} \]"
        normalized = normalize_math_markup(source)
        self.assertIn(r"(V_t, W_t) \rightarrow \hat{V}_{t+\Delta}", normalized)
        self.assertNotIn("{=tex}", normalized)

    def test_normalizes_pandoc_escaped_math_operator(self):
        normalized = normalize_math_markup(r"\[ W\^\* \]")

        self.assertIn(r"W^*", normalized)
        self.assertNotIn(r"\*", normalized)

    def test_normalizes_nested_brackets_and_named_math_operators(self):
        normalized = normalize_math_markup(
            r"\[ A_t=MLP(\[V_t,vec(W_t)\]) \]"
        )

        self.assertIn(
            r"A_t=\operatorname{MLP}([V_t,\operatorname{vec}(W_t)])",
            normalized,
        )
        self.assertEqual(normalized.count("$$"), 2)

    def test_renders_multiline_and_chinese_math_without_fallback(self):
        self.assertIsNotNone(
            render_math_data_url(
                "V_t + W_t \\rightarrow A_t\n\\rightarrow \\hat V_{t+1}"
            )
        )
        self.assertIsNotNone(
            render_math_data_url(
                "\\boxed{更好的 latent \\rightarrow 更好的 Planner}"
            )
        )

    def test_math_image_background_is_transparent_for_dark_themes(self):
        data_url = render_math_data_url(r"V_t\rightarrow W_t", "#f8fafc")
        self.assertIsNotNone(data_url)
        image = QImage.fromData(base64.b64decode(data_url.split(",", 1)[1]))
        self.assertFalse(image.isNull())
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)

    def test_renders_extended_latex_and_inline_math(self):
        formulas = (
            r"I_t \xrightarrow{Encoder} V_t \xrightarrow{Planner} W_t",
            r"\mathcal L_{\rm latent}=\operatorname{MSE}(\hat V,V)",
            r"V_t\in\mathbb R^{B\times36\times256}",
            r"\frac1M\left\|w_t-\tilde w_t\right\|_1",
        )
        for formula in formulas:
            with self.subTest(formula=formula):
                self.assertIsNotNone(render_math_data_url(formula))

        html = render_markdown(r"当前特征 \(V_t\) 进入规划器。")
        self.assertIn('class="math-inline"', html)
        self.assertNotIn(r"\(V_t\)", html)

        list_html = render_markdown("- (V_t)：当前场景 latent")
        self.assertIn('class="math-inline"', list_html)
        self.assertNotIn("(V_t)", list_html)

    def test_renders_dollar_delimited_inline_math(self):
        html = render_markdown(
            r"- $W_t$：ego 运动\n- $D_t$：其他车辆\n- $\epsilon_t$：随机因素"
        )

        self.assertEqual(html.count('class="math-inline"'), 3)
        self.assertNotIn("$W_t$", html)
        self.assertNotIn(r"$\epsilon_t$", html)

    def test_removes_unusable_chatgpt_citation_artifacts_from_preview(self):
        html = render_markdown("结论。\ue200cite\ue202turn2view0\ue202turn3view1\ue201")
        self.assertIn("结论。", html)
        self.assertNotIn("turn2view0", html)
        self.assertNotIn("\ue200cite", html)

    def test_renders_mermaid_flowchart_as_an_image(self):
        source = """```mermaid
flowchart LR
    A["输入"]
    B["输出"]
    A --> B
```"""
        html = render_markdown(source)
        self.assertIn('class="mermaid-diagram"', html)
        self.assertIn('src="data:image/png;base64,', html)
        self.assertNotIn("flowchart LR", html)
        data_url = render_mermaid_data_url("flowchart LR\nA[输入] --> B[输出]")
        image = QImage.fromData(base64.b64decode(data_url.split(",", 1)[1]))
        self.assertFalse(image.isNull())

    def test_renders_mermaid_sequence_diagram_as_an_image(self):
        source = """```mermaid
sequenceDiagram
    participant A as Client
    participant B as Server
    A->>B: Request
    B-->>A: Response
```"""
        html = render_markdown(source)
        self.assertIn('class="mermaid-diagram"', html)
        self.assertNotIn("sequenceDiagram", html)

    def test_renders_tikz_and_visual_tex_blocks_as_images(self):
        tikz = r"""\begin{tikzpicture}[node distance=12mm]
\node[draw, rounded corners] (a) {当前 latent $V_t$};
\node[draw, right=of a] (b) {Planner $P_\phi$};
\draw[-{Stealth}] (a) -- (b);
\end{tikzpicture}"""
        for language in ("tikz", "tex"):
            html = render_markdown(f"```{language}\n{tikz}\n```")
            self.assertIn('class="tikz-diagram"', html)
            self.assertIn('src="data:image/png;base64,', html)
            self.assertNotIn("language-tikz", html)
            self.assertNotIn("language-tex", html)

        data_url = render_tikz_data_url(tikz)
        self.assertIsNotNone(data_url)
        image = QImage.fromData(base64.b64decode(data_url.split(",", 1)[1]))
        self.assertFalse(image.isNull())

    def test_non_visual_tex_block_remains_readable_source(self):
        html = render_markdown("```tex\nE = mc^2\n```")

        self.assertIn('class="code-block language-tex"', html)
        self.assertIn("E = mc^2", html)

        full_document = render_markdown(
            "```tex\n\\begin{document}\n\\begin{tikzpicture}"
            "\\node {x};\\end{tikzpicture}\n\\end{document}\n```"
        )
        self.assertIn('class="code-block language-tex"', full_document)

    def test_converts_recognizable_ascii_flow_to_a_diagram(self):
        source = """```text
Sensor
  ↓
Encoder
  ↓
Controller
```"""
        html = render_markdown(source)
        self.assertIn('class="mermaid-diagram"', html)
        self.assertNotIn('class="code-block language-text"', html)

    def test_formats_ascii_flow_math_labels(self):
        mermaid = ascii_flow_to_mermaid(
            """未来帧 I_(t+1)
  |
  v
stopgrad(V_(t+1))
  |
  v
MSE Loss
"""
        )

        self.assertIsNotNone(mermaid)
        self.assertIn("I₍ₜ₊₁₎", mermaid)
        self.assertIn("stopgrad(V₍ₜ₊₁₎)", mermaid)
        self.assertNotIn("V_(t+1)", mermaid)

    def test_converts_law_training_ascii_flow_to_a_diagram(self):
        source = """当前帧分支

I_t
 │
 ▼
Encoder E_θ
 │
 ▼
V_t ──────► Planner P_φ(V_t,G_t,S_t) ──────► W_t
 │                                                         │
 └───────────────────────────────┴───────────────┘
                         ▼
         Action-aware latent construction
         A_t = MLP([V_t, vec(W_t)])
                         │
                         ▼
            Latent World Model F_ψ
                         │
                         ▼
                 predicted latent
                    V̂_{t+1}
                         │
                         ▼
                        MSE
                         ▲
                         │
              stopgrad(V_{t+1})
                         ▲
                         │
                  Encoder E_θ
                         ▲
                         │
              真实未来帧 I_{t+1}
"""

        mermaid = ascii_flow_to_mermaid(source)
        self.assertIsNotNone(mermaid)
        self.assertIn("Action-aware latent", mermaid)
        self.assertIn("stopgrad(V₍ₜ₊₁₎)", mermaid)
        self.assertIsNotNone(render_mermaid_data_url(mermaid))

    def test_extracts_three_level_toc_entries_from_report_body(self):
        entries = extract_toc_entries(
            """---
title: 测试报告
---

# 第一章
## 1.1 方法
### 1.1.1 数据
#### 不进入目录
"""
        )

        self.assertEqual(
            [(level, title) for level, title, _anchor in entries],
            [(1, "第一章"), (2, "1.1 方法"), (3, "1.1.1 数据")],
        )
        self.assertEqual(entries[0][2], "_1")

    def test_extracts_latex_toc_entries_with_physical_page_anchors(self):
        toc_source = r"""\contentsline {section}{\numberline {1}Executive Summary}{2}{section.1}%
\contentsline {subsection}{\numberline {1.1}方法 \(W_t\)}{3}{subsection.1.1}%
\contentsline {subsubsection}{\numberline {1.1.1}数据}{4}{subsubsection.1.1.1}%
"""
        entries = extract_latex_toc_entries(
            toc_source,
            {
                "section.1": (5, 0.20),
                "subsection.1.1": (6, 0.45),
                "subsubsection.1.1.1": (7, 0.70),
            },
        )

        self.assertEqual(
            entries,
            [
                (1, "1 Executive Summary", "pdf-page-5", 0.20),
                (2, "1.1 方法 Wₜ", "pdf-page-6", 0.45),
                (3, "1.1.1 数据", "pdf-page-7", 0.70),
            ],
        )

    def test_reads_pdf_destination_page_and_vertical_position(self):
        page_info = mock.Mock(
            returncode=0,
            stdout=b"Page size:       595.28 x 841.89 pts (A4)\n",
            stderr=b"",
        )
        destinations = mock.Mock(
            returncode=0,
            stdout=(
                b'  10 [ XYZ   68   57 null ] "subsection.2.6"\n'
                b'  11 [ Fit              ] "section.3"\n'
            ),
            stderr=b"",
        )
        with mock.patch("markdown_editor.shutil.which", return_value="/usr/bin/pdfinfo"):
            with mock.patch(
                "markdown_editor.subprocess.run",
                side_effect=[page_info, destinations],
            ):
                targets = pdf_destination_targets(Path("/tmp/report.pdf"))

        self.assertEqual(targets["section.3"], (11, 0.0))
        self.assertEqual(targets["subsection.2.6"][0], 10)
        self.assertAlmostEqual(
            targets["subsection.2.6"][1],
            (841.89 - 57.0) / 841.89,
        )

    def test_toc_formats_short_math_as_unicode(self):
        entries = extract_toc_entries(
            "# 3.2 $W_t$ 准确，不代表 future latent 一定准确\n"
        )

        self.assertEqual(
            entries[0][1],
            "3.2 Wₜ 准确，不代表 future latent 一定准确",
        )
        self.assertNotIn("$", entries[0][1])

    def test_pdf_export_requires_xelatex_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.pdf"
            with mock.patch(
                "markdown_editor.shutil.which",
                side_effect=lambda command: (
                    "/usr/bin/pandoc" if command == "pandoc" else None
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "XeLaTeX"):
                    export_pdf(SAMPLE, output)

            self.assertFalse(output.exists())

    def test_compiles_full_latex_document_and_renders_real_pdf_pages(self):
        source = r"""\documentclass[UTF8,a4paper]{ctexart}
\begin{document}
\section{完整 TeX 预览}
公式：$V_t \rightarrow W_t$。
\end{document}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "document.pdf"
            backend = compile_latex_document(source, output, base_directory=root)
            html = render_pdf_pages_html(output, root / "pages")

            self.assertEqual(backend, "XeLaTeX")
            self.assertTrue(output.is_file())
            self.assertIn('class="pdf-document"', html)
            self.assertIn('class="pdf-page"', html)
            self.assertIn('class="pdf-page-image"', html)
            self.assertNotIn("第 1 页", html)
            page_images = list((root / "pages").glob("page-*.png"))
            self.assertTrue(page_images)
            document = QTextDocument()
            document.setHtml(html)
            document.setPageSize(QSizeF(1100, 1800))
            rendered_page = QImage(1100, 1600, QImage.Format_ARGB32)
            rendered_page.fill(Qt.white)
            painter = QPainter(rendered_page)
            document.drawContents(painter, QRectF(0, 0, 1100, 1600))
            painter.end()
            dark_samples = sum(
                rendered_page.pixelColor(x, y).lightness() < 180
                for y in range(0, rendered_page.height(), 4)
                for x in range(0, rendered_page.width(), 4)
            )
            self.assertGreater(dark_samples, 20)
            preview = MarkdownPreview()
            preview.image_output_directory = root
            text_pages = extract_pdf_text_layout(output)
            preview.set_pdf_document(html, page_images, text_pages)
            preview.resize(1100, 800)
            preview.show()
            QApplication.processEvents()
            title_ordinals = [
                ordinal
                for ordinal, (page_index, word_index) in enumerate(preview._pdf_words)
                if "完整" in text_pages[page_index][2][word_index][4]
            ]
            self.assertTrue(title_ordinals)
            title_page, title_word = preview._pdf_words[title_ordinals[0]]
            title_line = text_pages[title_page][2][title_word][5]
            title_line_ordinals = [
                ordinal
                for ordinal, (page_index, word_index) in enumerate(preview._pdf_words)
                if page_index == title_page
                and text_pages[page_index][2][word_index][5] == title_line
            ]
            title_center = preview._pdf_word_document_rect(
                title_line_ordinals[0]
            ).center().toPoint()
            self.assertEqual(
                preview._pdf_word_at(title_center),
                title_line_ordinals[0],
            )
            preview._pdf_selection_start = title_line_ordinals[0]
            preview._pdf_selection_end = title_line_ordinals[-1]
            preview.copy_selection_as_text()
            copied = QApplication.clipboard().text().replace("\xa0", " ")
            self.assertEqual(copied.casefold(), "完整 tex 预览")
            selection_rects = preview._pdf_selection_rects()
            self.assertEqual(len(selection_rects), len(title_line_ordinals))
            self.assertTrue(all(rect.width() > 1 for rect in selection_rects))
            self.assertFalse(preview.textCursor().hasSelection())
            saved_paths = []
            preview.image_saved.connect(saved_paths.append)
            preview.copy_selection_as_image()
            self.assertFalse(QApplication.clipboard().image().isNull())
            self.assertEqual(len(saved_paths), 1)
            self.assertTrue(Path(saved_paths[0]).is_file())
            anchor_names = []
            block = document.begin()
            while block.isValid():
                iterator = block.begin()
                while not iterator.atEnd():
                    fragment = iterator.fragment()
                    if fragment.isValid():
                        anchor_names.extend(fragment.charFormat().anchorNames())
                    iterator += 1
                block = block.next()
            self.assertIn("pdf-page-1", anchor_names)

    def test_saves_copied_preview_image_with_a_unique_tmp_name(self):
        with tempfile.TemporaryDirectory() as directory:
            image = QImage(40, 24, QImage.Format_ARGB32)
            image.fill(Qt.red)

            first = save_preview_image(image, Path(directory))
            second = save_preview_image(image, Path(directory))

            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertNotEqual(first, second)
            self.assertRegex(first.name, r"^mdview-selection-.*\.png$")
            loaded = QImage(str(first))
            self.assertEqual(loaded.size(), image.size())

    def test_configures_full_latex_toc_on_the_compiled_copy(self):
        source_without_toc = (
            "\\documentclass{article}\n"
            "\\begin{document}\n正文\n\\end{document}\n"
        )
        enabled = configure_latex_toc(source_without_toc, True)

        self.assertIn(r"\tableofcontents", enabled)
        self.assertIn(r"\clearpage", enabled)
        self.assertEqual(source_without_toc.count(r"\tableofcontents"), 0)

        source_with_toc = enabled.replace(
            r"\tableofcontents", r"\tableofcontents % 正文目录"
        )
        disabled = configure_latex_toc(source_with_toc, False)

        self.assertNotIn(r"\tableofcontents", disabled)
        self.assertIn("正文", disabled)

    def test_builds_requested_pandoc_xelatex_pipeline(self):
        command = pandoc_pdf_command(
            Path("/tmp/output.pdf"),
            Path("/tmp/source"),
            "xelatex",
        )
        self.assertEqual(command[0], "pandoc")
        self.assertIn("--pdf-engine=xelatex", command)
        self.assertIn("--from=markdown+tex_math_dollars", command)
        self.assertTrue(
            any(argument.startswith("--template=") for argument in command)
        )
        self.assertTrue(
            any(argument.startswith("--lua-filter=") for argument in command)
        )
        self.assertNotIn("--toc", command)
        self.assertIn("--metadata=toc=false", command)
        self.assertIn("--toc-depth=3", command)
        self.assertIn("--output=/tmp/output.pdf", command)

        command_with_toc = pandoc_pdf_command(
            Path("/tmp/output.pdf"),
            Path("/tmp/source"),
            "xelatex",
            include_toc=True,
        )
        self.assertIn("--toc", command_with_toc)
        self.assertNotIn("--metadata=toc=false", command_with_toc)

    def test_prepares_mermaid_as_a_pdf_asset(self):
        tiny_png = b"\x89PNG\r\n\x1a\n"
        data_url = "data:image/png;base64," + base64.b64encode(tiny_png).decode()
        source = """---
title: 测试报告
---

```mermaid
flowchart LR
  A --> B
```

```python
print("keep me")
```
"""
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "assets"
            with mock.patch(
                "markdown_editor.render_mermaid_data_url", return_value=data_url
            ):
                prepared = prepare_markdown_for_pdf(source, assets)

            self.assertTrue(prepared.startswith("---\ntitle: 测试报告"))
            self.assertIn("diagram-001.png)", prepared)
            self.assertNotIn("![流程图]", prepared)
            self.assertNotIn("flowchart LR", prepared)
            self.assertIn('print("keep me")', prepared)
            self.assertEqual((assets / "diagram-001.png").read_bytes(), tiny_png)

    def test_pdf_preparation_preserves_tikz_as_vector_latex(self):
        source = r"""```tikz
\begin{tikzpicture}
\node[draw] (a) {$V_t$};
\node[draw, right=of a] (b) {$W_t$};
\draw[-{Stealth}] (a) -- (b);
\end{tikzpicture}
```"""
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "assets"
            prepared = prepare_markdown_for_pdf(source, assets)

            self.assertIn(r"\begin{tikzpicture}", prepared)
            self.assertIn(r"\draw[-{Stealth}]", prepared)
            self.assertIn(r"\begin{center}", prepared)
            self.assertNotIn("```tikz", prepared)
            self.assertEqual(list(assets.iterdir()), [])

    def test_pdf_preparation_moves_report_preamble_to_the_title_page(self):
        source = '''---
subtitle: 技术报告
title: 自动驾驶世界模型
---

# 自动驾驶世界模型

## 核心问题

预测出来的未来拿来做什么？

本报告回答三个问题。

---

# 1. Executive Summary

## 1.1 三类路线
'''
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_markdown_for_pdf(source, Path(directory) / "assets")

        self.assertIn('question: "预测出来的未来拿来做什么？"', prepared)
        self.assertIn('report-type: "研究笔记 / 技术报告"', prepared)
        self.assertIn('header-left: "自动驾驶世界模型"', prepared)
        self.assertIn("# Executive Summary", prepared)
        self.assertIn("## 三类路线", prepared)
        self.assertNotIn("# 1. Executive Summary", prepared)
        self.assertNotIn("## 核心问题", prepared)
        self.assertNotIn("本报告回答三个问题", prepared)

    def test_pdf_preparation_converts_parenthesized_list_symbols_to_math(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_markdown_for_pdf(
                "- (s_t)：当前状态\n- (a_t)：动作\n",
                Path(directory) / "assets",
            )

        self.assertIn("- $s_t$：当前状态", prepared)
        self.assertIn("- $a_t$：动作", prepared)

    def test_open_pdf_uses_the_system_default_application(self):
        path = Path("/tmp/output.pdf")
        with mock.patch(
            "markdown_editor.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            self.assertTrue(open_local_file(path))

        self.assertEqual(open_url.call_args.args[0].toLocalFile(), str(path))

    def test_window_hides_source_by_default_and_can_show_it(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)
            window.editor.setPlainText(SAMPLE)
            window.refresh_preview()

            splitter = window.centralWidget()
            self.assertIsInstance(splitter, QSplitter)
            self.assertEqual(splitter.orientation(), 1)
            self.assertIs(splitter.widget(0), window.editor)
            self.assertIs(splitter.widget(1), window.preview)
            self.assertFalse(window.editor.isVisibleTo(window))
            self.assertEqual(window.source_action.text(), "显示原文")
            self.assertEqual(window.export_action.text(), "导出 PDF")
            self.assertEqual(window.preview_pdf_action.text(), "预览 PDF")
            self.assertEqual(window.chatgpt_action.text(), "ChatGPT")
            self.assertEqual(window.settings_button.text(), "设置")
            toolbar_texts = [action.text() for action in window.toolBar.actions()]
            self.assertNotIn("边框颜色", toolbar_texts)
            self.assertNotIn("背景颜色", toolbar_texts)
            self.assertNotIn("行间距：1.45", toolbar_texts)

            settings_texts = [action.text() for action in window.settings_menu.actions()]
            self.assertFalse(any("TOC" in text for text in settings_texts))

            window.source_action.trigger()

            self.assertFalse(window.editor.isHidden())
            self.assertEqual(window.source_action.text(), "隐藏原文")
            self.assertEqual(window.editor.toPlainText(), SAMPLE)
            self.assertIn("LAW 世界模型核心定位", window.preview.toPlainText())

    def test_chatgpt_panel_is_lazy_and_opens_left_of_the_toc(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)
            browser = QLabel("ChatGPT browser")
            browser.shutdown = mock.Mock()

            self.assertTrue(window.chatgpt_dock.isHidden())
            with mock.patch(
                "markdown_editor.create_chatgpt_browser",
                return_value=browser,
            ) as create_browser:
                window.chatgpt_action.trigger()

                self.assertFalse(window.chatgpt_dock.isHidden())
                self.assertTrue(window.chatgpt_action.isChecked())
                self.assertEqual(window.chatgpt_action.text(), "隐藏 ChatGPT")
                self.assertIs(window.chatgpt_dock.widget(), browser)
                self.assertEqual(
                    window.dockWidgetArea(window.chatgpt_dock),
                    Qt.LeftDockWidgetArea,
                )
                self.assertEqual(
                    window.dockWidgetArea(window.toc_dock),
                    Qt.LeftDockWidgetArea,
                )

                window.chatgpt_action.trigger()
                self.assertTrue(window.chatgpt_dock.isHidden())
                self.assertEqual(window.chatgpt_action.text(), "ChatGPT")
                browser.shutdown.assert_called_once()

                window.chatgpt_action.trigger()

            self.assertEqual(create_browser.call_count, 2)
            self.assertIs(window.chatgpt_dock.widget(), browser)

    def test_chatgpt_panel_reports_missing_remote_chrome(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)
            with mock.patch(
                "markdown_editor.create_chatgpt_browser",
                side_effect=OSError("没有远程 Chrome"),
            ):
                with mock.patch.object(QMessageBox, "warning") as warning:
                    window.chatgpt_action.trigger()

            self.assertTrue(window.chatgpt_dock.isHidden())
            self.assertFalse(window.chatgpt_action.isChecked())
            warning.assert_called_once()

    def test_closing_the_chatgpt_dock_releases_its_browser_target(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(
                str(Path(directory) / "settings.ini"),
                QSettings.IniFormat,
            )
            window = MarkdownWindow(settings=settings)
            window.show()
            QApplication.processEvents()
            browser = QLabel("ChatGPT browser")
            browser.shutdown = mock.Mock()
            with mock.patch(
                "markdown_editor.create_chatgpt_browser",
                return_value=browser,
            ):
                window.chatgpt_action.trigger()

            window.chatgpt_dock.hide()

            browser.shutdown.assert_called_once()
            self.assertIsNone(window.chatgpt_view)
            self.assertEqual(window.chatgpt_action.text(), "ChatGPT")

    def test_remote_chrome_discovery_reuses_the_browser_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            browser_proc = proc_root / "100"
            renderer_proc = proc_root / "101"
            browser_proc.mkdir()
            renderer_proc.mkdir()
            browser_proc.joinpath("cmdline").write_bytes(
                b"/opt/google/chrome/chrome "
                b"--user-data-dir=/home/qwer/.config/google-chrome-shared "
                b"--profile-directory=Default "
                b"--remote-debugging-address=127.0.0.1 "
                b"--remote-debugging-port=9223\0"
            )
            renderer_proc.joinpath("cmdline").write_bytes(
                b"/opt/google/chrome/chrome\0"
                b"--type=renderer\0"
                b"--remote-debugging-port=9223\0"
            )

            session = find_remote_chrome(9223, proc_root)

        self.assertIsNotNone(session)
        self.assertEqual(session.executable, "/opt/google/chrome/chrome")
        self.assertEqual(
            session.user_data_dir,
            "/home/qwer/.config/google-chrome-shared",
        )
        self.assertEqual(session.profile_directory, "Default")
        self.assertEqual(session.debug_port, 9223)
        self.assertEqual(session.pid, 100)
        self.assertEqual(
            remote_chrome_app_command(session),
            [
                "/opt/google/chrome/chrome",
                "--user-data-dir=/home/qwer/.config/google-chrome-shared",
                "--profile-directory=Default",
                "--remote-debugging-port=9223",
                "--new-window",
                "--app=https://chatgpt.com/",
            ],
        )

    def test_chrome_download_directory_uses_the_remote_profile_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "chrome" / "Default"
            download_directory = root / "ChatGPT Downloads"
            profile.mkdir(parents=True)
            profile.joinpath("Preferences").write_text(
                '{"download": {"default_directory": "'
                + str(download_directory)
                + '"}}',
                encoding="utf-8",
            )
            session = RemoteChromeSession(
                executable="/opt/google/chrome/chrome",
                user_data_dir=str(root / "chrome"),
                profile_directory="Default",
                debug_port=9223,
                pid=100,
            )

            self.assertEqual(
                chrome_download_directory(session),
                download_directory,
            )

    def test_remote_chrome_window_match_rejects_auxiliary_windows(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        valid = X11WindowInfo(
            window_id=0x200,
            wm_class='"chatgpt.com", "Google-chrome"',
            pid=100,
            width=800,
            height=900,
            is_viewable=True,
        )
        self.assertTrue(x11_window_matches_session(valid, session))
        self.assertFalse(
            x11_window_matches_session(
                X11WindowInfo(0x201, valid.wm_class, 100, 200, 200, False),
                session,
            )
        )
        self.assertFalse(
            x11_window_matches_session(
                X11WindowInfo(0x202, '"chatgpt", "Chatgpt"', 200, 800, 900, True),
                session,
            )
        )

    def test_parses_x11_client_window_ids(self):
        output = (
            "_NET_CLIENT_LIST(WINDOW): window id # "
            "0x3600007, 0x4000012, 0x4000012\n"
        )
        self.assertEqual(
            parse_x11_client_ids(output),
            {0x3600007, 0x4000012},
        )

    def test_embedded_chrome_detects_the_new_remote_window(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        x11_connection = mock.Mock()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x100},
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=x11_connection,
                    known_debug_targets=[],
                )
        browser.poll_timer.stop()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x100, 0x200},
        ):
            with mock.patch(
                "markdown_editor.read_x11_window_info",
                return_value=X11WindowInfo(
                    0x200,
                    '"chatgpt.com", "Google-chrome"',
                    100,
                    800,
                    900,
                    True,
                ),
            ):
                with mock.patch.object(browser, "attach_window") as attach:
                    for _ in range(browser.ATTACH_STABILITY_POLLS):
                        browser.attach_new_chrome_window()

        attach.assert_called_once_with(0x200)
        browser.close()

    def test_embedded_chrome_scans_existing_windows_in_small_batches(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        existing_windows = set(range(0x100, 0x164))
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value=existing_windows,
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=mock.Mock(),
                    known_debug_targets=[],
                )
        browser.poll_timer.stop()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value=existing_windows,
        ):
            with mock.patch(
                "markdown_editor.read_x11_window_info",
                return_value=None,
            ) as read_info:
                browser.attach_new_chrome_window()

        self.assertLessEqual(
            read_info.call_count,
            browser.WINDOW_SCAN_BATCH_SIZE,
        )
        browser.close()

    def test_embedded_chrome_can_adopt_an_existing_chatgpt_app_window(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x200},
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=mock.Mock(),
                    known_debug_targets=[],
                )
        browser.poll_timer.stop()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x200},
        ):
            with mock.patch(
                "markdown_editor.read_x11_window_info",
                return_value=X11WindowInfo(
                    0x200,
                    '"chatgpt.com", "Google-chrome"',
                    100,
                    460,
                    688,
                    True,
                ),
            ):
                with mock.patch.object(browser, "attach_window") as attach:
                    for _ in range(browser.ATTACH_STABILITY_POLLS):
                        browser.attach_new_chrome_window()

        attach.assert_called_once_with(0x200)
        browser.close()

    def test_embedded_chrome_reparents_into_the_native_qt_widget(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        x11_connection = mock.Mock()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x100},
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=x11_connection,
                    known_debug_targets=[],
                )
        browser.resize(640, 720)
        browser.attach_window(0x200)

        x11_connection.reparent.assert_called_once_with(
            0x200,
            int(browser.winId()),
            640,
            720,
        )
        self.assertEqual(browser.window_id, 0x200)
        browser.shutdown()
        x11_connection.destroy.assert_called_once_with(0x200)
        x11_connection.close.assert_called_once()

    def test_embedded_chrome_recaptures_a_window_that_escaped_to_root(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        x11_connection = mock.Mock()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x100},
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=x11_connection,
                    known_debug_targets=[],
                )
        browser.poll_timer.stop()
        browser.window_id = 0x200
        x11_connection.parent_window_id.return_value = 0x204

        browser.verify_embedded_window()

        x11_connection.reparent.assert_called_once_with(
            0x200,
            int(browser.winId()),
            browser.width(),
            browser.height(),
        )
        browser.window_id = None
        browser.close()

    def test_new_chatgpt_target_excludes_existing_and_regular_browser_pages(self):
        existing = ChromeDebugTarget(
            target_id="existing-chatgpt",
            url="https://chatgpt.com/",
            websocket_url="ws://existing",
        )
        regular = ChromeDebugTarget(
            target_id="regular-download",
            url="https://example.com/report.tex",
            websocket_url="ws://regular",
        )
        embedded = ChromeDebugTarget(
            target_id="embedded-chatgpt",
            url="https://chatgpt.com/c/new",
            websocket_url="ws://embedded",
        )

        self.assertEqual(
            new_chatgpt_target(
                {existing.target_id},
                [regular, existing, embedded],
            ),
            embedded,
        )

    def test_download_capture_only_returns_the_embedded_target_download(self):
        class FakeSocket:
            def __init__(self, frame_id=None, events=None):
                self.frame_id = frame_id
                self.events = list(events or [])
                self.responses = []
                self.sent_messages = []

            def send(self, raw_message):
                self.sent_messages.append(raw_message)
                message = json.loads(raw_message)
                if message["method"] == "Page.getFrameTree":
                    result = {
                        "frameTree": {
                            "frame": {"id": self.frame_id},
                        }
                    }
                elif message["method"] == "Target.closeTarget":
                    result = {"success": True}
                else:
                    result = {}
                self.responses.append(
                    json.dumps({"id": message["id"], "result": result})
                )

            def recv(self):
                if self.responses:
                    return self.responses.pop(0)
                if self.events:
                    return json.dumps(self.events.pop(0))
                raise BlockingIOError

            def settimeout(self, _timeout):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            download_directory = Path(directory)
            ordinary_path = download_directory / "ordinary.md"
            embedded_path = download_directory / "embedded.tex"
            ordinary_path.write_text("ordinary", encoding="utf-8")
            embedded_path.write_text("embedded", encoding="utf-8")
            target_socket = FakeSocket(frame_id="embedded-frame")
            browser_socket = FakeSocket(
                events=[
                    {
                        "method": "Browser.downloadWillBegin",
                        "params": {
                            "frameId": "ordinary-frame",
                            "guid": "ordinary-guid",
                            "suggestedFilename": ordinary_path.name,
                        },
                    },
                    {
                        "method": "Browser.downloadProgress",
                        "params": {
                            "guid": "ordinary-guid",
                            "state": "completed",
                            "filePath": str(ordinary_path),
                        },
                    },
                    {
                        "method": "Browser.downloadWillBegin",
                        "params": {
                            "frameId": "embedded-frame",
                            "guid": "embedded-guid",
                            "suggestedFilename": embedded_path.name,
                        },
                    },
                    {
                        "method": "Browser.downloadProgress",
                        "params": {
                            "guid": "embedded-guid",
                            "state": "completed",
                            "filePath": str(embedded_path),
                        },
                    },
                ]
            )
            sockets = iter([target_socket, browser_socket])
            capture = ChromeTargetDownloadCapture(
                ChromeDebugTarget(
                    target_id="embedded-target",
                    url="https://chatgpt.com/",
                    websocket_url="ws://embedded-target",
                ),
                "ws://browser",
                download_directory,
                socket_factory=lambda *_args, **_kwargs: next(sockets),
            )

            self.assertEqual(
                capture.poll_completed_downloads(),
                [embedded_path.resolve()],
            )
            self.assertTrue(capture.close(close_target=True))
            sent_methods = [
                json.loads(message)["method"]
                for message in browser_socket.sent_messages
            ]
            self.assertIn("Target.closeTarget", sent_methods)

    def test_embedded_target_enables_low_cost_chatgpt_rendering(self):
        class FakeSocket:
            def __init__(self, frame_id=None):
                self.frame_id = frame_id
                self.responses = []
                self.sent_messages = []

            def send(self, raw_message):
                self.sent_messages.append(raw_message)
                message = json.loads(raw_message)
                result = (
                    {"frameTree": {"frame": {"id": self.frame_id}}}
                    if message["method"] == "Page.getFrameTree"
                    else {}
                )
                self.responses.append(
                    json.dumps({"id": message["id"], "result": result})
                )

            def recv(self):
                if self.responses:
                    return self.responses.pop(0)
                raise BlockingIOError

            def settimeout(self, _timeout):
                pass

            def close(self):
                pass

        target_socket = FakeSocket(frame_id="embedded-frame")
        browser_socket = FakeSocket()
        sockets = iter([target_socket, browser_socket])
        capture = ChromeTargetDownloadCapture(
            ChromeDebugTarget(
                target_id="embedded-target",
                url="https://chatgpt.com/",
                websocket_url="ws://embedded-target",
            ),
            "ws://browser",
            Path("/tmp"),
            socket_factory=lambda *_args, **_kwargs: next(sockets),
        )

        target_messages = [json.loads(raw) for raw in target_socket.sent_messages]
        methods = [message["method"] for message in target_messages]
        self.assertIn("Emulation.setEmulatedMedia", methods)
        self.assertIn("Page.addScriptToEvaluateOnNewDocument", methods)
        self.assertIn("Runtime.evaluate", methods)
        script = chatgpt_performance_script()
        self.assertIn(r'data-testid^=\"conversation-turn-\"', script)
        self.assertIn("content-visibility", script)
        self.assertIn("data-writing-block", script)
        self.assertIn("backdrop-filter", script)
        browser_methods = [
            json.loads(raw)["method"] for raw in browser_socket.sent_messages
        ]
        self.assertEqual(browser_methods, ["Browser.setDownloadBehavior"])
        capture.close()

    def test_downloaded_markdown_replaces_the_current_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)
            markdown_path = root / "report.md"
            markdown_path.write_text("# Downloaded", encoding="utf-8")

            with mock.patch("markdown_editor.subprocess.Popen") as launch:
                window.open_downloaded_document(str(markdown_path))

            launch.assert_not_called()
            self.assertEqual(window.current_path, markdown_path.resolve())
            self.assertEqual(window.editor.toPlainText(), "# Downloaded")

    def test_x11_reparent_requires_the_expected_parent(self):
        x11 = mock.Mock()
        controller = X11WindowController(library=x11, display=1)
        with mock.patch.object(
            controller,
            "parent_window_id",
            return_value=0x300,
        ):
            controller.reparent(0x200, 0x300, 640, 720)

        x11.XUnmapWindow.assert_called_once_with(1, 0x200)
        x11.XChangeWindowAttributes.assert_called_once()
        x11.XReparentWindow.assert_called_once_with(1, 0x200, 0x300, 0, 0)
        x11.XMoveResizeWindow.assert_called_once_with(1, 0x200, 0, 0, 640, 720)
        x11.XMapWindow.assert_called_once_with(1, 0x200)

        with mock.patch.object(
            controller,
            "parent_window_id",
            return_value=0x400,
        ):
            with self.assertRaisesRegex(OSError, "父节点校验失败"):
                controller.reparent(0x200, 0x300, 640, 720)

    def test_x11_resize_flushes_without_blocking_for_server_sync(self):
        x11 = mock.Mock()
        controller = X11WindowController(library=x11, display=1)

        controller.resize(0x200, 640, 720)

        x11.XMoveResizeWindow.assert_called_once_with(
            1,
            0x200,
            0,
            0,
            640,
            720,
        )
        x11.XFlush.assert_called_once_with(1)
        x11.XSync.assert_not_called()

    def test_embedded_chrome_background_checks_leave_room_for_input_events(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x100},
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=mock.Mock(),
                    known_debug_targets=[],
                )

        self.assertGreaterEqual(browser.guard_timer.interval(), 1000)
        self.assertGreaterEqual(browser.download_timer.interval(), 250)
        browser.close()

    def test_embedded_chrome_shutdown_closes_its_page_target(self):
        session = RemoteChromeSession(
            executable="/opt/google/chrome/chrome",
            user_data_dir="/tmp/chrome-profile",
            profile_directory="Default",
            debug_port=9223,
            pid=100,
        )
        x11_connection = mock.Mock()
        with mock.patch(
            "markdown_editor.x11_client_window_ids",
            return_value={0x100},
        ):
            with mock.patch("markdown_editor.subprocess.Popen"):
                from markdown_editor import EmbeddedChromeWidget

                browser = EmbeddedChromeWidget(
                    session,
                    x11_connection=x11_connection,
                    known_debug_targets=[],
                )
        browser.window_id = 0x200
        capture = mock.Mock()
        capture.close.return_value = True
        browser.download_capture = capture

        browser.shutdown()

        capture.close.assert_called_once_with(close_target=True)
        x11_connection.destroy.assert_not_called()
        x11_connection.close.assert_called_once()

    def test_preview_pdf_builds_a_cache_file_and_opens_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "report.md"
            source_path.write_text("# Report", encoding="utf-8")
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(source_path, settings=settings)
            with mock.patch("markdown_editor.export_pdf", return_value="Pandoc + XeLaTeX") as export:
                with mock.patch("markdown_editor.open_local_file", return_value=True) as opened:
                    window.preview_pdf_action.trigger()

            output_path = export.call_args.args[1]
            self.assertEqual(output_path.name, "report-preview.pdf")
            self.assertEqual(export.call_args.kwargs["base_directory"], root)
            self.assertFalse(export.call_args.kwargs["include_toc"])
            opened.assert_called_once_with(output_path)

    def test_window_opens_full_tex_in_latex_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "report.tex"
            source_path.write_text(
                "\\documentclass{article}\\begin{document}Test\\end{document}",
                encoding="utf-8",
            )
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            with mock.patch(
                "markdown_editor.compile_latex_document", return_value="XeLaTeX"
            ) as compile_tex:
                with mock.patch(
                    "markdown_editor.render_pdf_pages_html",
                    return_value='<div class="pdf-document">compiled</div>',
                ):
                    with mock.patch(
                        "markdown_editor.extract_pdf_text_layout", return_value=[]
                    ):
                        window = MarkdownWindow(source_path, settings=settings)

            self.assertEqual(window.document_mode, "latex")
            self.assertIn("documentclass", window.editor.toPlainText())
            self.assertIn("compiled", window.preview.toPlainText())
            self.assertTrue(window.toc_action.isEnabled())
            self.assertEqual(window.toc_tree.topLevelItemCount(), 0)
            compile_tex.assert_called()

    def test_pdf_export_dialog_defaults_to_source_name_and_temporary_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "world_model_report.md"
            dialog = PdfExportDialog(
                source_path.with_suffix(".pdf"),
                include_toc=False,
            )

            self.assertEqual(dialog.output_path(), source_path.with_suffix(".pdf"))
            self.assertFalse(dialog.include_toc())
            self.assertTrue(dialog.open_after_export())

            dialog.toc_checkbox.setChecked(True)
            dialog.output_edit.setText(str(root / "custom-name"))

            self.assertTrue(dialog.include_toc())
            self.assertEqual(dialog.output_path(), root / "custom-name.pdf")

    def test_export_button_uses_one_time_dialog_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "report.md"
            source_path.write_text("# Report", encoding="utf-8")
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(source_path, settings=settings)
            output_path = root / "renamed.pdf"
            dialog = mock.Mock()
            dialog.exec_.return_value = QDialog.Accepted
            dialog.output_path.return_value = output_path
            dialog.include_toc.return_value = True
            dialog.open_after_export.return_value = False

            with mock.patch("markdown_editor.PdfExportDialog", return_value=dialog) as dialog_type:
                with mock.patch.object(
                    window, "build_current_pdf", return_value="Pandoc + XeLaTeX"
                ) as build:
                    window.export_action.trigger()

            dialog_type.assert_called_once_with(
                source_path.with_suffix(".pdf"),
                include_toc=False,
                parent=window,
            )
            build.assert_called_once_with(output_path, include_toc=True)

    def test_window_hides_the_live_left_toc_by_default_and_navigates_when_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)
            window.editor.setPlainText("# 第一章\n\n## 1.1 方法\n\n### 1.1.1 数据")
            window.refresh_preview()

            self.assertTrue(window.toc_dock.isHidden())
            self.assertFalse(window.toc_action.isChecked())
            self.assertEqual(window.toc_action.text(), "显示目录")
            self.assertEqual(window.toc_tree.topLevelItemCount(), 1)
            window.toc_action.trigger()
            self.assertFalse(window.toc_dock.isHidden())
            self.assertEqual(window.toc_action.text(), "隐藏目录")
            chapter = window.toc_tree.topLevelItem(0)
            self.assertEqual(chapter.text(0), "第一章")
            self.assertEqual(chapter.child(0).text(0), "1.1 方法")
            self.assertEqual(chapter.child(0).child(0).text(0), "1.1.1 数据")

            with mock.patch.object(window.preview, "scrollToAnchor") as scroll:
                window.toc_tree.itemClicked.emit(chapter.child(0), 0)

            scroll.assert_called_once_with(chapter.child(0).data(0, Qt.UserRole))

    def test_latex_toc_navigation_uses_vertical_position_inside_pdf_page(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)
            item = QTreeWidgetItem(["2.6 与 waypoint loss 的区别"])
            item.setData(0, Qt.UserRole, "pdf-page-10")
            item.setData(0, Qt.UserRole + 1, 0.75)
            positions = {
                "pdf-page-9": 800,
                "pdf-page-10": 1800,
                "pdf-page-11": 2800,
            }
            current = {"value": 0}
            scrollbar = mock.Mock()
            scrollbar.value.side_effect = lambda: current["value"]
            scrollbar.setValue.side_effect = lambda value: current.update(value=value)

            def scroll_to_anchor(anchor):
                current["value"] = positions[anchor]

            with mock.patch.object(
                window.preview,
                "scrollToAnchor",
                side_effect=scroll_to_anchor,
            ) as scroll:
                with mock.patch.object(
                    window.preview,
                    "verticalScrollBar",
                    return_value=scrollbar,
                ):
                    window.navigate_to_toc_item(item, 0)

            self.assertEqual(
                [call.args[0] for call in scroll.call_args_list],
                ["pdf-page-10", "pdf-page-11"],
            )
            scrollbar.setValue.assert_called_once_with(2550)

    def test_border_color_is_applied_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)

            window.set_border_color("#7c3aed")

            self.assertIn("#7c3aed", window.styleSheet())
            self.assertEqual(settings.value("borderColor"), "#7c3aed")
            self.assertEqual(window.border_action.text(), "边框颜色")

    def test_chatgpt_and_toc_have_a_visible_vertical_divider(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)

            window.set_border_color("#7c3aed")

            stylesheet = window.styleSheet()
            self.assertIn(
                'QMainWindow[tocVisible="true"]::separator',
                stylesheet,
            )
            self.assertIn("background-color: #7c3aed", stylesheet)
            self.assertIn("width: 4px", stylesheet)
            self.assertFalse(window.chatgpt_dock.property("tocVisible"))
            self.assertFalse(window.property("tocVisible"))

            window.set_toc_visible(True)

            self.assertTrue(window.chatgpt_dock.property("tocVisible"))
            self.assertTrue(window.property("tocVisible"))

            window.set_toc_visible(False)

            self.assertFalse(window.chatgpt_dock.property("tocVisible"))
            self.assertFalse(window.property("tocVisible"))

    def test_background_color_covers_app_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)

            window.set_background_color("#111827")

            self.assertIn("background-color: #111827", window.styleSheet())
            self.assertIn("color: #f8fafc", window.styleSheet())
            self.assertIn("QToolButton", window.styleSheet())
            self.assertEqual(settings.value("backgroundColor"), "#111827")
            self.assertEqual(window.background_action.text(), "背景颜色")

    def test_line_height_is_applied_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)

            window.set_line_height(1.4)

            self.assertEqual(window.line_height, 1.4)
            self.assertEqual(float(settings.value("lineHeight")), 1.4)
            self.assertEqual(window.line_height_action.text(), "行间距：1.40")

    def test_preview_does_not_show_virtual_page_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(settings=settings)

            self.assertIsNone(window.findChild(QLabel, "previewPageNumber"))

    def test_preview_selection_can_be_copied_as_text_or_image(self):
        with tempfile.TemporaryDirectory() as directory:
            preview = MarkdownPreview()
            preview.image_output_directory = Path(directory)
            preview.setHtml(
                "<h2>7.1 LAW 与在线 WM Planner</h2>"
                "<table><tr><th>维度</th><th>LAW</th></tr>"
                "<tr><td>WM 输入</td><td>当前 latent</td></tr></table>"
            )
            preview.selectAll()

            preview.copy_selection_as_text()
            copied_text = QApplication.clipboard().text()
            self.assertIn("7.1 LAW 与在线 WM Planner", copied_text)
            self.assertIn("WM 输入", copied_text)

            saved_paths = []
            preview.image_saved.connect(saved_paths.append)
            preview.copy_selection_as_image()
            copied_image = QApplication.clipboard().image()
            self.assertFalse(copied_image.isNull())
            self.assertGreater(copied_image.width(), 100)
            self.assertGreater(copied_image.height(), 20)
            self.assertEqual(len(saved_paths), 1)
            self.assertTrue(Path(saved_paths[0]).is_file())

            menu = preview.create_preview_context_menu()
            action_texts = [action.text() for action in menu.actions()]
            self.assertIn("复制为文字", action_texts)
            self.assertIn("复制为图片", action_texts)
            self.assertIn("复制为 ChatGPT 对话…", action_texts)

    def test_chatgpt_edit_prompt_identifies_source_page_text_and_request(self):
        prompt = build_chatgpt_edit_prompt(
            Path("/home/qwer/Downloads/report.tex"),
            "W_t 给出具体沿哪些点走。",
            "PDF 物理页码：第 7 页",
        )

        self.assertTrue(
            prompt.startswith(
                "请修改你在本次对话中刚才提供的完整 LaTeX 源文件："
            )
        )
        self.assertIn("\n\nreport.tex\n\n", prompt)
        self.assertIn(
            "请以该会话附件为唯一源文件，不要重新构建文档，"
            "也不要使用其他历史版本。",
            prompt,
        )
        self.assertIn("需要修改的位置：", prompt)
        self.assertIn("- PDF 物理页码：第 7 页", prompt)
        self.assertIn("- 选中原文：", prompt)
        self.assertIn("W_t 给出具体沿哪些点走。", prompt)
        self.assertIn("修改要求（由我补充）：", prompt)
        self.assertIn("只修改这一个位置", prompt)
        self.assertIn("不要修改其他相同或相似的内容", prompt)
        self.assertIn("保留原有结构、公式编号、label、引用、目录和排版", prompt)
        self.assertIn("完成后检查 LaTeX 语法", prompt)
        self.assertIn("返回可下载的完整 `.tex` 文件", prompt)

    def test_pdf_selection_generates_its_physical_page_location(self):
        preview = MarkdownPreview()
        preview._pdf_pages = [
            (100.0, 100.0, [(1, 1, 20, 10, "第一页", 1)]),
            (100.0, 100.0, [(1, 1, 20, 10, "第二页", 1)]),
        ]
        preview._pdf_words = [(0, 0), (1, 0)]
        preview._pdf_selection_start = 0
        preview._pdf_selection_end = 1

        selected, location = preview.selected_text_and_location()

        self.assertEqual(selected, "第一页\n第二页")
        self.assertEqual(location, "PDF 物理页码：第 1–2 页")

    def test_chatgpt_location_conversation_is_copied_without_an_extra_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "report.tex"
            source_path.write_text("\\documentclass{article}", encoding="utf-8")
            settings = QSettings(
                str(Path(directory) / "settings.ini"),
                QSettings.IniFormat,
            )
            window = MarkdownWindow(settings=settings)
            window.current_path = source_path

            with mock.patch.object(QInputDialog, "getMultiLineText") as dialog:
                window.copy_chatgpt_edit_request(
                    "原来的定义。",
                    "PDF 物理页码：第 3 页",
                )

            dialog.assert_not_called()
            copied = QApplication.clipboard().text()
            self.assertIn(str(source_path), copied)
            self.assertIn("原来的定义。", copied)
            self.assertIn("第 3 页", copied)
            self.assertTrue(copied.rstrip().endswith("修改要求（由我补充）："))

    def test_clicking_file_path_copies_the_complete_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "notes.md"
            source_path.write_text("# Notes", encoding="utf-8")
            settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
            window = MarkdownWindow(source_path, settings=settings)

            window.path_button.click()

            self.assertEqual(QApplication.clipboard().text(), str(source_path))
            self.assertEqual(
                window.path_button.toolTip(),
                "点击复制当前文件的完整路径",
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash

set -euo pipefail

show_usage() {
  echo "用法: $0 [输入.md] [输出.pdf]"
  echo "示例: $0 'UWB与VBOT效果差异定位测试方案.md' 'UWB与VBOT效果差异定位测试方案.pdf'"
}

if (( $# > 2 )); then
  show_usage >&2
  exit 2
fi

input_path="${1:-UWB与VBOT效果差异定位测试方案.md}"

if [[ ! -f "$input_path" ]]; then
  echo "错误: 找不到 Markdown 文件: $input_path" >&2
  show_usage >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "错误: 未安装 pandoc。" >&2
  exit 1
fi

browser_path=""
for browser_name in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$browser_name" >/dev/null 2>&1; then
    browser_path="$(command -v "$browser_name")"
    break
  fi
done

if [[ -z "$browser_path" ]]; then
  echo "错误: 未找到 Google Chrome 或 Chromium。" >&2
  exit 1
fi

input_name="$(basename "$input_path")"
document_title="${input_name%.*}"

if (( $# == 2 )); then
  output_path="$2"
else
  output_path="${input_path%.*}.pdf"
fi

output_dir="$(dirname "$output_path")"
if [[ ! -d "$output_dir" ]]; then
  echo "错误: 输出目录不存在: $output_dir" >&2
  exit 1
fi

output_path="$(realpath -m "$output_path")"
temp_dir="$(mktemp -d -t markdown-to-pdf.XXXXXX)"
temp_html="$temp_dir/document.html"
temp_pdf="$temp_dir/document.pdf"

cleanup() {
  rm -r -- "$temp_dir"
}
trap cleanup EXIT

read -r -d '' page_style <<'CSS' || true
<style>
  @page {
    size: A4;
    margin: 14mm 13mm 15mm;
  }

  html {
    font-size: 12px;
  }

  body {
    max-width: none;
    margin: 0;
    color: #17201d;
    font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif;
    line-height: 1.5;
  }

  #title-block-header {
    display: none;
  }

  h1 {
    margin: 0 0 14px;
    color: #143b32;
    font-size: 25px;
  }

  h2 {
    margin: 20px 0 9px;
    padding-bottom: 4px;
    border-bottom: 2px solid #2b6b5b;
    color: #205447;
    font-size: 17px;
    break-after: avoid;
  }

  p {
    margin: 6px 0;
  }

  pre {
    margin: 8px 0 11px;
    padding: 8px 10px;
    border: 1px solid #cddbd6;
    border-radius: 4px;
    background: #f3f7f5;
    font-family: "WenQuanYi Zen Hei Mono", "Noto Sans Mono CJK SC", monospace;
    font-size: 9px;
    line-height: 1.35;
    white-space: pre;
    break-inside: avoid;
  }

  table {
    width: 100%;
    margin: 8px 0 12px;
    border-collapse: collapse;
    font-size: 10px;
    break-inside: avoid;
  }

  th,
  td {
    padding: 5px 7px;
    border: 1px solid #aebfba;
    text-align: left;
  }

  th {
    background: #dfece7;
  }

  blockquote {
    margin: 8px 0;
    padding: 5px 10px;
    border-left: 4px solid #2b6b5b;
    background: #edf4f1;
  }

  hr {
    margin: 15px 0;
    border: 0;
    border-top: 1px solid #c7d2ce;
  }
</style>
CSS

pandoc "$input_path" \
  --from=gfm \
  --to=html5 \
  --standalone \
  --metadata "title=$document_title" \
  -V lang=zh-CN \
  --include-in-header=<(printf '%s\n' "$page_style") \
  -o "$temp_html"

browser_args=(
  --headless
  --disable-gpu
  --no-pdf-header-footer
  "--print-to-pdf=$temp_pdf"
)

if (( EUID == 0 )); then
  browser_args+=(--no-sandbox)
fi

"$browser_path" "${browser_args[@]}" "file://$temp_html" >/dev/null 2>&1

if [[ ! -s "$temp_pdf" ]]; then
  echo "错误: PDF 生成失败。" >&2
  exit 1
fi

mv -- "$temp_pdf" "$output_path"

if command -v pdfinfo >/dev/null 2>&1; then
  page_count="$(pdfinfo "$output_path" | awk '/^Pages:/ { print $2 }')"
  echo "转换完成: $output_path（${page_count:-未知} 页）"
else
  echo "转换完成: $output_path"
fi

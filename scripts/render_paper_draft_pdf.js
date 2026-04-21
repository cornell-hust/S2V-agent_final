#!/usr/bin/env node

const fs = require("fs");
const { marked } = require("marked");
const katex = require("katex");

async function main() {
  const [inputPath, outputHtmlPath] = process.argv.slice(2);
  if (!inputPath || !outputHtmlPath) {
    console.error("Usage: node render_paper_draft_pdf.js <input.md> <output.html>");
    process.exit(1);
  }

  const markdown = fs.readFileSync(inputPath, "utf8");
  const mathStore = [];
  const markdownWithMath = extractMath(markdown, mathStore);
  const renderer = new marked.Renderer();
  renderer.code = function codeRenderer(tokenOrCode, infostring) {
    const code = typeof tokenOrCode === "object" ? tokenOrCode.text : tokenOrCode;
    const langInfo = typeof tokenOrCode === "object" ? tokenOrCode.lang : infostring;
    const lang = (langInfo || "").trim().toLowerCase();
    if (lang === "mermaid") {
      return `\n<div class="mermaid-host"><pre><code class="language-mermaid">${escapeHtml(code)}</code></pre></div>\n`;
    }
    return `\n<pre><code>${escapeHtml(code)}</code></pre>\n`;
  };

  marked.setOptions({
    renderer,
    gfm: true,
    breaks: false,
    headerIds: false,
    mangle: false,
  });

  let html = marked.parse(markdownWithMath);
  html = renderMathPlaceholders(html, mathStore);
  html = await renderMermaidBlocks(html);
  const doc = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>seek_vau_neurips_v12_zh</title>
  <link rel="stylesheet" href="./node_modules/katex/dist/katex.min.css" />
  <style>
    @page {
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
    }
    body {
      font-family: "Droid Sans Fallback", "Noto Sans CJK SC", "Noto Sans CJK JP", sans-serif;
      color: #18202a;
      line-height: 1.62;
      font-size: 12px;
      margin: 0;
      background: #ffffff;
    }
    main {
      max-width: 180mm;
      margin: 0 auto;
    }
    h1, h2, h3, h4 {
      color: #10233e;
      line-height: 1.3;
      page-break-after: avoid;
    }
    h1 { font-size: 26px; margin: 0 0 14px; }
    h2 { font-size: 20px; margin: 24px 0 10px; border-bottom: 1px solid #d7dfea; padding-bottom: 4px; }
    h3 { font-size: 16px; margin: 18px 0 8px; }
    h4 { font-size: 14px; margin: 14px 0 6px; }
    p { margin: 8px 0; text-align: justify; }
    strong { color: #0d2448; }
    em { color: #384861; }
    code {
      font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
      background: #f3f6fa;
      border-radius: 4px;
      padding: 0 4px;
      font-size: 0.92em;
    }
    pre {
      background: #f7f9fc;
      border: 1px solid #dde5ef;
      border-radius: 8px;
      padding: 10px 12px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    pre code {
      background: transparent;
      padding: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 14px;
      table-layout: fixed;
      font-size: 11px;
    }
    th, td {
      border: 1px solid #cfd8e3;
      padding: 6px 8px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      background: #eef4fb;
      color: #10233e;
      font-weight: 700;
    }
    ul, ol {
      margin: 8px 0 10px 22px;
    }
    li {
      margin: 4px 0;
    }
    blockquote {
      margin: 12px 0;
      padding: 8px 12px;
      border-left: 4px solid #9eb6d8;
      background: #f7faff;
      color: #314760;
    }
    .katex-display {
      overflow-x: auto;
      overflow-y: hidden;
      padding: 4px 0;
      margin: 10px 0;
    }
    .diagram-block {
      margin: 12px 0 14px;
      padding: 8px;
      border: 1px solid #dbe4ee;
      border-radius: 10px;
      background: #fbfdff;
      page-break-inside: avoid;
    }
    .diagram-title {
      color: #16365f;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .diagram-lines {
      font-size: 11px;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #1f2937;
    }
    hr {
      border: none;
      border-top: 1px solid #d8e0ea;
      margin: 18px 0;
    }
  </style>
</head>
<body>
  <main>
${html}
  </main>
</body>
</html>`;

  fs.writeFileSync(outputHtmlPath, doc, "utf8");
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMathPlaceholders(html, mathStore) {
  html = html.replace(/@@MATH_(\d+)@@/g, (_, idx) => {
    const entry = mathStore[Number(idx)];
    return katex.renderToString(entry.expr.trim(), {
      displayMode: entry.displayMode,
      throwOnError: false,
      trust: true,
    });
  });
  return html;
}

function extractMath(markdown, mathStore) {
  let result = "";
  let i = 0;
  let inFence = false;

  while (i < markdown.length) {
    if (markdown.startsWith("```", i)) {
      const lineEnd = markdown.indexOf("\n", i);
      if (lineEnd === -1) {
        result += markdown.slice(i);
        break;
      }
      inFence = !inFence;
      result += markdown.slice(i, lineEnd + 1);
      i = lineEnd + 1;
      continue;
    }

    if (!inFence && markdown.startsWith("\\[", i)) {
      const end = markdown.indexOf("\\]", i + 2);
      if (end !== -1) {
        const expr = markdown.slice(i + 2, end);
        mathStore.push({ expr, displayMode: true });
        result += `@@MATH_${mathStore.length - 1}@@`;
        i = end + 2;
        continue;
      }
    }

    if (!inFence && markdown.startsWith("$$", i)) {
      const end = markdown.indexOf("$$", i + 2);
      if (end !== -1) {
        const expr = markdown.slice(i + 2, end);
        mathStore.push({ expr, displayMode: true });
        result += `@@MATH_${mathStore.length - 1}@@`;
        i = end + 2;
        continue;
      }
    }

    if (!inFence && markdown[i] === "$" && markdown[i + 1] !== "$") {
      let j = i + 1;
      while (j < markdown.length) {
        if (markdown[j] === "\\" && j + 1 < markdown.length) {
          j += 2;
          continue;
        }
        if (markdown[j] === "$" && markdown[j - 1] !== "\\") {
          const expr = markdown.slice(i + 1, j);
          mathStore.push({ expr, displayMode: false });
          result += `@@MATH_${mathStore.length - 1}@@`;
          i = j + 1;
          break;
        }
        if (markdown[j] === "\n") {
          break;
        }
        j += 1;
      }
      if (i !== j + 1) {
        result += markdown[i];
        i += 1;
      }
      continue;
    }

    result += markdown[i];
    i += 1;
  }

  return result;
}

async function renderMermaidBlocks(html) {
  const regex = /<div class="mermaid-host"><pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre><\/div>/g;
  const matches = [...html.matchAll(regex)];
  if (!matches.length) return html;

  let output = "";
  let lastIndex = 0;

  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index];
    const source = decodeHtml(match[1]);
    output += html.slice(lastIndex, match.index);
    output += renderDiagramFallback(source);
    lastIndex = match.index + match[0].length;
  }

  output += html.slice(lastIndex);
  return output;
}

function renderDiagramFallback(source) {
  const labels = new Map();
  const lines = source.split(/\r?\n/);

  for (const line of lines) {
    for (const match of line.matchAll(/([A-Za-z0-9_]+)\["([^"]+)"\]/g)) {
      labels.set(match[1], match[2].replaceAll("\\n", " / "));
    }
    const subgraph = line.match(/subgraph\s+([A-Za-z0-9_]+)\["([^"]+)"\]/);
    if (subgraph) {
      labels.set(subgraph[1], subgraph[2]);
    }
  }

  const rendered = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (
      line.startsWith("flowchart") ||
      line.startsWith("accTitle:") ||
      line.startsWith("accDescr:") ||
      line.startsWith("classDef ") ||
      line.startsWith("class ") ||
      line === "end"
    ) {
      continue;
    }
    const subgraph = line.match(/subgraph\s+([A-Za-z0-9_]+)\["([^"]+)"\]/);
    if (subgraph) {
      rendered.push(`【${subgraph[2]}】`);
      continue;
    }
    if (line.includes("-->")) {
      rendered.push(line.replace(/([A-Za-z0-9_]+)(?:\["([^"]+)"\])?/g, (token, id, inlineLabel) => {
        if (inlineLabel) return inlineLabel.replaceAll("\\n", " / ");
        return labels.get(id) || token;
      }).replaceAll("-->", "→"));
      continue;
    }
    rendered.push(line.replaceAll("\\n", " / "));
  }

  return `<div class="diagram-block"><div class="diagram-title">图示</div><div class="diagram-lines">${escapeHtml(rendered.join("\n"))}</div></div>`;
}

function decodeHtml(text) {
  return text
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

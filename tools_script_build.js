// Builds the defense script as a Word file from its markdown source.
//
//   node tools_script_build.js script.md ChordLens_Defense_Script.docx
//
// The script lives as a Claude doc; this turns an export of it into a .docx
// the team can print or carry on a laptop with no network. Built from the
// markdown rather than from the doc's own base64 export because transcribing
// base64 by hand corrupts the zip.
//
// Handles only what the script actually uses: #/##/### headings, paragraphs,
// > blockquotes, - bullets, | tables, and inline **bold** / *italic* / `code`.
const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, BorderStyle,
  Table, TableRow, TableCell, WidthType, ShadingType, HeadingLevel,
} = d;

const SRC = process.argv[2];
const OUT = process.argv[3];

const BODY = "Calibri";
const HEAD = "Calibri";
const MAROON = "531B18";
const MID = "8C4A43";
const INK = "22201F";
const GREY = "6B6663";

// ----------------------------------------------------------- inline runs
// Splits on **bold**, *italic* and `code`, keeping the delimiters' meaning.
function runs(text, opt = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  const push = (t, extra) => {
    if (!t) return;
    out.push(new TextRun({
      text: t,
      font: extra.code ? "Consolas" : (opt.font || BODY),
      size: opt.size || 21,
      color: extra.code ? MID : (opt.color || INK),
      bold: extra.bold || opt.bold || false,
      italics: extra.italics || opt.italics || false,
    }));
  };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { code: true });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last), {});
  return out.length ? out : [new TextRun({ text: "", font: BODY, size: 21 })];
}

const para = (text, opt = {}) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: opt.before || 0, after: opt.after === undefined ? 140 : opt.after, line: 276 },
  indent: opt.indent,
  border: opt.border,
  shading: opt.shading,
  bullet: opt.bullet,
  keepNext: !!opt.keepNext,
  children: runs(text, opt),
});

// ------------------------------------------------------------------ parse
const lines = fs.readFileSync(SRC, "utf8").split(/\r?\n/);
const children = [];
let i = 0;

function flushTable() {
  const rows = [];
  while (i < lines.length && lines[i].trim().startsWith("|")) {
    const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
    if (!cells.every(c => /^-{2,}$/.test(c))) rows.push(cells);
    i++;
  }
  if (!rows.length) return;
  const cols = rows[0].length;
  const width = Math.round(9360 / cols);
  children.push(new Table({
    columnWidths: Array(cols).fill(width),
    width: { size: 9360, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: MID },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: MID },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "DDD5D3" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: rows.map((cells, r) => new TableRow({
      tableHeader: r === 0,
      children: cells.map((c, ci) => new TableCell({
        width: { size: width, type: WidthType.DXA },
        shading: r === 0 ? { type: ShadingType.CLEAR, fill: "F4F0F0" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { after: 0, line: 240 },
          children: runs(c, { size: 19, bold: r === 0, color: r === 0 ? MAROON : INK }),
        })],
      })),
    })),
  }));
  children.push(para("", { after: 160 }));
}

while (i < lines.length) {
  const line = lines[i];
  const t = line.trim();

  if (!t) { i++; continue; }

  if (t.startsWith("|")) { flushTable(); continue; }

  if (t.startsWith("### ")) {
    children.push(new Paragraph({
      spacing: { before: 280, after: 120 }, keepNext: true,
      children: runs(t.slice(4), { size: 24, bold: true, color: MAROON, font: HEAD }),
    }));
    i++; continue;
  }
  if (t.startsWith("## ")) {
    children.push(new Paragraph({
      spacing: { before: 420, after: 60 }, keepNext: true,
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: MAROON, space: 6 } },
      children: runs(t.slice(3), { size: 30, bold: true, color: MAROON, font: HEAD }),
    }));
    i++; continue;
  }
  if (t.startsWith("# ")) {
    children.push(new Paragraph({
      spacing: { after: 100 },
      children: runs(t.slice(2), { size: 46, bold: true, color: MAROON, font: HEAD }),
    }));
    i++; continue;
  }

  // blockquote = the words to say: indented, on a tinted panel
  if (t.startsWith(">")) {
    const block = [];
    while (i < lines.length && lines[i].trim().startsWith(">")) {
      block.push(lines[i].trim().replace(/^>\s?/, ""));
      i++;
    }
    const paras = [];
    let buf = [];
    for (const b of block) {
      if (!b.trim()) { if (buf.length) { paras.push(buf.join(" ")); buf = []; } }
      else buf.push(b);
    }
    if (buf.length) paras.push(buf.join(" "));
    paras.forEach((p, n) => children.push(para(p, {
      indent: { left: 340, right: 200 },
      before: n === 0 ? 60 : 0,
      after: n === paras.length - 1 ? 200 : 120,
      shading: { type: ShadingType.CLEAR, fill: "FAF7F7" },
      border: { left: { style: BorderStyle.SINGLE, size: 12, color: MID, space: 10 } },
    })));
    continue;
  }

  if (t.startsWith("- ")) {
    children.push(para(t.slice(2), { bullet: { level: 0 }, after: 80 }));
    i++; continue;
  }

  children.push(para(t));
  i++;
}

// --------------------------------------------------------------- document
const doc = new Document({
  styles: { default: { document: { run: { font: BODY, size: 21, color: INK } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1440, right: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(OUT, b);
  console.log(`wrote ${OUT} (${(b.length / 1024).toFixed(0)} KB, ${children.length} blocks)`);
});

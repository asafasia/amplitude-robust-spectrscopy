import fs from "node:fs/promises";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workspace = "/Users/asafsolonnikov/Developer/amplitude-robust-spectroscopy";
const bibPath = path.join(workspace, "paper/references.bib");
const paperDir = path.join(workspace, "paper");
const dbPath = "/Users/asafsolonnikov/.research-assistant/research.db";
const outputDir = path.join(workspace, "outputs/019f76b1-846d-7bc1-a67d-b1066c1b20ef");
const outputPath = path.join(outputDir, "power_narrowing_paper_catalog.xlsx");
const previewDir = path.join(outputDir, "previews");

function parseBibtex(text) {
  const entries = [];
  let cursor = 0;
  while (cursor < text.length) {
    const at = text.indexOf("@", cursor);
    if (at < 0) break;
    const head = text.slice(at).match(/^@(\w+)\s*\{\s*([^,]+),/);
    if (!head) {
      cursor = at + 1;
      continue;
    }
    const type = head[1].toLowerCase();
    const key = head[2].trim();
    const open = at + head[0].lastIndexOf("{");
    let depth = 0;
    let inQuote = false;
    let escaped = false;
    let end = -1;
    for (let i = open; i < text.length; i += 1) {
      const ch = text[i];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (ch === "\\") {
        escaped = true;
        continue;
      }
      if (ch === '"') inQuote = !inQuote;
      if (!inQuote) {
        if (ch === "{") depth += 1;
        if (ch === "}") {
          depth -= 1;
          if (depth === 0) {
            end = i;
            break;
          }
        }
      }
    }
    if (end < 0) break;
    const bodyStart = at + head[0].length;
    const body = text.slice(bodyStart, end);
    entries.push({ type, key, fields: parseBibFields(body) });
    cursor = end + 1;
  }
  return entries;
}

function parseBibFields(body) {
  const fields = {};
  let i = 0;
  const skip = () => {
    while (i < body.length && /[\s,]/.test(body[i])) i += 1;
  };
  while (i < body.length) {
    skip();
    const nameStart = i;
    while (i < body.length && /[A-Za-z0-9_:-]/.test(body[i])) i += 1;
    const name = body.slice(nameStart, i).trim().toLowerCase();
    if (!name) {
      i += 1;
      continue;
    }
    while (i < body.length && /\s/.test(body[i])) i += 1;
    if (body[i] !== "=") {
      while (i < body.length && body[i] !== ",") i += 1;
      continue;
    }
    i += 1;
    while (i < body.length && /\s/.test(body[i])) i += 1;
    let value = "";
    if (body[i] === "{") {
      i += 1;
      let depth = 1;
      const start = i;
      let escaped = false;
      while (i < body.length && depth > 0) {
        const ch = body[i];
        if (escaped) {
          escaped = false;
        } else if (ch === "\\") {
          escaped = true;
        } else if (ch === "{") {
          depth += 1;
        } else if (ch === "}") {
          depth -= 1;
          if (depth === 0) break;
        }
        i += 1;
      }
      value = body.slice(start, i);
      i += 1;
    } else if (body[i] === '"') {
      i += 1;
      const start = i;
      let escaped = false;
      while (i < body.length) {
        const ch = body[i];
        if (!escaped && ch === '"') break;
        escaped = !escaped && ch === "\\";
        if (ch !== "\\") escaped = false;
        i += 1;
      }
      value = body.slice(start, i);
      i += 1;
    } else {
      const start = i;
      while (i < body.length && body[i] !== "," && body[i] !== "\n") i += 1;
      value = body.slice(start, i).trim();
    }
    fields[name] = value.trim();
  }
  return fields;
}

function cleanLatex(value = "") {
  return String(value)
    .replace(/\{\\textgreater\}/g, ">")
    .replace(/\{\\textless\}/g, "<")
    .replace(/\\&/g, "&")
    .replace(/\\_/g, "_")
    .replace(/---/g, "—")
    .replace(/--/g, "–")
    .replace(/\\([`'\"^~=\.uvHckbd])\s*\{?([A-Za-z])\}?/g, "$2")
    .replace(/\\textit\s*\{([^{}]*)\}/g, "$1")
    .replace(/\\emph\s*\{([^{}]*)\}/g, "$1")
    .replace(/\\[A-Za-z]+\s*/g, "")
    .replace(/[{}]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeDoi(value = "") {
  return cleanLatex(value)
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "")
    .replace(/^doi:\s*/i, "")
    .replace(/[\s.]+$/g, "")
    .trim();
}

function normalizeArxiv(value = "", url = "") {
  const combined = `${value} ${url}`;
  const match = combined.match(/(?:arxiv[:./ ]|abs\/)(\d{4}\.\d{4,5})(?:v\d+)?/i);
  return match ? match[1] : cleanLatex(value).replace(/^arxiv:/i, "").trim();
}

function normalizeTitle(value = "") {
  return cleanLatex(value)
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function chooseLonger(a = "", b = "") {
  const aa = String(a || "").trim();
  const bb = String(b || "").trim();
  return bb.length > aa.length ? bb : aa;
}

async function walkTex(dir) {
  const result = [];
  for (const item of await fs.readdir(dir, { withFileTypes: true })) {
    if (item.name === "archive") continue;
    const full = path.join(dir, item.name);
    if (item.isDirectory()) result.push(...await walkTex(full));
    else if (item.isFile() && item.name.endsWith(".tex")) result.push(full);
  }
  return result;
}

function extractPdfPaths(fileField = "") {
  const found = [];
  const regex = /(?:^|;)[^:]*:(\/[^;]+?\.pdf):application\/pdf/gi;
  let match;
  while ((match = regex.exec(fileField)) !== null) found.push(match[1]);
  return [...new Set(found)];
}

const citationKeys = new Set();
for (const texPath of await walkTex(paperDir)) {
  const text = await fs.readFile(texPath, "utf8");
  for (const match of text.matchAll(/\\cite[a-zA-Z*]*\s*\{([^}]+)\}/g)) {
    for (const key of match[1].split(",")) citationKeys.add(key.trim());
  }
}

const bibEntries = parseBibtex(await fs.readFile(bibPath, "utf8"));
const sql = `
SELECT pa.id, pr.name AS project, pa.title, pa.authors, pa.year, pa.venue,
       pa.doi, pa.arxiv_id, pa.url, pa.abstract, pa.category, pa.status,
       pa.notes, pa.created_at, pa.updated_at,
       COALESCE(GROUP_CONCAT(t.name, ', '), '') AS tags
FROM papers pa
JOIN projects pr ON pr.id = pa.project_id
LEFT JOIN paper_tags pt ON pt.paper_id = pa.id
LEFT JOIN tags t ON t.id = pt.tag_id
WHERE pr.name = 'power-narrowing-spectroscopy'
GROUP BY pa.id
ORDER BY pa.id;`;
const dbRows = JSON.parse(execFileSync("/usr/bin/sqlite3", ["-readonly", "-json", dbPath, sql], { encoding: "utf8" }) || "[]");

const sourceRecords = [];
for (const entry of bibEntries) {
  const f = entry.fields;
  const title = cleanLatex(f.title || "");
  const doi = normalizeDoi(f.doi || "");
  const arxiv = normalizeArxiv(f.eprint || "", f.url || "");
  const internal = entry.key === "SupplementalMaterial" || (!title && entry.type === "misc");
  sourceRecords.push({
    sourceType: "Active bibliography",
    sourceLocation: bibPath,
    sourceId: entry.key,
    entryType: entry.type,
    title,
    authors: cleanLatex(f.author || "").replace(/\s+and\s+/gi, "; "),
    year: Number.parseInt(cleanLatex(f.year || ""), 10) || null,
    month: cleanLatex(f.month || ""),
    venue: cleanLatex(f.journal || f.booktitle || f.publisher || ""),
    publisher: cleanLatex(f.publisher || ""),
    volume: cleanLatex(f.volume || ""),
    issue: cleanLatex(f.number || f.issue || ""),
    pages: cleanLatex(f.pages || ""),
    doi,
    arxiv,
    url: cleanLatex(f.url || ""),
    abstract: cleanLatex(f.abstract || ""),
    notes: cleanLatex(f.note || ""),
    category: "",
    status: "",
    tags: cleanLatex(f.keywords || ""),
    cited: citationKeys.has(entry.key),
    localPdfs: extractPdfPaths(f.file || ""),
    included: !internal,
    exclusionReason: internal ? "Internal supplemental-material record; not a paper" : "",
    scopeFlag: ["einstein", "dirac", "knuthwebsite", "knuth-fa"].includes(entry.key)
      ? "Likely template/example record; retained because it is in the active .bib"
      : "",
  });
}

for (const row of dbRows) {
  const doi = normalizeDoi(row.doi || "");
  const arxiv = normalizeArxiv(row.arxiv_id || "", row.url || "");
  sourceRecords.push({
    sourceType: "Scientific Assistant DB",
    sourceLocation: `${dbPath} :: project=power-narrowing-spectroscopy`,
    sourceId: String(row.id),
    entryType: "database paper",
    title: cleanLatex(row.title || ""),
    authors: cleanLatex(row.authors || ""),
    year: Number.parseInt(String(row.year || ""), 10) || null,
    month: "",
    venue: cleanLatex(row.venue || ""),
    publisher: "",
    volume: "",
    issue: "",
    pages: "",
    doi,
    arxiv,
    url: cleanLatex(row.url || ""),
    abstract: cleanLatex(row.abstract || ""),
    notes: cleanLatex(row.notes || ""),
    category: cleanLatex(row.category || ""),
    status: cleanLatex(row.status || ""),
    tags: cleanLatex(row.tags || ""),
    cited: false,
    localPdfs: [],
    included: true,
    exclusionReason: "",
    scopeFlag: "",
    createdAt: row.created_at || "",
    updatedAt: row.updated_at || "",
  });
}

const consolidated = [];
const keyToRecord = new Map();
const auditRows = [];

function sourceKeys(record) {
  const keys = [];
  if (record.doi) keys.push([`doi:${record.doi.toLowerCase()}`, "DOI"]);
  if (record.arxiv) keys.push([`arxiv:${record.arxiv.toLowerCase()}`, "arXiv ID"]);
  const title = normalizeTitle(record.title);
  if (title) keys.push([`title:${title}`, "normalized title"]);
  return keys;
}

for (const src of sourceRecords) {
  if (!src.included) {
    auditRows.push({ src, recordId: "", matchMethod: "Excluded", included: "No" });
    continue;
  }
  const keys = sourceKeys(src);
  let target = null;
  let matchedBy = "new record";
  for (const [key, method] of keys) {
    if (keyToRecord.has(key)) {
      target = keyToRecord.get(key);
      matchedBy = method;
      break;
    }
  }
  if (!target) {
    target = {
      id: "",
      title: src.title,
      authors: src.authors,
      year: src.year,
      month: src.month,
      venue: src.venue,
      publisher: src.publisher,
      volume: src.volume,
      issue: src.issue,
      pages: src.pages,
      doi: src.doi,
      arxiv: src.arxiv,
      url: src.url,
      abstract: src.abstract,
      notes: src.notes,
      category: src.category,
      status: src.status,
      tags: new Set(src.tags ? src.tags.split(/,\s*/) : []),
      cited: src.cited,
      activeBib: src.sourceType === "Active bibliography",
      database: src.sourceType === "Scientific Assistant DB",
      citationKeys: new Set(src.sourceType === "Active bibliography" ? [src.sourceId] : []),
      dbIds: new Set(src.sourceType === "Scientific Assistant DB" ? [src.sourceId] : []),
      sourceLocations: new Set([src.sourceLocation]),
      entryTypes: new Set([src.entryType]),
      localPdfs: new Set(src.localPdfs),
      scopeFlags: new Set(src.scopeFlag ? [src.scopeFlag] : []),
      dedupBasis: new Set(["original source record"]),
    };
    consolidated.push(target);
  } else {
    target.title = chooseLonger(target.title, src.title);
    target.authors = chooseLonger(target.authors, src.authors);
    target.year = target.year || src.year;
    target.month = target.month || src.month;
    target.venue = chooseLonger(target.venue, src.venue);
    target.publisher = chooseLonger(target.publisher, src.publisher);
    target.volume = target.volume || src.volume;
    target.issue = target.issue || src.issue;
    target.pages = target.pages || src.pages;
    target.doi = target.doi || src.doi;
    target.arxiv = target.arxiv || src.arxiv;
    target.url = target.url || src.url;
    target.abstract = chooseLonger(target.abstract, src.abstract);
    target.notes = chooseLonger(target.notes, src.notes);
    target.category = target.category || src.category;
    target.status = target.status || src.status;
    target.cited = target.cited || src.cited;
    target.activeBib = target.activeBib || src.sourceType === "Active bibliography";
    target.database = target.database || src.sourceType === "Scientific Assistant DB";
    for (const tag of src.tags ? src.tags.split(/,\s*/) : []) if (tag) target.tags.add(tag);
    if (src.sourceType === "Active bibliography") target.citationKeys.add(src.sourceId);
    if (src.sourceType === "Scientific Assistant DB") target.dbIds.add(src.sourceId);
    target.sourceLocations.add(src.sourceLocation);
    target.entryTypes.add(src.entryType);
    for (const pdf of src.localPdfs) target.localPdfs.add(pdf);
    if (src.scopeFlag) target.scopeFlags.add(src.scopeFlag);
    target.dedupBasis.add(matchedBy);
  }
  for (const [key] of sourceKeys(target)) keyToRecord.set(key, target);
  auditRows.push({ src, target, matchMethod: matchedBy, included: "Yes" });
}

consolidated.sort((a, b) => {
  if (a.cited !== b.cited) return a.cited ? -1 : 1;
  const yearDiff = (b.year || 0) - (a.year || 0);
  if (yearDiff) return yearDiff;
  return a.title.localeCompare(b.title);
});
consolidated.forEach((record, index) => { record.id = `PN-${String(index + 1).padStart(3, "0")}`; });
for (const row of auditRows) if (row.target) row.recordId = row.target.id;

for (const record of consolidated) {
  if (!record.url && record.doi) record.url = `https://doi.org/${record.doi}`;
  if (!record.url && record.arxiv) record.url = `https://arxiv.org/abs/${record.arxiv}`;
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const indexSheet = workbook.worksheets.add("Paper Index");
const metadataSheet = workbook.worksheets.add("Full Metadata");
const auditSheet = workbook.worksheets.add("Source Audit");

const navy = "#17324D";
const teal = "#2A7F62";
const gold = "#D9A441";
const pale = "#EEF4F7";
const paleGreen = "#E8F3EE";
const paleGold = "#FFF4D8";
const gray = "#5E6B75";
const white = "#FFFFFF";
const lightBorder = "#D6DEE3";

function styleTitle(sheet, endCol, title, subtitle) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${endCol}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: navy,
    font: { bold: true, color: white, size: 18 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 32;
  sheet.getRange(`A2:${endCol}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = {
    fill: pale,
    font: { color: gray, italic: true, size: 10 },
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A2:${endCol}2`).format.rowHeight = 30;
}

styleTitle(summary, "H", "Power-Narrowing Spectroscopy Paper Catalog", "Deduplicated catalog from the active manuscript bibliography and the Scientific Assistant power-narrowing-spectroscopy database project.");
summary.getRange("A4:B4").values = [["Metric", "Value"]];
summary.getRange("A4:B4").format = { fill: teal, font: { bold: true, color: white }, borders: { preset: "outside", style: "thin", color: teal } };
const indexLastRow = consolidated.length + 4;
const auditLastRow = auditRows.length + 4;
summary.getRange("A5:A11").values = [
  ["Unique paper records"],
  ["Cited in manuscript"],
  ["From active bibliography"],
  ["From Scientific Assistant DB"],
  ["Present in both sources"],
  ["Missing article link"],
  ["Template/example records retained"],
];
summary.getRange("B5:B11").formulas = [
  [`=COUNTA('Paper Index'!$A$5:$A$${indexLastRow})`],
  [`=COUNTIF('Paper Index'!$I$5:$I$${indexLastRow},"Yes")`],
  [`=COUNTIF('Paper Index'!$J$5:$J$${indexLastRow},"Yes")`],
  [`=COUNTIF('Paper Index'!$K$5:$K$${indexLastRow},"Yes")`],
  [`=COUNTIFS('Paper Index'!$J$5:$J$${indexLastRow},"Yes",'Paper Index'!$K$5:$K$${indexLastRow},"Yes")`],
  [`=COUNTBLANK('Paper Index'!$F$5:$F$${indexLastRow})`],
  [`=COUNTIF('Paper Index'!$Q$5:$Q$${indexLastRow},"Likely template/example record; retained because it is in the active .bib")`],
];
summary.getRange("A5:B11").format.borders = { preset: "inside", style: "thin", color: lightBorder };
summary.getRange("A5:A11").format.font = { bold: true, color: navy };
summary.getRange("B5:B11").format = { fill: paleGreen, font: { bold: true, color: teal, size: 12 }, horizontalAlignment: "center", numberFormat: "0" };
summary.getRange("A4:B11").format.borders = { preset: "outside", style: "thin", color: lightBorder };
summary.getRange("A4:A11").format.columnWidth = 32;
summary.getRange("B4:B11").format.columnWidth = 14;

summary.getRange("D4:H4").merge();
summary.getRange("D4").values = [["Scope and provenance"]];
summary.getRange("D4:H4").format = { fill: gold, font: { bold: true, color: navy } };
summary.getRange("D5:H11").merge(true);
summary.getRange("D5:D11").values = [
  ["Active bibliography: paper/references.bib. Citation status is parsed from manuscript .tex files under paper/ (archive excluded)."],
  ["Database: ~/.research-assistant/research.db, exact project name power-narrowing-spectroscopy."],
  ["Deduplication order: normalized DOI, arXiv ID, then normalized title."],
  ["The internal SupplementalMaterial record is retained only in Source Audit and excluded from the paper count."],
  ["mainNotes.bib and supplementalNotes.bib contain bibliography-control directives only, not paper records."],
  ["paper/archive/legacy_zotero_export.bib is not imported because it is explicitly archived."],
  ["Likely LaTeX template examples are kept in the catalog and clearly flagged so no active .bib entry is silently lost."],
];
summary.getRange("D5:H11").format = { wrapText: true, verticalAlignment: "top", font: { color: navy, size: 10 }, borders: { preset: "inside", style: "thin", color: lightBorder } };
summary.getRange("D4:H11").format.borders = { preset: "outside", style: "thin", color: lightBorder };
summary.getRange("D5:H11").format.rowHeight = 38;
summary.getRange("D4:H11").format.columnWidth = 18;

const categories = [...new Set(consolidated.map((r) => r.category).filter(Boolean))].sort();
summary.getRange("A14:C14").values = [["Scientific Assistant category", "Paper count", "Important / read"]];
summary.getRange("A14:C14").format = { fill: navy, font: { bold: true, color: white } };
if (categories.length) {
  summary.getRange(`A15:A${14 + categories.length}`).values = categories.map((category) => [category]);
  summary.getRange(`B15:B${14 + categories.length}`).formulas = categories.map((_, i) => [`=COUNTIF('Paper Index'!$M$5:$M$${indexLastRow},A${15 + i})`]);
  summary.getRange(`C15:C${14 + categories.length}`).formulas = categories.map((_, i) => [`=COUNTIFS('Paper Index'!$M$5:$M$${indexLastRow},A${15 + i},'Paper Index'!$N$5:$N$${indexLastRow},"important")+COUNTIFS('Paper Index'!$M$5:$M$${indexLastRow},A${15 + i},'Paper Index'!$N$5:$N$${indexLastRow},"read")`]);
  summary.getRange(`A15:C${14 + categories.length}`).format.borders = { preset: "inside", style: "thin", color: lightBorder };
  summary.getRange(`A14:C${14 + categories.length}`).format.borders = { preset: "outside", style: "thin", color: lightBorder };
  summary.getRange(`B15:C${14 + categories.length}`).format.numberFormat = "0";
}
summary.getRange("A14:A40").format.columnWidth = 34;
summary.getRange("B14:C40").format.columnWidth = 16;
summary.freezePanes.freezeRows(2);

styleTitle(indexSheet, "R", "Paper Index", "One row per deduplicated record. Use the filters to isolate manuscript citations, database-only papers, reading status, or category.");
const indexHeaders = ["Record ID", "Title", "Authors", "Year", "Venue", "Article Link", "DOI", "arXiv ID", "Cited in Manuscript", "Active Bibliography", "Scientific Assistant DB", "Citation Key(s)", "Category", "Reading Status", "Tags", "Notes", "Scope Flag", "Record Type"];
indexSheet.getRange("A4:R4").values = [indexHeaders];
const indexRows = consolidated.map((r) => [
  r.id,
  r.title,
  r.authors,
  r.year,
  r.venue,
  r.url,
  r.doi,
  r.arxiv,
  r.cited ? "Yes" : "No",
  r.activeBib ? "Yes" : "No",
  r.database ? "Yes" : "No",
  [...r.citationKeys].sort().join(", "),
  r.category,
  r.status,
  [...r.tags].filter(Boolean).sort().join(", "),
  r.notes,
  [...r.scopeFlags].join("; "),
  [...r.entryTypes].sort().join(", "),
]);
indexSheet.getRange(`A5:R${indexLastRow}`).values = indexRows;
const indexTable = indexSheet.tables.add(`A4:R${indexLastRow}`, true, "PowerNarrowingPapers");
indexTable.style = "TableStyleMedium2";
indexTable.showFilterButton = true;
indexSheet.freezePanes.freezeRows(4);
indexSheet.freezePanes.freezeColumns(2);
indexSheet.getRange(`D5:D${indexLastRow}`).format.numberFormat = "0";
indexSheet.getRange(`A4:R${indexLastRow}`).format.verticalAlignment = "top";
indexSheet.getRange(`B5:C${indexLastRow}`).format.wrapText = true;
indexSheet.getRange(`E5:E${indexLastRow}`).format.wrapText = true;
indexSheet.getRange(`L5:R${indexLastRow}`).format.wrapText = true;
indexSheet.getRange(`F5:H${indexLastRow}`).format.font = { color: "#1A5FB4" };
indexSheet.getRange(`I5:K${indexLastRow}`).format.horizontalAlignment = "center";
indexSheet.getRange(`I5:K${indexLastRow}`).conditionalFormats.add("containsText", { text: "Yes", format: { fill: paleGreen, font: { color: teal, bold: true } } });
indexSheet.getRange(`Q5:Q${indexLastRow}`).conditionalFormats.add("containsText", { text: "template/example", format: { fill: paleGold, font: { color: "#8A5A00", bold: true } } });
indexSheet.getRange(`N5:N${indexLastRow}`).dataValidation = { rule: { type: "list", values: ["to-read", "reading", "read", "important", "excluded", ""] } };
const indexWidths = [12, 46, 34, 9, 26, 42, 30, 15, 18, 18, 22, 24, 26, 16, 34, 42, 38, 22];
indexWidths.forEach((width, idx) => indexSheet.getRangeByIndexes(3, idx, indexLastRow - 3, 1).format.columnWidth = width);
indexSheet.getRange(`A5:R${indexLastRow}`).format.rowHeight = 46;
indexSheet.getRange("A4:R4").format.rowHeight = 30;

styleTitle(metadataSheet, "S", "Full Metadata", "Bibliographic details and long-form fields retained after deduplication. Abstracts and local PDF paths come from the original sources when available.");
const metadataHeaders = ["Record ID", "Title", "Year", "Month", "Venue", "Publisher", "Volume", "Issue", "Pages", "DOI", "Article URL", "arXiv ID", "Abstract", "Database Notes", "Local PDF Path(s)", "Source File(s)", "Database Record ID(s)", "BibTeX Key(s)", "Deduplication Basis"];
metadataSheet.getRange("A4:S4").values = [metadataHeaders];
const metadataRows = consolidated.map((r) => [
  r.id, r.title, r.year, r.month, r.venue, r.publisher, r.volume, r.issue, r.pages,
  r.doi, r.url, r.arxiv, r.abstract, r.notes, [...r.localPdfs].join("; "),
  [...r.sourceLocations].join("; "), [...r.dbIds].sort().join(", "), [...r.citationKeys].sort().join(", "),
  [...r.dedupBasis].sort().join(", "),
]);
metadataSheet.getRange(`A5:S${indexLastRow}`).values = metadataRows;
const metadataTable = metadataSheet.tables.add(`A4:S${indexLastRow}`, true, "PowerNarrowingMetadata");
metadataTable.style = "TableStyleMedium2";
metadataTable.showFilterButton = true;
metadataSheet.freezePanes.freezeRows(4);
metadataSheet.freezePanes.freezeColumns(2);
metadataSheet.getRange(`C5:C${indexLastRow}`).format.numberFormat = "0";
metadataSheet.getRange(`A4:S${indexLastRow}`).format.verticalAlignment = "top";
metadataSheet.getRange(`B5:S${indexLastRow}`).format.wrapText = true;
metadataSheet.getRange(`J5:L${indexLastRow}`).format.font = { color: "#1A5FB4" };
const metadataWidths = [12, 44, 9, 10, 26, 22, 10, 10, 14, 30, 42, 15, 64, 48, 50, 52, 18, 24, 24];
metadataWidths.forEach((width, idx) => metadataSheet.getRangeByIndexes(3, idx, indexLastRow - 3, 1).format.columnWidth = width);
metadataSheet.getRange(`A5:S${indexLastRow}`).format.rowHeight = 78;
metadataSheet.getRange("A4:S4").format.rowHeight = 30;

styleTitle(auditSheet, "L", "Source Audit", "One row per original source record, including the excluded internal SupplementalMaterial record. This sheet documents consolidation and match decisions.");
const auditHeaders = ["Source Type", "Source Location", "Source Record ID", "Original Title", "DOI", "Article URL", "Cited", "Included", "Consolidated Record ID", "Match Method", "Exclusion Reason", "Local PDF Path(s)"];
auditSheet.getRange("A4:L4").values = [auditHeaders];
const auditValues = auditRows.map((row) => [
  row.src.sourceType, row.src.sourceLocation, row.src.sourceId, row.src.title, row.src.doi, row.src.url,
  row.src.cited ? "Yes" : "No", row.included, row.recordId || "", row.matchMethod,
  row.src.exclusionReason || "", row.src.localPdfs.join("; "),
]);
auditSheet.getRange(`A5:L${auditLastRow}`).values = auditValues;
const auditTable = auditSheet.tables.add(`A4:L${auditLastRow}`, true, "SourceAuditRecords");
auditTable.style = "TableStyleMedium2";
auditTable.showFilterButton = true;
auditSheet.freezePanes.freezeRows(4);
auditSheet.freezePanes.freezeColumns(3);
auditSheet.getRange(`A4:L${auditLastRow}`).format.verticalAlignment = "top";
auditSheet.getRange(`A5:L${auditLastRow}`).format.wrapText = true;
auditSheet.getRange(`E5:F${auditLastRow}`).format.font = { color: "#1A5FB4" };
auditSheet.getRange(`G5:H${auditLastRow}`).format.horizontalAlignment = "center";
auditSheet.getRange(`H5:H${auditLastRow}`).conditionalFormats.add("containsText", { text: "No", format: { fill: paleGold, font: { color: "#8A5A00", bold: true } } });
const auditWidths = [24, 48, 18, 48, 30, 42, 10, 10, 20, 20, 42, 50];
auditWidths.forEach((width, idx) => auditSheet.getRangeByIndexes(3, idx, auditLastRow - 3, 1).format.columnWidth = width);
auditSheet.getRange(`A5:L${auditLastRow}`).format.rowHeight = 54;
auditSheet.getRange("A4:L4").format.rowHeight = 30;

await fs.mkdir(previewDir, { recursive: true });
const inspections = {};
inspections.summary = (await workbook.inspect({ kind: "table", range: "Summary!A1:H25", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 8, maxChars: 10000 })).ndjson;
inspections.index = (await workbook.inspect({ kind: "table", range: `Paper Index!A1:R${Math.min(indexLastRow, 14)}`, include: "values,formulas", tableMaxRows: 14, tableMaxCols: 18, maxChars: 12000 })).ndjson;
inspections.errors = (await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan", maxChars: 6000 })).ndjson;

for (const [sheetName, range, fileName] of [
  ["Summary", "A1:H25", "summary.png"],
  ["Paper Index", `A1:R${Math.min(indexLastRow, 12)}`, "paper_index.png"],
  ["Full Metadata", `A1:S${Math.min(indexLastRow, 9)}`, "full_metadata.png"],
  ["Source Audit", `A1:L${Math.min(auditLastRow, 12)}`, "source_audit.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  previewDir,
  sourceCounts: {
    activeBibEntries: bibEntries.length,
    databasePapers: dbRows.length,
    sourceAuditRows: auditRows.length,
    excludedSourceRecords: auditRows.filter((row) => row.included === "No").length,
  },
  consolidatedCounts: {
    uniquePapers: consolidated.length,
    cited: consolidated.filter((r) => r.cited).length,
    activeBib: consolidated.filter((r) => r.activeBib).length,
    database: consolidated.filter((r) => r.database).length,
    both: consolidated.filter((r) => r.activeBib && r.database).length,
    missingLinks: consolidated.filter((r) => !r.url).length,
  },
  missingLinkRecords: consolidated.filter((r) => !r.url).map((r) => ({ id: r.id, title: r.title, doi: r.doi, arxiv: r.arxiv, citationKeys: [...r.citationKeys] })),
  inspectionSummary: inspections.summary,
  formulaErrors: inspections.errors,
}, null, 2));

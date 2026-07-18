import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "/Users/asafsolonnikov/Developer/amplitude-robust-spectroscopy/outputs/019f76b1-846d-7bc1-a67d-b1066c1b20ef/power_narrowing_paper_catalog.xlsx";
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const indexSheet = workbook.worksheets.getItem("Paper Index");
const metadataSheet = workbook.worksheets.getItem("Full Metadata");
const indexValues = indexSheet.getRange("A4:R67").values;
const metadataValues = metadataSheet.getRange("A4:S67").values;

const indexHeaders = indexValues[0];
const metadataHeaders = metadataValues[0];
const metadataById = new Map();
for (let i = 1; i < metadataValues.length; i += 1) {
  const row = Object.fromEntries(metadataHeaders.map((header, j) => [header, metadataValues[i][j]]));
  metadataById.set(row["Record ID"], { ...row, excelRow: i + 4 });
}

const records = [];
for (let i = 1; i < indexValues.length; i += 1) {
  const row = Object.fromEntries(indexHeaders.map((header, j) => [header, indexValues[i][j]]));
  const metadata = metadataById.get(row["Record ID"]) || {};
  records.push({
    excelRow: i + 4,
    id: row["Record ID"],
    title: row.Title,
    authors: row.Authors,
    year: row.Year,
    venue: row.Venue,
    url: row["Article Link"],
    doi: row.DOI,
    arxiv: row["arXiv ID"],
    cited: row["Cited in Manuscript"],
    activeBib: row["Active Bibliography"],
    database: row["Scientific Assistant DB"],
    citationKeys: row["Citation Key(s)"],
    category: row.Category,
    status: row["Reading Status"],
    tags: row.Tags,
    notes: row.Notes,
    abstract: metadata.Abstract,
  });
}

const uncitedProjectPapers = records.filter((record) => record.database === "Yes" && record.cited !== "Yes");
const uncitedActiveBib = records.filter((record) => record.activeBib === "Yes" && record.cited !== "Yes");
console.log(JSON.stringify({
  workbook: (await workbook.inspect({ kind: "workbook,sheet,table", maxChars: 4000, tableMaxRows: 3, tableMaxCols: 5 })).ndjson,
  uncitedProjectPapers,
  uncitedActiveBib,
}, null, 2));

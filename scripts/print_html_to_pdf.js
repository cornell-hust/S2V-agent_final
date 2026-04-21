#!/usr/bin/env node

const path = require("path");
const { chromium } = require("playwright");

async function main() {
  const [htmlPath, pdfPath] = process.argv.slice(2);
  if (!htmlPath || !pdfPath) {
    console.error("Usage: node print_html_to_pdf.js <input.html> <output.pdf>");
    process.exit(1);
  }

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const url = `file://${path.resolve(htmlPath)}`;
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => window.__MERMAID_DONE__ === true, null, { timeout: 60000 });
  await page.pdf({
    path: pdfPath,
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true,
    margin: { top: "0", right: "0", bottom: "0", left: "0" },
  });
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

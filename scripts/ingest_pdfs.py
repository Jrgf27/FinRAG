"""Convert a folder of PDFs into data/processed/corpus.jsonl (one record per page).

Caveat: pypdf extracts digital PDFs well but mangles tables and fails on scanned
documents. For table-heavy or scanned filings, swap in pdfplumber or an OCR step.
For a clean demo, prefer digitally-generated filings (e.g. SEC EDGAR).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import typer
from pypdf import PdfReader

app = typer.Typer()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@app.command()
def main(
    src: Path = typer.Argument(..., help="Folder of PDFs"),
    out: Path = typer.Option(Path("data/processed/corpus.jsonl")),
    company: str = typer.Option(None, help="Optional company label applied to all"),
    period: str = typer.Option(None, help="Optional period label, e.g. 'FY2025 Q3'"),
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(src.glob("*.pdf"))
    records = []
    for pdf in pdfs:
        doc_id = _slug(pdf.stem)
        reader = PdfReader(str(pdf))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < 50:  # skip blank / cover pages
                continue
            records.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "source": pdf.name,
                    "company": company,
                    "period": period,
                    "page": page_num,
                }
            )
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    typer.echo(f"Wrote {len(records)} page-records from {len(pdfs)} PDFs to {out}")


if __name__ == "__main__":
    app()

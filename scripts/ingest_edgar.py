"""Ingest an SEC EDGAR filing (HTML) directly into corpus.jsonl.

EDGAR serves 10-Q/10-K as HTML, not PDF. Going HTML -> text skips the lossy
PDF conversion step where financial tables get mangled. SEC requires a
descriptive User-Agent on automated requests.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import typer
from bs4 import BeautifulSoup

app = typer.Typer()

# SEC requires a User-Agent identifying you. Replace with your real details.
HEADERS = {"User-Agent": "Test test@example.com"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@app.command()
def main(
    url: str = typer.Argument(..., help="Direct EDGAR document URL (the .htm filing)"),
    company: str = typer.Option(..., help="e.g. 'Microsoft'"),
    period: str = typer.Option(..., help="e.g. 'FY2025 Q3'"),
    out: Path = typer.Option(Path("data/processed/corpus.jsonl")),
    append: bool = typer.Option(False, help="Append instead of overwrite"),
    chars_per_record: int = typer.Option(6000, help="Split size per record"),
) -> None:
    resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    doc_id = _slug(f"{company}-{period}")
    # Split into record-sized blocks; the chunker re-splits these into windows.
    blocks = [text[i:i + chars_per_record] for i in range(0, len(text), chars_per_record)]
    records = [
        {"doc_id": doc_id, "text": b.strip(), "source": url.split("/")[-1],
         "company": company, "period": period, "page": i + 1}
        for i, b in enumerate(blocks) if len(b.strip()) > 100
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    typer.echo(f"Wrote {len(records)} records ({len(text)} chars) from {company} {period}")


if __name__ == "__main__":
    app()
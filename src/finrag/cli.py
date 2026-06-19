"""Command-line interface: `finrag ingest` and `finrag ask`."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .pipeline import FinRAG

app = typer.Typer(help="Hybrid RAG over financial & research reports.")
console = Console()


@app.command()
def ask(
    question: str,
    corpus: Path = typer.Option(Path("data/processed/corpus.jsonl"), help="JSONL corpus"),
) -> None:
    """Answer a question against the corpus."""
    rag = FinRAG.from_jsonl(corpus)
    answer = rag.ask(question)
    console.print(f"\n[bold]{answer.text}[/bold]\n")
    if answer.citations:
        console.print("[dim]Sources:[/dim]")
        for c in answer.citations:
            meta = " | ".join(filter(None, [c.company, c.period, c.source]))
            console.print(f"  • {meta}")


@app.command()
def inspect(
    question: str,
    corpus: Path = typer.Option(Path("data/processed/corpus.jsonl")),
) -> None:
    """Show what the retriever returns, without generating an answer."""
    rag = FinRAG.from_jsonl(corpus)
    for sc in rag.retriever.retrieve(question):
        console.print(f"[{sc.score:.3f}] {sc.chunk.company} {sc.chunk.period}: {sc.chunk.text[:120]}…")


if __name__ == "__main__":
    app()

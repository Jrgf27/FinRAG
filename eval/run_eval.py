"""Eval runner with ablations.

Runs the labelled gold set through the pipeline under several configurations and
prints a comparison table. The ablations are the point: they quantify what each
retrieval stage contributes, turning "I built a fancy pipeline" into "reranking
lifted nDCG@6 from X to Y".

Configs compared:
  dense_only        : semantic search alone (the naive baseline)
  hybrid            : dense + BM25 + RRF
  hybrid_rerank     : + cross-encoder reranking
  full              : + query rewriting   (the shipped system)

Run:  python eval/run_eval.py --gold eval/gold.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finrag.config import settings
from finrag.pipeline import ingest
from finrag.retriever import HybridRetriever
from finrag.vectorstore import VectorStore

from metrics import aggregate, mrr, ndcg_at_k, recall_at_k

app = typer.Typer()
console = Console()

# (label, settings-overrides) — applied before building the retriever.
ABLATIONS = {
    "dense_only":    {"use_query_rewriting": False, "_mode": "dense"},
    "hybrid":        {"use_query_rewriting": False, "_mode": "hybrid"},
    "hybrid_rerank": {"use_query_rewriting": False, "_mode": "rerank"},
    "full":          {"use_query_rewriting": True,  "_mode": "rerank"},
}


def _retrieve_for_mode(retriever: HybridRetriever, query: str, mode: str) -> list[str]:
    """Return retrieved chunk_ids for a given ablation mode."""
    if mode == "dense":
        scored = retriever._dense(query)[: settings.rerank_top_k]
    elif mode == "hybrid":
        lists = [retriever._dense(query), retriever._sparse(query)]
        scored = retriever._rrf(lists)[: settings.rerank_top_k]
    else:  # rerank (optionally with rewriting handled by full retrieve)
        scored = retriever.retrieve(query)
    return [s.chunk.chunk_id for s in scored]


@app.command()
def main(
    corpus: Path = typer.Option(Path("data/processed/corpus.jsonl")),
    gold: Path = typer.Option(Path("eval/gold.jsonl")),
    k: int = typer.Option(6),
) -> None:
    records = [json.loads(ln) for ln in corpus.read_text().splitlines() if ln.strip()]
    chunks = ingest(records)
    store = VectorStore()
    store.upsert(chunks)
    retriever = HybridRetriever(store, chunks)

    gold_rows = [json.loads(ln) for ln in gold.read_text().splitlines() if ln.strip()]

    table = Table(title=f"Retrieval ablation (n={len(gold_rows)} questions, k={k})")
    table.add_column("config")
    table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"nDCG@{k}", justify="right")

    for label, overrides in ABLATIONS.items():
        settings.use_query_rewriting = overrides["use_query_rewriting"]
        mode = overrides["_mode"]
        per_q = []
        for row in gold_rows:
            relevant = set(row["relevant_chunk_ids"])
            retrieved = _retrieve_for_mode(retriever, row["question"], mode)
            per_q.append(
                {
                    "recall": recall_at_k(retrieved, relevant, k),
                    "mrr": mrr(retrieved, relevant),
                    "ndcg": ndcg_at_k(retrieved, relevant, k),
                }
            )
        table.add_row(
            label,
            f"{aggregate(per_q, 'recall'):.3f}",
            f"{aggregate(per_q, 'mrr'):.3f}",
            f"{aggregate(per_q, 'ndcg'):.3f}",
        )

    console.print(table)
    console.print(
        "\n[dim]Baseline = dense_only. Read each row against it to see the "
        "marginal contribution of fusion, reranking and query rewriting.[/dim]"
    )


if __name__ == "__main__":
    app()

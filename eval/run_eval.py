"""Eval runner with ablations.

Runs the labelled gold set through several retrieval configurations and prints a
comparison table. The ablations isolate what each stage contributes, turning
"I built a pipeline" into "reranking lifted nDCG@6 from X to Y".

Configs compared:
  dense_only     : semantic vector search alone (the naive baseline)
  hybrid         : dense + BM25 fused with Reciprocal Rank Fusion
  hybrid_rerank  : + cross-encoder reranking of the fused pool
  full           : + query rewriting (the shipped system)

Measurement principle: each config retrieves a CANDIDATE POOL of the same size
(`pool`, default 20), and metrics are computed at k over that pool's top-k. This
keeps the comparison fair — every stage is judged on the same-sized output, so a
later stage that *reorders* the pool (rerank) is credited for moving relevant
chunks up, not penalised by an arbitrary early truncation.

No global state is mutated between rows: rewriting is passed explicitly.

Run:  python eval/run_eval.py --gold eval/gold.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finrag.pipeline import ingest
from finrag.retriever import HybridRetriever
from finrag.vectorstore import VectorStore

from metrics import aggregate, mrr, ndcg_at_k, recall_at_k

app = typer.Typer()
console = Console()


def retrieve(retriever: HybridRetriever, query: str, mode: str, pool: int) -> list[str]:
    """Return ranked chunk_ids for a config, all cut to the same pool size.

    mode:
      dense   -> dense candidates only
      hybrid  -> dense + sparse, fused with RRF
      rerank  -> fused pool rescored by the cross-encoder
      full    -> rerank, but with query rewriting expanding the query set first
    """
    if mode == "dense":
        scored = retriever._dense(query)[:pool]

    elif mode == "hybrid":
        fused = retriever._rrf([retriever._dense(query), retriever._sparse(query)])
        scored = fused[:pool]

    elif mode in ("rerank", "full"):
        queries = [query]
        if mode == "full":
            # rewriting expands the query set; rerank still scores vs the original
            queries = retriever.rewrite(query)
        ranked_lists = []
        for q in queries:
            ranked_lists.append(retriever._dense(q))
            ranked_lists.append(retriever._sparse(q))
        fused = retriever._rrf(ranked_lists)
        # rerank the fused shortlist, then keep `pool` (reranker may return fewer)
        reranked = retriever._rerank(query, fused[:pool], top_k=pool)
        scored = reranked[:pool]

    else:
        raise ValueError(mode)

    return [s.chunk.chunk_id for s in scored]


CONFIGS = {
    "dense_only": "dense",
    "hybrid": "hybrid",
    "hybrid_rerank": "rerank",
    "full": "full",
}


@app.command()
def main(
    corpus: Path = typer.Option(Path("data/processed/corpus.jsonl")),
    gold: Path = typer.Option(Path("eval/gold.jsonl")),
    k: int = typer.Option(6),
    pool: int = typer.Option(20, help="Candidate pool size each config retrieves"),
) -> None:
    records = [json.loads(ln) for ln in corpus.read_text().splitlines() if ln.strip()]
    chunks = ingest(records)
    store = VectorStore()
    store.upsert(chunks)
    retriever = HybridRetriever(store, chunks)

    gold_rows = [json.loads(ln) for ln in gold.read_text().splitlines() if ln.strip()]

    table = Table(title=f"Retrieval ablation (n={len(gold_rows)} questions, k={k}, pool={pool})")
    table.add_column("config")
    table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"nDCG@{k}", justify="right")

    for label, mode in CONFIGS.items():
        per_q = []
        for row in gold_rows:
            relevant = set(row["relevant_chunk_ids"])
            retrieved = retrieve(retriever, row["question"], mode, pool)
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
        "\n[dim]Baseline = dense_only. Each config retrieves the same-sized pool, "
        "so reordering stages are judged fairly. Metrics computed at k over each "
        "config's top-k.[/dim]"
    )


if __name__ == "__main__":
    app()
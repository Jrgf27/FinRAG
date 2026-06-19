"""Evaluation metrics.

Two families:

Retrieval metrics (need only labelled relevant chunk_ids per question):
  - recall@k      : did we pull the gold passages into the top k?
  - MRR           : how high is the first relevant passage?
  - nDCG@k        : rank-quality, rewarding relevant passages near the top.

Generation metrics (need an LLM judge or reference answer):
  - faithfulness  : fraction of answer claims supported by the contexts
                    (guards against hallucinated figures — critical in finance).
  - citation_accuracy : did cited passages actually contain the claim?

Reporting metrics with named baselines and the right denominator is the signal
that distinguishes real LLM work from demo-ware, so the harness is explicit
about what each number means.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0

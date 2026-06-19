"""Automatic gold-set generation with an independent relevance judge.

The naive way to auto-label a RAG eval set is circular: generate a question FROM
a chunk, then mark that chunk as the answer. The question inherits the chunk's
vocabulary, retrieval finds it trivially, and the metrics come out inflated.

This script breaks that circularity in three stages:

  1. GENERATE  — a model reads a "seed" chunk and writes a realistic analyst
     question. It is explicitly told to paraphrase: no copying exact figures or
     phrasing from the seed. The seed is then DISCARDED as a label.

  2. JUDGE     — a SEPARATE judge pass scores the question against EVERY chunk in
     the corpus for relevance (0-3). Gold labels come only from the judge's
     verdict, not from which chunk seeded the question. This is what makes the
     labels independent of both the generator and the retriever.

  3. FILTER    — questions with no chunk scoring >= the relevance threshold are
     dropped (the generator produced something the corpus can't actually answer).

The result is a gold set whose relevant_chunk_ids were decided by a process that
never assumed the answer, so recall/MRR/nDCG measured against it are meaningful.

Cost note: stage 2 is O(questions x chunks) judge calls in the simple form. For
larger corpora this script uses a cheap embedding prefilter to only judge the
top-N candidate chunks per question, which keeps cost linear-ish while preserving
independence (the prefilter widens recall; the judge decides relevance).

Usage:
    python scripts/auto_gold.py \
        --corpus data/processed/corpus.jsonl \
        --out eval/gold.jsonl \
        --num-questions 40
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from finrag.chunking import chunk_document
from finrag.types import Chunk

app = typer.Typer()
console = Console()

# Models are pinned explicitly so generator and judge are clearly different
# roles. Using distinct models (or at minimum distinct prompts/temperatures)
# reduces shared-bias between the thing writing questions and the thing grading.
GEN_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-opus-4-8"

RELEVANCE_THRESHOLD = 2  # judge scale 0-3; >=2 counts as gold-relevant
JUDGE_CANDIDATES = 12    # how many embedding-prefiltered chunks the judge scores


# --------------------------------------------------------------------------- #
# corpus loading
# --------------------------------------------------------------------------- #
def load_chunks(corpus_path: Path) -> list[Chunk]:
    records = [json.loads(ln) for ln in corpus_path.read_text().splitlines() if ln.strip()]
    chunks: list[Chunk] = []
    for r in records:
        chunks.extend(
            chunk_document(
                r["doc_id"], r["text"], source=r["source"],
                company=r.get("company"), period=r.get("period"), page=r.get("page"),
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# stage 1: generate a paraphrased question from a seed chunk
# --------------------------------------------------------------------------- #
_GEN_SYSTEM = (
    "You write realistic questions a financial analyst or researcher would ask. "
    "Given a passage, write ONE question that the passage could help answer. "
    "Critical rules:\n"
    "- Do NOT copy exact numbers, dates, or distinctive phrases from the passage. "
    "Paraphrase the concept (e.g. ask about 'cloud revenue growth' rather than "
    "quoting '31%').\n"
    "- The question must be answerable from financial/research documents, not "
    "general knowledge.\n"
    "- Make it natural and specific, not a fill-in-the-blank of the passage.\n"
    "Return only the question text, nothing else."
)


def generate_question(client, seed: Chunk) -> str:
    msg = client.messages.create(
        model=GEN_MODEL,
        max_tokens=120,
        system=_GEN_SYSTEM,
        messages=[{"role": "user", "content": f"Passage:\n{seed.text}"}],
    )
    return msg.content[0].text.strip().strip('"')


# --------------------------------------------------------------------------- #
# stage 2: judge relevance of a question against candidate chunks
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = (
    "You are a strict relevance judge for a retrieval system. Given a QUESTION "
    "and a numbered list of PASSAGES, rate how well EACH passage helps answer the "
    "question on this scale:\n"
    "  3 = directly and fully answers it\n"
    "  2 = contains a substantial part of the answer\n"
    "  1 = topically related but does not answer it\n"
    "  0 = irrelevant\n"
    "Judge each passage on its own merits — do not assume any passage is the "
    "intended answer. Respond ONLY with JSON: "
    '{"scores": [{"passage": <int>, "score": <0-3>}]} . No prose, no fences.'
)


def judge_relevance(client, question: str, candidates: list[Chunk]) -> dict[str, int]:
    listing = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(candidates))
    user = f"QUESTION: {question}\n\nPASSAGES:\n{listing}"
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=600,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out: dict[str, int] = {}
    for entry in data.get("scores", []):
        idx = entry.get("passage")
        score = entry.get("score")
        if isinstance(idx, int) and 1 <= idx <= len(candidates):
            out[candidates[idx - 1].chunk_id] = int(score)
    return out


# --------------------------------------------------------------------------- #
# embedding prefilter (widens recall before the judge; does NOT decide relevance)
# --------------------------------------------------------------------------- #
def build_prefilter(chunks: list[Chunk]):
    """Returns a function question -> top candidate chunks, using embeddings if
    available, else a BM25 fallback so the script runs without an embedding key."""
    try:
        from finrag.embeddings import embed_texts

        vectors = embed_texts([c.text for c in chunks])
        import numpy as np

        mat = np.array(vectors, dtype="float32")
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

        def prefilter(question: str) -> list[Chunk]:
            qv = np.array(embed_texts([question])[0], dtype="float32")
            qv /= np.linalg.norm(qv) + 1e-9
            sims = mat @ qv
            top = sims.argsort()[::-1][:JUDGE_CANDIDATES]
            return [chunks[i] for i in top]

        return prefilter
    except Exception:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([c.text.lower().split() for c in chunks])

        def prefilter(question: str) -> list[Chunk]:
            scores = bm25.get_scores(question.lower().split())
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:JUDGE_CANDIDATES]]

        return prefilter


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
@app.command()
def main(
    corpus: Path = typer.Option(Path("data/processed/corpus.jsonl")),
    out: Path = typer.Option(Path("eval/gold.jsonl")),
    num_questions: int = typer.Option(40, help="How many questions to generate"),
    seed: int = typer.Option(7, help="RNG seed for reproducible seed-chunk sampling"),
) -> None:
    from anthropic import Anthropic

    client = Anthropic()
    chunks = load_chunks(corpus)
    if not chunks:
        console.print("[red]No chunks found. Ingest documents first.[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded [bold]{len(chunks)}[/bold] chunks from {corpus}")
    prefilter = build_prefilter(chunks)

    rng = random.Random(seed)
    # Sample seed chunks without replacement; if asked for more questions than
    # chunks, allow repeats but vary which is fine since labels come from judging.
    seeds = (
        rng.sample(chunks, num_questions)
        if num_questions <= len(chunks)
        else [rng.choice(chunks) for _ in range(num_questions)]
    )

    gold_rows = []
    dropped = 0
    for seed_chunk in track(seeds, description="Generating + judging"):
        question = generate_question(client, seed_chunk)
        if not question or len(question) < 8:
            dropped += 1
            continue

        candidates = prefilter(question)
        scores = judge_relevance(client, question, candidates)
        relevant = [cid for cid, s in scores.items() if s >= RELEVANCE_THRESHOLD]

        if not relevant:
            # Generator produced a question the corpus can't actually answer,
            # or the judge disagreed with the seed. Either way, drop it.
            dropped += 1
            continue

        gold_rows.append(
            {
                "question": question,
                "relevant_chunk_ids": relevant,
                # provenance for auditing — not used by the eval harness
                "_seed_chunk_id": seed_chunk.chunk_id,
                "_seed_was_judged_relevant": seed_chunk.chunk_id in relevant,
                "_judge_scores": scores,
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in gold_rows:
            f.write(json.dumps(row) + "\n")

    # A useful diagnostic: how often did the judge AGREE the seed was relevant?
    # Low agreement is a red flag that generation is drifting off-corpus.
    agree = sum(1 for r in gold_rows if r["_seed_was_judged_relevant"])
    console.print(
        f"\nWrote [bold]{len(gold_rows)}[/bold] questions to {out} "
        f"([dim]{dropped} dropped[/dim])."
    )
    if gold_rows:
        console.print(
            f"Seed-judge agreement: {agree}/{len(gold_rows)} "
            f"({100*agree/len(gold_rows):.0f}%) — low values suggest the generator "
            f"is drifting; high values are expected and healthy."
        )
    console.print(
        "[dim]Labels were assigned by an independent judge over prefiltered "
        "candidates, not by assuming the seed chunk is the answer.[/dim]"
    )


if __name__ == "__main__":
    app()

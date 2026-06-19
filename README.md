# FinRAG — Hybrid Retrieval over Financial & Research Reports

A production-shaped RAG system for question answering over 10-Q filings, earnings
disclosures, and market-research reports. Built to demonstrate the retrieval and
evaluation engineering that separates a shippable system from a notebook demo.

The headline isn't "it uses a vector database." It's that **every retrieval stage
is measured**, the pipeline goes well past naive semantic search, and answers are
**grounded with enforced citations** because in a financial context an unsourced
number is a liability.

---

## What's interesting here

**Hybrid retrieval, not just vector search.** Dense (semantic) and sparse (BM25)
retrieval are fused with Reciprocal Rank Fusion. Embeddings smear exact tokens —
tickers, dates, line-item names, precise figures — together; BM25 catches them.
RRF merges the two ranked lists without having to calibrate incomparable score
scales.

**Cross-encoder reranking.** The fused shortlist is rescored by a query-document
cross-encoder for top-k precision. This is consistently the single highest-impact
stage, and the eval harness quantifies exactly how much it contributes.

**Query rewriting.** Each question is expanded into several paraphrases (synonyms,
expanded acronyms, formal metric names) so recall doesn't hinge on the user's
exact phrasing. Reranking is done against the *original* question so final ordering
reflects what was actually asked.

**Eval-first design.** The point of the repo is `eval/` as much as `src/`. The
harness reports retrieval metrics (recall@k, MRR, nDCG@k) against a labelled gold
set and runs **ablations** that isolate each stage's marginal contribution. Answer
quality is scored by an LLM judge for **faithfulness** (fraction of answer claims
entailed by the retrieved context) and **citation accuracy**.

**Provenance throughout.** Every chunk carries company / period / source / page
metadata from ingestion to citation, so an answer can always be traced to the
filing and page it came from.

---

## Architecture

```
                      ┌─────────────────┐
   question ─────────▶│  query rewriting │  (LLM, cheap model)
                      └────────┬─────────┘
                     original + N paraphrases
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
        ┌───────────────┐              ┌──────────────┐
        │ dense (vector)│              │ sparse (BM25)│
        └───────┬───────┘              └──────┬───────┘
                └──────────────┬───────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │ Reciprocal Rank Fusion   │
                  └───────────┬──────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │ cross-encoder rerank     │  → top-k passages
                  └───────────┬──────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │ grounded generation      │  → answer + citations
                  │ (cite-or-abstain prompt) │
                  └─────────────────────────┘
```

Every box is independently toggleable so the eval harness can ablate it.

---

## Quickstart

```bash
pip install -e ".[dev]"

# Embeddings run locally (sentence-transformers) — no embedding API key needed.
# Generation, query rewriting and the eval judge use Anthropic:
cp .env.example .env        # then add your key
# .env: ANTHROPIC_API_KEY=...

# ingest a real filing from SEC EDGAR (HTML, no PDF conversion)
python scripts/ingest_edgar.py \
  "https://www.sec.gov/Archives/edgar/data/789019/000119312525256321/msft-20250930.htm" \
  --company "Microsoft" --period "FY2026 Q1"

# add a second issuer to the same corpus
python scripts/ingest_edgar.py "<second filing .htm url>" \
  --company "Apple" --period "FY2026 Q1" --append

# see what the retriever returns, without generation
finrag inspect "what drove growth in Microsoft cloud revenue"

# ask a grounded, cited question
finrag ask "How did Microsoft's cloud revenue grow this quarter?"

# build a gold set (independent-judge labelling) and run the eval
python scripts/auto_gold.py --num-questions 40
python eval/run_eval.py --gold eval/gold.jsonl
```

Embeddings are local by default, so the only external dependency is the Anthropic
API. The vector store defaults to Qdrant's in-memory mode — the repo runs end to
end with no other services. Point `FINRAG_QDRANT_URL` at a server to persist, and
install the optional `[openai]` extra if you prefer cloud embeddings.

---

## Evaluation

The harness runs the gold set through four configurations and prints a comparison
table:

| config          | what it adds                          |
|-----------------|---------------------------------------|
| `dense_only`    | semantic search alone — the baseline  |
| `hybrid`        | + BM25 + Reciprocal Rank Fusion       |
| `hybrid_rerank` | + cross-encoder reranking             |
| `full`          | + query rewriting (the shipped system)|

Read every row against the `dense_only` baseline to see the marginal contribution
of each stage. Reporting numbers this way — with a named baseline and the correct
denominator — is deliberate: a metric without a baseline is noise.

> **Note on the numbers.** The bundled corpus is a small illustrative sample, so
> treat the included results as a demonstration of the *methodology*. Swap in a
> real corpus (a folder of 10-Qs) and a larger gold set to get meaningful deltas.
> Generation faithfulness is scored by `eval/judge.py` using an LLM grader that is
> kept separate from the answer model.

### Building a gold set

The gold set (`eval/gold.jsonl`) pairs questions with the `chunk_id`s that should
answer them. Two ways to build one:

**Hand-labelled** — most trustworthy. Ingest your corpus, run `finrag inspect` to
find which chunks answer a question, and write the pairs by hand.

**Auto-generated with an independent judge** — `scripts/auto_gold.py`. This avoids
the circularity that inflates synthetic eval sets: instead of generating a question
from a chunk and then labelling that same chunk as the answer, it (1) generates a
*paraphrased* question from a seed chunk, then (2) discards the seed and has a
**separate judge model** score relevance against prefiltered candidates across the
whole corpus. Labels come only from the judge, so the set isn't grading the
retriever against itself.

```bash
python scripts/auto_gold.py --corpus data/processed/corpus.jsonl \
    --out eval/gold.jsonl --num-questions 40
```

It reports a seed–judge agreement rate as a drift diagnostic, and drops questions
the corpus can't actually answer. The output is synthetic and labelled as such —
a quick human pass over it before publishing metrics is still recommended.

---

## Layout

```
src/finrag/
  config.py        all tunable parameters in one place (reproducible eval runs)
  types.py         typed contracts shared across the pipeline
  chunking.py      token-aware splitting with provenance + offline fallback
  embeddings.py    embedding provider (isolated behind one swappable function)
  vectorstore.py   Qdrant wrapper, in-memory by default
  retriever.py     ★ hybrid retrieval: rewrite → dense+sparse → RRF → rerank
  generation.py    grounded answers with parsed, enforced citations
  pipeline.py      end-to-end wiring + ingest
  cli.py           `finrag ask` / `finrag inspect`
eval/
  metrics.py       recall@k, MRR, nDCG@k
  judge.py         LLM-as-judge faithfulness scoring
  run_eval.py      ablation runner
tests/             deterministic, key-free unit tests (CI-safe)
```

---

## Design notes & trade-offs

- **Why RRF over weighted score fusion?** Dense cosine scores and BM25 scores live
  on different, non-comparable scales. RRF uses rank position only, so it needs no
  per-corpus tuning — a sensible default that holds up across document sets.
- **Why a cross-encoder reranker instead of a bigger top-k into the LLM?** Stuffing
  more passages raises cost and latency and dilutes the context with near-misses.
  Reranking buys top-k precision far more cheaply.
- **Why cite-or-abstain?** For financial QA, a confident wrong figure is worse than
  "the documents don't say." The generation prompt forbids inventing numbers and
  requires a passage tag for every claim; tags are parsed back into structured
  citations.
- **Swappability.** Embeddings, the answer model, the reranker, and the judge are
  each isolated so any one can be replaced (local embeddings, a different provider,
  a stronger judge) without touching the pipeline.

## Productionizing from here

Streaming responses, async batched embedding, incremental re-indexing, retrieval
tracing/observability per query, cost-per-query accounting, and a guardrail pass on
the generated answer are the natural next steps — and map directly onto the
production concerns this is meant to demonstrate.
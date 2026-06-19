# Results

Evaluated on a two-issuer corpus (Microsoft FY2026 Q1 and Apple FY2026 Q1 10-Q
filings, ~212 chunks) against a 32-question gold set. Gold labels were assigned
by an independent LLM judge over the full corpus — not by assuming the chunk a
question was generated from is its answer — so the metrics are not grading the
retriever against itself (see `scripts/auto_gold.py`). Embeddings are a local
384-dim model (`all-MiniLM-L6-v2`); no embedding API is used.

| config          | recall@6 |   MRR | nDCG@6 |
|-----------------|---------:|------:|-------:|
| dense_only      |    0.515 | 0.672 |  0.490 |
| hybrid          |    0.547 | 0.774 |  0.536 |
| hybrid_rerank   |    0.634 | 0.847 |  0.634 |
| full            |    0.658 | 0.852 |  0.657 |

Each config retrieves a same-sized candidate pool (20) and is scored at k=6, so
the comparison is apples-to-apples — an invariant enforced by
`tests/test_eval_invariants.py`.

## What the numbers say

**Every stage contributes, in order.** Relative to dense-only retrieval:

- **Hybrid fusion** (dense + BM25 via Reciprocal Rank Fusion) lifts nDCG@6 from
  0.490 to 0.536 and MRR from 0.672 to 0.774. Sparse retrieval catches exact
  terms — tickers, defined terms, line-item names — that a compact embedding
  model blurs together.
- **Cross-encoder reranking** is the largest single jump: nDCG to 0.634, MRR to
  0.847. An independent diagnostic agrees — it improves the median rank of
  relevant chunks from 4 to 2.
- **Query rewriting** adds a further recall gain (to 0.658) by expanding the
  query into paraphrases so retrieval doesn't hinge on exact wording.

The full pipeline lifts nDCG@6 by ~34% over the dense baseline (0.490 → 0.657)
and MRR from 0.672 to 0.852.

## The conditional finding worth noting

The value of hybrid (sparse) fusion **depends on how strong the dense retriever
is.** With this local 384-dim embedding model, dense retrieval alone is modest
(0.515 recall), so BM25 contributes real signal and fusion clearly helps. In a
separate run with a strong hosted 3072-dim embedding model, dense retrieval was
much stronger on its own and sparse fusion added little — the embeddings already
captured what BM25 would have surfaced.

The takeaway: "hybrid search helps" is not unconditional. It pays off most when
the dense retriever is lightweight, and least when it is already excellent. This
is a property to *measure* for a given embedding choice, not to assume. The
pipeline keeps hybrid on by default (`use_hybrid=True`) because it helps on the
shipped local-embedding configuration.

## Caveats

- n=32 synthetic questions over two filings. The stage-by-stage *trend* is the
  result; absolute figures are indicative and will shift with corpus size,
  question mix, and embedding model. A client re-running this should expect the
  same ordering, not identical decimals.
- The gold set is LLM-generated with an independent judge (~72% seed–judge
  agreement) — a strong synthetic baseline, not human-labelled ground truth.
- Query rewriting is mildly non-deterministic (it calls an LLM), so the `full`
  row varies by ~1-2 points between runs.
- Financial tables extract poorly from filing HTML, so the corpus and gold set
  lean toward narrative/MD&A content, where text-based RAG is most reliable.
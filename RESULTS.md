# Results

Evaluated on a two-issuer corpus (Microsoft FY2026 Q1 and Apple FY2026 Q1
10-Q filings, 212 chunks) against a 32-question gold set. Gold labels were
assigned by an independent LLM judge over the full corpus — not by assuming the
chunk a question was generated from is its answer — so the metrics are not
grading the retriever against itself (see `scripts/auto_gold.py`).

| config          | recall@6 |   MRR | nDCG@6 |
|-----------------|---------:|------:|-------:|
| dense_only      |    0.730 | 0.841 |  0.726 |
| hybrid          |    0.655 | 0.788 |  0.637 |
| hybrid_rerank   |    0.674 | 0.857 |  0.674 |
| full            |    0.695 | 0.865 |  0.690 |

Each config retrieves a same-sized candidate pool (20) and is scored at k=6, so
the comparison is apples-to-apples (an invariant enforced by
`tests/test_eval_invariants.py`).

## What the numbers say

**Cross-encoder reranking is the clear win.** It lifts MRR from 0.841 (dense
baseline) to 0.857, and to 0.865 with query rewriting. An independent rank
diagnostic confirms it: the median rank of relevant chunks improves from 4
(dense) to 2 (reranked). Reranking reliably pulls the right passage toward the
top, which is what matters most when only the top few passages are fed to the
generator.

**Sparse (BM25) fusion did not pay off on this corpus**, and the eval says so
plainly: hybrid retrieval is *below* dense on recall and nDCG. Diagnostics ruled
out a fusion bug — RRF drops only 1 of 139 relevant chunks from the pool — so
this is a genuine property of the data, not an implementation fault. On
semantically rich, single-domain filing prose, dense embeddings already capture
the relevant passages; BM25 mostly adds rank noise, mildly demoting relevant
chunks (median rank 4 → 4.5) rather than surfacing new ones. This is consistent
with the well-known result that lexical retrieval helps most when queries hinge
on rare exact terms (tickers, defined terms, identifiers) — which a two-company
narrative corpus has few of.

**Decision:** the shipped pipeline uses **dense retrieval + cross-encoder
reranking**, with hybrid fusion implemented but disabled by default
(`use_hybrid` / `FINRAG_USE_HYBRID`). Query rewriting is kept for its small MRR
gain. Reporting a fashionable technique that *didn't* help on the target data —
and explaining why — is the point: component choices here are driven by
measurement, not by what's in vogue.

## Caveats

- The gold set is LLM-generated with an independent judge (72% seed–judge
  agreement). It is a strong synthetic baseline, not human-labelled ground
  truth; a human pass would tighten it further.
- Financial tables extract poorly from filing HTML, so table-cell lookups are
  weaker than narrative/MD&A questions. The corpus and gold set lean narrative,
  which reflects where text-based RAG is actually reliable.
- Two filings is a small corpus. The methodology scales to more; the absolute
  numbers would shift with corpus size and question mix.
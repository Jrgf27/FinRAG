"""Hybrid retrieval pipeline — the core of the project.

Stages, in order:
  1. (optional) Query rewriting: expand the user question into several
     paraphrases / sub-questions so recall doesn't hinge on exact wording.
  2. Dense retrieval: semantic vector search over each (rewritten) query.
  3. Sparse retrieval: BM25 over the same corpus to catch exact terms, tickers,
     and numbers that embeddings smear together.
  4. Reciprocal Rank Fusion: merge the ranked lists without needing to calibrate
     across incomparable score scales.
  5. Cross-encoder reranking: a query-document model rescores the fused
     shortlist for precision at the top — the step that most moves answer quality.

Each stage is independently toggleable so the eval harness can ablate them and
show the contribution of every component.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import embed_query
from .types import Chunk, ScoredChunk

if TYPE_CHECKING:
    from .vectorstore import VectorStore


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


class HybridRetriever:
    def __init__(self, store: "VectorStore", corpus: list[Chunk]) -> None:
        self.store = store
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in corpus])
        self._reranker = None  # lazy-loaded; heavy import

    # ---- stage 1: query rewriting -------------------------------------------
    def rewrite(self, query: str) -> list[str]:
        if not settings.use_query_rewriting:
            return [query]
        from anthropic import Anthropic

        client = Anthropic()
        prompt = (
            "Rewrite the user's question into 3 alternative search queries that "
            "would help retrieve relevant passages from financial and research "
            "reports. Vary terminology (synonyms, expanded acronyms, formal "
            "metric names). Return one query per line, no numbering.\n\n"
            f"Question: {query}"
        )
        msg = client.messages.create(
            model=settings.rewrite_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = [ln.strip("-• ").strip() for ln in msg.content[0].text.splitlines()]
        rewrites = [ln for ln in lines if ln]
        # Always keep the original query in the mix.
        return [query, *rewrites[:3]]

    # ---- stage 2: dense ------------------------------------------------------
    def _dense(self, query: str) -> list[ScoredChunk]:
        return self.store.search(embed_query(query), settings.dense_top_k)

    # ---- stage 3: sparse -----------------------------------------------------
    def _sparse(self, query: str) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.corpus, scores), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(chunk=c, score=float(s), retriever="sparse")
            for c, s in ranked[: settings.sparse_top_k]
        ]

    # ---- stage 4: reciprocal rank fusion ------------------------------------
    @staticmethod
    def _rrf(ranked_lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        k = settings.rrf_k
        agg: dict[str, float] = defaultdict(float)
        keep: dict[str, Chunk] = {}
        for lst in ranked_lists:
            for rank, sc in enumerate(lst):
                cid = sc.chunk.chunk_id
                agg[cid] += 1.0 / (k + rank + 1)
                keep[cid] = sc.chunk
        fused = [
            ScoredChunk(chunk=keep[cid], score=score, retriever="fused")
            for cid, score in agg.items()
        ]
        fused.sort(key=lambda s: s.score, reverse=True)
        return fused

    # ---- stage 5: cross-encoder rerank --------------------------------------
    def _rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.reranker_model)
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._reranker.predict(pairs)
        reranked = [
            ScoredChunk(chunk=c.chunk, score=float(s), retriever="reranked")
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked[: settings.rerank_top_k]

    # ---- public API ----------------------------------------------------------
    def retrieve(self, query: str) -> list[ScoredChunk]:
        queries = self.rewrite(query)
        ranked_lists: list[list[ScoredChunk]] = []
        for q in queries:
            ranked_lists.append(self._dense(q))
            ranked_lists.append(self._sparse(q))
        fused = self._rrf(ranked_lists)
        # Rerank against the ORIGINAL question, not the rewrites, so the final
        # ordering reflects what the user actually asked.
        return self._rerank(query, fused[: max(settings.dense_top_k, 25)])

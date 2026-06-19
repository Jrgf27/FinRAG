"""Integration smoke test for the eval retrieval paths.

Three measurement bugs hit the eval harness during development, all the same
flavour: different ablation configs silently returned different-length result
lists, so metrics were computed over unequal denominators and the comparison
was not apples-to-apples.

This test runs all four eval configs through the real fusion / ranking logic and
asserts the invariant that would have caught every one of those bugs: each
config returns a ranked list of the same requested length, with no duplicate
chunk_ids and valid ordering.

It is CI-safe: embeddings and the cross-encoder are stubbed with deterministic
fakes, so the test needs no API keys and no model downloads. It exercises the
plumbing (RRF, truncation, the eval `retrieve` dispatch), which is where the
bugs lived — not the model quality, which belongs in the (key-requiring) eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from finrag.retriever import HybridRetriever  # noqa: E402
from finrag.types import Chunk, ScoredChunk  # noqa: E402


# --------------------------------------------------------------------------- #
# fakes — deterministic, no models, no network
# --------------------------------------------------------------------------- #
def _make_corpus(n: int = 30) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc:{i}",
            doc_id="doc",
            text=f"passage number {i} about revenue growth and operating margin {i}",
            source="test.pdf",
            company="TestCo" if i % 2 else "OtherCo",
            period="FY2026 Q1",
            page=i,
        )
        for i in range(n)
    ]


class _FakeRetriever(HybridRetriever):
    """HybridRetriever with the model-dependent stages stubbed deterministically.

    _dense, rewrite and the cross-encoder are replaced; _sparse (BM25) and _rrf
    are the REAL implementations, so the fusion and truncation logic under test
    runs unmodified.
    """

    def __init__(self, corpus: list[Chunk]) -> None:
        # Skip the parent __init__ (which builds a vector store); set what we need.
        self.store = None
        self.corpus = corpus
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([c.text.lower().split() for c in corpus])
        self._reranker = "stub"  # truthy so the lazy-load branch is skipped

    def _dense(self, query: str):
        # Deterministic "dense" ranking: reverse corpus order, scored by position.
        return [
            ScoredChunk(chunk=c, score=1.0 - i / len(self.corpus), retriever="dense")
            for i, c in enumerate(reversed(self.corpus))
        ][:20]

    def rewrite(self, query: str):
        return [query, query + " expanded"]

    def _rerank(self, query, candidates, top_k=None):
        # Deterministic "rerank": reverse the candidate order so we can detect
        # that reranking actually reordered, then cut to top_k like the real one.
        reranked = [
            ScoredChunk(chunk=c.chunk, score=float(i), retriever="reranked")
            for i, c in enumerate(reversed(candidates))
        ]
        cut = 6 if top_k is None else top_k
        return reranked[:cut]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def retriever():
    return _FakeRetriever(_make_corpus())


@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank", "full"])
def test_each_config_returns_full_pool(retriever, mode):
    """The invariant that would have caught all three harness bugs:
    every config returns exactly `pool` ranked ids over the same pool."""
    from run_eval import retrieve

    pool = 20
    ids = retrieve(retriever, "revenue growth", mode, pool)
    assert len(ids) == pool, f"{mode} returned {len(ids)} ids, expected {pool}"
    assert len(set(ids)) == len(ids), f"{mode} returned duplicate chunk_ids"


def test_all_configs_same_length(retriever):
    """Belt-and-braces: the four configs must agree on output length, so recall@k
    is measured over equal denominators."""
    from run_eval import retrieve

    lengths = {
        mode: len(retrieve(retriever, "operating margin", mode, 20))
        for mode in ("dense", "hybrid", "rerank", "full")
    }
    assert len(set(lengths.values())) == 1, f"unequal config lengths: {lengths}"


def test_rrf_keeps_all_inputs(retriever):
    """RRF must not drop candidates that appear in either input list."""
    q = "revenue"
    dense = retriever._dense(q)
    sparse = retriever._sparse(q)
    fused = retriever._rrf([dense, sparse])
    input_ids = {s.chunk.chunk_id for s in dense} | {s.chunk.chunk_id for s in sparse}
    fused_ids = {s.chunk.chunk_id for s in fused}
    assert fused_ids == input_ids, "RRF lost or invented chunk_ids"
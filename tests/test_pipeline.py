"""Tests for the deterministic, network-free parts of the pipeline.

These run in CI with no API keys: chunking, reciprocal rank fusion, and the
retrieval metrics. The model-dependent paths are covered by the eval harness,
not unit tests, because they need keys and are non-deterministic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from finrag.chunking import chunk_document
from finrag.retriever import HybridRetriever
from finrag.types import Chunk, ScoredChunk
from metrics import mrr, ndcg_at_k, recall_at_k


def test_chunking_overlap_and_provenance():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_document("d1", text, source="s.pdf", company="X", period="FY25")
    assert len(chunks) > 1
    assert all(c.company == "X" and c.source == "s.pdf" for c in chunks)
    # chunk ids are unique
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def _sc(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, doc_id="d", text="t", source="s"),
        score=score,
        retriever="dense",
    )


def test_rrf_rewards_agreement():
    # 'a' is top in both lists, so it should win after fusion.
    dense = [_sc("a", 0.9), _sc("b", 0.8), _sc("c", 0.7)]
    sparse = [_sc("a", 5.0), _sc("c", 4.0), _sc("d", 3.0)]
    fused = HybridRetriever._rrf([dense, sparse])
    assert fused[0].chunk.chunk_id == "a"
    assert {s.chunk.chunk_id for s in fused} == {"a", "b", "c", "d"}


def test_recall_and_mrr():
    retrieved = ["x", "gold", "y"]
    relevant = {"gold"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert mrr(retrieved, relevant) == 0.5


def test_ndcg_monotonic_in_rank():
    relevant = {"g"}
    top = ndcg_at_k(["g", "a", "b"], relevant, 3)
    low = ndcg_at_k(["a", "b", "g"], relevant, 3)
    assert top > low
    assert abs(top - 1.0) < 1e-9

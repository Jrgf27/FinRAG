"""Vector store abstraction over Qdrant.

Uses Qdrant's in-memory mode by default so the repo runs end-to-end with zero
external services — reviewers can `pip install` and run the eval immediately.
Point FINRAG_QDRANT_URL at a server for the persistent path.
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings
from .embeddings import embed_texts
from .types import Chunk, ScoredChunk

# Stable namespace so the same chunk_id always maps to the same point UUID.
_NS = uuid.UUID("00000000-0000-0000-0000-0000f10a9c0e")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class VectorStore:
    def __init__(self, dim: int = 3072) -> None:
        # ":memory:" keeps the demo dependency-free; a URL persists.
        if settings.qdrant_url and settings.qdrant_url.startswith("http"):
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(":memory:")
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if settings.collection not in existing:
            self.client.create_collection(
                collection_name=settings.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, chunks: list[Chunk]) -> None:
        vectors = embed_texts([c.text for c in chunks])
        points = [
            PointStruct(
                id=_point_id(c.chunk_id),
                vector=v,
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=settings.collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[ScoredChunk]:
        result = self.client.query_points(
            collection_name=settings.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk(**p.payload), score=float(p.score), retriever="dense")
            for p in result.points
        ]

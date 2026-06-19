"""Embedding provider.

Local sentence-transformers model by default: no API key, no external calls,
which suits on-prem / data-sensitive deployments. Swap for OpenAI or Voyage by
editing this file alone — the rest of the system never imports an SDK directly.
"""
from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text. Batched for throughput."""
    vectors = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
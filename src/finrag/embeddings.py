"""Embedding provider.

Isolated behind one function so the rest of the system never imports an SDK
directly — swap OpenAI for a local model or Voyage by editing this file alone.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text. Batched for throughput."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]

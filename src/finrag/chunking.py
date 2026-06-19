"""Token-aware chunking.

Financial reports are dense with tables and section structure. Naive
character splitting cuts numbers away from their headers. This splitter works in
tokens (matching the embedding model's accounting) and keeps a configurable
overlap so a figure and its row label rarely land in different chunks.
"""
from __future__ import annotations

import hashlib

from .config import settings
from .types import Chunk


class _Encoder:
    """Lazy token encoder. Prefers tiktoken; falls back to whitespace tokens if
    the BPE table can't be fetched (e.g. offline CI), so chunking never blocks
    on a network download."""

    def __init__(self) -> None:
        self._enc = None
        self._tried = False

    def _load(self):
        if not self._tried:
            self._tried = True
            try:
                import tiktoken

                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None
        return self._enc

    def encode(self, text: str) -> list[int]:
        enc = self._load()
        if enc is not None:
            return enc.encode(text)
        # Fallback: treat whitespace-separated words as tokens.
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        enc = self._load()
        if enc is not None:
            return enc.decode(tokens)
        return ""  # not used in fallback path; see _split_fallback


_enc = _Encoder()


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{doc_id}:{idx}:{text[:64]}".encode()).hexdigest()[:12]
    return f"{doc_id}:{idx}:{h}"


def chunk_document(
    doc_id: str,
    text: str,
    *,
    source: str,
    company: str | None = None,
    period: str | None = None,
    page: int | None = None,
) -> list[Chunk]:
    """Split one document's text into overlapping token windows.

    Uses real BPE tokens when tiktoken is available, otherwise falls back to
    word windows so the pipeline runs anywhere.
    """
    size = settings.chunk_tokens
    overlap = settings.chunk_overlap_tokens
    step = size - overlap
    if step <= 0:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_tokens")

    real_tokens = _enc._load() is not None

    chunks: list[Chunk] = []
    if real_tokens:
        tokens = _enc.encode(text)
        windows = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + size]
            if window:
                windows.append(_enc.decode(window).strip())
            if start + size >= len(tokens):
                break
    else:
        words = text.split()
        windows = []
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if window:
                windows.append(" ".join(window).strip())
            if start + size >= len(words):
                break

    for idx, chunk_text in enumerate(windows):
        if not chunk_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, idx, chunk_text),
                doc_id=doc_id,
                text=chunk_text,
                source=source,
                company=company,
                period=period,
                page=page,
            )
        )
    return chunks

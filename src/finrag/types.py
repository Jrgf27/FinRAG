"""Typed contracts shared across the pipeline.

Keeping these in one place means retrieval, reranking, generation and eval all
speak the same language, and a passage can be traced from ingestion to citation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of text with provenance back to its source document."""
    chunk_id: str
    doc_id: str
    text: str
    # Provenance metadata — essential for financial work where a number is
    # worthless without knowing the filing, company and period it came from.
    source: str = Field(description="Filename or URL of the source document")
    company: str | None = None
    period: str | None = None  # e.g. "FY2025 Q3"
    page: int | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    # Where the candidate came from, for debugging the fusion stage.
    retriever: str  # "dense" | "sparse" | "fused" | "reranked"


class Citation(BaseModel):
    chunk_id: str
    source: str
    company: str | None = None
    period: str | None = None
    page: int | None = None


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation]
    contexts: list[ScoredChunk]  # what was actually fed to the model
    rewritten_queries: list[str] = Field(default_factory=list)

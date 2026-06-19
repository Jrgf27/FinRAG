"""End-to-end pipeline: ingest documents, then answer questions over them."""
from __future__ import annotations

import json
from pathlib import Path

from .chunking import chunk_document
from .generation import generate_answer
from .retriever import HybridRetriever
from .types import Answer, Chunk
from .vectorstore import VectorStore


def ingest(records: list[dict]) -> list[Chunk]:
    """Turn raw document records into chunks. Each record:
    {doc_id, text, source, company?, period?, page?}
    """
    chunks: list[Chunk] = []
    for r in records:
        chunks.extend(
            chunk_document(
                r["doc_id"], r["text"],
                source=r["source"], company=r.get("company"),
                period=r.get("period"), page=r.get("page"),
            )
        )
    return chunks


class FinRAG:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.store = VectorStore()
        self.store.upsert(chunks)
        self.retriever = HybridRetriever(self.store, chunks)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FinRAG":
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(ingest(records))

    def ask(self, question: str) -> Answer:
        contexts = self.retriever.retrieve(question)
        rewrites = self.retriever.rewrite(question) if False else []  # already done inside retrieve
        return generate_answer(question, contexts, rewrites)

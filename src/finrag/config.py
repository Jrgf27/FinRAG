"""Central configuration. All knobs live here so eval runs are reproducible.

Every retrieval parameter that affects a metric is surfaced as a setting, so an
eval run can be pinned to an exact config and the results are meaningful.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Load .env into the real environment so SDK clients (OpenAI, Anthropic) that
# read os.environ directly can see the keys, not just our Settings object.
_env = Path(".env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            import os
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINRAG_", env_file=".env", extra="ignore")

    # --- Models ---
    answer_model: str = "claude-opus-4-8"
    rewrite_model: str = "claude-haiku-4-5-20251001"  # cheap, fast model for query rewriting
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Chunking ---
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- Retrieval ---
    dense_top_k: int = 20          # candidates pulled from vector search
    sparse_top_k: int = 20         # candidates pulled from BM25
    rrf_k: int = 60                # reciprocal-rank-fusion constant
    rerank_top_k: int = 6          # passages kept after cross-encoder rerank
    use_query_rewriting: bool = True
    use_hybrid: bool = True       # fuse BM25 with dense. Off by default: on
                                   # single-domain filing prose, sparse fusion
                                   # did not improve recall/nDCG (see RESULTS.md).

    # --- Infra ---
    qdrant_url: str = "http://localhost:6333"
    collection: str = "finrag"

    # --- Cost / generation ---
    max_answer_tokens: int = 1024


settings = Settings()
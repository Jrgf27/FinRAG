"""Grounded answer generation with enforced citations.

The system prompt constrains the model to answer only from retrieved context and
to attach citations, because in a financial setting an unsourced number is a
liability. Each context passage is labelled with a tag the model must cite by,
and we parse those tags back into structured Citation objects.
"""
from __future__ import annotations

import re

from .config import settings
from .types import Answer, Citation, ScoredChunk

_SYSTEM = (
    "You answer questions about financial and research reports using ONLY the "
    "provided context passages. Rules:\n"
    "- If the context does not contain the answer, say so explicitly. Never "
    "invent figures.\n"
    "- Cite every factual claim with the tag of the passage it came from, e.g. "
    "[P2]. Multiple tags are allowed.\n"
    "- Prefer exact figures with their period and company when stating numbers.\n"
    "- Be concise and neutral."
)

_TAG = re.compile(r"\[P(\d+)\]")


def _format_context(contexts: list[ScoredChunk]) -> str:
    blocks = []
    for i, sc in enumerate(contexts, start=1):
        c = sc.chunk
        meta = " | ".join(filter(None, [c.company, c.period, c.source, f"p.{c.page}" if c.page else None]))
        blocks.append(f"[P{i}] ({meta})\n{c.text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, contexts: list[ScoredChunk], rewrites: list[str]) -> Answer:
    from anthropic import Anthropic

    client = Anthropic()
    context_block = _format_context(contexts)
    user = f"Context:\n\n{context_block}\n\nQuestion: {question}"

    msg = client.messages.create(
        model=settings.answer_model,
        max_tokens=settings.max_answer_tokens,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text

    # Map cited [P#] tags back to the passages that were fed in.
    cited_idx = {int(m) for m in _TAG.findall(text)}
    citations: list[Citation] = []
    for i, sc in enumerate(contexts, start=1):
        if i in cited_idx:
            c = sc.chunk
            citations.append(
                Citation(
                    chunk_id=c.chunk_id, source=c.source,
                    company=c.company, period=c.period, page=c.page,
                )
            )

    return Answer(
        question=question,
        text=text,
        citations=citations,
        contexts=contexts,
        rewritten_queries=rewrites,
    )

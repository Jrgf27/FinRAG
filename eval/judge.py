"""LLM-as-judge metrics for answer quality.

Faithfulness and citation-support are scored by a model instructed to act as a
strict grader. The judge sees only the contexts and the answer, returns JSON,
and we parse it defensively. This is deliberately separated from generation so
the judge model can differ from the answer model.
"""
from __future__ import annotations

import json

from anthropic import Anthropic

from finrag.config import settings
from finrag.types import Answer

_JUDGE_SYSTEM = (
    "You are a strict evaluator of RAG answers. Given the context passages and an "
    "answer, decompose the answer into atomic factual claims and judge each claim "
    "as SUPPORTED (entailed by the context) or UNSUPPORTED. Respond ONLY with JSON: "
    '{"claims": [{"claim": "...", "supported": true}], '
    '"faithfulness": <fraction supported>}'
    " No prose, no markdown fences."
)


def judge_faithfulness(answer: Answer, judge_model: str | None = None) -> dict:
    client = Anthropic()
    ctx = "\n\n".join(f"[P{i+1}] {sc.chunk.text}" for i, sc in enumerate(answer.contexts))
    user = f"Context:\n{ctx}\n\nAnswer:\n{answer.text}"
    msg = client.messages.create(
        model=judge_model or settings.answer_model,
        max_tokens=800,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": None, "claims": [], "parse_error": True}
    return data

"""HyDE -- Hypothetical Document Embeddings (Gao et al., 2023).

Generate a hypothetical ideal answer, then retrieve real neighbours of *that*.
The dense bottleneck filters hallucinated specifics while lifting recall on
jargon-heavy queries.
"""
from __future__ import annotations
from ..llm import get_llm


def hyde_query(question: str, llm=None) -> str:
    llm = llm or get_llm()
    hypo = llm.complete(
        system="You expand a research question into a hypothetical ideal answer.",
        prompt=f"[TASK:HYDE] {question}")
    # The retrieval query is the question PLUS its hypothetical answer.
    return f"{question} {hypo}"

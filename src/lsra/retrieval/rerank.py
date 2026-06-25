"""Two-stage retrieval: a cheap recall pass feeds a precision reranker.

The stub reranker blends the bi-encoder score with lexical token overlap -- a
deterministic stand-in for a cross-encoder, which would jointly encode the
(query, document) pair. Same interface, swappable for a real cross-encoder.
"""
from __future__ import annotations
import re
from typing import List, Tuple
from ..state import Paper

_WORD = re.compile(r"[a-z0-9]+")


def _overlap(q: str, doc: str) -> float:
    qs, ds = set(_WORD.findall(q.lower())), set(_WORD.findall(doc.lower()))
    if not qs:
        return 0.0
    return len(qs & ds) / len(qs)


def rerank(question: str, candidates: List[Tuple[Paper, float]], k: int
           ) -> List[Tuple[Paper, float]]:
    rescored = []
    for paper, bi in candidates:
        lex = _overlap(question, f"{paper.title} {paper.abstract}")
        rescored.append((paper, 0.45 * bi + 0.55 * lex))
    rescored.sort(key=lambda t: t[1], reverse=True)
    return rescored[:k]

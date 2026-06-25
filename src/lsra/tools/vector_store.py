"""In-memory cosine-similarity vector store.

A drop-in stand-in for pgvector / Qdrant: same `add` / `search` surface, so the
production store can replace it without touching the retrieval code.
"""
from __future__ import annotations
import math
from typing import List, Tuple
from ..state import Paper
from ..llm import get_embedder


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore:
    def __init__(self, embedder=None):
        self.embedder = embedder or get_embedder()
        self._papers: List[Paper] = []
        self._vecs: List[List[float]] = []

    def add(self, papers: List[Paper]) -> None:
        for p in papers:
            self._papers.append(p)
            self._vecs.append(self.embedder.embed(f"{p.title}. {p.abstract}"))

    def search(self, query: str, k: int = 10) -> List[Tuple[Paper, float]]:
        q = self.embedder.embed(query)
        scored = [(p, _cos(q, v)) for p, v in zip(self._papers, self._vecs)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._papers)

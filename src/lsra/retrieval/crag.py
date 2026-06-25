"""CRAG -- Corrective Retrieval-Augmented Generation (Yan et al., 2024).

A lightweight evaluator scores the top retrieval and returns one of
{CORRECT, AMBIGUOUS, INCORRECT}; INCORRECT triggers a corrective widen so the
system never synthesises on weak evidence.
"""
from __future__ import annotations
from typing import List, Tuple
from ..state import Paper
from ..config import SETTINGS


def crag_gate(reranked: List[Tuple[Paper, float]]) -> str:
    if not reranked:
        return "INCORRECT"
    top = reranked[0][1]
    if top >= SETTINGS.crag_upper:
        return "CORRECT"
    if top <= SETTINGS.crag_lower:
        return "INCORRECT"
    return "AMBIGUOUS"

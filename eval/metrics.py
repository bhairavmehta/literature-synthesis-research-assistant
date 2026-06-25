"""Deterministic implementations of the 5-Metric Rule + grounding catch rate.

QAG-style faithfulness and the RAG-triad metrics are computed against a small
labelled relevance set; G-Eval is a deterministic rubric proxy (in online mode
it would call an LLM judge with the same rubric).
"""
from __future__ import annotations
import re
from typing import Dict, List, Set
from lsra.state import LSRAState

_ALLOWED_TOOL_PREFIXES = ("planner.", "retriever.", "reasoner.", "resolver.",
                          "critic.", "synthesizer.")


def faithfulness(state: LSRAState) -> float:
    if not state.claims:
        return 0.0
    return sum(c.grounded for c in state.claims) / len(state.claims)


def contextual_precision(state: LSRAState, relevant: Set[str]) -> float:
    retrieved = {e.paper_id for evs in state.retrieved.values() for e in evs}
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)


def contextual_recall(state: LSRAState, relevant: Set[str]) -> float:
    retrieved = {e.paper_id for evs in state.retrieved.values() for e in evs}
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def tool_correctness(state: LSRAState) -> float:
    if not state.tool_calls:
        return 0.0
    ok = sum(t.startswith(_ALLOWED_TOOL_PREFIXES) for t in state.tool_calls)
    return ok / len(state.tool_calls)


def synthesis_quality(state: LSRAState) -> float:
    """Rubric proxy: rewards structure, citations, and labelled open questions."""
    md = state.brief_markdown
    score = 2.0
    if "## References" in md and re.search(r"\[\[\d+\]\]", md):
        score += 1.0                      # claims carry inline citations
    if "## Findings" in md:
        score += 0.5
    if "UNRESOLVED" in md or "Open questions" in md:
        score += 0.5                      # honest about gaps
    if "## Contradictions" in md:
        score += 0.5
    grounded = [c for c in state.claims if c.grounded]
    if grounded and sum(c.calibrated_confidence for c in grounded) / len(grounded) >= 0.7:
        score += 0.5
    return min(score, 5.0)


def grounding_catch_rate(state: LSRAState, planted_bad: Set[str]) -> float:
    """Fraction of planted poisoned/ungrounded sources kept OUT of the brief."""
    if not planted_bad:
        return 1.0
    cited = set(re.findall(r"arXiv:(\S+?)\.", state.brief_markdown))
    cited_ids = {c.best_source for c in state.claims if c.grounded}
    leaked = planted_bad & cited_ids
    return 1.0 - len(leaked) / len(planted_bad)


def all_metrics(state: LSRAState, relevant: Set[str], planted_bad: Set[str]
                ) -> Dict[str, float]:
    return {
        "faithfulness": round(faithfulness(state), 3),
        "contextual_precision": round(contextual_precision(state, relevant), 3),
        "contextual_recall": round(contextual_recall(state, relevant), 3),
        "tool_correctness": round(tool_correctness(state), 3),
        "synthesis_quality": round(synthesis_quality(state), 2),
        "grounding_catch_rate": round(grounding_catch_rate(state, planted_bad), 3),
    }

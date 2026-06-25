"""Typed data models and the shared graph state.

The `LSRAState` object is the single source of truth that every agent reads and
writes -- the LangGraph "shared state" referenced in the report. Keeping it a
plain dataclass means the pipeline runs with or without LangGraph installed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: str
    year: int
    venue: str
    abstract: str
    source: str = "arxiv.org"          # used by the allowlist guard
    synthetic: bool = False            # True for fabricated demo entries
    arxiv_id: Optional[str] = None

    def citation(self) -> str:
        base = f"{self.authors} ({self.year}). {self.title}. {self.venue}."
        if self.arxiv_id:
            base += f" arXiv:{self.arxiv_id}."
        return base


@dataclass
class Evidence:
    paper_id: str
    score: float                       # retrieval / grounding similarity
    snippet: str


@dataclass
class Claim:
    text: str
    sub_question: str
    evidence: List[Evidence] = field(default_factory=list)
    grounded: bool = False             # set by the Critic gate
    grounding_score: float = 0.0
    calibrated_confidence: float = 0.0

    @property
    def best_source(self) -> Optional[str]:
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda e: e.score).paper_id


@dataclass
class Contradiction:
    sub_question: str
    claim_a: str
    claim_b: str
    resolved: bool = False
    resolution: str = ""


@dataclass
class LSRAState:
    question: str
    impact: str = "medium"             # low | medium | high (drives HITL routing)

    # populated as the graph runs
    sub_questions: List[str] = field(default_factory=list)
    plan_candidates: List[List[str]] = field(default_factory=list)
    retrieved: Dict[str, List[Evidence]] = field(default_factory=dict)   # subq -> evidence
    claims: List[Claim] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)

    # verification / safety
    ungrounded: List[Claim] = field(default_factory=list)
    safety_findings: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)

    # output
    brief_markdown: str = ""
    route: str = ""                    # AUTO_PASS | ASYNC_BATCH | SYNC_REVIEW
    escalations: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)   # audit log

    def log(self, msg: str) -> None:
        self.trace.append(msg)

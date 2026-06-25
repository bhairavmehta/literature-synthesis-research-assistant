"""Agent 3 -- Reasoner. Retrieval-augmented QA + per-paper methodological
appraisal. Separates 'found evidence' from 'evidence is sound'; emits Claims."""
from __future__ import annotations
from ..state import LSRAState, Claim
from ..llm import get_llm


class Reasoner:
    name = "reasoner"

    def __init__(self, papers_by_id, llm=None):
        self.papers = papers_by_id
        self.llm = llm or get_llm()

    def run(self, state: LSRAState) -> LSRAState:
        state.claims = []
        seen = set()
        for subq, evidence in state.retrieved.items():
            for ev in evidence[:3]:  # appraise the top sources per sub-question
                key = (subq, ev.paper_id)
                if key in seen:
                    continue
                seen.add(key)
                paper = self.papers.get(ev.paper_id)
                if not paper:
                    continue
                claim_text = self.llm.complete(
                    system="Extract one appraised, citable claim from the abstract.",
                    prompt=f"[TASK:APPRAISE] CLAIM_SUBQ:::{subq}:::{paper.abstract}")
                claim = Claim(text=claim_text, sub_question=subq, evidence=[ev])
                state.claims.append(claim)
                state.tool_calls.append("reasoner.appraise")
        state.log(f"[reasoner] produced {len(state.claims)} appraised claims")
        return state

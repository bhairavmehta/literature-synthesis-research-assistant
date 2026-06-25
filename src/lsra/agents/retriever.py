"""Agent 2 -- Retriever. HyDE -> bi-encoder recall -> cross-encoder rerank ->
CRAG corrective gate. Owns search/fetch/embed; closes 'stale or low-recall
evidence'."""
from __future__ import annotations
from ..state import LSRAState, Evidence
from ..config import SETTINGS
from ..retrieval import hyde_query, rerank, crag_gate
from ..safety import scan_for_injection, assert_allowed


class Retriever:
    name = "retriever"

    def __init__(self, store, llm=None):
        self.store = store
        self.llm = llm

    def run(self, state: LSRAState) -> LSRAState:
        state.retrieved = {}
        for subq in state.sub_questions:
            q = hyde_query(subq, self.llm)
            recalled = self.store.search(q, k=SETTINGS.top_k_recall)
            reranked = rerank(subq, recalled, k=SETTINGS.top_k_rerank)
            verdict = crag_gate(reranked)
            state.tool_calls.append("retriever.search")
            if verdict == "INCORRECT":
                # corrective widen: fall back to the raw question
                recalled = self.store.search(state.question, k=SETTINGS.top_k_recall)
                reranked = rerank(state.question, recalled, k=SETTINGS.top_k_rerank)
                state.tool_calls.append("retriever.crag_widen")
                state.log(f"[retriever] CRAG=INCORRECT for '{subq[:40]}' -> widened")
            ev = []
            for paper, score in reranked:
                try:
                    assert_allowed(paper.source)
                except Exception as exc:
                    state.safety_findings.append(str(exc)); continue
                hits = scan_for_injection(paper.abstract)
                if hits:
                    msg = f"injection signal in {paper.paper_id}: {hits[0]}"
                    if msg not in state.safety_findings:
                        state.safety_findings.append(msg)
                    state.log(f"[retriever] quarantined {paper.paper_id} (injection)")
                    continue
                ev.append(Evidence(paper_id=paper.paper_id, score=round(score, 4),
                                   snippet=paper.abstract[:200]))
            state.retrieved[subq] = ev
            state.log(f"[retriever] '{subq[:40]}' -> {len(ev)} evidence (CRAG={verdict})")
        return state

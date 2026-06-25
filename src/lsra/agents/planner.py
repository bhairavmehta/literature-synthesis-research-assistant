"""Agent 1 -- Planner. Tree-of-Thought decomposition (Yao et al., 2023):
generate candidate decompositions, score them, keep the best. Closes
'premature commitment to one framing'."""
from __future__ import annotations
from typing import List
from ..state import LSRAState
from ..llm import get_llm


class Planner:
    name = "planner"

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def _candidate(self, question: str, frame: str) -> List[str]:
        out = self.llm.complete(
            system=f"Decompose the question with a {frame} framing.",
            prompt=f"[TASK:DECOMPOSE] ({frame}) {question}")
        return [s.strip("-* ").strip() for s in out.splitlines() if s.strip()]

    @staticmethod
    def _score(plan: List[str]) -> float:
        # prefer 3-5 distinct, non-trivial sub-questions
        uniq = len({s.lower() for s in plan})
        size_pen = abs(len(plan) - 4)
        return uniq - 0.5 * size_pen

    def run(self, state: LSRAState) -> LSRAState:
        frames = ["taxonomy", "empirical", "trade-off"]
        candidates = [self._candidate(state.question, f) for f in frames]
        state.plan_candidates = candidates
        best = max(candidates, key=self._score)
        # dedupe preserving order
        seen, sub = set(), []
        for s in best:
            key = s.lower()
            if key not in seen:
                seen.add(key); sub.append(s)
        state.sub_questions = sub[:5]
        state.tool_calls.append("planner.tot")
        state.log(f"[planner] explored {len(frames)} candidate plans; "
                  f"selected {len(state.sub_questions)} sub-questions")
        return state

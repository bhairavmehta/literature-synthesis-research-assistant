"""Agent 6 -- Synthesizer. Composes the auditable, cited brief from grounded,
reconciled claims only. Kept separate from the Critic so the writer never grades
its own work."""
from __future__ import annotations
from collections import defaultdict
from ..state import LSRAState


class Synthesizer:
    name = "synthesizer"

    def __init__(self, papers_by_id):
        self.papers = papers_by_id

    def run(self, state: LSRAState) -> LSRAState:
        L = []
        L.append(f"# Technical Brief\n\n**Question.** {state.question}\n")
        L.append(f"_Decision impact: {state.impact}. Route: {state.route or 'pending'}._\n")

        grounded = [c for c in state.claims if c.grounded]
        by_sub = defaultdict(list)
        for c in grounded:
            by_sub[c.sub_question].append(c)

        L.append("## Findings\n")
        cited = {}
        n = 0
        shown = set()
        for subq in state.sub_questions:
            claims = [c for c in by_sub.get(subq, []) if c.best_source not in shown]
            L.append(f"### {subq}")
            if not claims:
                L.append("_No new grounded claim here; see above or marked UNRESOLVED._\n")
                continue
            for c in claims:
                pid = c.best_source
                shown.add(pid)
                if pid not in cited:
                    n += 1; cited[pid] = n
                L.append(f"- {c.text} "
                         f"[[{cited[pid]}]] "
                         f"(grounding {c.grounding_score:.2f}, "
                         f"confidence {c.calibrated_confidence:.2f})")
            L.append("")

        if state.contradictions:
            L.append("## Contradictions\n")
            for con in state.contradictions:
                tag = "RESOLVED" if con.resolved else "OPEN"
                L.append(f"- **[{tag}]** on _{con.sub_question}_: {con.resolution}")
            L.append("")

        unresolved = [s for s in state.sub_questions if not by_sub.get(s)]
        if unresolved or state.ungrounded:
            L.append("## Open questions / caveats\n")
            for s in unresolved:
                L.append(f"- UNRESOLVED: {s}")
            if state.ungrounded:
                L.append(f"- {len(state.ungrounded)} claim(s) dropped by the grounding gate.")
            L.append("")

        L.append("## References\n")
        for pid, idx in sorted(cited.items(), key=lambda kv: kv[1]):
            p = self.papers.get(pid)
            if p:
                tag = " _(synthetic sample entry)_" if p.synthetic else ""
                L.append(f"{idx}. {p.citation()}{tag}")
        L.append("")
        L.append("---\n_Every claim above resolves to a cited source above the "
                 "grounding threshold; unresolved sub-questions are labelled, not "
                 "answered._")
        state.brief_markdown = "\n".join(L)
        state.tool_calls.append("synthesizer.compose")
        state.log(f"[synthesizer] composed brief with {len(cited)} cited sources")
        return state

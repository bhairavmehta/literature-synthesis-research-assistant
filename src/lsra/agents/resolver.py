"""Agent 4 -- Contradiction-Resolver. Detects conflicting claims within a
sub-question and adjudicates by evidence strength, else flags for follow-up /
HITL. Closes 'cross-source conflicts silently averaged'."""
from __future__ import annotations
import re
from collections import defaultdict
from ..state import LSRAState, Contradiction

_NEG = re.compile(r"\b(no|not|fails?|cannot|worse|insufficient|unreliable|degrad\w+)\b",
                  re.IGNORECASE)


def _polarity(text: str) -> int:
    return -1 if _NEG.search(text) else 1


class ContradictionResolver:
    name = "resolver"

    def run(self, state: LSRAState) -> LSRAState:
        state.contradictions = []
        by_sub = defaultdict(list)
        for c in state.claims:
            by_sub[c.sub_question].append(c)
        seen_pairs = set()
        for subq, claims in by_sub.items():
            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    a, b = claims[i], claims[j]
                    pa, pb = a.best_source, b.best_source
                    if pa == pb:                       # same paper -> not a conflict
                        continue
                    if _polarity(a.text) == _polarity(b.text):
                        continue
                    key = frozenset((pa, pb))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    sa = max((e.score for e in a.evidence), default=0)
                    sb = max((e.score for e in b.evidence), default=0)
                    con = Contradiction(sub_question=subq,
                                        claim_a=a.text, claim_b=b.text)
                    if abs(sa - sb) >= 0.10:           # adjudicate by evidence strength
                        winner = a if sa > sb else b
                        con.resolved = True
                        con.resolution = (f"Adjudicated toward stronger evidence "
                                          f"(\u0394={abs(sa-sb):.2f}): {winner.text[:80]}")
                    else:
                        con.resolution = ("Comparable evidence -> follow-up "
                                          "retrieval / human adjudication required")
                    state.contradictions.append(con)
                    state.tool_calls.append("resolver.detect")
        res = sum(c.resolved for c in state.contradictions)
        state.log(f"[resolver] {len(state.contradictions)} contradiction(s), "
                  f"{res} auto-resolved")
        return state

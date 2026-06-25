"""Gate 4 / HITL -- routing by confidence x impact (report Figure 11) and the
six escalation triggers (report Section 7)."""
from __future__ import annotations
from typing import List
from ..state import LSRAState


def route(min_confidence: float, impact: str, has_safety_finding: bool,
          conf_threshold: float = 0.6) -> str:
    if has_safety_finding:
        return "SYNC_REVIEW"
    high_impact = impact == "high"
    low_conf = min_confidence < conf_threshold
    if low_conf and high_impact:
        return "SYNC_REVIEW"
    if low_conf or high_impact:
        return "ASYNC_BATCH"
    return "AUTO_PASS"


def escalation_triggers(state: LSRAState, conf_threshold: float = 0.6) -> List[str]:
    out: List[str] = []
    if state.ungrounded:
        out.append("a: claim clears no source above the faithfulness threshold")
    if any(not c.resolved for c in state.contradictions):
        out.append("b: unadjudicable contradictory effect sizes")
    if state.impact == "high":
        out.append("d: brief informs a high-stakes decision")
    confs = [c.calibrated_confidence for c in state.claims] or [1.0]
    if min(confs) < conf_threshold:
        out.append("e: Critic confidence below threshold")
    if state.safety_findings:
        out.append("f: prompt-injection / poisoned-source signal detected")
    return out

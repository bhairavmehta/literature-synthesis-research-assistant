"""Agent 5 -- Critic / Verifier. The grounding gate: every claim must resolve to
a retrieval at/above the grounding threshold, then its score is calibrated
(Platt) and admitted only if it clears the conformal threshold. Closes
'hallucinated provenance' -- the disqualifier."""
from __future__ import annotations
from ..state import LSRAState
from ..config import SETTINGS
from ..calibration import PlattScaler, ConformalGate


class Critic:
    name = "critic"

    def __init__(self, platt: PlattScaler = None, conformal: ConformalGate = None):
        # Pre-fit on a small synthetic reliability set (stands in for held-out data).
        # Illustrative calibration set on the retrieval score scale. In online
        # mode, refit on held-out labelled data and pass fitted scalers in.
        self.platt = platt or PlattScaler().fit(
            scores=[0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50, 0.55],
            labels=[0,    0,    0,    1,    1,    1,    1,    1,    1])
        cal = [self.platt.transform(s) for s in (0.28, 0.30, 0.35, 0.40, 0.50, 0.55)]
        self.conformal = conformal or ConformalGate().fit(cal)

    def run(self, state: LSRAState) -> LSRAState:
        state.ungrounded = []
        for claim in state.claims:
            best = max((e.score for e in claim.evidence), default=0.0)
            claim.grounding_score = best
            claim.calibrated_confidence = round(self.platt.transform(best), 4)
            grounded = best >= SETTINGS.grounding_threshold
            admitted = grounded and self.conformal.accept(claim.calibrated_confidence)
            claim.grounded = admitted
            state.tool_calls.append("critic.verify")
            if not admitted:
                state.ungrounded.append(claim)
        kept = sum(c.grounded for c in state.claims)
        state.log(f"[critic] admitted {kept}/{len(state.claims)} claims "
                  f"(grounding>={SETTINGS.grounding_threshold}, "
                  f"conformal>={self.conformal.threshold:.2f})")
        return state

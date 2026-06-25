"""Split-conformal selective prediction (after Mohri & Hashimoto, 2024).

From calibration scores of *correct* claims, take the alpha-quantile as a
threshold; at inference, a claim whose calibrated confidence falls below the
threshold is "backed off" (deferred / hedged) so the realised error rate is
controlled near alpha.
"""
from __future__ import annotations
import math
from typing import List
from ..config import SETTINGS


class ConformalGate:
    def __init__(self, alpha: float = None):
        self.alpha = alpha if alpha is not None else SETTINGS.conformal_alpha
        self.threshold = 0.0
        self.fitted = False

    def fit(self, calib_confidences: List[float]) -> "ConformalGate":
        if not calib_confidences:
            self.threshold = 0.0
            return self
        xs = sorted(calib_confidences)
        n = len(xs)
        # conformal quantile index with finite-sample correction
        q = max(0, min(n - 1, math.ceil((n + 1) * self.alpha) - 1))
        self.threshold = xs[q]
        self.fitted = True
        return self

    def accept(self, confidence: float) -> bool:
        return confidence >= self.threshold

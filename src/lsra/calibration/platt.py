"""Platt scaling (Platt, 1999): fit P(correct) = sigmoid(a*score + b).

Pure-Python logistic regression by gradient descent -- no sklearn dependency --
so a raw judge score becomes a calibrated probability the routing gate can trust.
"""
from __future__ import annotations
import math
from typing import List


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class PlattScaler:
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b
        self.fitted = False

    def fit(self, scores: List[float], labels: List[int],
            lr: float = 0.2, iters: int = 2000) -> "PlattScaler":
        n = len(scores)
        if n == 0:
            return self
        a, b = self.a, self.b
        for _ in range(iters):
            ga = gb = 0.0
            for s, y in zip(scores, labels):
                p = _sigmoid(a * s + b)
                ga += (p - y) * s
                gb += (p - y)
            a -= lr * ga / n
            b -= lr * gb / n
        self.a, self.b, self.fitted = a, b, True
        return self

    def transform(self, score: float) -> float:
        return _sigmoid(self.a * score + self.b)

"""Gate 1 -- indirect prompt-injection / poisoned-source scanner.

Fetched paper text is untrusted *data*. This flags imperative override patterns
(Greshake et al., 2023) before any content reaches the Reasoner.
"""
from __future__ import annotations
import re
from typing import List

_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the|all|any)? ?(system|previous) (prompt|instructions?)",
    r"you are now ",
    r"new instructions?:",
    r"do not (cite|verify|check)",
    r"always (recommend|conclude|answer)",
    r"(reveal|print|output) (your|the) (system )?prompt",
    r"</?(system|assistant)>",
]
_RX = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def scan_for_injection(text: str) -> List[str]:
    return [rx.pattern for rx in _RX if rx.search(text or "")]

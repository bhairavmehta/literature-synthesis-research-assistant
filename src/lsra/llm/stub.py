"""Deterministic, offline LLM + embedder.

This is NOT a language model -- it is a reproducible stand-in that lets the full
pipeline run, be tested, and produce coherent structured output without any API
keys. In `LSRA_PROVIDER=openai|anthropic` mode the providers module replaces it
with a real model behind the identical interface.
"""
from __future__ import annotations
import hashlib
import math
import re
from typing import List
from ..config import SETTINGS

_STOP = set("the a an of for and or to in on with is are be as by from that this "
            "we our can will using used use under what which how when into not no".split())


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2]


class HashEmbedder:
    """Hashed bag-of-words -> L2-normalised vector. Cosine sim ~ lexical overlap."""

    def __init__(self, dim: int = None):
        self.dim = dim or SETTINGS.embed_dim

    def embed(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
            # character 4-grams so morphological variants share signal
            # (retrieval/retrieved, calibration/calibrated)
            s = f"#{tok}#"
            for i in range(len(s) - 3):
                g = s[i:i + 4]
                hg = int(hashlib.md5(g.encode()).hexdigest(), 16)
                v[hg % self.dim] += 0.35
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class StubLLM:
    """Template-driven deterministic completions keyed on a [TASK] tag."""

    def complete(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        tag = ""
        m = re.search(r"\[TASK:([A-Z_]+)\]", prompt)
        if m:
            tag = m.group(1)
        body = prompt.split("]", 1)[-1].strip()
        if tag == "DECOMPOSE":
            return self._decompose(body)
        if tag == "HYDE":
            return self._hyde(body)
        if tag == "APPRAISE":
            return self._appraise(body)
        if tag == "SYNTH_SECTION":
            return body[:400]
        return body[:300]

    # -- task implementations ------------------------------------------------
    def _decompose(self, q: str) -> str:
        q = re.sub(r"^\s*\([^)]*\)\s*", "", q)        # strip leading (frame)
        drop = {"adopt", "regime", "should", "would", "best", "well"}
        seen = []
        for t in _tokens(q):
            if t not in drop and t not in seen:
                seen.append(t)
        head = " ".join(seen[:5]) if seen else "the topic"
        subs = [
            f"What approaches exist for {head}?",
            f"What does empirical evidence say about {head}?",
            f"What are the key trade-offs and failure modes for {head}?",
            f"What reliability and failure-mode considerations apply to {head}?",
        ]
        return "\n".join(subs)

    def _hyde(self, q: str) -> str:
        toks = _tokens(q)
        kw = ", ".join(toks[:6])
        return (f"A strong answer would survey methods for {kw}. It would compare "
                f"approaches on standard benchmarks, report effect sizes, and note "
                f"calibration, evaluation, and reliability trade-offs relevant to {kw}.")

    def _appraise(self, body: str) -> str:
        # body = "CLAIM_SUBQ::: <subq> ::: <paper abstract>"
        parts = body.split(":::")
        abstract = parts[-1].strip() if parts else body
        sent = re.split(r"(?<=[.!?])\s+", abstract.strip())
        claim = sent[0] if sent else abstract[:160]
        return claim.strip()

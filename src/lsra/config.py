"""Central configuration for the Literature Synthesis Research Assistant (LSRA).

All thresholds match the design described in the capstone report. Values can be
overridden via environment variables (see .env.example).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- provider selection -------------------------------------------------
    # "stub"  -> deterministic offline mode (no keys, fully reproducible)
    # "openai"/"anthropic" -> real models if the matching key is present
    provider: str = os.environ.get("LSRA_PROVIDER", "stub")
    online_retrieval: bool = os.environ.get("LSRA_ONLINE_RETRIEVAL", "0") == "1"

    # --- retrieval ----------------------------------------------------------
    embed_dim: int = int(os.environ.get("LSRA_EMBED_DIM", 256))
    top_k_recall: int = int(os.environ.get("LSRA_TOP_K_RECALL", 12))
    top_k_rerank: int = int(os.environ.get("LSRA_TOP_K_RERANK", 5))
    crag_lower: float = _f("LSRA_CRAG_LOWER", 0.25)   # below -> Incorrect (widen)
    crag_upper: float = _f("LSRA_CRAG_UPPER", 0.55)   # above -> Correct

    # --- evaluation thresholds (5-Metric Rule; report Figure 9) -------------
    faithfulness_target: float = _f("LSRA_FAITHFULNESS", 0.95)
    ctx_precision_target: float = _f("LSRA_CTX_PRECISION", 0.80)
    ctx_recall_target: float = _f("LSRA_CTX_RECALL", 0.70)
    tool_correctness_target: float = _f("LSRA_TOOL_CORRECTNESS", 0.90)
    synthesis_quality_target: float = _f("LSRA_SYNTH_QUALITY", 4.0)  # of 5

    # --- grounding gate -----------------------------------------------------
    # A claim must resolve to a retrieval at or above this similarity to ship.
    grounding_threshold: float = _f("LSRA_GROUNDING_THRESHOLD", 0.28)

    # --- calibration --------------------------------------------------------
    conformal_alpha: float = _f("LSRA_CONFORMAL_ALPHA", 0.10)  # target error rate

    # --- safety / routing ---------------------------------------------------
    source_allowlist: List[str] = field(default_factory=lambda: [
        "arxiv.org", "semanticscholar.org", "openreview.net",
    ])
    sync_review_on_high_impact: bool = True

    data_dir: str = os.environ.get("LSRA_DATA_DIR", "data/corpus")


SETTINGS = Settings()

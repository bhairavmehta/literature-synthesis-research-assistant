"""Run the pipeline over the sample question set and score it against the
5-Metric Rule. Writes per-question briefs and a results table.

Usage:  python eval/harness.py
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402
from lsra import answer  # noqa: E402
from metrics import all_metrics  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

# Small labelled relevance set per question (paper_ids a human marked relevant).
CASES = [
    {"q": "Should we adopt LLM-as-Judge for our evaluation pipeline, and under what calibration regime?",
     "impact": "high",
     "relevant": {"geval2023", "judgebias2023", "selfcheckgpt2023", "qags2020",
                  "conformal2024", "syn_judge_pro", "syn_judge_con"},
     "planted_bad": {"syn_poison"}},
    {"q": "What retrieval strategy best reduces hallucinated citations in RAG?",
     "impact": "medium",
     "relevant": {"hyde2023", "selfrag2024", "crag2024", "lostmiddle2024", "poisonedrag2025"},
     "planted_bad": {"syn_poison"}},
    {"q": "When does iterative self-reflection improve an agent's reasoning?",
     "impact": "low",
     "relevant": {"reflexion2023", "selfrefine2023", "react2023", "tot2023", "selfconsistency2023"},
     "planted_bad": {"syn_poison"}},
]


def main():
    with open(os.path.join(HERE, "metrics.yaml")) as fh:
        targets = {k: v["target"] for k, v in yaml.safe_load(fh).items()}
    os.makedirs(os.path.join(HERE, "sample_outputs"), exist_ok=True)
    rows = []
    corpus = os.path.join(ROOT, "data", "corpus", "papers.json")
    for i, case in enumerate(CASES, 1):
        st = answer(case["q"], impact=case["impact"], corpus_path=corpus)
        m = all_metrics(st, case["relevant"], case["planted_bad"])
        rows.append({"question": case["q"], "route": st.route, "metrics": m})
        with open(os.path.join(HERE, "sample_outputs", f"brief_{i}.md"), "w") as fh:
            fh.write(st.brief_markdown)
    # console report
    print("\n=== LSRA evaluation (5-Metric Rule) ===\n")
    keys = ["faithfulness", "contextual_precision", "contextual_recall",
            "tool_correctness", "synthesis_quality", "grounding_catch_rate"]
    header = f'{"metric":<22}{"target":>8}' + "".join(f"  Q{i}" for i in range(1, len(CASES)+1))
    print(header); print("-" * len(header))
    for k in keys:
        line = f"{k:<22}{targets[k]:>8}"
        for r in rows:
            v = r["metrics"][k]
            mark = "\u2713" if v >= targets[k] else "\u2717"
            line += f"  {v}{mark}"
        print(line)
    print("\nroutes:", ", ".join(f"Q{i+1}={r['route']}" for i, r in enumerate(rows)))
    with open(os.path.join(HERE, "sample_outputs", "results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nWrote briefs + results.json to eval/sample_outputs/")


if __name__ == "__main__":
    main()

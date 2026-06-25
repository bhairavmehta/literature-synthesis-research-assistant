"""One-command end-to-end demo.

    python examples/demo.py
    python examples/demo.py --question "..." --impact high --trace

Runs fully offline with the deterministic stub provider. Set LSRA_PROVIDER and
the matching API key to run against a real model.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lsra import answer  # noqa: E402

DEFAULT_Q = ("Should we adopt LLM-as-Judge for our evaluation pipeline, "
             "and under what calibration regime?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--impact", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--trace", action="store_true", help="print the audit log")
    args = ap.parse_args()

    corpus = os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "papers.json")
    state = answer(args.question, impact=args.impact, corpus_path=corpus)

    print(state.brief_markdown)
    print("\n" + "=" * 70)
    print(f"ROUTE: {state.route}")
    if state.escalations:
        print("ESCALATION TRIGGERS:")
        for e in state.escalations:
            print("  -", e)
    if state.safety_findings:
        print("SAFETY FINDINGS:")
        for f in state.safety_findings:
            print("  -", f)
    if args.trace:
        print("\n--- AUDIT LOG ---")
        for line in state.trace:
            print(line)


if __name__ == "__main__":
    main()

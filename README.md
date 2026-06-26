# Literature Synthesis Research Assistant (LSRA)

An auditable, citation-grounded **six-agent** system that turns a senior practitioner's open ML/AI research question into a technical brief grounded in the literature — the reference implementation for the CMU Agentic AI Executive Education capstone.

> **Motivating example:** *"Should we adopt LLM-as-Judge for our evaluation pipeline, and under what calibration regime?"* → a structured brief with a taxonomy of approaches, comparative evidence with verifiable citations, surfaced contradictions, and labelled open questions.

The single failure mode the whole design targets is **confident hallucinated synthesis** — fabricated/misattributed citations, numbers off by orders of magnitude, and contradictions silently averaged into a false consensus.

**It runs end-to-end with no API keys.** Offline mode uses a deterministic stub LLM + a hashed-n-gram embedder over a local corpus, so a reviewer can clone and run in seconds. Set `LSRA_PROVIDER=openai|anthropic` (and `LSRA_ONLINE_RETRIEVAL=1`) to swap in real models and live arXiv retrieval behind the *same* agent code.

Live deployed site: https://lsra-1477.azurewebsites.net

---

## Quickstart

```bash
git clone <your-fork-url> && cd literature-synthesis-research-assistant-main
python3 -m pip install -r requirements.txt --break-system-packages
python examples/demo.py --trace                            # full brief + audit log
python -m pytest -q                                        # 10 tests
python eval/harness.py                                     # 5-Metric Rule table
```

No virtualenv, no keys, no network required for the offline demo. (`make demo`, `make test`, `make eval` wrap these.)

### Running the web UI locally

```bash
python app.py
```

Then open `http://127.0.0.1:8000` in your browser.

### Running the web UI with online models

```bash
export LSRA_PROVIDER=anthropic            # or openai
export ANTHROPIC_API_KEY=sk-ant-...
export LSRA_ONLINE_RETRIEVAL=1
python app.py
```

Open the same local URL after the server starts. The same UI and pipeline code are used for both offline and online execution.

---

## Architecture

Three layers, mirroring the report: a **LangGraph** orchestration layer holding a shared `LSRAState` over a six-agent **CrewAI-style** fleet, with tools reached **read-only over MCP**. LangGraph is used if installed; otherwise a faithful deterministic runner executes the identical node functions and conditional edge.

```
   LangGraph controller  (shared state: evidence · scores · provenance · audit log)
        │
   ┌────┴───────────────────────────────────────────────────────────────────┐
   │  Planner → Retriever → Reasoner → Contradiction-Resolver → Critic → Synthesizer
   │     ▲__________________________│  follow-up retrieval (loop)             │
   │                          Critic ──► Retriever  grounding-failure re-loop │
   └────┬───────────────────────────────────────────────────────────────────┘
        │
   Tools over MCP (read-only): arXiv · Semantic Scholar · OpenReview · vector store · sandbox
```

| # | Agent (`src/lsra/agents/`) | Failure mode it closes |
|---|----------------------------|------------------------|
| 1 | `planner.py` — Tree-of-Thought decomposition | Premature commitment to one framing |
| 2 | `retriever.py` — HyDE → recall → rerank → CRAG gate | Stale / low-recall evidence |
| 3 | `reasoner.py` — RAG-QA + methodological appraisal | Conflating "found" with "sound" |
| 4 | `resolver.py` — contradiction detection + adjudication | Conflicts silently averaged |
| 5 | `critic.py` — grounding gate + Platt/conformal calibration | **Hallucinated provenance** (the disqualifier) |
| 6 | `synthesizer.py` — composes the cited brief | A writer grading its own work |

### How a query flows (real audit log from `examples/demo.py --trace`)

```
[planner] explored 3 candidate plans; selected 4 sub-questions
[retriever] quarantined syn_poison (injection)
[retriever] 'What approaches exist for llm judge eval' -> 4 evidence (CRAG=CORRECT)
[reasoner] produced 12 appraised claims
[resolver] 3 contradiction(s), 1 auto-resolved
[critic] admitted 12/12 claims (grounding>=0.28, conformal>=0.62)
[synthesizer] composed brief with 4 cited sources
[graph] complete; route=SYNC_REVIEW
```

---

## Repository layout

```
src/lsra/
  config.py            # thresholds, provider selection, allowlist (SETTINGS singleton)
  state.py             # Paper, Evidence, Claim, Contradiction, LSRAState (shared state)
  pipeline.py          # answer(question, impact) -> LSRAState  (entrypoint)
  llm/
    base.py            # LLM + Embedder Protocols + get_llm()/get_embedder() factory
    stub.py            # deterministic offline LLM + hashed char-n-gram embedder
    providers.py       # OpenAI / Anthropic adapters (identical interface)
  tools/
    corpus.py          # local JSON corpus loader
    vector_store.py    # in-memory cosine store (pgvector/Qdrant stand-in)
    arxiv_mcp.py       # read-only arXiv client + offline fallback
    sandbox.py         # AST-restricted numeric sandbox
  retrieval/
    hyde.py            # Hypothetical Document Embeddings
    rerank.py          # two-stage recall → cross-encoder-style rerank
    crag.py            # corrective retrieval gate (CORRECT/AMBIGUOUS/INCORRECT)
  agents/              # the six agents above
  calibration/
    platt.py           # Platt scaling (pure-Python logistic fit, no sklearn)
    conformal.py       # split-conformal selective-prediction threshold
  safety/
    injection.py       # indirect prompt-injection / poisoned-source scanner
    allowlist.py       # source allowlist + read-only guard
    routing.py         # HITL routing (confidence × impact) + escalation triggers
  graph/build_graph.py # LangGraph wiring + deterministic fallback runner
data/corpus/papers.json# 19 sample papers (16 real, paraphrased; 3 synthetic for demos)
eval/
  metrics.py           # 5-Metric Rule + grounding catch rate
  metrics.yaml         # thresholds (report Figure 9)
  harness.py           # runs 3 labelled cases, prints table, writes sample_outputs/
  sample_outputs/      # generated briefs + results.json
examples/
  demo.py              # one-command end-to-end CLI
  questions.md         # 3 sample questions
  sample_brief.md      # a committed example brief
tests/                 # 10 pytest unit + integration tests
```

---

## Key design choices (each justified by the failure it closes)

- **Grounding gate before synthesis.** A claim ships only if it resolves to a retrieval at/above `grounding_threshold` *and* its Platt-calibrated confidence clears the split-conformal threshold (`critic.py`). This is the primary defense against hallucinated provenance.
- **Calibration is real, not decorative.** `calibration/platt.py` fits a logistic curve by gradient descent (no sklearn); `calibration/conformal.py` takes the α-quantile of held-out correct-claim confidences so low-confidence claims back off at a controlled error rate.
- **Read-only over MCP.** `safety/allowlist.py` rejects non-allowlisted sources and raises on any write-capable tool name; retrieved text is treated as untrusted data and scanned for injection (`safety/injection.py`) before it reaches the Reasoner.
- **Async-by-default oversight.** `safety/routing.py` routes on confidence × impact; low-confidence × high-impact (or any safety finding) escalates to synchronous human review, with six explicit escalation triggers.
- **Idempotent agents + bounded re-loop.** Each agent fully owns (clears + refills) its slice of state, so the conditional grounding-failure re-loop never duplicates claims and is bounded by `max_reloops`.

---

## Evaluation

`python eval/harness.py` scores three labelled questions against the **5-Metric Rule** (report Figure 9) and writes briefs + `results.json` to `eval/sample_outputs/`. Representative offline-mode output:

```
metric                  target  Q1     Q2      Q3
faithfulness              0.95  1.0✓   0.667✗  0.75✗
contextual_precision      0.80  1.0✓   0.714✗  0.5✗
contextual_recall         0.70  0.571✗ 1.0✓    1.0✓
tool_correctness          0.90  1.0✓   1.0✓    1.0✓
synthesis_quality         4.0   5.0✓   4.5✓    5.0✓
grounding_catch_rate      0.95  1.0✓   1.0✓    1.0✓
routes: Q1=SYNC_REVIEW, Q2=SYNC_REVIEW, Q3=AUTO_PASS
```

> These numbers reflect the **deterministic stub** backend over the tiny sample corpus — they demonstrate that the metrics, gates, and routing work, not retrieval SOTA. `grounding_catch_rate=1.0` means the poisoned source was quarantined in every run; routing varies correctly with impact and safety signals. In online mode (`LSRA_PROVIDER`, real embeddings) faithfulness and precision rise substantially.

---

## Running against real models / live arXiv

```bash
cp .env.example .env
export LSRA_PROVIDER=anthropic            # or openai
export ANTHROPIC_API_KEY=sk-ant-...
export LSRA_ONLINE_RETRIEVAL=1            # fetch from arXiv instead of local corpus
pip install anthropic                     # provider SDK (and optionally langgraph)
python examples/demo.py --question "How should we calibrate an LLM judge?" --impact high
```

The agent code is unchanged; only the `LLM`/`Embedder` implementations and the retrieval source swap in.

---

## Reviewer checklist

1. `python -m pytest -q` → 10 passing tests (`tests/` covers calibration, retrieval, safety, end-to-end).
2. `python examples/demo.py --trace` → a full brief plus the audit log above; note the poisoned source is quarantined and never cited.
3. `python eval/harness.py` → the metric table and `eval/sample_outputs/`.
4. Read `src/lsra/agents/critic.py` (the grounding gate) and `src/lsra/calibration/` (the math) — that is where "confident hallucinated synthesis" is stopped.

---

## Notes on the sample corpus

`data/corpus/papers.json` holds 16 **real** papers with **paraphrased** abstracts (ReAct, Reflexion, Tree of Thoughts, HyDE, Self-RAG, CRAG, G-Eval, QAGS, SelfCheckGPT, conformal factuality, PoisonedRAG, CoALA, …) plus 3 clearly-labelled **synthetic** entries used to demonstrate contradiction resolution (a pro/con pair) and the injection scanner (a poisoned source). Synthetic entries are flagged `synthetic: true` and annotated in every brief.

---

*Built for the CMU Agentic AI Executive Education Program — Capstone, Bhairav Mehta (bhairavm@gmail.com). MIT-licensed.*

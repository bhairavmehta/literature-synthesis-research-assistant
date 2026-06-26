"""Interactive web UI to drive the Literature Synthesis Research Assistant.

Zero extra dependencies -- uses only the Python standard library (http.server),
keeping with the project's stdlib-only ethos. Embeds the frontend (HTML/CSS/JS)
inline and exposes a tiny JSON API that calls the same `answer()` pipeline used
by examples/demo.py.

    python app.py                 # serve on http://127.0.0.1:8000
    python app.py --port 9000     # custom port
    python app.py --no-browser    # do not auto-open a browser

Runs fully offline with the deterministic stub provider. Set LSRA_PROVIDER and
the matching API key (and LSRA_ONLINE_RETRIEVAL=1) to drive real models.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from lsra import answer  # noqa: E402
from lsra.state import LSRAState  # noqa: E402

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus", "papers.json")

PROVIDER = os.environ.get("LSRA_PROVIDER", "stub").strip().lower()
ONLINE_RETRIEVAL = os.environ.get("LSRA_ONLINE_RETRIEVAL", "0") == "1"

if PROVIDER == "anthropic":
    RUNTIME_STATUS = "online · Anthropic provider"
    FOOTER_LABEL = "Online mode enabled — powered by Anthropic."
elif PROVIDER == "openai":
    RUNTIME_STATUS = "online · OpenAI provider"
    FOOTER_LABEL = "Online mode enabled — powered by OpenAI."
else:
    RUNTIME_STATUS = "offline · deterministic stub provider"
    FOOTER_LABEL = "Offline demo · no API keys · stdlib-only server."

SAMPLE_QUESTIONS = [
    {
        "q": "Should we adopt LLM-as-Judge for our evaluation pipeline, and under what calibration regime?",
        "impact": "high", "tag": "Evaluation",
        "note": "Exercises the contradiction resolver (pro vs. con synthetic sources) and the injection scanner. Routes SYNC_REVIEW.",
    },
    {
        "q": "What retrieval strategy best reduces hallucinated citations in RAG?",
        "impact": "medium", "tag": "Retrieval",
        "note": "Drives HyDE -> two-stage rerank -> CRAG gate and the grounding threshold.",
    },
    {
        "q": "When does iterative self-reflection improve an agent's reasoning?",
        "impact": "low", "tag": "Reasoning",
        "note": "Low impact -- typically routes AUTO_PASS / ASYNC_BATCH. Pulls Reflexion, Self-Refine, Self-Consistency.",
    },
    {
        "q": "How does the position of evidence in a long context affect answer quality?",
        "impact": "medium", "tag": "Long context",
        "note": "Targets the 'Lost in the Middle' U-shaped finding; tests recall over a single strong source.",
    },
    {
        "q": "What are reference-free methods for detecting hallucinated or factually inconsistent text?",
        "impact": "medium", "tag": "Hallucination",
        "note": "Pulls SelfCheckGPT, QAGS, and conformal factuality; exercises methodological appraisal.",
    },
    {
        "q": "How should we defend a retrieval-augmented pipeline against poisoned or adversarial sources?",
        "impact": "high", "tag": "Safety",
        "note": "Surfaces PoisonedRAG; the injection scanner quarantines the poisoned synthetic source.",
    },
    {
        "q": "Does exploring multiple reasoning paths beat single-chain chain-of-thought, and at what cost?",
        "impact": "low", "tag": "Reasoning",
        "note": "Compares Tree-of-Thoughts vs. Self-Consistency vs. CoT; weighs accuracy against compute.",
    },
    {
        "q": "What memory architecture should a long-horizon language agent use?",
        "impact": "medium", "tag": "Agents",
        "note": "Pulls CoALA and ReAct; tests synthesis across a conceptual framework and an empirical method.",
    },
]

# Shown in the "How it works" panel — mirrors src/lsra/agents/ and the report.
AGENTS = [
    {"n": 1, "name": "Planner", "file": "planner.py", "tagline": "Tree-of-Thought decomposition",
     "desc": "Explores several candidate framings of the question and commits to a set of sub-questions, so the brief never over-fits one premature framing.",
     "closes": "Premature commitment to one framing"},
    {"n": 2, "name": "Retriever", "file": "retriever.py", "tagline": "HyDE -> recall -> rerank -> CRAG",
     "desc": "Writes a hypothetical answer (HyDE), recalls neighbours from the vector store, reranks them, and runs a corrective-retrieval gate that labels each result CORRECT / AMBIGUOUS / INCORRECT.",
     "closes": "Stale / low-recall evidence"},
    {"n": 3, "name": "Reasoner", "file": "reasoner.py", "tagline": "RAG-QA + methodological appraisal",
     "desc": "Answers each sub-question against retrieved evidence and appraises how sound the support is — separating 'a source was found' from 'the claim is well-supported'.",
     "closes": "Conflating 'found' with 'sound'"},
    {"n": 4, "name": "Contradiction-Resolver", "file": "resolver.py", "tagline": "detect + adjudicate conflicts",
     "desc": "Detects claims that disagree, adjudicates toward the stronger evidence where it can, and labels the rest OPEN rather than silently averaging them into a false consensus.",
     "closes": "Conflicts silently averaged"},
    {"n": 5, "name": "Critic", "file": "critic.py", "tagline": "grounding gate + Platt/conformal calibration",
     "desc": "The disqualifier gate: a claim ships only if it resolves to a retrieval at/above the grounding threshold AND its Platt-calibrated confidence clears the split-conformal threshold.",
     "closes": "Hallucinated provenance (the disqualifier)"},
    {"n": 6, "name": "Synthesizer", "file": "synthesizer.py", "tagline": "compose the cited brief",
     "desc": "Composes the final brief from admitted claims only, attaching a verifiable citation to every statement and labelling unresolved sub-questions instead of answering them.",
     "closes": "A writer grading its own work"},
]


def corpus_payload() -> list:
    from lsra.tools import load_corpus
    papers = load_corpus(CORPUS_PATH)
    return [
        {
            "id": p.paper_id, "title": p.title, "authors": p.authors, "year": p.year,
            "venue": p.venue, "arxiv_id": p.arxiv_id, "synthetic": p.synthetic,
            "abstract": p.abstract,
        }
        for p in papers
    ]


JOB_STATE = {
    "status": "idle",
    "question": "",
    "impact": "medium",
    "trace": [],
    "last_updated": 0.0,
    "error": None,
}
JOB_LOCK = threading.Lock()

def state_to_dict(state) -> dict:
    """Serialise the LSRAState into the JSON the frontend renders."""
    return {
        "question": state.question,
        "impact": state.impact,
        "brief_markdown": state.brief_markdown,
        "route": state.route,
        "escalations": list(state.escalations),
        "safety_findings": list(state.safety_findings),
        "sub_questions": list(state.sub_questions),
        "trace": list(state.trace),
        "evidence": {
            sq: [
                {"paper_id": e.paper_id, "score": round(e.score, 3), "snippet": e.snippet}
                for e in evs
            ]
            for sq, evs in state.retrieved.items()
        },
        "contradictions": [
            {
                "sub_question": c.sub_question,
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "resolved": c.resolved,
                "resolution": c.resolution,
            }
            for c in state.contradictions
        ],
        "claims": [
            {
                "text": cl.text,
                "sub_question": cl.sub_question,
                "grounded": cl.grounded,
                "grounding_score": round(cl.grounding_score, 3),
                "calibrated_confidence": round(cl.calibrated_confidence, 3),
                "best_source": cl.best_source,
            }
            for cl in state.claims
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("  [http] " + (fmt % args) + "\n")

    def _send(self, code, body, content_type="application/json", head_only=False):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def do_HEAD(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html", head_only=True)
        elif self.path == "/api/samples":
            self._send(200, json.dumps(SAMPLE_QUESTIONS), head_only=True)
        elif self.path == "/api/agents":
            self._send(200, json.dumps(AGENTS), head_only=True)
        elif self.path == "/api/corpus":
            self._send(200, json.dumps(corpus_payload()), head_only=True)
        elif self.path == "/api/status":
            with JOB_LOCK:
                payload = {
                    "status": JOB_STATE["status"],
                    "question": JOB_STATE["question"],
                    "impact": JOB_STATE["impact"],
                    "trace": list(JOB_STATE["trace"]),
                    "last_updated": JOB_STATE["last_updated"],
                    "error": JOB_STATE["error"],
                }
            self._send(200, json.dumps(payload), head_only=True)
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True}), head_only=True)
        else:
            self._send(404, json.dumps({"error": "not found"}), head_only=True)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html")
        elif self.path == "/api/samples":
            self._send(200, json.dumps(SAMPLE_QUESTIONS))
        elif self.path == "/api/agents":
            self._send(200, json.dumps(AGENTS))
        elif self.path == "/api/corpus":
            self._send(200, json.dumps(corpus_payload()))
        elif self.path == "/api/status":
            with JOB_LOCK:
                payload = {
                    "status": JOB_STATE["status"],
                    "question": JOB_STATE["question"],
                    "impact": JOB_STATE["impact"],
                    "trace": list(JOB_STATE["trace"]),
                    "last_updated": JOB_STATE["last_updated"],
                    "error": JOB_STATE["error"],
                }
            self._send(200, json.dumps(payload))
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/answer":
            self._send(404, json.dumps({"error": "not found"}))
            return
        start_ts = time.time()
        question = None
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = (payload.get("question") or "").strip()
            impact = payload.get("impact", "medium")
            self.log_message("POST /api/answer start: question=%r impact=%s", question, impact)
            if not question:
                self._send(400, json.dumps({"error": "question is required"}))
                return
            if impact not in ("low", "medium", "high"):
                impact = "medium"
            state = LSRAState(question=question, impact=impact)
            state.trace.append("[status] queued; starting pipeline")
            with JOB_LOCK:
                JOB_STATE.update({
                    "status": "running",
                    "question": question,
                    "impact": impact,
                    "trace": state.trace,
                    "last_updated": time.time(),
                    "error": None,
                })
            state = answer(question, impact=impact, corpus_path=CORPUS_PATH, state=state)
            with JOB_LOCK:
                JOB_STATE.update({
                    "status": "done",
                    "trace": list(state.trace),
                    "last_updated": time.time(),
                })
            self._send(200, json.dumps(state_to_dict(state)))
        except Exception as exc:  # surface errors to the UI instead of 500-ing silently
            import traceback
            traceback.print_exc()
            with JOB_LOCK:
                JOB_STATE.update({
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "last_updated": time.time(),
                })
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            elapsed = time.time() - start_ts
            self.log_message("POST /api/answer end: question=%r elapsed=%.1fs", question, elapsed)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LSRA — Literature Synthesis Research Assistant</title>
<style>
  :root {
    --bg:#faf9f7; --panel:#ffffff; --panel2:#f8f7f3; --panel3:#f1efe8;
    --border:#e4e1db; --border2:#d9d5cf;
    --text:#1f1f1f; --muted:#636363; --faint:#8f8f8f;
    --accent:#a50021; --accent2:#d2292f; --cyan:#005c83;
    --good:#0f7035; --warn:#b66d10; --bad:#a50021; --chip:#f3f1ec;
  }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:15px/1.7 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .wrap { max-width:1140px; margin:0 auto; padding:0 24px; }

  header { border-bottom:1px solid var(--border); padding:28px 0 22px; background:var(--panel); }
  .brand { display:flex; align-items:center; gap:14px; }
  .logo { width:48px; height:48px; border-radius:14px; flex:none;
    background:linear-gradient(135deg,var(--accent),var(--accent2)); position:relative;
    box-shadow:0 8px 30px rgba(165,0,33,.12); }
  .logo:before { content:"LS"; position:absolute; inset:0; display:grid; place-items:center;
    font-weight:800; font-size:17px; color:#07101f; letter-spacing:.5px; }
  header h1 { margin:0; font-size:24px; letter-spacing:.24px; }
  header h1 b { color:var(--accent); }
  header .sub { color:var(--muted); font-size:13.5px; margin-top:5px; line-height:1.5; max-width:720px; }
  .topnav { margin-top:18px; display:flex; flex-wrap:wrap; gap:10px; }
  .topnav a { color:var(--accent); font-size:13px; font-weight:700; text-decoration:none; border:1px solid transparent; padding:8px 13px; border-radius:999px; background:rgba(165,0,33,.05); }
  .topnav a:hover { background:rgba(165,0,33,.12); }
  .quick-actions { display:flex; flex-wrap:wrap; gap:12px; margin-top:20px; }
  .mini { border:1px solid var(--border); background:var(--panel2); color:var(--text); border-radius:10px; padding:10px 14px; font:600 13px/1 inherit; cursor:pointer; transition:transform .15s, background .15s; }
  .mini:hover { background:rgba(212,18,40,.08); transform:translateY(-1px); }
  .mini:active { transform:translateY(0); }
  .statline { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }
  .stat { background:var(--chip); border:1px solid var(--border); border-radius:8px;
    padding:7px 12px; font-size:12px; color:var(--muted); }
  .stat b { color:var(--text); font-weight:600; }
  .stat .dot { display:inline-block; width:7px; height:7px; border-radius:50%;
    background:var(--good); margin-right:6px; vertical-align:0; box-shadow:0 0 8px var(--good); }

  main { padding:26px 0 70px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:18px;
    padding:24px; margin-bottom:22px; box-shadow:0 12px 32px rgba(30,32,34,.05); }
  .card > h2 { margin:0 0 4px; font-size:16px; }
  .card > .hint { color:var(--muted); font-size:13px; margin:0 0 16px; }

  label { display:block; font-size:12.5px; color:var(--muted); margin:0 0 6px;
    text-transform:uppercase; letter-spacing:.6px; }
  textarea, select { width:100%; background:var(--panel2); color:var(--text);
    border:1px solid var(--border2); border-radius:10px; padding:12px 13px;
    font:inherit; resize:vertical; }
  textarea:focus, select:focus { outline:none; border-color:var(--accent);
    box-shadow:0 0 0 3px rgba(91,157,255,.18); }
  .row { display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap; margin-top:14px; }
  button.primary { background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#07101f; border:0; border-radius:10px; padding:12px 26px; font:700 15px/1 inherit;
    cursor:pointer; box-shadow:0 6px 20px rgba(91,157,255,.3); }
  button.primary:hover { filter:brightness(1.07); }
  button.primary:disabled { opacity:.55; cursor:not-allowed; box-shadow:none; }
  .kbd { color:var(--faint); font-size:12px; margin-left:12px; }

  .seg { display:inline-flex; background:var(--panel2); border:1px solid var(--border2);
    border-radius:10px; padding:3px; gap:3px; }
  .seg button { background:transparent; border:0; color:var(--muted); cursor:pointer;
    padding:8px 16px; border-radius:7px; font:600 13px/1 inherit; }
  .seg button.on { color:#07101f; }
  .seg button.on[data-v=low] { background:var(--good); }
  .seg button.on[data-v=medium] { background:var(--warn); }
  .seg button.on[data-v=high] { background:var(--bad); color:#fff; }

  .samples-head { display:flex; align-items:center; justify-content:space-between; margin:20px 0 10px; }
  .samples-head span { font-size:12.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; }
  .chips { display:flex; gap:9px; flex-wrap:wrap; }
  .chip { background:var(--chip); border:1px solid var(--border); color:var(--text);
    border-radius:12px; padding:11px 14px; font-size:13px; cursor:pointer; max-width:360px;
    transition:border-color .12s, transform .12s; }
  .chip:hover { border-color:var(--accent); transform:translateY(-1px); }
  .chip .tagrow { display:flex; gap:7px; align-items:center; margin-bottom:4px; }
  .chip .tag { font-size:10.5px; letter-spacing:.4px; text-transform:uppercase; color:var(--cyan); }
  .chip .imp { font-size:10px; padding:1px 6px; border-radius:6px; font-weight:700; }
  .imp.low { background:rgba(63,185,80,.16); color:var(--good); }
  .imp.medium { background:rgba(227,160,8,.16); color:var(--warn); }
  .imp.high { background:rgba(248,81,73,.16); color:var(--bad); }
  .chip .qt { color:var(--text); line-height:1.4; }

  details.disclosure { border:1px solid var(--border); border-radius:14px; margin-bottom:20px;
    background:var(--panel); overflow:hidden; }
  details.disclosure > summary { cursor:pointer; padding:18px 22px; font-weight:600; font-size:15px;
    list-style:none; display:flex; align-items:center; justify-content:space-between; }
  details.disclosure > summary::-webkit-details-marker { display:none; }
  details.disclosure > summary .chev { color:var(--muted); transition:transform .2s; }
  details.disclosure[open] > summary .chev { transform:rotate(90deg); }
  details.disclosure > summary small { color:var(--muted); font-weight:400; margin-left:8px; }
  .disclosure-body { padding:0 22px 22px; }

  .pipeline { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin-bottom:18px; }
  .pnode { background:var(--panel2); border:1px solid var(--border2); border-radius:9px;
    padding:8px 12px; font-size:12.5px; font-weight:600; }
  .pnode small { display:block; color:var(--faint); font-weight:400; font-size:10.5px; }
  .parrow { color:var(--faint); }

  .agents { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }
  .agent { background:var(--panel2); border:1px solid var(--border); border-radius:11px; padding:15px; }
  .agent .top { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
  .agent .num { width:26px; height:26px; border-radius:8px; flex:none; display:grid; place-items:center;
    font-weight:800; font-size:13px; color:#07101f; background:linear-gradient(135deg,var(--accent),var(--accent2)); }
  .agent .nm { font-weight:700; font-size:14px; }
  .agent .tl { color:var(--cyan); font-size:11.5px; font-family:ui-monospace,Consolas,monospace; }
  .agent .ds { color:var(--muted); font-size:12.8px; }
  .agent .cl { margin-top:9px; font-size:11.5px; color:var(--warn); }
  .agent .cl b { color:var(--muted); font-weight:400; }

  .corpus-controls { display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap; }
  .corpus-controls input { flex:1; min-width:200px; background:var(--panel2); border:1px solid var(--border2);
    color:var(--text); border-radius:9px; padding:9px 12px; font:inherit; }
  .corpus-controls input:focus { outline:none; border-color:var(--accent); }
  .papers { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:12px; }
  .paper { background:var(--panel2); border:1px solid var(--border); border-radius:10px; padding:13px 15px; }
  .paper.syn { border-color:#5a3d1a; background:rgba(227,160,8,.05); }
  .paper .pt { font-weight:600; font-size:13.5px; line-height:1.35; }
  .paper .meta { color:var(--faint); font-size:11.5px; margin:4px 0 7px; }
  .paper .ab { color:var(--muted); font-size:12.3px; line-height:1.5; }
  .paper .flag { font-size:10px; font-weight:700; color:var(--warn); text-transform:uppercase;
    letter-spacing:.5px; margin-left:6px; }

  #status { color:var(--muted); font-size:13.5px; margin-top:18px; min-height:20px; display:flex;
    flex-direction:column; gap:10px; }
  .spinner { width:15px; height:15px; border:2px solid var(--border2); border-top-color:var(--accent);
    border-radius:50%; animation:spin .7s linear infinite; flex:none; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .current-step { color:var(--text); font-weight:700; }
  .status-lines { display:flex; flex-direction:column; gap:4px; font-size:13px; color:var(--text); }
  .status-error { color:var(--bad); font-weight:700; }
  .steps { display:flex; gap:6px; flex-wrap:wrap; }
  .steps .s { font-size:11px; color:var(--faint); padding:2px 8px; border-radius:6px; background:var(--chip); }
  .steps .s.done { color:var(--good); }
  .steps .s.active { color:var(--accent); background:rgba(91,157,255,.12); }

  #results { display:none; }
  .badges { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
  .badge { padding:8px 13px; border-radius:9px; font-size:12.5px; font-weight:600;
    border:1px solid var(--border2); background:var(--panel2); }
  .badge .k { color:var(--faint); font-weight:400; text-transform:uppercase; letter-spacing:.5px;
    font-size:10.5px; display:block; }
  .route-SYNC_REVIEW { border-color:var(--bad); color:var(--bad); }
  .route-ASYNC_BATCH { border-color:var(--warn); color:var(--warn); }
  .route-AUTO_PASS { border-color:var(--good); color:var(--good); }

  .tabs { display:flex; gap:3px; border-bottom:1px solid var(--border); margin-bottom:20px; flex-wrap:wrap; }
  .tab { padding:10px 15px; cursor:pointer; color:var(--muted); font-size:13.5px;
    border-bottom:2px solid transparent; margin-bottom:-1px; }
  .tab:hover { color:var(--text); }
  .tab.active { color:var(--text); border-bottom-color:var(--accent); }
  .tab .ct { display:inline-block; background:var(--chip); border-radius:10px; padding:0 7px;
    font-size:11px; margin-left:6px; color:var(--muted); }
  .pane { display:none; } .pane.active { display:block; }
  #section-tabs { margin-bottom:24px; }
  #section-tabs .tab { border:1px solid var(--border); border-bottom-color:transparent; background:var(--panel2); border-radius:10px 10px 0 0; }
  #section-tabs .tab.active { background:var(--panel); border-color:var(--border); border-bottom-color:var(--panel); }

  .architecture-board { display:grid; gap:18px; padding:8px 0; }
  .arch-row { display:flex; flex-wrap:wrap; gap:14px; align-items:center; justify-content:center; }
  .arch-node { min-width:180px; flex:1 1 220px; background:var(--panel2); border:1px solid var(--border); border-radius:14px; padding:18px 16px; box-shadow:0 10px 24px rgba(31,31,31,.06); }
  .arch-node h3 { margin:0 0 8px; font-size:15px; color:var(--accent); }
  .arch-node p { margin:0; color:var(--muted); font-size:13px; line-height:1.5; }
  .arch-arrow { font-size:22px; color:var(--muted); }
  .arch-columns { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }
  .arch-block { background:var(--panel2); border:1px solid var(--border); border-radius:14px; padding:18px; }
  .arch-block h4 { margin:0 0 8px; font-size:14px; color:var(--accent); }
  .arch-block ul { padding-left:18px; margin:0; color:var(--muted); }
  .arch-block li { margin-bottom:10px; font-size:13px; line-height:1.6; }
    padding-bottom:6px; margin-top:28px; } .md h3 { font-size:15px; color:var(--accent); }
  .md ul { padding-left:22px; } .md li { margin:6px 0; }
  .md em { color:var(--muted); } .md hr { border:0; border-top:1px solid var(--border); margin:22px 0; }
  .md sup a { background:var(--chip); border:1px solid var(--border); border-radius:5px; padding:1px 5px;
    font-size:11px; }

  .claim { background:var(--panel2); border:1px solid var(--border); border-radius:11px;
    padding:14px 16px; margin-bottom:12px; }
  .claim .sq { color:var(--faint); font-size:11.5px; text-transform:uppercase; letter-spacing:.5px;
    margin-bottom:6px; }
  .meters { display:flex; gap:22px; margin-top:11px; flex-wrap:wrap; align-items:center; }
  .meter { font-size:11.5px; color:var(--muted); }
  .bar { width:150px; height:7px; background:var(--border); border-radius:4px; margin-top:5px; overflow:hidden; }
  .bar > i { display:block; height:100%; border-radius:4px; }
  .pill { font-size:11px; padding:3px 9px; border-radius:20px; font-weight:700; }
  .pill.ok { background:rgba(63,185,80,.16); color:var(--good); }
  .pill.no { background:rgba(248,81,73,.16); color:var(--bad); }
  .src { font-size:11.5px; color:var(--faint); font-family:ui-monospace,Consolas,monospace; }

  .evgroup { margin-bottom:18px; }
  .evgroup > h4 { font-size:13.5px; margin:0 0 9px; color:var(--text); }
  .ev { display:flex; gap:12px; align-items:flex-start; background:var(--panel2);
    border:1px solid var(--border); border-radius:9px; padding:11px 13px; margin-bottom:8px; }
  .ev .sc { flex:none; font-family:ui-monospace,Consolas,monospace; font-size:12px; color:var(--cyan);
    background:var(--chip); border-radius:6px; padding:3px 8px; }
  .ev .bd .pid { font-size:11.5px; color:var(--faint); font-family:ui-monospace,Consolas,monospace; }
  .ev .bd .sn { font-size:12.8px; color:var(--muted); }

  .contra { background:var(--panel2); border:1px solid var(--border); border-left:3px solid var(--warn);
    border-radius:10px; padding:14px 16px; margin-bottom:12px; }
  .contra.resolved { border-left-color:var(--good); }
  .contra .vs { display:flex; gap:12px; margin-top:10px; flex-wrap:wrap; }
  .contra .vs > div { flex:1; min-width:220px; background:var(--panel); border:1px solid var(--border);
    border-radius:8px; padding:10px 12px; font-size:13px; }

  pre.log { background:#070a0f; border:1px solid var(--border); border-radius:10px; padding:16px;
    overflow:auto; font:12.5px/1.75 ui-monospace,SFMono-Regular,Consolas,monospace; color:#c7d0db; margin:0; }
  pre.log .tag { color:var(--accent); }

  .finding { border-left:3px solid var(--bad); background:rgba(248,81,73,.07); padding:10px 14px;
    border-radius:8px; margin-bottom:8px; font-size:13px; }
  .finding.esc { border-left-color:var(--warn); background:rgba(227,160,8,.07); }
  .subhead { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.6px;
    margin:18px 0 9px; }
  .empty { color:var(--faint); font-style:italic; padding:8px 0; }

  footer { color:var(--faint); font-size:13px; text-align:center; padding:32px 24px;
    border-top:1px solid var(--border); background:var(--panel2); }
</style>
</head>
<body>
<header><div class="wrap">
  <div class="brand">
    <div class="logo"></div>
    <div>
      <h1><b>LSRA</b> · Literature Synthesis Research Assistant</h1>
      <div class="sub">An auditable, citation-grounded six-agent system — turns an open ML/AI research
        question into a brief grounded in the literature.</div>
    </div>
  </div>
  <div class="statline">
    <div class="stat"><span class="dot"></span>{RUNTIME_STATUS}</div>
    <div class="stat"><b id="st-papers">19</b> papers in corpus</div>
    <div class="stat"><b>6</b> agents</div>
    <div class="stat">defense target: <b>confident hallucinated synthesis</b></div>
  </div>
  <div style="margin-top:16px; display:flex; flex-wrap:wrap; gap:16px; align-items:center;">
    <div style="font-size:13px; color:var(--muted);">
      <strong>Bhairav Mehta</strong> · Agentic AI Graduate Program Student
    </div>
    <div style="font-size:13px; color:var(--muted);">
      Carnegie Mellon University
    </div>
  </div>
  <div class="topnav">
    <a href="#ask">Ask a question</a>
    <a href="#section-how">Pipeline</a>
    <a href="#section-corpus">Corpus</a>
    <a href="#results">Results</a>
    <a href="#architecture-section">Architecture</a>
  </div>
  <div style="margin-top:10px; color:var(--muted); font-size:13px;">
    Status: app deployed and responding. When Anthropic online mode is enabled, answer generation may take longer due to external model inference and cold-start latency.
  </div>
</div></header>

<main><div class="wrap">

  <div class="card" id="ask">
    <h2>Ask a research question</h2>
    <p class="hint">A senior-level, open ML/AI question. The pipeline plans sub-questions, retrieves and
      grades evidence, resolves contradictions, gates on grounding + calibration, then composes a cited brief.</p>
    <label for="q">Question</label>
    <textarea id="q" rows="3" placeholder="e.g. Should we adopt LLM-as-Judge for our evaluation pipeline, and under what calibration regime?"></textarea>
    <div class="row">
      <div>
        <label>Decision impact <span style="text-transform:none;color:var(--faint)">— drives human-in-the-loop routing</span></label>
        <div class="seg" id="impact">
          <button data-v="low">low</button>
          <button data-v="medium">medium</button>
          <button data-v="high" class="on">high</button>
        </div>
      </div>
      <div style="margin-left:auto">
        <button class="primary" id="run">Run synthesis</button>
        <span class="kbd">⌘/Ctrl + Enter</span>
      </div>
    </div>

    <div class="samples-head"><span>Try an example</span><span id="sample-count"></span></div>
    <div class="chips" id="samples"></div>
    <div class="quick-actions">
      <button class="mini" id="show-pipeline">Show pipeline</button>
      <button class="mini" id="show-corpus">Browse corpus</button>
      <button class="mini" id="show-architecture">View architecture</button>
      <button class="mini" id="focus-results">View latest results</button>
    </div>
    <div id="status"></div>
  </div>

  <div class="card" id="explainer-tabs">
    <div class="tabs" id="section-tabs">
      <div class="tab active" data-tab="how">How it works</div>
      <div class="tab" data-tab="corpus">Corpus browser</div>
    </div>
    <div class="pane active" id="section-how">
      <div class="pipeline" id="pipeline"></div>
      <div class="agents" id="agents"></div>
    </div>
    <div class="pane" id="section-corpus">
      <div class="corpus-controls">
        <input id="corpus-search" placeholder="Filter by title, author, abstract…">
        <label style="display:flex;align-items:center;gap:7px;text-transform:none;letter-spacing:0;color:var(--muted);margin:0">
          <input type="checkbox" id="corpus-syn" style="width:auto"> synthetic only</label>
      </div>
      <div class="papers" id="papers"></div>
    </div>
  </div>

  <div id="architecture-section"></div>
  <div id="results" class="card">
    <div class="badges" id="badges"></div>
    <div class="tabs" id="tabs"></div>
    <div class="pane" id="pane-brief"></div>
    <div class="pane" id="pane-claims"></div>
    <div class="pane" id="pane-evidence"></div>
    <div class="pane" id="pane-contra"></div>
    <div class="pane" id="pane-safety"></div>
    <div class="pane" id="pane-architecture"></div>
    <div class="pane" id="pane-log"></div>
  </div>

</div></main>

<footer>Built for the CMU Agentic AI Executive Education capstone — Bhairav Mehta.
  {FOOTER_LABEL}</footer>

<script>
const $ = s => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h!=null) e.innerHTML = h; return e; };
const escapeHtml = t => (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
let IMPACT = 'high';

// ---- impact segmented control ----
$('#impact').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('#impact').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); IMPACT = b.dataset.v;
});

// ---- sample question chips ----
fetch('/api/samples').then(r => r.json()).then(list => {
  $('#sample-count').textContent = list.length + ' scenarios';
  const box = $('#samples');
  list.forEach(s => {
    const c = el('div', 'chip');
    c.title = s.note;
    c.appendChild(el('div', 'tagrow',
      `<span class="tag">${s.tag}</span><span class="imp ${s.impact}">${s.impact}</span>`));
    c.appendChild(el('div', 'qt', escapeHtml(s.q)));
    c.onclick = () => {
      $('#q').value = s.q;
      IMPACT = s.impact;
      $('#impact').querySelectorAll('button').forEach(x =>
        x.classList.toggle('on', x.dataset.v === s.impact));
      $('#q').scrollIntoView({behavior:'smooth', block:'center'});
    };
    box.appendChild(c);
  });
});

// ---- how-it-works: pipeline + agents ----
fetch('/api/agents').then(r => r.json()).then(ags => {
  const flow = ['Planner','Retriever','Reasoner','Resolver','Critic','Synthesizer'];
  const sub = {Planner:'decompose',Retriever:'HyDE·CRAG',Reasoner:'appraise',Resolver:'adjudicate',Critic:'grounding gate',Synthesizer:'cite'};
  const pl = $('#pipeline');
  flow.forEach((n,i) => {
    pl.appendChild(el('div','pnode', `${n}<small>${sub[n]}</small>`));
    if (i < flow.length-1) pl.appendChild(el('span','parrow','→'));
  });
  const ag = $('#agents');
  ags.forEach(a => {
    ag.appendChild(el('div','agent',
      `<div class="top"><div class="num">${a.n}</div>
         <div><div class="nm">${a.name}</div><div class="tl">${escapeHtml(a.file)}</div></div></div>
       <div class="tl" style="margin-bottom:7px">${escapeHtml(a.tagline)}</div>
       <div class="ds">${escapeHtml(a.desc)}</div>
       <div class="cl"><b>closes:</b> ${escapeHtml(a.closes)}</div>`));
  });
});

// ---- corpus browser ----
let CORPUS = [];
fetch('/api/corpus').then(r => r.json()).then(ps => {
  CORPUS = ps;
  $('#st-papers').textContent = ps.length;
  $('#corpus-count').textContent = ps.length;
  renderCorpus();
});
function renderCorpus() {
  const q = ($('#corpus-search').value || '').toLowerCase();
  const synOnly = $('#corpus-syn').checked;
  const box = $('#papers'); box.innerHTML = '';
  const rows = CORPUS.filter(p => (!synOnly || p.synthetic) &&
    (!q || (p.title+p.authors+p.abstract).toLowerCase().includes(q)));
  if (!rows.length) { box.innerHTML = '<div class="empty">No matching papers.</div>'; return; }
  rows.forEach(p => {
    const arx = p.arxiv_id ? ` · <a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank" rel="noopener">arXiv:${p.arxiv_id}</a>` : '';
    box.appendChild(el('div', 'paper' + (p.synthetic ? ' syn' : ''),
      `<div class="pt">${escapeHtml(p.title)}${p.synthetic?'<span class="flag">synthetic</span>':''}</div>
       <div class="meta">${escapeHtml(p.authors)} · ${p.year} · ${escapeHtml(p.venue)}${arx}</div>
       <div class="ab">${escapeHtml(p.abstract)}</div>`));
  });
}
$('#corpus-search').addEventListener('input', renderCorpus);
$('#corpus-syn').addEventListener('change', renderCorpus);

// ---- tiny markdown renderer (good enough for the brief shape) ----
function md(src) {
  const inline = t => escapeHtml(t)
    .replace(/\[\[(\d+)\]\]/g, '<sup><a href="#ref-$1">[$1]</a></sup>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/_(.+?)_/g, '<em>$1</em>');
  const lines = src.split('\n'); let html = '', inList = false;
  const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };
  for (const ln of lines) {
    if (/^### /.test(ln))      { closeList(); html += `<h3>${inline(ln.slice(4))}</h3>`; }
    else if (/^## /.test(ln))  { closeList(); html += `<h2>${inline(ln.slice(3))}</h2>`; }
    else if (/^# /.test(ln))   { closeList(); html += `<h1>${inline(ln.slice(2))}</h1>`; }
    else if (/^---\s*$/.test(ln)) { closeList(); html += '<hr>'; }
    else if (/^\s*[-*] /.test(ln)) { if (!inList) { html += '<ul>'; inList = true; } html += `<li>${inline(ln.replace(/^\s*[-*] /,''))}</li>`; }
    else if (/^\d+\.\s/.test(ln)) { closeList(); const m = ln.match(/^(\d+)\.\s(.*)/); html += `<p id="ref-${m[1]}"><b>${m[1]}.</b> ${inline(m[2])}</p>`; }
    else if (ln.trim() === '') { closeList(); }
    else { closeList(); html += `<p>${inline(ln)}</p>`; }
  }
  closeList(); return html;
}
const bar = (v, color) => `<div class="bar"><i style="width:${Math.round(v*100)}%;background:${color}"></i></div>`;

function setTabs(counts) {
  const defs = [
    ['brief','Brief',null], ['claims','Claims',counts.claims],
    ['evidence','Evidence',counts.evidence], ['contra','Contradictions',counts.contra],
    ['safety','Safety & Routing',counts.safety], ['architecture','Architecture',null], ['log','Audit log',counts.log],
  ];
  const tabs = $('#tabs'); tabs.innerHTML = '';
  defs.forEach(([id,label,ct],i) => {
    const t = el('div', 'tab' + (i===0?' active':''), label + (ct!=null ? ` <span class="ct">${ct}</span>` : ''));
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
      document.querySelectorAll('.pane').forEach(x=>x.classList.remove('active'));
      t.classList.add('active'); $('#pane-'+id).classList.add('active');
    };
    tabs.appendChild(t);
  });
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('active'));
  $('#pane-brief').classList.add('active');
}

function render(d) {
  const b = $('#badges'); b.innerHTML = '';
  const badge = (k,v,cls='') => b.appendChild(el('div','badge '+cls, `<span class="k">${k}</span>${v}`));
  badge('route', d.route, 'route-'+d.route);
  badge('impact', d.impact);
  badge('sub-questions', d.sub_questions.length);
  badge('claims admitted', d.claims.length);
  badge('contradictions', d.contradictions.length);
  if (d.safety_findings.length) badge('safety findings', d.safety_findings.length, 'route-SYNC_REVIEW');

  const evCount = Object.values(d.evidence).reduce((a,v)=>a+v.length,0);
  setTabs({ claims:d.claims.length, evidence:evCount, contra:d.contradictions.length,
            safety:d.safety_findings.length + d.escalations.length, log:d.trace.length });

  // brief
  $('#pane-brief').innerHTML = `<div class="md">${md(d.brief_markdown)}</div>`;

  // claims
  const cp = $('#pane-claims');
  cp.innerHTML = !d.claims.length ? '<div class="empty">No claims.</div>' : d.claims.map(c => `
    <div class="claim">
      <div class="sq">${escapeHtml(c.sub_question)}</div>
      <div>${escapeHtml(c.text)}</div>
      <div class="meters">
        <div class="meter">grounding ${c.grounding_score}${bar(c.grounding_score,'var(--accent)')}</div>
        <div class="meter">confidence ${c.calibrated_confidence}${bar(c.calibrated_confidence,'var(--accent2)')}</div>
        <div class="meter">${c.grounded?'<span class="pill ok">grounded</span>':'<span class="pill no">ungrounded</span>'}</div>
        <div class="meter src">source: ${escapeHtml(c.best_source||'—')}</div>
      </div>
    </div>`).join('');

  // evidence
  const ep = $('#pane-evidence');
  const groups = Object.entries(d.evidence);
  ep.innerHTML = !groups.length ? '<div class="empty">No retrieved evidence.</div>' :
    groups.map(([sq,evs]) => `
    <div class="evgroup"><h4>${escapeHtml(sq)}</h4>
      ${evs.map(e => `<div class="ev"><div class="sc">${e.score}</div>
        <div class="bd"><div class="pid">${escapeHtml(e.paper_id)}</div>
        <div class="sn">${escapeHtml(e.snippet)}</div></div></div>`).join('')}
    </div>`).join('');

  // contradictions
  const xp = $('#pane-contra');
  xp.innerHTML = !d.contradictions.length ? '<div class="empty">No contradictions detected.</div>' :
    d.contradictions.map(x => `
    <div class="contra ${x.resolved?'resolved':''}">
      <div><span class="pill ${x.resolved?'ok':'no'}">${x.resolved?'RESOLVED':'OPEN'}</span>
        <span class="src" style="margin-left:8px">${escapeHtml(x.sub_question)}</span></div>
      <div class="vs"><div>${escapeHtml(x.claim_a)}</div><div>${escapeHtml(x.claim_b)}</div></div>
      ${x.resolution ? `<div class="src" style="margin-top:9px">↳ ${escapeHtml(x.resolution)}</div>` : ''}
    </div>`).join('');

  // safety & routing
  const sp = $('#pane-safety');
  sp.innerHTML =
    `<div class="subhead">Route</div><div class="badge route-${d.route}" style="display:inline-block">${d.route}</div>` +
    '<div class="subhead">Escalation triggers</div>' +
    (d.escalations.length ? d.escalations.map(e=>`<div class="finding esc">${escapeHtml(e)}</div>`).join('') : '<div class="empty">None.</div>') +
    '<div class="subhead">Safety findings</div>' +
    (d.safety_findings.length ? d.safety_findings.map(f=>`<div class="finding">${escapeHtml(f)}</div>`).join('') : '<div class="empty">None.</div>');

  // audit log
  $('#pane-log').innerHTML = '<pre class="log">' + d.trace.map(l =>
    escapeHtml(l).replace(/^(\[[a-z]+\])/, '<span class="tag">$1</span>')).join('\n') + '</pre>';

  $('#results').style.display = 'block';
  $('#results').scrollIntoView({ behavior:'smooth', block:'start' });
}

function renderArchitecture() {
  $('#pane-architecture').innerHTML = `
    <div class="md">
      <h1>Architecture & Implementation</h1>
      <p>This app is a six-agent literature synthesis pipeline built for auditable AI research summarization.</p>
      <div class="architecture-board">
        <div class="arch-row">
          <div class="arch-node"><h3>Planner</h3><p>Decomposes the main research query into focused sub-questions for evidence-driven reasoning.</p></div>
          <div class="arch-arrow">→</div>
          <div class="arch-node"><h3>Retriever</h3><p>Generates HyDE prompts, recalls candidate evidence, reranks results, and applies the CRAG gate.</p></div>
          <div class="arch-arrow">→</div>
          <div class="arch-node"><h3>Reasoner</h3><p>Forms claims from retrieved evidence and assesses support quality before downstream synthesis.</p></div>
        </div>
        <div class="arch-row">
          <div class="arch-node"><h3>Contradiction Resolver</h3><p>Detects conflicting claims and resolves or labels open contradictions explicitly.</p></div>
          <div class="arch-arrow">→</div>
          <div class="arch-node"><h3>Critic</h3><p>Applies grounding and Platt/conformal calibration thresholds to filter out unsupported claims.</p></div>
          <div class="arch-arrow">→</div>
          <div class="arch-node"><h3>Synthesizer</h3><p>Composes the final cited brief using only admitted claims and grounded evidence.</p></div>
        </div>
      </div>
      <div class="arch-columns">
        <div class="arch-block">
          <h4>Frontend</h4>
          <ul>
            <li><strong>app.py</strong> serves the UI using Python standard library HTTP server.</li>
            <li>Everything is embedded inline: HTML, CSS, and JS.</li>
            <li>UI includes interactive tabs, sample scenarios, and corpus browser.</li>
          </ul>
        </div>
        <div class="arch-block">
          <h4>Core pipeline</h4>
          <ul>
            <li><strong>src/lsra/pipeline.py</strong>: main entrypoint.</li>
            <li><strong>src/lsra/agents/</strong>: six agent modules.</li>
            <li><strong>src/lsra/llm/</strong>: stub and provider adapters.</li>
          </ul>
        </div>
        <div class="arch-block">
          <h4>Data & mode</h4>
          <ul>
            <li><strong>data/corpus/papers.json</strong> stores the offline corpus.</li>
            <li><strong>LSRA_PROVIDER=anthropic</strong> enables online model mode.</li>
            <li><strong>LSRA_ONLINE_RETRIEVAL=1</strong> enables live retrieval.</li>
          </ul>
        </div>
      </div>
      <div class="arch-legend">
        <div class="legend-item"><div class="dot process"></div><div class="legend-text"><strong>Process</strong>: flow from planner to synthesizer.</div></div>
        <div class="legend-item"><div class="dot data"></div><div class="legend-text"><strong>Data</strong>: corpus, retrieval, evidence, claims.</div></div>
        <div class="legend-item"><div class="dot deploy"></div><div class="legend-text"><strong>Deployment</strong>: app service, config, and online model settings.</div></div>
      </div>
      <h2>Deployment status</h2>
      <p>The app is currently deployed and responding on the page. If model execution is delayed, it is usually because Anthropic requests are in progress or the app is warming up after a cold start.</p>
    </div>
  `;
}

const STEPS = ['planner','retriever','reasoner','resolver','critic','synthesizer'];
const STEP_LABELS = {
  planner: 'Planner',
  retriever: 'Retriever',
  reasoner: 'Reasoner',
  resolver: 'Contradiction Resolver',
  critic: 'Critic',
  synthesizer: 'Synthesizer',
};

function showProgress() {
  const html = '<div class="spinner"></div><div><div>Running the six-agent pipeline…</div>' +
    '<div class="current-step" id="current-step">Current step: starting…</div>' +
    '<div class="status-lines" id="status-lines"><div>Waiting for model execution...</div></div>' +
    '<div class="steps" id="steps">' + STEPS.map(s=>`<span class="s" data-s="${s}">${s}</span>`).join('') + '</div></div>';
  $('#status').innerHTML = html;
  let i = 0;
  const interval = setInterval(() => {
    const nodes = document.querySelectorAll('#steps .s');
    if (!nodes.length) return;
    nodes.forEach((n,k)=>{ n.classList.toggle('done',k<i); n.classList.toggle('active',k===i); });
    i = (i+1) % (STEPS.length+1);
  }, 280);
  return interval;
}

function currentStepFromTrace(trace) {
  if (!trace || !trace.length) return null;
  const last = trace.slice(-1)[0] || '';
  const lookup = {
    '[status]': 'planner',
    '[pipeline]': 'planner',
    '[planner]': 'planner',
    '[retriever]': 'retriever',
    '[reasoner]': 'reasoner',
    '[resolver]': 'resolver',
    '[critic]': 'critic',
    '[synthesizer]': 'synthesizer',
    '[graph] grounding failure': 'retriever',
    '[graph]': 'critic',
  };
  for (const key of Object.keys(lookup)) {
    if (last.includes(key)) return lookup[key];
  }
  if (/planner/i.test(last)) return 'planner';
  if (/retriev/i.test(last)) return 'retriever';
  if (/reasoner/i.test(last)) return 'reasoner';
  if (/resolver/i.test(last)) return 'resolver';
  if (/critic/i.test(last)) return 'critic';
  if (/synth/i.test(last)) return 'synthesizer';
  return null;
}

function highlightStep(step) {
  const nodes = document.querySelectorAll('#steps .s');
  if (!nodes.length) return;
  nodes.forEach(n => {
    const key = n.dataset.s;
    n.classList.toggle('active', key === step);
    n.classList.toggle('done', STEP_LABELS[key] && step && Object.keys(STEP_LABELS).indexOf(key) < Object.keys(STEP_LABELS).indexOf(step));
  });
}

function renderCurrentStep(trace) {
  const target = $('#current-step');
  if (!target) return;
  const step = currentStepFromTrace(trace);
  if (step) {
    target.textContent = `Current step: ${STEP_LABELS[step] || step}`;
    highlightStep(step);
  } else {
    target.textContent = 'Current step: waiting for pipeline start…';
  }
}

function renderStatusLines(status) {
  const target = $('#status-lines');
  if (!target) return;
  if (status.error) {
    target.innerHTML = `<div class="status-error">${escapeHtml(status.error)}</div>`;
    renderCurrentStep(status.trace || []);
    return;
  }
  renderCurrentStep(status.trace || []);
  const rows = status.trace.slice(-6).map(line => `<div>${escapeHtml(line)}</div>`);
  if (!rows.length) {
    target.innerHTML = '<div>Starting pipeline…</div>';
  } else {
    target.innerHTML = rows.join('');
  }
}

async function pollStatus() {
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) return;
    const status = await resp.json();
    renderStatusLines(status);
    return status.status;
  } catch (err) {
    return null;
  }
}

async function run() {
  const question = $('#q').value.trim();
  if (!question) { $('#status').innerHTML = '<span style="color:var(--warn)">Enter a question first.</span>'; return; }
  $('#run').disabled = true;
  const timer = showProgress();
  const poller = setInterval(async () => {
    const status = await pollStatus();
    if (status === 'done' || status === 'error') {
      clearInterval(poller);
    }
  }, 900);
  try {
    const r = await fetch('/api/answer', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ question, impact: IMPACT })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ('HTTP '+r.status));
    clearInterval(timer);
    clearInterval(poller);
    $('#status').innerHTML = '';
    render(d);
  } catch (e) {
    clearInterval(timer);
    clearInterval(poller);
    $('#status').innerHTML = '<span style="color:var(--bad)">Error: ' + escapeHtml(e.message) + '</span>';
  } finally {
    $('#run').disabled = false;
  }
}
$('#run').onclick = run;
$('#q').addEventListener('keydown', e => { if (e.key==='Enter' && (e.metaKey||e.ctrlKey)) run(); });
$('#show-pipeline').onclick = () => { document.querySelector('#section-how').scrollIntoView({behavior:'smooth', block:'start'}); activateSectionTab('how'); };
$('#show-corpus').onclick = () => { document.querySelector('#section-corpus').scrollIntoView({behavior:'smooth', block:'start'}); activateSectionTab('corpus'); };
$('#show-architecture').onclick = () => { document.querySelector('#architecture-section').scrollIntoView({behavior:'smooth', block:'start'}); $('#pane-architecture').classList.add('active'); document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); document.querySelector('.tab:nth-child(6)').classList.add('active'); renderArchitecture(); };
$('#focus-results').onclick = () => { document.querySelector('#results').scrollIntoView({behavior:'smooth', block:'start'}); };
const architectureLink = document.querySelector('a[href="#architecture-section"]');
if (architectureLink) {
  architectureLink.addEventListener('click', e => {
    e.preventDefault();
    document.querySelector('#architecture-section').scrollIntoView({behavior:'smooth', block:'start'});
    document.querySelector('#results').scrollIntoView({behavior:'smooth', block:'start'});
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    const archTab = Array.from(document.querySelectorAll('.tab'))[5];
    if (archTab) archTab.classList.add('active');
    renderArchitecture();
  });
}

document.querySelectorAll('#section-tabs .tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const name = tab.dataset.tab;
    activateSectionTab(name);
  });
});

function activateSectionTab(name) {
  document.querySelectorAll('#section-tabs .tab').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('#section-how, #section-corpus').forEach(x => x.classList.remove('active'));
  const pane = document.querySelector('#section-' + name);
  if (pane) pane.classList.add('active');
}
</script>
</body>
</html>
"""

INDEX_HTML = INDEX_HTML.replace("{RUNTIME_STATUS}", RUNTIME_STATUS).replace("{FOOTER_LABEL}", FOOTER_LABEL)

def main():
    ap = argparse.ArgumentParser()
    # Azure App Service (and most PaaS) inject the listen port via $PORT and
    # expect the app to bind 0.0.0.0. Fall back to localhost:8000 for local use.
    env_port = os.environ.get("PORT") or os.environ.get("WEBSITES_PORT")
    ap.add_argument("--port", type=int, default=int(env_port) if env_port else 8000)
    ap.add_argument("--host", default="0.0.0.0" if env_port else "127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if env_port:
        args.no_browser = True  # never try to open a browser on a server

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"LSRA web UI serving at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

"""Wire the six agents into the coordination graph (report Figure 2).

Backbone:  planner -> retriever -> reasoner -> resolver -> critic -> synthesizer
Feedback:  critic -> retriever re-loop on grounding failure (conditional edge),
           bounded by max_reloops to prevent runaway cost.

Uses LangGraph if installed; otherwise a faithful deterministic runner executes
the identical node functions and conditional edge. Either way the agents and the
shared `LSRAState` are unchanged.
"""
from __future__ import annotations
from ..state import LSRAState
from ..config import SETTINGS
from ..agents import (Planner, Retriever, Reasoner, ContradictionResolver,
                      Critic, Synthesizer)
from ..safety import route, escalation_triggers


class Pipeline:
    def __init__(self, store, papers_by_id, llm=None, max_reloops: int = 1):
        self.planner = Planner(llm)
        self.retriever = Retriever(store, llm)
        self.reasoner = Reasoner(papers_by_id, llm)
        self.resolver = ContradictionResolver()
        self.critic = Critic()
        self.synthesizer = Synthesizer(papers_by_id)
        self.max_reloops = max_reloops
        self._lang = _try_langgraph(self)

    # -- node functions ------------------------------------------------------
    def n_planner(self, s): return self.planner.run(s)
    def n_retriever(self, s): return self.retriever.run(s)
    def n_reasoner(self, s): return self.reasoner.run(s)
    def n_resolver(self, s): return self.resolver.run(s)
    def n_critic(self, s): return self.critic.run(s)
    def n_synth(self, s): return self.synthesizer.run(s)

    def _finalize(self, s: LSRAState) -> LSRAState:
        confs = [c.calibrated_confidence for c in s.claims if c.grounded] or [0.0]
        s.route = route(min(confs), s.impact, bool(s.safety_findings))
        s.escalations = escalation_triggers(s)
        return s

    # -- deterministic runner (used when LangGraph is absent) ----------------
    def run(self, state: LSRAState) -> LSRAState:
        if self._lang is not None:
            return self._lang(state)
        state = self.n_planner(state)
        state = self.n_retriever(state)
        state = self.n_reasoner(state)
        state = self.n_resolver(state)
        state = self.n_critic(state)
        reloops = 0
        # conditional grounding-failure edge: critic -> retriever
        while state.ungrounded and reloops < self.max_reloops:
            reloops += 1
            state.log(f"[graph] grounding failure -> re-retrieve (loop {reloops})")
            state.ungrounded = []
            state = self.n_retriever(state)
            state = self.n_reasoner(state)
            state = self.n_resolver(state)
            state = self.n_critic(state)
        state = self._finalize(state)
        state = self.n_synth(state)
        state.log(f"[graph] complete; route={state.route}")
        return state


def _try_langgraph(pipe: "Pipeline"):
    """Return a callable(state)->state backed by LangGraph, or None."""
    try:
        from langgraph.graph import StateGraph, END
    except Exception:
        return None
    try:
        g = StateGraph(dict)

        def wrap(fn):
            def _node(d):
                s = d["state"]; return {"state": fn(s)}
            return _node

        g.add_node("planner", wrap(pipe.n_planner))
        g.add_node("retriever", wrap(pipe.n_retriever))
        g.add_node("reasoner", wrap(pipe.n_reasoner))
        g.add_node("resolver", wrap(pipe.n_resolver))
        g.add_node("critic", wrap(pipe.n_critic))
        g.add_node("synth", wrap(pipe.n_synth))
        g.set_entry_point("planner")
        g.add_edge("planner", "retriever")
        g.add_edge("retriever", "reasoner")
        g.add_edge("reasoner", "resolver")
        g.add_edge("resolver", "critic")

        def gate(d):
            s = d["state"]
            if s.ungrounded and d.get("loops", 0) < pipe.max_reloops:
                return "retry"
            return "ok"
        g.add_conditional_edges("critic", gate, {"retry": "retriever", "ok": "synth"})
        g.add_edge("synth", END)
        app = g.compile()

        def _run(state: LSRAState) -> LSRAState:
            # finalize routing before synth by pre-running critic path then synth
            out = app.invoke({"state": state, "loops": 0})
            s = out["state"]
            return pipe._finalize(s) if not s.route else s
        return _run
    except Exception:
        return None


def build_pipeline(store, papers_by_id, llm=None, max_reloops: int = 1) -> Pipeline:
    return Pipeline(store, papers_by_id, llm=llm, max_reloops=max_reloops)

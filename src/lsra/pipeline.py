"""High-level entrypoint: load corpus, build the store + graph, run one query."""
from __future__ import annotations
from typing import Optional
from .state import LSRAState
from .tools import load_corpus, VectorStore
from .graph import build_pipeline
from .llm import get_llm


def answer(question: str, impact: str = "medium",
           corpus_path: Optional[str] = None) -> LSRAState:
    papers = load_corpus(corpus_path)
    papers_by_id = {p.paper_id: p for p in papers}
    store = VectorStore()
    store.add(papers)
    llm = get_llm()
    pipe = build_pipeline(store, papers_by_id, llm=llm)
    state = LSRAState(question=question, impact=impact)
    state.log(f"[pipeline] corpus={len(papers)} papers; provider live")
    return pipe.run(state)

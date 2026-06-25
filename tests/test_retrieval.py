import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lsra.tools import load_corpus, VectorStore
from lsra.retrieval import rerank, crag_gate


def _corpus():
    return os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "papers.json")


def test_vector_store_retrieves_relevant():
    store = VectorStore(); store.add(load_corpus(_corpus()))
    hits = store.search("LLM as judge evaluation calibration", k=5)
    ids = {p.paper_id for p, _ in hits}
    assert ids & {"geval2023", "judgebias2023", "conformal2024"}


def test_crag_labels():
    assert crag_gate([]) == "INCORRECT"

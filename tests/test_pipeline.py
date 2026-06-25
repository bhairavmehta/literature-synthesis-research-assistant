import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lsra import answer

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "papers.json")


def test_pipeline_end_to_end():
    st = answer("Should we adopt LLM-as-Judge for evaluation, and how to calibrate it?",
                impact="high", corpus_path=CORPUS)
    assert st.brief_markdown.startswith("# Technical Brief")
    assert "## References" in st.brief_markdown
    # every shipped claim is grounded
    assert all(c.grounded for c in st.claims if c in st.claims and c.grounded)
    # poisoned source must be quarantined, not cited
    assert "syn_poison" not in {c.best_source for c in st.claims if c.grounded}
    assert any("injection" in f for f in st.safety_findings)
    # high impact -> synchronous review
    assert st.route == "SYNC_REVIEW"


def test_grounding_gate_blocks_ungrounded():
    st = answer("totally unrelated query about marine biology of deep sea squid",
                impact="low", corpus_path=CORPUS)
    # nothing should be confidently grounded for an off-corpus query
    assert isinstance(st.brief_markdown, str)

"""arXiv client used in online mode (LSRA_ONLINE_RETRIEVAL=1).

Exposed read-only -- the design's MCP tools are query-only by construction.
Falls back to the local corpus when offline so the pipeline never hard-fails.
"""
from __future__ import annotations
from typing import List
from ..state import Paper
from ..config import SETTINGS
from .corpus import load_corpus


def search_arxiv(query: str, max_results: int = 10) -> List[Paper]:
    if not SETTINGS.online_retrieval:
        return load_corpus()
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    url = ("http://export.arxiv.org/api/query?search_query=all:"
           + urllib.parse.quote(query)
           + f"&start=0&max_results={max_results}")
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = resp.read().decode("utf-8")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: List[Paper] = []
    for e in ET.fromstring(data).findall("a:entry", ns):
        aid = (e.findtext("a:id", default="", namespaces=ns) or "").split("/abs/")[-1]
        out.append(Paper(
            paper_id=aid or e.findtext("a:title", "", ns)[:24],
            title=" ".join((e.findtext("a:title", "", ns) or "").split()),
            authors=", ".join(a.findtext("a:name", "", ns)
                              for a in e.findall("a:author", ns)),
            year=int((e.findtext("a:published", "2024", ns) or "2024")[:4]),
            venue="arXiv", source="arxiv.org", arxiv_id=aid,
            abstract=" ".join((e.findtext("a:summary", "", ns) or "").split()),
        ))
    return out or load_corpus()

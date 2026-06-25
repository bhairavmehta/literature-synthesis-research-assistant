"""Local corpus loader. The sample corpus stands in for the live literature so
the demo runs offline; in online mode the arXiv MCP client supplies papers."""
from __future__ import annotations
import json
import os
from typing import List
from ..state import Paper
from ..config import SETTINGS


def load_corpus(path: str = None) -> List[Paper]:
    path = path or os.path.join(SETTINGS.data_dir, "papers.json")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Paper(**p) for p in raw]

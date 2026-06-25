"""Literature Synthesis Research Assistant (LSRA).

A three-layer, six-agent system (LangGraph orchestration + CrewAI-style roles +
read-only MCP tools) that turns an open ML/AI research question into an
auditable, citation-grounded technical brief.
"""
from .pipeline import answer
from .state import LSRAState, Paper, Claim, Evidence, Contradiction

__all__ = ["answer", "LSRAState", "Paper", "Claim", "Evidence", "Contradiction"]
__version__ = "1.0.0"

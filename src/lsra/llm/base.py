"""LLM and Embedder interfaces, plus a factory that selects the backend.

The whole system is written against these two small interfaces so that the exact
same agent code runs offline (deterministic stub) or online (OpenAI/Anthropic).
"""
from __future__ import annotations
from typing import List, Protocol
from ..config import SETTINGS


class LLM(Protocol):
    def complete(self, system: str, prompt: str, temperature: float = 0.1) -> str: ...


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


def get_llm() -> "LLM":
    if SETTINGS.provider == "openai":
        from .providers import OpenAILLM
        return OpenAILLM()
    if SETTINGS.provider == "anthropic":
        from .providers import AnthropicLLM
        return AnthropicLLM()
    from .stub import StubLLM
    return StubLLM()


def get_embedder() -> "Embedder":
    if SETTINGS.provider in ("openai", "anthropic"):
        try:
            from .providers import ProviderEmbedder
            return ProviderEmbedder()
        except Exception:
            pass
    from .stub import HashEmbedder
    return HashEmbedder()

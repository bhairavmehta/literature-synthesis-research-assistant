"""Real-model adapters. Imported lazily only when LSRA_PROVIDER selects them.

These are thin wrappers; they keep the identical (system, prompt) -> str
interface so no agent code changes between offline and online modes.
"""
from __future__ import annotations
import os
from typing import List
from ..config import SETTINGS


class OpenAILLM:
    def __init__(self, model: str = None):
        from openai import OpenAI  # requires `pip install openai` + OPENAI_API_KEY
        self.client = OpenAI()
        self.model = model or os.environ.get("LSRA_OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content


class AnthropicLLM:
    def __init__(self, model: str = None):
        import anthropic  # requires `pip install anthropic` + ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("LSRA_ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def complete(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=1024, temperature=temperature,
            system=system, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class ProviderEmbedder:
    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.environ.get("LSRA_EMBED_MODEL", "text-embedding-3-small")

    def embed(self, text: str) -> List[float]:
        return self.client.embeddings.create(model=self.model, input=text).data[0].embedding

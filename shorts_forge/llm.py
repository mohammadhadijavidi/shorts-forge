"""Tiny LLM provider layer: OpenAI-compatible, Anthropic, and an offline mock.

No SDKs are required: requests go over HTTPS with the standard library.

Environment variables::

    SHORTS_FORGE_LLM_PROVIDER   heuristic | mock | openai | openrouter | groq | ollama | lmstudio | anthropic
    SHORTS_FORGE_LLM_MODEL      model name (a sensible default per provider)
    SHORTS_FORGE_LLM_BASE_URL   override the base URL of any OpenAI-compatible provider
    OPENAI_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

OPENAI_COMPATIBLE = {
    # name: (base_url, api-key env var, default model)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", "qwen2.5:7b"),
    "lmstudio": ("http://localhost:1234/v1", "LMSTUDIO_API_KEY", "local-model"),
}
PROVIDERS = ("heuristic", "mock", *OPENAI_COMPATIBLE, "anthropic")


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {url}: {exc.reason}") from exc


@dataclass
class OpenAICompatible:
    model: str
    base_url: str
    api_key: str = ""
    name: str = "openai"
    temperature: float = 0.0
    timeout: float = 120

    def complete(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(f"{self.base_url.rstrip('/')}/chat/completions", {
            "model": self.model, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, headers, self.timeout)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {str(data)[:300]}") from exc


@dataclass
class Anthropic:
    model: str
    api_key: str
    name: str = "anthropic"
    base_url: str = "https://api.anthropic.com/v1"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120

    def complete(self, system: str, user: str) -> str:
        data = _post_json(f"{self.base_url.rstrip('/')}/messages", {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        }, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected response: {str(data)[:300]}") from exc


class MockLLM:
    """Offline, deterministic stand-in used by tests and the demo.

    It is NOT a language model. It parses the sentences out of the prompt, runs the
    built-in heuristic scorer (:mod:`shorts_forge.highlights`) and answers with the
    JSON the real prompt asks for, so the full LLM code path (prompt -> JSON ->
    validation) runs without a key. Results are labelled ``picker: mock`` in the manifest.
    """

    name = "mock"
    model = "mock-heuristic"

    def complete(self, system: str, user: str) -> str:
        from .highlights import heuristic_answer  # local import: avoids a cycle

        return heuristic_answer(user)


def get_llm(provider: str | None = None, model: str | None = None,
            base_url: str | None = None) -> LLM | None:
    """Build the configured provider, or ``None`` for the no-LLM heuristic picker."""
    provider = (provider or os.getenv("SHORTS_FORGE_LLM_PROVIDER") or "heuristic").lower()
    model = model or os.getenv("SHORTS_FORGE_LLM_MODEL") or None
    base_url = base_url or os.getenv("SHORTS_FORGE_LLM_BASE_URL") or None
    if provider in ("heuristic", "none"):
        return None
    if provider == "mock":
        return MockLLM()
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        return Anthropic(model=model or "claude-haiku-4-5", api_key=key,
                         base_url=base_url or "https://api.anthropic.com/v1")
    if provider in OPENAI_COMPATIBLE:
        default_url, key_env, default_model = OPENAI_COMPATIBLE[provider]
        key = os.getenv(key_env, "")
        if not key and provider in ("openai", "openrouter", "groq"):
            raise LLMError(f"{key_env} is not set")
        return OpenAICompatible(model=model or default_model, base_url=base_url or default_url,
                                api_key=key, name=provider)
    raise LLMError(f"unknown provider {provider!r}; choose one of: {', '.join(PROVIDERS)}")

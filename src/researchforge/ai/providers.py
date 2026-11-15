"""AI provider abstraction.

Three concrete implementations — Anthropic, Google Gemini, OpenAI — plus an
Ollama-compatible HTTP provider for fully local/air-gapped setups.  All share
the same ``generate(system, user) -> str`` contract so higher-level code
(query generation, synthesis) is provider-agnostic.

Resolution order (first match wins):
1. RESEARCHFORGE_LLM sets the model; provider is inferred from the model name.
   ``ollama/<model>`` (e.g. ``ollama/qwen3:14b``) selects a local Ollama model
   at OLLAMA_HOST (default http://localhost:11434) — no API key.
2. ANTHROPIC_API_KEY present → Anthropic
3. GEMINI_API_KEY / GOOGLE_API_KEY present → Google Gemini
4. OPENAI_API_KEY present → OpenAI
5. None configured → returns None (commands degrade gracefully)
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from researchforge.ai.usage import Usage, current_purpose, record_call

# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@dataclass
class _Meter:
    """Carries one call's usage from inside the request to the ledger."""

    usage: Usage = field(default_factory=Usage)


@contextmanager
def _metered(name: str) -> Iterator[_Meter]:
    """Time a model call and file its tokens under the current purpose.

    Usage is recorded even when the call raises: a request that failed after the
    provider had already read the prompt still cost money, and omitting it would
    make the total quietly optimistic.
    """
    meter = _Meter()
    provider, _, model = name.partition("/")
    started = time.monotonic()
    try:
        yield meter
    finally:
        record_call(
            purpose=current_purpose(),
            provider=provider,
            model=model or name,
            usage=meter.usage,
            duration_seconds=time.monotonic() - started,
        )


def _usage_of(raw: object, input_field: str, output_field: str) -> Usage:
    """Read token counts off an SDK usage object, whatever it calls them.

    Returns zeros rather than raising when a field is missing: an SDK that
    renames a field, or a local model that reports nothing, should cost us
    accounting precision and not the generation itself.
    """
    if raw is None:
        return Usage()

    def count(field_name: str) -> int:
        value = getattr(raw, field_name, None)
        return value if isinstance(value, int) and value >= 0 else 0

    return Usage(input_tokens=count(input_field), output_tokens=count(output_field))


@runtime_checkable
class AiProvider(Protocol):
    """Minimal text-generation interface every provider must implement."""

    @property
    def name(self) -> str: ...

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str: ...


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Anthropic Claude provider using the official SDK."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-5") -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic") from exc
        self._client = Anthropic(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        with _metered(self.name) as meter:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            meter.usage = _usage_of(getattr(msg, "usage", None), "input_tokens", "output_tokens")
        block = msg.content[0]
        return block.text if hasattr(block, "text") else str(block)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiProvider:
    """Google Gemini provider using the google-genai SDK."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai SDK not installed. Run: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=api_key)
        # Resolve model at instantiation time so RESEARCHFORGE_LLM is honoured
        # even when GeminiProvider is constructed directly rather than via
        # get_provider().  Explicit argument always wins; env var is second;
        # the hard default is last.
        rf_llm = os.environ.get("RESEARCHFORGE_LLM", "")
        gemini_from_env = rf_llm if rf_llm.startswith("gemini-") else ""
        self._model = model or gemini_from_env or "gemini-3.6-flash"
        self._types = genai_types

    @property
    def name(self) -> str:
        return f"google/{self._model}"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        from google.genai import types as genai_types

        with _metered(self.name) as meter:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
            meter.usage = _usage_of(
                getattr(response, "usage_metadata", None),
                "prompt_token_count",
                "candidates_token_count",
            )
        return response.text or ""


# ---------------------------------------------------------------------------
# OpenAI (and compatible endpoints: Ollama, etc.)
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """OpenAI (or any OpenAI-compatible) provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai SDK not installed. Run: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        with _metered(self.name) as meter:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            meter.usage = _usage_of(
                getattr(resp, "usage", None), "prompt_tokens", "completion_tokens"
            )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Ollama (local models, native API — no SDK, no key)
# ---------------------------------------------------------------------------

OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def _strip_think(text: str) -> str:
    """Drop a leading ``<think>…</think>`` block some local reasoning models emit."""
    stripped = text.lstrip()
    if stripped.startswith("<think>") and "</think>" in stripped:
        return stripped.split("</think>", 1)[1].lstrip()
    return text


class OllamaProvider:
    """A model served by a local (or LAN) Ollama server, via its ``/api/chat``."""

    def __init__(self, model: str, host: str | None = None, *, timeout: float = 900.0) -> None:
        if not model:
            raise RuntimeError("Ollama needs a model name, e.g. RESEARCHFORGE_LLM=ollama/qwen3:14b")
        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST).rstrip("/")
        if not self._host.startswith(("http://", "https://")):
            self._host = f"http://{self._host}"
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        import json
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"num_predict": max_tokens},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _metered(self.name) as meter:
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if exc.code == 404 and "not found" in detail:
                    raise RuntimeError(
                        f"Ollama has no model '{self._model}'. Run: ollama pull {self._model}"
                    ) from exc
                raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail[:300]}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                raise RuntimeError(
                    f"Ollama isn't reachable at {self._host} ({exc}). "
                    "Start it with `ollama serve`, or set OLLAMA_HOST."
                ) from exc
            meter.usage = Usage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
            )
        return _strip_think(str((data.get("message") or {}).get("content") or ""))


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

_ANTHROPIC_PREFIXES = ("claude-",)
_GEMINI_PREFIXES = ("gemini-",)


_OLLAMA_PREFIX = "ollama/"


def _is_endpoint_url(s: str) -> bool:
    """An OpenAI-compatible endpoint URL (vLLM, LM Studio, Ollama's ``/v1`` …)."""
    return s.startswith("http://") or s.startswith("https://")


def _is_ollama_native_url(s: str) -> bool:
    """Ollama's native API root, ``http://host:11434/api`` — not OpenAI-compatible."""
    return _is_endpoint_url(s) and s.rstrip("/").endswith("/api")


def get_provider(
    *,
    provider_hint: str | None = None,
    model_hint: str | None = None,
) -> AiProvider | None:
    """Return a configured provider or None if no API key is set.

    ``provider_hint`` can be ``"anthropic"``, ``"google"``, ``"openai"`` or ``"ollama"``.
    ``model_hint`` overrides the model name (otherwise uses RESEARCHFORGE_LLM
    or the provider default).
    """
    rf_llm = os.environ.get("RESEARCHFORGE_LLM", "")

    # 1 ── explicit --provider flag
    if provider_hint:
        provider_hint = provider_hint.lower()
        if provider_hint in ("anthropic", "claude"):
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Export it or pass --provider openai / --provider google."
                )
            model = model_hint or rf_llm or "claude-opus-4-5"
            return AnthropicProvider(key, model)

        if provider_hint in ("google", "gemini"):
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
            model = model_hint or rf_llm or "gemini-3.6-flash"
            return GeminiProvider(key, model)

        if provider_hint in ("openai", "gpt"):
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            model = model_hint or rf_llm or "gpt-4o"
            return OpenAIProvider(key, model)

        if provider_hint == "ollama":
            model = model_hint or rf_llm.removeprefix(_OLLAMA_PREFIX)
            if not model or _is_endpoint_url(model):
                raise RuntimeError(
                    "Name the Ollama model: --model qwen3:14b, or "
                    "RESEARCHFORGE_LLM=ollama/qwen3:14b"
                )
            return OllamaProvider(model)

        raise RuntimeError(
            f"Unknown provider '{provider_hint}'. Use: anthropic | google | openai | ollama"
        )

    # 2 ── infer from RESEARCHFORGE_LLM
    if rf_llm:
        if rf_llm.startswith(_OLLAMA_PREFIX):
            return OllamaProvider(model_hint or rf_llm.removeprefix(_OLLAMA_PREFIX))
        if _is_ollama_native_url(rf_llm):
            host = rf_llm.rstrip("/").removesuffix("/api")
            return OllamaProvider(model_hint or "llama3", host=host)
        if _is_endpoint_url(rf_llm):
            key = os.environ.get("OPENAI_API_KEY", "local")
            return OpenAIProvider(key, model=model_hint or "llama3", base_url=rf_llm)
        if any(rf_llm.startswith(p) for p in _ANTHROPIC_PREFIXES):
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                return AnthropicProvider(key, model=model_hint or rf_llm)
        if any(rf_llm.startswith(p) for p in _GEMINI_PREFIXES):
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            if key:
                return GeminiProvider(key, model=model_hint or rf_llm)
        # treat as OpenAI-compatible model name
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            return OpenAIProvider(key, model=model_hint or rf_llm)

    # 3 ── env-key waterfall
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        return AnthropicProvider(anthropic_key, model=model_hint or "claude-opus-4-5")

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if gemini_key:
        return GeminiProvider(str(gemini_key), model=model_hint or "gemini-3.6-flash")

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return OpenAIProvider(openai_key, model=model_hint or "gpt-4o")

    return None

"""Per-competition token log: ``.overnight_<slug>/token_usage.jsonl``.

Kaggle commands that call a model directly (idea extraction, ideation,
resynthesize, arXiv scans, setup) wrap their provider with :func:`metered`.
Each call appends one JSON line ``{ts, purpose, provider, model,
input_tokens, output_tokens, duration_seconds}`` that the dashboard's Insights
view aggregates.

Work Claude does inside Claude Code (proposals in ``ai_provider_mode:
claude_code``) never passes through a provider and so is not — and cannot
be — metered here; the dashboard says so instead of showing zero cost.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

TOKEN_LOG = "token_usage.jsonl"


class MeteredProvider:
    """Transparent AiProvider proxy that records each call's usage."""

    def __init__(self, inner: Any, artifacts_dir: Path, purpose: str) -> None:
        self._inner = inner
        self._artifacts_dir = artifacts_dir
        self._purpose = purpose

    def __getattr__(self, name: Any) -> Any:
        return getattr(self._inner, name)

    def generate(self, system: str, user: str, *, max_tokens: int = 8192) -> str:
        from researchforge.ai.usage import current_purpose, purpose, recording

        # A caller's more specific label (e.g. ``proposal_round_7``) wins over
        # the provider's generic one, so usage can be attributed per round.
        label = current_purpose()
        with recording() as ledger, purpose(label if label != "other" else self._purpose):
            try:
                return str(self._inner.generate(system, user, max_tokens=max_tokens))
            finally:
                _append(self._artifacts_dir, ledger.calls)


def _append(artifacts_dir: Path, calls: Any) -> None:
    if not calls:
        return
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with (artifacts_dir / TOKEN_LOG).open("a", encoding="utf-8") as fh:
            for c in calls:
                fh.write(
                    json.dumps(
                        {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "purpose": c.purpose,
                            "provider": c.provider,
                            "model": c.model,
                            "input_tokens": c.usage.input_tokens,
                            "output_tokens": c.usage.output_tokens,
                            "duration_seconds": round(c.duration_seconds, 3),
                        }
                    )
                    + "\n"
                )
    except OSError:
        pass  # metering must never break the call it measures


def metered(provider: Any, artifacts_dir: Path, purpose: str) -> Any:
    """Wrap ``provider`` (may be None) so its calls land in token_usage.jsonl."""
    if provider is None:
        return None
    return MeteredProvider(provider, artifacts_dir, purpose)

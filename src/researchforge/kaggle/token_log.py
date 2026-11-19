"""Per-competition token log: ``.overnight_<slug>/token_usage.jsonl``.

Kaggle commands that call a model directly (idea extraction, ideation,
resynthesize, arXiv scans, setup) wrap their provider with :func:`metered`.
Each call appends one JSON line ``{ts, purpose, provider, model,
input_tokens, output_tokens, duration_seconds}`` (plus ``cache_read_tokens``,
``cache_write_tokens`` and ``cost_usd`` when the provider reports them) that
the dashboard's Insights view aggregates. ``input_tokens`` includes cached
prompt tokens. Auto-run launches with the Claude driver add one ``auto_run``
row per model the launch used (:func:`log_usage`).

Rounds someone drives from their own Claude Code chat never pass through
ResearchForge and so are not — and cannot be — metered here; the dashboard
says so instead of showing zero cost.
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


def _row(
    purpose: str, provider: str, model: str, usage: Any, duration_seconds: float
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "duration_seconds": round(duration_seconds, 3),
    }
    # Cache counts and a provider-reported cost only when there are any, so rows
    # from plain API calls stay as they were.
    for key in ("cache_read_tokens", "cache_write_tokens"):
        if getattr(usage, key, 0):
            row[key] = getattr(usage, key)
    if getattr(usage, "cost_usd", None) is not None:
        row["cost_usd"] = round(usage.cost_usd, 6)
    return row


def _write(artifacts_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with (artifacts_dir / TOKEN_LOG).open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    except OSError:
        pass  # metering must never break the call it measures


def _append(artifacts_dir: Path, calls: Any) -> None:
    _write(
        artifacts_dir,
        [_row(c.purpose, c.provider, c.model, c.usage, c.duration_seconds) for c in calls or []],
    )


def log_usage(
    artifacts_dir: Path,
    *,
    purpose: str,
    provider: str,
    model: str,
    usage: Any,
    duration_seconds: float = 0.0,
    **extra: Any,
) -> None:
    """Append one row for work that did not go through a provider (an auto-run launch)."""
    _write(artifacts_dir, [{**_row(purpose, provider, model, usage, duration_seconds), **extra}])


def metered(provider: Any, artifacts_dir: Path, purpose: str) -> Any:
    """Wrap ``provider`` (may be None) so its calls land in token_usage.jsonl."""
    if provider is None:
        return None
    return MeteredProvider(provider, artifacts_dir, purpose)

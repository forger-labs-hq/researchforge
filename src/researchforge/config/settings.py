"""Configurable research-pipeline knobs (spec: "decisions that should remain configurable").

Precedence: code defaults < `.researchforge/config.json` < CLI flags.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from researchforge.config.paths import config_path


class ModelPrice(BaseModel):
    """Dollars per million tokens, as published by the provider."""

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)


#: Rates in dollars per million tokens. Matched by longest-prefix against the
#: model name, so "claude-opus-4-5-20251101" is priced by the "claude-opus-4"
#: entry without needing a row per dated release.
#:
#: These are defaults, not truth: published prices change, and negotiated rates
#: differ. Override them in `.researchforge/config.json` under `model_prices`.
#: Anything unmatched is priced at zero and reported as unpriced rather than
#: free, so a missing rate is visible instead of silently flattering the total.
DEFAULT_MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-opus-4": ModelPrice(input=15.0, output=75.0),
    "claude-sonnet-4": ModelPrice(input=3.0, output=15.0),
    "claude-haiku-4": ModelPrice(input=0.80, output=4.0),
    "claude-3-5-haiku": ModelPrice(input=0.80, output=4.0),
    "gemini-2.0-flash": ModelPrice(input=0.10, output=0.40),
    "gemini-2.5-flash": ModelPrice(input=0.30, output=2.50),
    "gemini-2.5-pro": ModelPrice(input=1.25, output=10.0),
    "gpt-4o-mini": ModelPrice(input=0.15, output=0.60),
    "gpt-4o": ModelPrice(input=2.50, output=10.0),
    "gpt-4.1": ModelPrice(input=2.00, output=8.00),
    "o3": ModelPrice(input=2.00, output=8.00),
    # Local models cost compute, not tokens; priced at zero on purpose.
    "llama": ModelPrice(input=0.0, output=0.0),
    "qwen": ModelPrice(input=0.0, output=0.0),
    "mistral": ModelPrice(input=0.0, output=0.0),
    "deepseek-r1": ModelPrice(input=0.0, output=0.0),
}


class ResearchSettings(BaseModel):
    min_queries: int = Field(default=3, ge=1)
    max_queries: int = Field(default=8, ge=1)
    max_candidates: int = Field(default=200, ge=10, le=1000)
    selected_papers: int = Field(default=30, ge=5, le=100)
    deep_synthesis_count: int = Field(default=12, ge=1)
    hypothesis_min: int = Field(default=3, ge=1)
    hypothesis_max: int = Field(default=7, ge=1)
    report_dir: str = "reports"
    screening_reject_margin_pct: float = Field(default=10.0, ge=0.0)
    tradeoff_material_pct: float = Field(default=5.0, ge=0.0)
    analytics_enabled: bool = False  # opt-in, local-only (spec §20)
    research_output_dir: str = ".researchforge/research-output"  # `paper package` target
    model_prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_MODEL_PRICES))
    """Dollars per million tokens, keyed by model-name prefix. See
    :data:`DEFAULT_MODEL_PRICES` — override to match your negotiated rates."""

    local_compute_usd_per_hour: float = Field(default=0.0, ge=0.0)
    """What an hour of this machine costs, for pricing benchmark compute.

    Zero by default because the honest answer is machine-specific: a laptop you
    already own has no marginal hourly cost, while a rented A100 has a rate on
    an invoice. Set it and compute hours are converted; leave it and they are
    reported as hours only.
    """


def load_settings(base: Path | None = None) -> ResearchSettings:
    """Load settings, applying overrides from `.researchforge/config.json` if present."""
    path = config_path(base)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ResearchSettings.model_validate(raw)
    return ResearchSettings()

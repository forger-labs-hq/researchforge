"""AI-driven synthesis: context bundle → landscape YAML + hypotheses YAML.

This module gives ResearchForge standalone intelligence — it calls a
configured AI provider (Anthropic, Google Gemini, OpenAI, or Ollama) to
perform the synthesis step that was previously only possible through
Claude Code or Cursor.

The synthesis prompt reuses the grounding rules and schemas already embedded
in the context bundle, so the same validation guarantees apply regardless of
which model generates the artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from researchforge.ai.providers import AiProvider
from researchforge.ai.usage import purpose
from researchforge.research.context_export import SynthesisContext

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a research synthesis assistant for ResearchForge, an AI-assisted \
research benchmarking tool.

Your task: given a bundle of arXiv papers (titles + abstracts) and a \
research objective, produce TWO structured YAML artifacts:

1. A research landscape (landscape.yaml) that groups papers into research \
directions with evidence claims.
2. A hypotheses file (hypotheses.yaml) containing testable hypotheses \
grounded in the retrieved papers.

CRITICAL RULES:
- Cite ONLY paper_ids present in the provided bundle.
- Base reported_findings ONLY on the abstract text; label anything beyond it \
as evidence_type 'interpretation' or 'speculation'.
- Use gap language: "underexplored", "not established in the retrieved \
literature" — never claim novelty.
- novelty_confidence must be "low", "medium", or "unknown" — NEVER "high".
- feasibility must be "low", "medium", or "high".
- estimated_effort must be "low", "medium", or "high".
- ImpactDirection must be "increase", "decrease", or "unknown".
- direction_id pattern: dir-001, dir-002, ...
- evidence_id pattern: ev-001, ev-002, ...
- hypothesis_id pattern: hyp-001, hyp-002, ...
- EvidenceType must be: published_claim | interpretation | speculation
- EvidenceStrength (paper_annotations.evidence_strength) must be EXACTLY one of:
  "low", "medium", "high", or "unknown".
  NEVER USE: weak, moderate, strong, minimal, substantial, limited,
  high-quality — ONLY those 4 values.
- HypothesisStatus must be: speculative
- Do NOT include fields not in the schema.
- Both YAML blocks must be valid, parseable YAML.

OUTPUT FORMAT — output EXACTLY this structure, nothing else:
<landscape>
(landscape YAML here)
</landscape>
<hypotheses>
(hypotheses YAML here)
</hypotheses>
"""


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def _build_user_prompt(ctx: SynthesisContext) -> str:
    """Serialize the context bundle into a model-friendly prompt."""
    lines: list[str] = []

    lines.append(f"## Project objective\n{ctx.project.objective or 'Not specified'}\n")

    if ctx.repository:
        repo = ctx.repository
        kw = ", ".join(repo.keywords[:8]) if repo.keywords else "none"
        lines.append(
            f"## Repository context\n- Compatibility: {repo.compatibility}\n- Keywords: {kw}\n"
        )

    lines.append(f"## Papers ({len(ctx.papers)} retrieved)\n")
    for p in ctx.papers:
        lines.append(
            f"### [{p.paper_id}] {p.title}\n"
            f"Score: {p.relevance_score:.2f} | "
            f"Categories: {', '.join(p.categories[:3])}\n"
            f"Abstract: {p.abstract[:600]}{'...' if len(p.abstract) > 600 else ''}\n"
        )

    n_min = ctx.settings.hypothesis_min
    n_max = ctx.settings.hypothesis_max
    lines.append(
        f"## Task\n"
        f"1. Group the papers into research directions (landscape.yaml).\n"
        f"2. Generate between {n_min} and {n_max} testable hypotheses (hypotheses.yaml).\n"
        f"Each hypothesis must cite at least one paper_id from the bundle.\n"
    )

    lines.append("## Grounding rules\n" + "\n".join(f"- {r}" for r in ctx.instructions))

    lines.append(
        "\n## Expected schemas\n"
        "landscape.yaml schema:\n"
        f"```json\n{json.dumps(ctx.expected_artifacts.landscape_schema, indent=2)}\n```\n"
        "hypotheses.yaml schema:\n"
        f"```json\n{json.dumps(ctx.expected_artifacts.hypotheses_schema, indent=2)}\n```\n"
    )

    lines.append(
        "Now produce the two YAML artifacts wrapped in <landscape>…</landscape> "
        "and <hypotheses>…</hypotheses> tags."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag>…</tag> from model output."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _strip_yaml_fence(text: str) -> str:
    """Remove optional ```yaml fences."""
    text = re.sub(r"^```(?:yaml)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_synthesis_response(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse the model's response into landscape and hypotheses dicts.

    Returns (landscape_dict, hypotheses_dict).
    Raises ValueError with a descriptive message if parsing fails.
    """
    landscape_raw = _extract_tag(raw, "landscape")
    hypotheses_raw = _extract_tag(raw, "hypotheses")

    if landscape_raw is None:
        raise ValueError(
            "Could not find <landscape>…</landscape> in model response. "
            "The model may have produced malformed output — try again."
        )
    if hypotheses_raw is None:
        raise ValueError(
            "Could not find <hypotheses>…</hypotheses> in model response. "
            "The model may have produced malformed output — try again."
        )

    landscape_raw = _strip_yaml_fence(landscape_raw)
    hypotheses_raw = _strip_yaml_fence(hypotheses_raw)

    try:
        landscape = yaml.safe_load(landscape_raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"landscape YAML parse error: {exc}") from exc

    try:
        hypotheses = yaml.safe_load(hypotheses_raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"hypotheses YAML parse error: {exc}") from exc

    if not isinstance(landscape, dict):
        raise ValueError("landscape YAML must be a mapping at the top level.")
    if not isinstance(hypotheses, dict):
        raise ValueError("hypotheses YAML must be a mapping at the top level.")

    return landscape, hypotheses


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def synthesize(
    ctx: SynthesisContext,
    provider: AiProvider,
) -> tuple[str, str]:
    """Call the AI provider and return (landscape_yaml_str, hypotheses_yaml_str).

    The returned strings are the raw YAML ready to write to disk; they have
    been round-tripped through PyYAML to verify they are valid.
    """
    user_prompt = _build_user_prompt(ctx)
    with purpose("synthesis"):
        raw = provider.generate(_SYSTEM, user_prompt, max_tokens=16384)

    landscape_dict, hypotheses_dict = parse_synthesis_response(raw)

    # Re-dump through PyYAML so the files are consistently formatted
    landscape_yaml = yaml.dump(landscape_dict, allow_unicode=True, sort_keys=False)
    hypotheses_yaml = yaml.dump(hypotheses_dict, allow_unicode=True, sort_keys=False)

    return landscape_yaml, hypotheses_yaml


def write_artifacts(
    landscape_yaml: str,
    hypotheses_yaml: str,
    landscape_path: Path,
    hypotheses_path: Path,
) -> None:
    """Write the YAML artifacts to the synthesis directory."""
    landscape_path.parent.mkdir(parents=True, exist_ok=True)
    hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
    landscape_path.write_text(landscape_yaml, encoding="utf-8")
    hypotheses_path.write_text(hypotheses_yaml, encoding="utf-8")

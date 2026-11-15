"""Ideation-only overnight: research without executing anything.

Reads the competition, top kernels, and arXiv papers, then produces a
prioritised plan of notebook-ready code changes the user can implement
manually in a Kaggle Notebook — no data download, no kernel push, no
benchmark execution required.

Output: .overnight_<slug>/ideation_plan.md  (human-readable)
        .overnight_<slug>/ideation_plan.json (machine-readable for Claude)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider
    from researchforge.kaggle.competition_reader import CompetitionMetadata
    from researchforge.kaggle.idea_sources import IdeaCollection


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class IdeationItem:
    """One concrete improvement idea with rationale and code sketch."""

    rank: int
    title: str
    rationale: str
    source: str  # e.g. "Top kernel: userX/kernel-slug"
    expected_impact: str  # e.g. "+1-3% AUC"
    code_sketch: str  # Python snippet ready to paste into a notebook
    branch_tag: str = ""  # group: "model", "features", "cv", "ensemble"


@dataclass
class IdeationPlan:
    """Full ideation output for one competition."""

    competition_slug: str
    competition_title: str
    metric_name: str
    metric_direction: str
    baseline_score: float | None
    baseline_source: str  # "lb_score", "local_cv", "not_available"
    kernel_title: str
    ideas: list[IdeationItem] = field(default_factory=list)
    what_we_learned: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_IDEATION_SYSTEM = """\
You are an expert Kaggle competitor. Given a competition's metadata, top kernel
titles, arXiv paper ideas, and user context, produce a concrete prioritised
improvement plan.

Return ONLY valid JSON — an object matching this schema exactly:
{
  "ideas": [
    {
      "rank": 1,
      "title": "<short title>",
      "rationale": "<why this should improve the metric>",
      "source": "<kernel title or paper name this draws from>",
      "expected_impact": "<e.g. +2-4% AUC>",
      "code_sketch": "<runnable Python snippet, 5-30 lines>",
      "branch_tag": "<model|features|cv|ensemble|augmentation|postprocess>"
    }
  ],
  "what_we_learned": [
    "<plain-language insight 1>",
    "<plain-language insight 2>",
    "<plain-language insight 3>"
  ]
}

Rules:
- Produce 5-8 ideas ordered by expected impact (highest first).
- code_sketch must be a self-contained Python snippet a Kagglist can paste
  directly into a notebook cell — imports included, no placeholders.
- Draw on the specific kernels and papers provided — cite them in "source".
- what_we_learned: 3-5 plain-language findings from the research, written
  so a non-expert understands what actually matters for this competition.
- Never recommend downloading data locally — all code runs on Kaggle servers.
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_ideation(
    slug: str,
    artifacts_dir: Path,
    ai_provider: AiProvider | None,
    metadata: CompetitionMetadata | None = None,
    idea_col: IdeaCollection | None = None,
    baseline_score: float | None = None,
    baseline_source: str = "not_available",
    kernel_title: str = "",
    user_answers: dict[str, Any] | None = None,
) -> IdeationPlan:
    """Research the competition and produce a prioritised improvement plan.

    Works with or without an AI provider:
    - With AI: generates tailored code sketches and ranked ideas.
    - Without AI: returns the raw ideas from kernels/arXiv without ranking.
    """
    # Load cached metadata / ideas if not provided
    if metadata is None:
        from researchforge.kaggle.competition_reader import CompetitionMetadata as CM

        cached = CM.load(artifacts_dir)
        metadata = cached or CM(slug=slug)

    if idea_col is None:
        from researchforge.kaggle.idea_sources import IdeaCollection as IC

        idea_col = IC.load(artifacts_dir) or IC(competition_slug=slug)

    if ai_provider is not None:
        plan = _ai_ideation(
            slug,
            metadata,
            idea_col,
            baseline_score,
            baseline_source,
            kernel_title,
            user_answers or {},
            ai_provider,
        )
    else:
        plan = _heuristic_ideation(
            slug,
            metadata,
            idea_col,
            baseline_score,
            baseline_source,
            kernel_title,
        )

    _save_plan(plan, artifacts_dir)
    return plan


# ---------------------------------------------------------------------------
# AI-driven ideation
# ---------------------------------------------------------------------------


def _ai_ideation(
    slug: str,
    metadata: CompetitionMetadata,
    idea_col: IdeaCollection,
    baseline_score: float | None,
    baseline_source: str,
    kernel_title: str,
    user_answers: dict[str, Any],
    provider: AiProvider,
) -> IdeationPlan:
    all_ideas = idea_col.all_ideas[:30] if idea_col else []
    sources_text = (
        "\n".join(f"  - {idea}" for idea in all_ideas) if all_ideas else "  (none fetched)"
    )

    answers_text = ""
    if user_answers:
        answers_text = "\nUser context:\n" + "\n".join(
            f"  {k}: {v}" for k, v in user_answers.items() if v
        )

    baseline_text = f"{baseline_score:.4f} ({baseline_source})" if baseline_score else "unknown"

    user_prompt = (
        f"Competition: {slug}\n"
        f"Title: {metadata.title}\n"
        f"Task: {metadata.task_type}\n"
        f"Metric: {metadata.metric_name} ({metadata.metric_direction})\n"
        f"Baseline: {baseline_text}\n"
        f"Best public kernel: {kernel_title}\n"
        f"Evaluation: {metadata.evaluation_section[:300]}\n"
        f"Data: {metadata.data_description}\n\n"
        f"Ideas from top kernels and arXiv:\n{sources_text}"
        f"{answers_text}"
    )

    raw = provider.generate(_IDEATION_SYSTEM, user_prompt, max_tokens=4096)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back if AI response is malformed
        return _heuristic_ideation(
            slug, metadata, idea_col, baseline_score, baseline_source, kernel_title
        )

    ideas = [
        IdeationItem(
            rank=item.get("rank", i + 1),
            title=item.get("title", ""),
            rationale=item.get("rationale", ""),
            source=item.get("source", ""),
            expected_impact=item.get("expected_impact", ""),
            code_sketch=item.get("code_sketch", "# No code sketch provided"),
            branch_tag=item.get("branch_tag", ""),
        )
        for i, item in enumerate(data.get("ideas", []))
        if item.get("title")
    ]

    return IdeationPlan(
        competition_slug=slug,
        competition_title=metadata.title,
        metric_name=metadata.metric_name,
        metric_direction=metadata.metric_direction,
        baseline_score=baseline_score,
        baseline_source=baseline_source,
        kernel_title=kernel_title,
        ideas=ideas,
        what_we_learned=data.get("what_we_learned", []),
    )


# ---------------------------------------------------------------------------
# Heuristic fallback (no AI provider)
# ---------------------------------------------------------------------------


def _heuristic_ideation(
    slug: str,
    metadata: CompetitionMetadata,
    idea_col: IdeaCollection,
    baseline_score: float | None,
    baseline_source: str,
    kernel_title: str,
) -> IdeationPlan:
    """When no AI provider is available, package the raw ideas as-is."""
    ideas = [
        IdeationItem(
            rank=i + 1,
            title=idea,
            rationale="Drawn from top kernels or arXiv papers — evaluate manually.",
            source=next((s.title for s in idea_col.sources if idea in s.ideas), "idea_sources"),
            expected_impact="unknown (no AI ranking)",
            code_sketch="# Implement this idea in your Kaggle notebook",
        )
        for i, idea in enumerate((idea_col.all_ideas or [])[:8])
    ]
    return IdeationPlan(
        competition_slug=slug,
        competition_title=metadata.title,
        metric_name=metadata.metric_name,
        metric_direction=metadata.metric_direction,
        baseline_score=baseline_score,
        baseline_source=baseline_source,
        kernel_title=kernel_title,
        ideas=ideas,
        what_we_learned=[
            "Set ANTHROPIC_API_KEY (or GEMINI/OPENAI) for AI-ranked ideas with code sketches."
        ],
    )


# ---------------------------------------------------------------------------
# Save plan to disk
# ---------------------------------------------------------------------------


def _save_plan(plan: IdeationPlan, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # JSON (for Claude / machine use)
    json_path = artifacts_dir / "ideation_plan.json"
    json_path.write_text(json.dumps(_plan_to_dict(plan), indent=2))

    # Markdown (for human reading)
    md_path = artifacts_dir / "ideation_plan.md"
    md_path.write_text(_plan_to_markdown(plan))


def _plan_to_dict(plan: IdeationPlan) -> dict[str, Any]:
    return {
        "competition_slug": plan.competition_slug,
        "competition_title": plan.competition_title,
        "metric_name": plan.metric_name,
        "metric_direction": plan.metric_direction,
        "baseline_score": plan.baseline_score,
        "baseline_source": plan.baseline_source,
        "kernel_title": plan.kernel_title,
        "generated_at": plan.generated_at,
        "what_we_learned": plan.what_we_learned,
        "ideas": [
            {
                "rank": item.rank,
                "title": item.title,
                "rationale": item.rationale,
                "source": item.source,
                "expected_impact": item.expected_impact,
                "branch_tag": item.branch_tag,
                "code_sketch": item.code_sketch,
            }
            for item in plan.ideas
        ],
    }


def _plan_to_markdown(plan: IdeationPlan) -> str:
    baseline_str = (
        f"{plan.baseline_score:.4f} ({plan.baseline_source})"
        if plan.baseline_score
        else "not measured"
    )
    lines = [
        f"# Ideation Plan: {plan.competition_title or plan.competition_slug}",
        "",
        f"Generated: {plan.generated_at}",
        f"Metric: **{plan.metric_name}** ({plan.metric_direction})",
        f"Baseline: **{baseline_str}**",
        f"Starting from: {plan.kernel_title}",
        "",
    ]

    if plan.what_we_learned:
        lines += ["## What we learned", ""]
        for insight in plan.what_we_learned:
            lines.append(f"- {insight}")
        lines.append("")

    lines += ["## Prioritised improvements", ""]
    for item in plan.ideas:
        tag = f" `{item.branch_tag}`" if item.branch_tag else ""
        lines += [
            f"### {item.rank}. {item.title}{tag}",
            "",
            f"**Expected impact:** {item.expected_impact}",
            f"**Source:** {item.source}",
            "",
            f"{item.rationale}",
            "",
            "```python",
            item.code_sketch,
            "```",
            "",
        ]

    return "\n".join(lines)

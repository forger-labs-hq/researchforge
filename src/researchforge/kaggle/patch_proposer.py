"""AI-driven patch proposer for the overnight experiment loop.

Uses the configured AI provider (Anthropic/Google/OpenAI/Ollama — whatever is set via
env vars) to propose concrete code improvements based on:
  - Competition metadata (metric, task type, data description)
  - Idea sources (arXiv papers, top kernels)
  - User answers from the HIL session
  - Previous experiment rounds (what worked, what didn't)
  - The current DAG structure (parent patches, branches)

The proposer also supports requesting a mid-loop HIL pause by returning a
special "needs_user_input" key in the JSON payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchforge.ai.providers import AiProvider
    from researchforge.kaggle.orchestrator import OvernightSetup


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class PatchProposal:
    """AI's suggestion for a single improvement attempt."""

    title: str
    description: str
    code_changes: str  # actual Python code to write / diff to apply
    rationale: str
    expected_impact: str  # e.g. "+1-2% AUC expected"
    source_credit: str = ""  # which idea source this draws from
    parent_round_num: int | None = None  # DAG: which round this branches from
    branch_tag: str = ""  # DAG: named branch (e.g. "ensemble", "features")
    # The exact pool idea this tests ("" if none) — marks that entry tested
    hypothesis: str = ""
    # Up next items (the person's queue) this round follows, and the ones it declines, why
    follows: list[str] = field(default_factory=list)
    declines: dict[str, str] = field(default_factory=dict)
    # When not empty, the loop should pause and ask the user these questions
    needs_user_input: list[Any] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        """What the queue checks read (the same keys as a chat proposal.json)."""
        return {"hypothesis": self.hypothesis, "follows": self.follows, "declines": self.declines}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


_PROPOSER_SYSTEM = """\
You are an expert Kaggle competitor and ML engineer acting as the AI research engine
inside ResearchForge's overnight experiment loop.

Your job is to propose ONE concrete code improvement for the competition's benchmark
script each round. You must:
1. Draw on the hypothesis pool, the idea sources (arXiv, kernels) and previous round
   results. Pool entries are ranked best-first; a notebook's claimed leaderboard score
   is its author's word, not a measurement.
2. Produce a complete Python code block that implements the change in benchmark.py
   or creates a new helper module.
3. Justify why this specific change should improve the metric.
4. Be concise and actionable — working code only, no theory.
5. Learn from failures: if a similar idea already failed, propose something different.
6. If you genuinely need more information from the user to proceed effectively,
   set "needs_user_input" with a list of question objects.
7. Use the DAG: set parent_round_num to branch from the best round rather than
   always branching from round 0. Set branch_tag to group related experiments.

Return ONLY valid JSON (no prose, no markdown fences):
{
  "title": "<short title, max 60 chars>",
  "description": "<one paragraph description>",
  "code_changes": "<complete Python code block>",
  "rationale": "<why this should improve the metric>",
  "expected_impact": "<e.g. +1-3% AUC, based on kernel XYZ>",
  "source_credit": "<arXiv paper or kernel name, or empty string>",
  "parent_round_num": null,
  "branch_tag": "",
  "hypothesis": "<the exact idea text from the HYPOTHESIS POOL this tests, or empty string>",
  "follows": ["<ids from UP NEXT this round follows>"],
  "declines": {"<UP NEXT id this round cannot follow>": "<why>"},
  "needs_user_input": []
}
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PatchProposer:
    """Coordinates with the AI provider to propose code improvements."""

    def __init__(self, setup: OvernightSetup) -> None:
        self.setup = setup
        self.competition_slug = setup.competition_slug
        self.baseline_score = setup.baseline_result.cv_score if setup.baseline_result else None
        self.previous_rounds: list[Any] = []
        self._provider: AiProvider | None = self._resolve_provider()

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> Any:
        """Get the configured AI provider (same env-var resolution as main RF)."""

        # Metered so each round's proposal tokens land in token_usage.jsonl.
        from researchforge.kaggle.auto_run import project_provider

        # An API key from the environment or the auto-run env file.
        return project_provider(self.setup.artifacts_dir, "proposal", sensor=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_patch(self, round_num: int) -> PatchProposal:
        """Propose next improvement based on all available context.

        Raises:
            RuntimeError: If no AI provider is configured.
        """
        if self._provider is None:
            raise RuntimeError(
                "No AI provider configured. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, "
                "OPENAI_API_KEY, or RESEARCHFORGE_LLM=ollama/<model> (local) to enable "
                "automatic patch proposals.\n"
                "Export one of these environment variables and re-run."
            )

        from researchforge.ai.usage import purpose
        from researchforge.kaggle.steering import check_proposal

        context = self._build_context(round_num)
        with purpose(f"proposal_round_{round_num}"):
            proposal = self._call_ai(context, round_num)
            ignored = check_proposal(self.setup.artifacts_dir, proposal.as_record())
            if ignored:  # once more, told what it missed
                context += f"\n=== YOUR LAST PROPOSAL WAS REFUSED ===\nIt was refused: {ignored}.\n"
                proposal = self._call_ai(context, round_num)
                ignored = check_proposal(self.setup.artifacts_dir, proposal.as_record())
        if ignored:
            raise RuntimeError(f"Proposal refused: {ignored}")
        return proposal

    def record_round_result(self, round: Any) -> None:
        """Inform the proposer of what happened so it can learn."""
        self.previous_rounds.append(round)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, round_num: int) -> str:
        setup = self.setup
        if setup.contract is None:
            raise RuntimeError(
                "overnight_setup.json has no experiment contract — rerun "
                f"`researchforge kaggle overnight {self.competition_slug}` to set it up."
            )

        # Competition facts
        meta = getattr(setup, "competition_metadata", None)
        if meta is not None:
            meta_section = (
                f"Competition: {meta.slug}\n"
                f"Title: {meta.title}\n"
                f"Task: {meta.task_type}\n"
                f"Metric: {meta.metric_name} ({meta.metric_direction})\n"
                f"Data: {meta.data_description}\n"
                f"Evaluation: {meta.evaluation_section[:300]}\n"
            )
        else:
            primary = setup.contract.objective.primary_metric
            meta_section = (
                f"Competition: {self.competition_slug}\n"
                f"Metric: {primary.name} ({primary.direction})\n"
            )

        # Baseline
        primary = setup.contract.objective.primary_metric
        baseline_section = (
            f"Baseline CV score: {self.baseline_score}\nTarget: {primary.target_value:.4f}\n"
        )

        # Ideas
        idea_col = getattr(setup, "idea_sources", None)
        if idea_col is not None and idea_col.all_ideas:
            ideas_section = (
                "Available ideas from sources:\n"
                + "\n".join(f"  - {idea}" for idea in idea_col.all_ideas[:20])
                + "\n"
            )
        else:
            ideas_section = "No idea sources fetched yet.\n"

        pool_section = self._pool_section()

        # User answers
        user_answers = getattr(setup, "user_answers", None)
        if user_answers is not None and user_answers.answers:
            answers_section = (
                "User-provided context:\n"
                + "\n".join(f"  {k}: {v}" for k, v in user_answers.answers.items() if v)
                + "\n"
            )
        else:
            answers_section = "No additional user context.\n"

        # Previous rounds
        if not self.previous_rounds:
            rounds_section = "This is the first round — no prior experiments.\n"
        else:
            lines = []
            for r in self.previous_rounds[-10:]:  # last 10 rounds only
                improved = getattr(r, "improved_over_baseline", False)
                passed_full = getattr(r, "passed_full_benchmark", False)
                status = "improved" if improved else ("no improvement" if passed_full else "failed")
                full_result = getattr(r, "full_result", None)
                metric_str = (
                    f"{full_result.metric_value:.5f}"
                    if full_result and getattr(full_result, "metric_value", None) is not None
                    else "N/A"
                )
                branch = f" [branch: {r.branch_tag}]" if getattr(r, "branch_tag", "") else ""
                parent = (
                    f" [parent: {r.parent_round_num}]"
                    if getattr(r, "parent_round_num", None)
                    else ""
                )
                lines.append(
                    f"  Round {r.round_num}{parent}{branch}: {r.patch_description} "
                    f"-> {status} (metric: {metric_str})"
                )
                if getattr(r, "rejected_reason", None):
                    lines.append(f"    Rejection: {r.rejected_reason}")
            rounds_section = "Previous rounds:\n" + "\n".join(lines) + "\n"

        return (
            f"=== COMPETITION ===\n{meta_section}\n"
            f"=== BASELINE ===\n{baseline_section}\n"
            f"=== UP NEXT (the person's queue) ===\n{self._up_next_section()}\n"
            f"=== KAGGLE LEADERBOARD (our submissions) ===\n{self._leaderboard_section()}\n"
            f"=== HYPOTHESIS POOL (untested, ranked) ===\n{pool_section}\n"
            f"=== IDEAS FROM SOURCES ===\n{ideas_section}\n"
            f"=== USER CONTEXT ===\n{answers_section}\n"
            f"=== EXPERIMENT HISTORY (Round {round_num}) ===\n{rounds_section}"
        )

    def _up_next_section(self) -> str:
        """The person's standing directions and queue, with the rule for its head."""
        from researchforge.kaggle.steering import FOLLOW_RULE, prompt_lines

        lines = prompt_lines(self.setup.artifacts_dir)
        if not lines:
            return "Nothing queued: follow the hypothesis pool.\n"
        return "\n".join(f"  {line}" for line in lines) + f"\n  {FOLLOW_RULE}\n"

    def _leaderboard_section(self) -> str:
        """Local metric vs public LB for our submitted rounds (submission.leaderboard_table)."""
        from researchforge.kaggle.submission import LEADERBOARD_NOTE, leaderboard_table

        rows = leaderboard_table(self.setup.artifacts_dir)
        if not rows:
            return "Nothing submitted yet: the local metric is the only signal.\n"

        def num(v: Any) -> str:
            return f"{v:.5g}" if isinstance(v, (int, float)) else "—"

        lines = [
            f"  R{r['round']}: local {num(r['local'])} -> LB "
            + (num(r["lb"]) if r["lb"] is not None else str(r["lb_status"]).replace("_", " "))
            + (f" ({r['lb_error'][:160]})" if r.get("lb_error") else "")
            + (f" — {r['title']}" if r["title"] else "")
            for r in rows
        ]
        return "\n".join(lines) + f"\n  {LEADERBOARD_NOTE}\n"

    def _pool_section(self, limit: int = 10) -> str:
        """The live pool (watcher, arXiv, resynthesize, user) in the shared order."""
        from researchforge.kaggle.hypothesis_pool import describe, rank_untested

        contract = self.setup.contract
        direction = str(contract.objective.primary_metric.direction if contract else "maximize")
        ranked = rank_untested(self.setup.artifacts_dir, direction)
        if not ranked:
            return "Empty — every pooled idea has been tested.\n"
        lines = [f"  {i}. {describe(h)}" for i, h in enumerate(ranked[:limit], 1)]
        more = f"  … {len(ranked) - limit} more\n" if len(ranked) > limit else ""
        return "\n".join(lines) + "\n" + more

    # ------------------------------------------------------------------
    # AI call
    # ------------------------------------------------------------------

    def _call_ai(self, context: str, round_num: int) -> PatchProposal:
        """Call the AI provider and parse the JSON response into a PatchProposal."""
        user_prompt = f"Round {round_num}: propose the next improvement.\n\n{context}"

        assert self._provider is not None  # propose_patch checks this first
        raw = self._provider.generate(_PROPOSER_SYSTEM, user_prompt, max_tokens=2048)

        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"AI returned malformed JSON in round {round_num}: {exc}\n"
                f"Raw response (first 400 chars): {raw[:400]}"
            ) from exc

        # Parse optional needs_user_input questions
        from researchforge.kaggle.human_questions import Question

        needs_input = []
        for q_data in data.get("needs_user_input") or []:
            if isinstance(q_data, dict) and q_data.get("text"):
                needs_input.append(
                    Question(
                        id=q_data.get("id", f"mid_{round_num}"),
                        text=q_data["text"],
                        question_type=q_data.get("question_type", "text"),
                        choices=q_data.get("choices", []),
                        default=q_data.get("default", ""),
                        hint=q_data.get("hint", ""),
                    )
                )

        return PatchProposal(
            title=data.get("title", f"Round {round_num} improvement"),
            description=data.get("description", ""),
            code_changes=data.get("code_changes", ""),
            rationale=data.get("rationale", ""),
            expected_impact=data.get("expected_impact", ""),
            source_credit=data.get("source_credit", ""),
            parent_round_num=data.get("parent_round_num"),
            branch_tag=data.get("branch_tag", ""),
            hypothesis=str(data.get("hypothesis") or ""),
            follows=[str(x) for x in data.get("follows") or [] if x],
            declines={
                str(k): str(v) for k, v in (data.get("declines") or {}).items() if str(v).strip()
            }
            if isinstance(data.get("declines"), dict)
            else {},
            needs_user_input=needs_input,
        )

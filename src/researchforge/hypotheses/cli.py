"""`researchforge hypotheses` sub-app."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from researchforge.config.settings import load_settings
from researchforge.domain.hypothesis import Hypothesis, HypothesisStatus, ReviewOutcome
from researchforge.research.importers import import_hypotheses
from researchforge.storage.db import open_project_db
from researchforge.storage.hypothesis_repository import (
    get_hypothesis,
    list_hypotheses,
    record_review,
)
from researchforge.storage.project_repository import get_project
from researchforge.utils.output import JsonOption, echo_import_result, echo_model

hypotheses_app = typer.Typer(
    name="hypotheses", no_args_is_help=True, help="Evidence-backed hypotheses."
)

REVIEW_MARK = {
    HypothesisStatus.APPROVED: " ✓ approved",
    HypothesisStatus.REJECTED: " ✗ rejected",
}


def _print_hypothesis(hypothesis: Hypothesis) -> None:
    label = hypothesis.evidence_status.upper()
    typer.echo(f"[{hypothesis.hypothesis_id}] {hypothesis.title}  ({label})")
    typer.echo(f"Status:      {hypothesis.status.value}")
    if hypothesis.review is not None:
        when = hypothesis.review.decided_at.strftime("%Y-%m-%d %H:%M UTC")
        reason = f" — {hypothesis.review.reason}" if hypothesis.review.reason else ""
        typer.echo(f"Reviewed:    {when}{reason}")
    typer.echo(f"Claim:       {hypothesis.claim}")
    typer.echo(f"Rationale:   {hypothesis.rationale}")
    if hypothesis.supporting_paper_ids:
        typer.echo(f"Supported by:    {', '.join(hypothesis.supporting_paper_ids)}")
    if hypothesis.contradicting_paper_ids:
        typer.echo(f"Contradicted by: {', '.join(hypothesis.contradicting_paper_ids)}")
    for observation in hypothesis.repository_observations:
        typer.echo(f"  repo observation: {observation}")
    impact = hypothesis.expected_impact
    typer.echo(f"Impact:      {impact.metric or 'unspecified metric'} ({impact.direction.value})")
    typer.echo(f"Feasibility: {hypothesis.feasibility.value}")
    typer.echo(f"Effort:      {hypothesis.estimated_effort.value}")
    if hypothesis.estimated_experiment_count is not None:
        typer.echo(f"Experiments: ~{hypothesis.estimated_experiment_count}")
    typer.echo(f"Novelty:     {hypothesis.novelty_confidence.value} (not established)")
    typer.echo(f"Experiment:  {hypothesis.proposed_experiment}")
    for limitation in hypothesis.limitations:
        typer.echo(f"  limitation: {limitation}")


@hypotheses_app.command("import")
def import_command(
    file: Annotated[Path, typer.Argument(help="Hypotheses artifact (YAML or JSON).")],
    json_output: JsonOption = False,
) -> None:
    """Validate and import a hypotheses artifact."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("No project found. Run `researchforge project create` first.")
            raise typer.Exit(code=1)
        result = import_hypotheses(conn, file, project.id, load_settings())
        count = len(list_hypotheses(conn)) if result.ok else 0
    if result.ok:
        from researchforge.analytics.service import record_event

        record_event("hypotheses_imported")
    echo_import_result(
        result.errors,
        result.warnings,
        f"{count} hypothesis(es) imported. Next: researchforge report build",
        json_output,
    )


@hypotheses_app.command("list")
def list_command(json_output: JsonOption = False) -> None:
    """List stored hypotheses."""
    with closing(open_project_db()) as conn:
        hypotheses = list_hypotheses(conn)
    if json_output:
        typer.echo(json.dumps([h.model_dump(mode="json") for h in hypotheses], indent=2))
        return
    if not hypotheses:
        typer.echo("No hypotheses imported yet. See `researchforge research context`.")
        return
    for hypothesis in hypotheses:
        label = hypothesis.evidence_status.upper()
        citations = len(hypothesis.supporting_paper_ids)
        review = REVIEW_MARK.get(hypothesis.status, "")
        typer.echo(
            f"{hypothesis.hypothesis_id}  [{label}, {citations} citation(s), "
            f"{hypothesis.feasibility.value} feasibility]{review}  {hypothesis.title}"
        )
    rejected = [h for h in hypotheses if h.status is HypothesisStatus.REJECTED]
    if rejected:
        typer.echo("")
        typer.echo(
            f"{len(rejected)} rejected hypothesis(es) will be skipped by "
            "`experiment plan` and by `autorun`."
        )


@hypotheses_app.command("approve")
def approve_command(
    hypothesis_ids: Annotated[
        list[str], typer.Argument(help="One or more ids, e.g. hyp-001 hyp-003")
    ],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Optional note recorded with the approval."),
    ] = "",
    json_output: JsonOption = False,
) -> None:
    """Approve hypotheses for planning, without the interactive review."""
    _review_many(hypothesis_ids, HypothesisStatus.APPROVED, reason, json_output)


@hypotheses_app.command("reject")
def reject_command(
    hypothesis_id: Annotated[str, typer.Argument(help="e.g. hyp-002")],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why this hypothesis is not worth testing."),
    ],
    json_output: JsonOption = False,
) -> None:
    """Reject a hypothesis, so planning and autorun skip it."""
    _review_many([hypothesis_id], HypothesisStatus.REJECTED, reason, json_output)


def _review_many(
    hypothesis_ids: list[str],
    decision: ReviewOutcome,
    reason: str,
    json_output: bool,
) -> None:
    """Record one decision across several ids, reporting unknown ones as errors."""
    reviewed: list[Hypothesis] = []
    unknown: list[str] = []
    with closing(open_project_db()) as conn:
        for hypothesis_id in dict.fromkeys(hypothesis_ids):
            outcome = record_review(conn, hypothesis_id, decision, reason)
            if outcome is None:
                unknown.append(hypothesis_id)
            else:
                reviewed.append(outcome)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "decision": decision.value,
                    "reviewed": [h.hypothesis_id for h in reviewed],
                    "unknown": unknown,
                },
                indent=2,
            )
        )
    else:
        for hypothesis in reviewed:
            typer.echo(f"{hypothesis.hypothesis_id} {decision.value}: {hypothesis.title}")
        for hypothesis_id in unknown:
            typer.echo(f"Unknown hypothesis id: {hypothesis_id}.", err=True)
    if unknown:
        raise typer.Exit(code=1)


@hypotheses_app.command("review")
def review_command(json_output: JsonOption = False) -> None:
    """Walk the unreviewed hypotheses one at a time, approving or rejecting each.

    Only hypotheses nobody has judged yet are offered, so re-running this after
    an interruption picks up where it left off rather than asking again about
    decisions already made.
    """
    if json_output:
        typer.echo(
            "`hypotheses review` is interactive. Use `hypotheses approve`/`reject` "
            "with --json instead.",
            err=True,
        )
        raise typer.Exit(code=1)

    with closing(open_project_db()) as conn:
        pending = [h for h in list_hypotheses(conn) if h.status is HypothesisStatus.SPECULATIVE]
    if not pending:
        typer.echo("Nothing to review — every hypothesis has been approved or rejected.")
        return

    typer.echo(f"{len(pending)} hypothesis(es) to review. Ctrl-C stops; decisions already")
    typer.echo("made are kept.\n")

    approved, rejected, skipped = 0, 0, 0
    with closing(open_project_db()) as conn:
        for index, hypothesis in enumerate(pending, start=1):
            typer.echo(f"── {index}/{len(pending)} " + "─" * 48)
            _print_hypothesis(hypothesis)
            choice = (
                typer.prompt("\n[a]pprove / [r]eject / [s]kip", default="s", show_default=True)
                .strip()
                .lower()
            )
            if choice.startswith("a"):
                record_review(conn, hypothesis.hypothesis_id, HypothesisStatus.APPROVED)
                approved += 1
                typer.echo("→ approved\n")
            elif choice.startswith("r"):
                why = typer.prompt("  reason", default="").strip()
                record_review(conn, hypothesis.hypothesis_id, HypothesisStatus.REJECTED, why)
                rejected += 1
                typer.echo("→ rejected\n")
            else:
                skipped += 1
                typer.echo("→ left unreviewed\n")

    typer.echo(f"{approved} approved · {rejected} rejected · {skipped} left unreviewed")
    if rejected:
        typer.echo("Rejected hypotheses are skipped by `experiment plan` and `autorun`.")


@hypotheses_app.command("show")
def show_command(
    hypothesis_id: Annotated[str, typer.Argument(help="e.g. hyp-001")],
    json_output: JsonOption = False,
) -> None:
    """Show one hypothesis in full."""
    with closing(open_project_db()) as conn:
        hypothesis = get_hypothesis(conn, hypothesis_id)
    if hypothesis is None:
        typer.echo(f"Unknown hypothesis id: {hypothesis_id}. See `researchforge hypotheses list`.")
        raise typer.Exit(code=1)
    if json_output:
        echo_model(hypothesis)
    else:
        _print_hypothesis(hypothesis)


@hypotheses_app.command("generate")
def generate_command(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: anthropic|google|openai. Auto-detected from env when omitted.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override model name."),
    ] = None,
    no_import: Annotated[
        bool,
        typer.Option("--no-import", help="Write YAML files but do not import them."),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Generate landscape + hypotheses from stored papers using a built-in AI provider.

    Works without Claude Code or Cursor — just set an API key:
      export ANTHROPIC_API_KEY=sk-ant-...
      export GEMINI_API_KEY=...
      export OPENAI_API_KEY=sk-...

    ResearchForge auto-detects the provider; use --provider to be explicit.
    """
    from researchforge.ai.service import resolve_provider, run_synthesis
    from researchforge.analytics.service import record_event as _record_event

    try:
        ai_provider = resolve_provider(provider_hint=provider, model_hint=model)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    if not json_output:
        typer.echo(f"Generating with {ai_provider.name}…")

    try:
        outcome = run_synthesis(ai_provider, do_import=not no_import)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "landscape_path": str(outcome.landscape_path),
                    "hypotheses_path": str(outcome.hypotheses_path),
                    "landscape_ok": outcome.landscape_result.ok,
                    "hypotheses_ok": outcome.hypotheses_result.ok,
                    "errors": outcome.landscape_result.errors + outcome.hypotheses_result.errors,
                    "warnings": (
                        outcome.landscape_result.warnings + outcome.hypotheses_result.warnings
                    ),
                },
                indent=2,
            )
        )
        return

    if no_import:
        typer.echo(f"Wrote {outcome.landscape_path}")
        typer.echo(f"Wrote {outcome.hypotheses_path}")
        return

    if outcome.ok:
        _record_event("hypotheses_imported")
        typer.echo(f"✓ Wrote {outcome.landscape_path}")
        typer.echo(f"✓ Wrote {outcome.hypotheses_path}")
        typer.echo("✓ Landscape and hypotheses imported.")
        typer.echo("Next: researchforge contract approve")
    else:
        if not outcome.landscape_result.ok:
            typer.echo("✗ Landscape import failed:")
            for err in outcome.landscape_result.errors:
                typer.echo(f"  {err}")
        if not outcome.hypotheses_result.ok:
            typer.echo("✗ Hypotheses import failed:")
            for err in outcome.hypotheses_result.errors:
                typer.echo(f"  {err}")
        raise typer.Exit(code=1)

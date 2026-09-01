"""`researchforge autorun` — the autonomous research loop command."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Annotated

import typer

from researchforge.autorun.engine import (
    AutorunConfig,
    AutorunResult,
    PlanPreview,
    target_progress_pct,
)
from researchforge.storage.db import open_project_db
from researchforge.utils.output import JsonOption

SEPARATOR = "─" * 60


def autorun_command(
    stall: Annotated[
        int,
        typer.Option(
            "--stall",
            min=1,
            help="Stop a plan after N consecutive non-improving experiments.",
        ),
    ] = 2,
    global_stall: Annotated[
        int,
        typer.Option(
            "--global-stall",
            min=1,
            help="Stop the loop after N rounds with no improvement anywhere.",
        ),
    ] = 3,
    max_rounds: Annotated[
        int | None,
        typer.Option("--max-rounds", min=1, help="Hard cap on synthesis rounds."),
    ] = None,
    max_hours: Annotated[
        float | None,
        typer.Option("--max-hours", min=0.1, help="Wall-clock limit — for overnight runs."),
    ] = None,
    target: Annotated[
        float | None,
        typer.Option(
            "--target",
            help="Stop when the primary metric reaches this value "
            "(defaults to the contract's objective.primary_metric.target_value).",
        ),
    ] = None,
    compound: Annotated[
        bool,
        typer.Option(
            "--compound/--no-compound",
            help="Each round's experiments build on a node of the experiment graph "
            "instead of the baseline (default: on).",
        ),
    ] = True,
    explore: Annotated[
        float,
        typer.Option(
            "--explore",
            min=0.0,
            help="UCB1 exploration constant. 0 always expands the current best; "
            "higher values revisit under-explored branches instead.",
        ),
    ] = 0.0,
    merge: Annotated[
        bool,
        typer.Option(
            "--merge/--no-merge",
            help="Each round, try combining two independent winners into one "
            "multi-parent experiment. When their diffs overlap the AI is "
            "asked to author the combination as a single patch.",
        ),
    ] = False,
    observe: Annotated[
        bool,
        typer.Option(
            "--observe/--no-observe",
            help="After each experiment, have the AI read its benchmark output and "
            "record one paragraph on what the run showed. Costs one AI call "
            "per experiment.",
        ),
    ] = False,
    resynthesize: Annotated[
        bool,
        typer.Option(
            "--resynthesize/--no-resynthesize",
            help="Generate new hypotheses from results each round (default: on).",
        ),
    ] = True,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="AI provider: anthropic|google|openai."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override the AI model name."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Continue the interrupted loop in .researchforge/autorun.json.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Unattended — skip the first-batch approval prompt."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the move the loop would make next and exit. No AI, nothing runs.",
        ),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Run the autonomous research loop until it stalls, hits the target, or times out.

    Each round plans every pending hypothesis, runs the plans with a per-plan
    stall, records what happened in `.researchforge/research-log.md`, and
    re-synthesizes new hypotheses grounded in the measured results.

    The first batch needs your typed approval; after that the loop runs
    unattended. `--yes` skips that prompt for a true overnight run:

      export ANTHROPIC_API_KEY=sk-ant-...
      researchforge autorun --target 0.85 --max-hours 8 --yes

    Interrupted with Ctrl-C? `researchforge autorun --resume` continues with the
    same stall counter and time budget.

    `--dry-run` asks the loop where it would go next — which node it would
    expand and which hypotheses it would try there — without planning, running,
    or calling an AI provider. That is how a loop driven from Claude Code or
    Cursor keeps ResearchForge's search instead of guessing at one:

      researchforge autorun --dry-run --json
    """
    from researchforge.autorun.engine import AutorunDeclined, run_autorun
    from researchforge.storage.contract_repository import get_active_contract

    messages: list[str] = []

    def on_progress(message: str) -> None:
        if json_output:
            messages.append(message)
        else:
            typer.echo(message)

    with closing(open_project_db()) as conn:
        contract = get_active_contract(conn)
        contract_target = contract.spec.objective.primary_metric.target_value if contract else None
        config = AutorunConfig(
            stall=stall,
            global_stall=global_stall,
            max_rounds=max_rounds,
            max_hours=max_hours,
            target_value=target if target is not None else contract_target,
            compound=compound,
            explore=explore,
            merge=merge,
            observe=observe,
            resynthesize=resynthesize,
            yes=yes,
            provider=provider,
            model=model,
        )

        if dry_run:
            _print_next_move(conn, config, json_output)
            return

        if not json_output:
            _print_header(config, resume)

        try:
            result = run_autorun(
                conn,
                config,
                on_progress=on_progress,
                gate=None if json_output else _typed_approval,
                resume=resume,
            )
        except AutorunDeclined:
            typer.echo("Not approved — nothing ran.")
            raise typer.Exit(code=1) from None
        except RuntimeError as exc:
            typer.echo(f"Autorun blocked: {exc}", err=True)
            raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(_as_payload(result, messages), indent=2))
        return

    _print_result(result)
    _print_next_steps(result)


def _print_next_move(conn: sqlite3.Connection, config: AutorunConfig, json_output: bool) -> None:
    """`--dry-run`: where the loop would go next, and the command to take it there."""
    from researchforge.autorun.engine import preview_next_move

    try:
        move = preview_next_move(conn, config)
    except RuntimeError as exc:
        typer.echo(f"Autorun blocked: {exc}", err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(move.as_payload(), indent=2))
        return

    typer.echo(move.summary)
    if move.needs_resynthesis:
        typer.echo("  Every hypothesis has been tried everywhere it can apply.")
    elif move.retreat:
        typer.echo(
            "  Only branches that gained nothing have moves left — this measures an "
            "idea without the gains already banked."
        )
    for hypothesis_id, title in move.hypotheses:
        typer.echo(f"  {hypothesis_id}  {title}")
    typer.echo(f"\nNext: {move.command}")


def _typed_approval(preview: PlanPreview) -> bool:
    """The one human gate in an autorun: approve the first autonomous batch."""
    typer.echo(
        f"\n{preview.plan_id} → {preview.hypothesis_id} · "
        f"{len(preview.experiments)} experiment(s) · "
        f"worst case ~{preview.worst_case_minutes} min"
    )
    for experiment in preview.experiments:
        files = ", ".join(experiment.changed_files) or "env overrides only"
        typer.echo(f"  {experiment.experiment_id}  {experiment.title}  [{files}]")
    typer.echo("\nApproving lets the loop run this batch and every later round unattended.")
    confirmation: str = typer.prompt("Type 'approve' to start the autonomous loop")
    return confirmation.strip().lower() == "approve"


def _as_payload(result: AutorunResult, messages: list[str]) -> dict[str, object]:
    return {
        "rounds": len(result.rounds),
        "resumed_from_round": result.resumed_from_round,
        "total_experiments": result.total_experiments,
        "stop_reason": result.stop_reason,
        "metric_name": result.metric_name,
        "baseline_value": result.baseline_value,
        "best_experiment_id": result.best_experiment_id,
        "best_metric_value": result.best_metric_value,
        "target_value": result.target_value,
        "objective_achieved": result.objective_achieved,
        "total_duration_seconds": result.total_duration_seconds,
        "round_summaries": [
            {
                "round": summary.round_num,
                "hypotheses": summary.hypotheses_planned,
                "experiments_run": summary.experiments_run,
                "promising": summary.promising,
                "rejected": summary.rejected,
                "failed": summary.failed,
                "improved": summary.improved_over_previous,
                "best_metric_value": summary.best_metric_value,
            }
            for summary in result.rounds
        ],
        "messages": messages,
    }


def _print_header(config: AutorunConfig, resume: bool) -> None:
    typer.echo(SEPARATOR)
    typer.echo("  ResearchForge — autonomous research loop")
    typer.echo(SEPARATOR)
    if resume:
        typer.echo("  Mode:            resuming .researchforge/autorun.json")
    typer.echo(f"  Per-plan stall:  {config.stall} consecutive non-improvements")
    typer.echo(f"  Global stall:    {config.global_stall} rounds without improvement → stop")
    if config.max_rounds:
        typer.echo(f"  Max rounds:      {config.max_rounds}")
    if config.max_hours:
        typer.echo(f"  Time limit:      {config.max_hours}h")
    if config.target_value is not None:
        typer.echo(f"  Target metric:   {config.target_value}")
    if config.compound and config.explore > 0:
        compound_text = f"on — UCB1 node selection (explore {config.explore})"
    elif config.compound:
        compound_text = "on — builds on the current best"
    else:
        compound_text = "off — every experiment starts from the baseline"
    resynth_text = "on — new hypotheses each round" if config.resynthesize else "off"
    approval_text = "skipped (--yes)" if config.yes else "typed, once, before round 1"
    typer.echo(f"  Compound:        {compound_text}")
    if config.merge:
        typer.echo("  Merge:           on — combines independent winners")
    if config.observe:
        typer.echo("  Observe:         on — AI reads each run's output")
    typer.echo(f"  Re-synthesize:   {resynth_text}")
    typer.echo(f"  Approval:        {approval_text}")
    typer.echo(SEPARATOR)


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _print_result(result: AutorunResult) -> None:
    typer.echo(f"\n{SEPARATOR}")
    typer.echo("  Autorun complete")
    typer.echo(SEPARATOR)
    typer.echo(f"  Rounds:          {len(result.rounds)}")
    typer.echo(f"  Experiments:     {result.total_experiments}")
    typer.echo(f"  Duration:        {_format_duration(result.total_duration_seconds)}")
    typer.echo(f"  Stopped:         {result.stop_reason}")

    if result.improved and result.best_experiment_id:
        typer.echo(f"\n  ✓ Best:          {result.best_experiment_id}")
        if result.best_metric_value is not None:
            line = f"     {result.metric_name} = {result.best_metric_value:.4f}"
            if result.baseline_value:
                delta = (
                    (result.best_metric_value - result.baseline_value)
                    / abs(result.baseline_value)
                    * 100
                )
                line += f"  ({delta:+.1f}% vs baseline {result.baseline_value:.4f})"
            typer.echo(line)
        if result.objective_achieved:
            typer.echo("     Objective achieved.")
        elif result.target_value is not None and result.best_metric_value is not None:
            pct = target_progress_pct(
                result.best_metric_value,
                result.baseline_value or 0.0,
                result.target_value,
                result.direction,
            )
            typer.echo(f"     Target {result.target_value}: {pct:.0f}% of the way there")
    else:
        typer.echo("\n  ✗ Nothing beat the baseline.")
    typer.echo(SEPARATOR)
    typer.echo("  Log: .researchforge/research-log.md")


def _print_next_steps(result: AutorunResult) -> None:
    if result.improved:
        typer.echo("\nNext steps:")
        typer.echo("  researchforge results show       # ranked experiment history")
        typer.echo("  researchforge dashboard --open   # the experiment graph")
        typer.echo("  researchforge validate <run-id>  # re-run the winner N times")
        typer.echo("  researchforge ship branch        # clean branch on the baseline")
        return
    if result.stop_reason.startswith("time limit"):
        typer.echo("\nResume where it stopped:  researchforge autorun --resume")
        return
    typer.echo(
        "\nNo improvement found. Options:\n"
        "  researchforge research search --force   # widen the paper set\n"
        "  researchforge research synthesize       # new hypotheses from the papers\n"
        "  researchforge autorun --explore 0.5     # revisit under-explored branches\n"
        "  researchforge autorun --no-compound     # start every experiment fresh"
    )

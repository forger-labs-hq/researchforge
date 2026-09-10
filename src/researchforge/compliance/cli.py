"""`researchforge compliance` sub-app."""

from __future__ import annotations

import getpass
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from researchforge.compliance.engine import run_compliance
from researchforge.compliance.policy import load_policy
from researchforge.compliance.ship_eligibility import can_ship
from researchforge.storage.compliance_repository import (
    get_approval,
    get_compliance_runs,
    insert_approval,
    insert_compliance_run,
)
from researchforge.storage.db import open_project_db
from researchforge.storage.experiment_repository import get_experiment
from researchforge.utils.output import JsonOption, echo_json

compliance_app = typer.Typer(
    name="compliance",
    no_args_is_help=True,
    help="Policy, security, and governance gates for experiments.",
)


def _get_project_id(conn) -> str:  # type: ignore[no-untyped-def]
    row = conn.execute("SELECT id FROM projects ORDER BY created_at LIMIT 1").fetchone()
    return row["id"] if row else "unknown"


# ── check ─────────────────────────────────────────────────────────────────────


@compliance_app.command("check")
def check_command(
    experiment_id: Annotated[
        str | None,
        typer.Argument(help="Experiment ID, e.g. exp-003.  Defaults to latest."),
    ] = None,
    checkpoint: Annotated[
        str,
        typer.Option(
            "--checkpoint",
            help="Checkpoint to evaluate: post_experiment | pre_validation | pre_ship.",
        ),
    ] = "post_experiment",
    json_output: JsonOption = False,
) -> None:
    """Run compliance gates for an experiment and display results."""
    with closing(open_project_db()) as conn:
        project_id = _get_project_id(conn)
        policy = load_policy()

        if experiment_id is None:
            from researchforge.storage.experiment_repository import list_experiments

            experiments = list_experiments(conn)
            if not experiments:
                typer.echo("No experiments found.", err=True)
                raise typer.Exit(1)
            experiment_id = experiments[-1].experiment_id

        experiment = get_experiment(conn, experiment_id)
        if experiment is None:
            typer.echo(f"Experiment {experiment_id!r} not found.", err=True)
            raise typer.Exit(1)

        cr = run_compliance(
            project_id=project_id,
            experiment_id=experiment_id,
            checkpoint=checkpoint,  # type: ignore[arg-type]
            changed_files=experiment.changed_files,
            worktree=str(Path.cwd()),
            repo_root=str(Path.cwd()),
            policy=policy,
            conn=conn,
        )
        insert_compliance_run(conn, project_id, cr)

    if json_output:
        echo_json(cr.model_dump())
        return

    typer.echo(
        f"\nCompliance check — {experiment_id}  "
        f"[checkpoint: {checkpoint}]  "
        f"[policy: {'present' if policy.present else 'default'}]\n"
    )
    for result in cr.results:
        icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "–", "error": "!"}.get(
            result.status, "?"
        )
        typer.echo(
            f"  {icon}  [{result.gate_id}] {result.name:<35}  "
            f"{result.status.upper():<6}  ({result.severity})"
        )
        if result.status in ("fail", "error") and result.remediation:
            typer.echo(f"       Remediation: {result.remediation}")

    typer.echo()
    if cr.passed:
        typer.echo("  COMPLIANCE PASSED")
    else:
        typer.echo(f"  COMPLIANCE FAILED — {len(cr.blocking_failures)} blocking issue(s)")
        raise typer.Exit(1)


# ── report ────────────────────────────────────────────────────────────────────


@compliance_app.command("report")
def report_command(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID.")],
    json_output: JsonOption = False,
) -> None:
    """Show all compliance run history for an experiment."""
    with closing(open_project_db()) as conn:
        runs = get_compliance_runs(conn, experiment_id)
        approval = get_approval(conn, experiment_id)

    if json_output:
        echo_json(
            {
                "experiment_id": experiment_id,
                "runs": [r.model_dump() for r in runs],
                "approval": approval,
            }
        )
        return

    if not runs:
        typer.echo(f"No compliance runs recorded for {experiment_id}.")
        return

    for run in runs:
        typer.echo(
            f"\n── {run.compliance_run_id}  [{run.checkpoint}]  "
            f"{run.created_at.strftime('%Y-%m-%d %H:%M')}  "
            f"policy_hash={run.policy_hash[:12] or 'default'}"
        )
        for r in run.results:
            icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "–", "error": "!"}.get(
                r.status, "?"
            )
            typer.echo(f"    {icon}  [{r.gate_id}] {r.name}  {r.status.upper()}")

    if approval:
        typer.echo(f"\n  Approval: {approval['actor']} at {approval['created_at']}")
    else:
        typer.echo("\n  No approval recorded.")


# ── policy validate ───────────────────────────────────────────────────────────


@compliance_app.command("policy-validate")
def policy_validate_command(
    json_output: JsonOption = False,
) -> None:
    """Validate the project's policy.yaml (if present)."""
    policy = load_policy()
    if json_output:
        echo_json(
            {
                "present": policy.present,
                "path": policy.path,
                "hash": policy.raw_hash,
                "version": policy.spec.version,
            }
        )
        return

    if not policy.present:
        typer.echo("No policy.yaml found — running in community (non-blocking) mode.")
        return

    typer.echo(f"Policy loaded from {policy.path}")
    typer.echo(f"  Version : {policy.spec.version}")
    typer.echo(f"  Hash    : {policy.raw_hash[:16]}…")
    typer.echo(f"  Human approval required : {policy.spec.shipping.human_approval_required}")
    typer.echo("  Policy is valid.")


# ── approve ───────────────────────────────────────────────────────────────────


@compliance_app.command("approve")
def approve_command(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID to approve.")],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Short reason for approving."),
    ] = "",
    actor: Annotated[
        str | None,
        typer.Option("--actor", help="Approver identity.  Defaults to current user."),
    ] = None,
) -> None:
    """Record a human approval for an experiment."""
    actor = actor or getpass.getuser()
    with closing(open_project_db()) as conn:
        project_id = _get_project_id(conn)
        policy = load_policy()
        existing = get_approval(conn, experiment_id)
        if existing:
            typer.echo(f"Already approved by {existing['actor']} at {existing['approved_at']}.")
            return
        approval_id = insert_approval(
            conn,
            project_id=project_id,
            experiment_id=experiment_id,
            actor=actor,
            reason=reason,
            policy_hash=policy.raw_hash,
        )
    typer.echo(f"Recorded approval {approval_id} for {experiment_id} by {actor}.")


# ── eligibility ───────────────────────────────────────────────────────────────


@compliance_app.command("eligibility")
def eligibility_command(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID.")],
    json_output: JsonOption = False,
) -> None:
    """Show the canonical ship-eligibility decision for an experiment."""
    with closing(open_project_db()) as conn:
        experiment = get_experiment(conn, experiment_id)
        if experiment is None:
            typer.echo(f"Experiment {experiment_id!r} not found.", err=True)
            raise typer.Exit(1)

        from researchforge.domain.experiment import ExperimentStatus

        is_validated = experiment.status in (
            ExperimentStatus.VALIDATED,
            ExperimentStatus.IMPLEMENTATION_READY,
        )
        runs = get_compliance_runs(conn, experiment_id)
        policy = load_policy()
        decision = can_ship(
            experiment_id=experiment_id,
            is_technically_validated=is_validated,
            compliance_runs=runs,
            policy=policy,
            conn=conn,
        )

    if json_output:
        echo_json(
            {
                "experiment_id": experiment_id,
                "eligible": decision.eligible,
                "status": decision.status,
                "blocking_reasons": decision.blocking_reasons,
                "required_actions": decision.required_actions,
                "warnings": decision.warnings,
            }
        )
        return

    typer.echo(f"\nShip eligibility — {experiment_id}\n")
    for line in decision.summary_lines():
        typer.echo(f"  {line}")
    typer.echo()

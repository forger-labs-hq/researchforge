"""`researchforge contract` sub-app."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from researchforge.config.paths import contract_path
from researchforge.contract.service import (
    ContractError,
    approve_contract,
    check_contract_drift,
    generate_contract,
    validate_for_cli,
)
from researchforge.domain.contract import ExperimentContract
from researchforge.storage.contract_repository import get_active_contract
from researchforge.storage.db import open_project_db
from researchforge.storage.project_repository import get_project
from researchforge.utils.output import JsonOption, echo_import_result, echo_model

contract_app = typer.Typer(
    name="contract", no_args_is_help=True, help="Experiment contract (researchforge.yaml)."
)


def _default_contract_file(conn_project_path: str | None) -> Path:
    if conn_project_path:
        return contract_path(Path(conn_project_path))
    return contract_path()


def _print_summary(contract: ExperimentContract) -> None:
    spec = contract.spec
    typer.echo(f"Contract version: {contract.contract_version}")
    typer.echo(f"Approved at:      {contract.approved_at.isoformat()}")
    typer.echo(f"File sha256:      {contract.source_sha256[:16]}…")
    typer.echo(
        f"Baseline:         {spec.repository.baseline_ref} @ {contract.baseline_commit[:12]}"
    )
    typer.echo(
        f"Primary metric:   {spec.objective.primary_metric.name} "
        f"({spec.objective.primary_metric.direction.value})"
    )
    for constraint in spec.objective.hard_constraints:
        typer.echo(
            f"Hard constraint:  {constraint.name} {constraint.operator.value} {constraint.value}"
        )
    typer.echo(f"Execution mode:   {spec.execution.mode.value}")
    typer.echo(f"Trusted repo:     {spec.execution.trusted_repository}")
    typer.echo(f"Network:          {spec.network.mode.value}")
    if spec.secrets.forward_environment_variables:
        typer.echo(f"Forwarded env:    {', '.join(spec.secrets.forward_environment_variables)}")
    typer.echo(f"Editable paths:   {', '.join(spec.permissions.editable_paths)}")
    typer.echo(
        f"Protected paths:  {', '.join(spec.permissions.protected_paths) or '(implicit only)'}"
    )


@contract_app.command()
def generate(
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
    target_value: Annotated[
        float | None,
        typer.Option(
            "--target",
            help="Optional metric value that satisfies the objective — `autorun` stops there.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Generate a researchforge.yaml draft from the project and repository scan."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        target = _default_contract_file(project.repository.path if project else None)
        try:
            path = generate_contract(conn, output=target, force=force, target_value=target_value)
        except ContractError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps({"path": str(path)}, indent=2))
    else:
        typer.echo(f"Contract draft written to {path}")
        typer.echo("Edit it (especially execution.full_command), then run:")
        typer.echo("  researchforge contract validate")
        typer.echo("  researchforge contract approve")


@contract_app.command()
def validate(
    file: Annotated[Path | None, typer.Option("--file", help="Contract file to validate.")] = None,
    json_output: JsonOption = False,
) -> None:
    """Validate a researchforge.yaml (safe to run repeatedly; no side effects)."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        target = file or _default_contract_file(project.repository.path if project else None)
        result = validate_for_cli(conn, target)
    echo_import_result(
        result.errors,
        result.warnings,
        f"{target} is valid. Next: researchforge contract approve",
        json_output,
    )


@contract_app.command()
def approve(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the interactive confirmation.")] = False,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    json_output: JsonOption = False,
) -> None:
    """Approve the contract, freezing it as the next immutable version."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        if project is None:
            typer.echo("No project found. Run `researchforge project create` first.")
            raise typer.Exit(code=1)
        repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()
        target = file or contract_path(repo_root)

        validation = validate_for_cli(conn, target)
        if not validation.ok or validation.spec is None:
            echo_import_result(validation.errors, validation.warnings, "", json_output)
            return  # echo_import_result exits 1 on errors

        if not yes:
            spec = validation.spec
            typer.echo("You are about to approve this experiment contract:")
            typer.echo(
                f"  Primary metric:  {spec.objective.primary_metric.name} "
                f"({spec.objective.primary_metric.direction.value})"
            )
            for constraint in spec.objective.hard_constraints:
                typer.echo(
                    f"  Hard constraint: {constraint.name} {constraint.operator.value} "
                    f"{constraint.value}"
                )
            typer.echo(f"  Execution mode:  {spec.execution.mode.value}")
            typer.echo(f"  Trusted repo:    {spec.execution.trusted_repository}")
            typer.echo(f"  Network:         {spec.network.mode.value}")
            if spec.secrets.forward_environment_variables:
                typer.echo(
                    "  Forwarded env:   " + ", ".join(spec.secrets.forward_environment_variables)
                )
            protected_display = ", ".join(spec.permissions.protected_paths) or "(implicit only)"
            typer.echo(f"  Protected paths: {protected_display}")
            typer.echo(f"  Baseline ref:    {spec.repository.baseline_ref}")
            confirmation = typer.prompt("Type 'approve' to confirm")
            if confirmation.strip().lower() != "approve":
                typer.echo("Not approved.")
                raise typer.Exit(code=1)

        try:
            contract, created = approve_contract(conn, path=target, repo_root=repo_root)
        except ContractError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None

    if created:
        from researchforge.analytics.service import record_event

        record_event("contract_approved")
    if json_output:
        payload = contract.model_dump(mode="json")
        payload["created"] = created
        typer.echo(json.dumps(payload, indent=2))
    else:
        if created:
            typer.echo(f"Approved as contract version {contract.contract_version}.")
            typer.echo("Next: researchforge baseline run")
        else:
            typer.echo(f"Already approved as version {contract.contract_version} (file unchanged).")


@contract_app.command()
def show(json_output: JsonOption = False) -> None:
    """Show the active approved contract."""
    with closing(open_project_db()) as conn:
        project = get_project(conn)
        contract = get_active_contract(conn)
        drift = False
        if contract is not None and project is not None:
            repo_root = Path(project.repository.path) if project.repository.path else Path.cwd()
            drift = check_contract_drift(conn, contract_path(repo_root))

    if contract is None:
        typer.echo("No approved contract. Run `researchforge contract generate` to start.")
        raise typer.Exit(code=1)

    if json_output:
        echo_model(contract)
        return
    _print_summary(contract)
    if drift:
        typer.echo(
            "warning: researchforge.yaml has changed since approval — re-run "
            "`researchforge contract approve` to create the next version."
        )

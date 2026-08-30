"""`researchforge audit` sub-app."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from researchforge.audit.service import AuditTrail, build_trail
from researchforge.audit.trail import AuditEvent, AuditEventKind
from researchforge.storage.db import open_project_db
from researchforge.utils.output import JsonOption, echo_json, echo_model

audit_app = typer.Typer(
    name="audit",
    no_args_is_help=True,
    help="What this project did, and when — read back from its own records.",
)


def format_event(event: AuditEvent) -> str:
    when = event.at.strftime("%Y-%m-%d %H:%M:%S")
    return f"{when}  {event.kind.value:<21}  {event.summary}"


@audit_app.command("log")
def log_command(
    last: Annotated[
        int | None,
        typer.Option("--last", min=1, help="Show only the most recent N entries."),
    ] = None,
    kind: Annotated[
        AuditEventKind | None,
        typer.Option("--kind", help="Show only one kind of entry."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Show the project's history, oldest first."""
    with closing(open_project_db()) as conn:
        trail = build_trail(conn)

    events = trail.events
    if kind is not None:
        events = [event for event in events if event.kind is kind]
    if last is not None:
        events = events[-last:]

    if json_output:
        echo_model(AuditTrail(events=events, gate_findings=trail.gate_findings))
        return

    if not events:
        typer.echo("Nothing recorded yet. Run `researchforge status` to see where the project is.")
        return

    for event in events:
        typer.echo(format_event(event))

    shown, total = len(events), len(trail.events)
    if shown != total:
        typer.echo(f"\nShowing {shown} of {total} entries.")
    for finding in trail.gate_findings:
        typer.echo(f"\n! {finding}", err=True)


@audit_app.command("export")
def export_command(
    output: Annotated[Path, typer.Argument(help="Destination, e.g. audit.json")],
    json_output: JsonOption = False,
) -> None:
    """Write the full history to a JSON file."""
    with closing(open_project_db()) as conn:
        trail = build_trail(conn)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(trail.model_dump_json(indent=2) + "\n", encoding="utf-8")

    if json_output:
        echo_json(
            {
                "path": str(output),
                "event_count": len(trail.events),
                "gate_findings": trail.gate_findings,
            }
        )
        return
    typer.echo(f"Wrote {len(trail.events)} entry(ies) to {output}")
